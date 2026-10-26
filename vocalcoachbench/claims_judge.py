"""Optional explicit-endpoint runner for the frozen corrected515 judge protocol.

Importing this module makes no requests. The ``run`` command requires an
endpoint, served judge model and an existing local tokenizer snapshot. Raw
attempts are preserved; unresolved judge failures never become model labels.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from .claims import (BUDGETS, _unique, _validate_outcome, digest,
                     parse_json_object, require, sha256, validate_judgments,
                     build_diagnosis_prompt, build_prescription_prompt)
from .io import read_jsonl
from .protocol import INVENTORY_ID_SHA256
import hashlib


def judge_request(job: dict, judge_model: str, budget: int) -> dict:
    """Preserve the original asymmetric diagnosis/prescription settings."""
    request = {"model": judge_model, "messages": job["messages"]}
    if job["phase"] == "diagnosis":
        request.update(temperature=0, max_tokens=budget, response_format={"type": "json_object"},
                       extra_body={"reasoning_effort": "low"})
    else:
        request.update(max_completion_tokens=budget, reasoning_effort="low")
    return request


def _save_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _token_count(tokenizer, job: dict) -> int:
    encoded = tokenizer.apply_chat_template(job["messages"], tokenize=True, return_dict=True,
        add_generation_prompt=True, reasoning_effort="low")
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        require(len(ids) == 1, "Expected one prompt, not a batch")
        ids = ids[0]
    require(isinstance(ids, (list, tuple)) and ids and all(isinstance(x, int) for x in ids), "Invalid tokenizer input_ids")
    return len(ids)


def _validate_job_population(jobs: dict[str, dict]) -> None:
    """Reject malformed or noncanonical jobs before tokenizer/model loading."""
    require(len(jobs) == 1030, "Expected the full 515-recording, two-phase job set")
    models = {job.get("model") for job in jobs.values()}
    require(len(models) == 1 and all(isinstance(m, str) and m for m in models), "Expected one source model")
    source_model = next(iter(models))
    by_sample = {}
    for job in jobs.values():
        sid, phase = job.get("sample_id"), job.get("phase")
        require(isinstance(sid, str) and sid.startswith(("A_", "B_")), "Expected full canonical sample_id")
        require(phase in BUDGETS and job.get("budgets") == BUDGETS[phase], "Judge phase/budget mismatch")
        require(job["id"] == f"{source_model}:{sid}:{phase}", "Judge ID disagrees with model/sample/phase")
        phases = by_sample.setdefault(sid, {})
        require(phase not in phases, f"Duplicate sample phase: {sid}/{phase}")
        phases[phase] = job
        for key in ("gt", "pred"):
            require(isinstance(job.get(key), list), f"Judge {key} must be a list")
            _unique(job[key], "id", f"judge {key}")
        require(len(job["pred"]) <= 5, "Judge model claims exceed five slots")
        require(all(isinstance(g.get("category"), list) and g["category"] and isinstance(g.get("text"), str)
                    for g in job["gt"]), "Invalid expert claim content/category")
        if phase == "diagnosis":
            messages = build_diagnosis_prompt("diagnosis", job["gt"], job["pred"])
        else:
            issues = [{"id": g["id"], "category": g["category"], "text": g["text"]} for g in job["gt"]]
            messages = build_prescription_prompt(job["pred"], issues)
        require(job.get("messages") == messages, "Frozen judge prompt mismatch")
    require(len(by_sample) == 515 and all(set(phases) == set(BUDGETS) for phases in by_sample.values()),
            "Expected 515 unique recordings with both diagnosis and correction jobs")
    identities = sorted((sid[2:], sid) for sid in by_sample)
    identity_hash = hashlib.sha256(json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(identity_hash == INVENTORY_ID_SHA256, "Judge samples do not match the canonical recording inventory")
    require(sum(len(phases["diagnosis"]["gt"]) for phases in by_sample.values()) == 6162, "Judge expert population mismatch")
    require(all(phases["diagnosis"]["gt"] == phases["correction"]["gt"] for phases in by_sample.values()),
            "Diagnosis and correction expert inputs differ")


def run_claim_judge(jobs_path: str | Path, outcomes_path: str | Path, *, endpoint: str,
                    model: str, tokenizer_path: str | Path, api_key_env: str = "OPENAI_API_KEY",
                    timeout: float = 240) -> dict:
    """Call a user-selected server; resume completed jobs without overwriting.

    Up to three full-context attempts per phase use the frozen token budgets.
    A second invocation resumes never-started jobs. Any unresolved outcome or
    orphan attempt stops resume so a new audited run must use a new output path.
    No server is launched and no model or tokenizer is downloaded.
    """
    parsed_url = urlparse(endpoint)
    require(parsed_url.scheme in {"http", "https"} and bool(parsed_url.netloc), "An explicit HTTP(S) endpoint is required")
    require(parsed_url.username is None and parsed_url.password is None, "Supply credentials through the API-key environment variable")
    require(bool(model), "An explicit served judge model is required")
    require(Path(tokenizer_path).is_dir(), "Provide an existing local judge tokenizer snapshot directory")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        require(parsed_url.hostname in {"localhost", "127.0.0.1", "::1"}, f"Set {api_key_env} for this remote endpoint")
        api_key = "EMPTY"
    jobs = _unique(read_jsonl(jobs_path), "id", "judge job")
    _validate_job_population(jobs)
    # Optional dependencies are imported only after explicit validated inputs.
    from openai import OpenAI
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    token_counts = {}
    for job in jobs.values():
        require(job.get("phase") in BUDGETS and job.get("budgets") == BUDGETS[job["phase"]], "Judge phase/budget mismatch")
        token_counts[job["id"]] = _token_count(tokenizer, job)
        require(token_counts[job["id"]] + max(job["budgets"]) <= 16384, f"Full-context budget overflow: {job['id']}")
    output = Path(outcomes_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"jobs_sha256": sha256(jobs_path), "endpoint": endpoint, "judge_model": model,
                "tokenizer_path": str(Path(tokenizer_path).resolve()), "context_limit": 16384,
                "tokenizer_file_sha256": {p.name: sha256(p) for p in Path(tokenizer_path).iterdir()
                                          if p.is_file() and p.suffix in {".json", ".jinja"}},
                "input_tokens": token_counts, "request_policy": "corrected515_strict_judge_v2"}
    metadata_path = output.with_suffix(".run.json")
    attempt_root = output.with_suffix(".attempts")
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if metadata_path.exists():
            require(json.loads(metadata_path.read_text(encoding="utf-8")) == metadata, "Judge run configuration changed; choose a new output path")
        else:
            require(not output.exists() and not attempt_root.exists(), "Existing outcomes lack a matching run manifest")
            _save_new(metadata_path, metadata)
        previous = _unique(read_jsonl(output), "id", "judge outcome") if output.exists() else {}
        for identity, outcome in previous.items():
            require(identity in jobs, f"Unexpected prior job: {identity}")
            _validate_outcome(outcome, jobs[identity])
        pending = [job for identity, job in jobs.items() if identity not in previous]
        for job in pending:
            require(not (attempt_root / digest(job["id"])).exists(),
                    f"Interrupted raw attempts exist for {job['id']}; preserve them and start a new audited output")
        client = OpenAI(base_url=endpoint, api_key=api_key, max_retries=0, timeout=timeout)
        unresolved = []
        for job in pending:
            result = {"id": job["id"], "job_sha256": digest(job), "attempts": []}
            if not job["pred"]:
                judgments = ([{"gt_id": g["id"], "label": "MISS", "matched_model_claim_ids": [],
                               "reason": "No model claims emitted."} for g in job["gt"]]
                             if job["phase"] == "diagnosis" else [])
                result.update(valid=True, no_model_claims=True, judgments=judgments)
            else:
                for index, budget in enumerate(job["budgets"]):
                    request = judge_request(job, model, budget)
                    attempt = {"request": request, "request_sha256": digest(request), "started_at": time.time()}
                    # This durable marker precedes the network request. A kill
                    # during the call must not turn it into a never-started job
                    # that resume would silently issue a second time.
                    _save_new(attempt_root / digest(job["id"]) / f"{index}.started.json",
                              {"job_id": job["id"], **attempt})
                    try:
                        response = client.chat.completions.create(**request)
                        attempt["response"] = response.model_dump(mode="json")
                        obj = parse_json_object(response.choices[0].message.content or "")
                        errors = validate_judgments(obj, job)
                        if response.choices[0].finish_reason != "stop":
                            errors.append("non_stop_finish_reason")
                        attempt.update(parsed=obj, validation_errors=errors)
                    except Exception as exc:
                        # Exception text may contain transport headers. Preserve
                        # the error type without copying credentials into logs.
                        attempt.update(validation_errors=["backend_exception"], error=type(exc).__name__)
                    attempt["finished_at"] = time.time()
                    _save_new(attempt_root / digest(job["id"]) / f"{index}.json", {"job_id": job["id"], **attempt})
                    result["attempts"].append(attempt)
                    if not attempt["validation_errors"]:
                        result.update(valid=True, judgments=attempt["parsed"]["judgments"])
                        break
                if not result.get("valid"):
                    result["valid"] = False
                    unresolved.append(job["id"])
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return {"status": "needs_judge_repair" if unresolved else "complete", "jobs": len(jobs),
                "completed": len(previous) + len(pending), "unresolved": unresolved,
                "outcomes_path": str(output.resolve()), "outcomes_sha256": sha256(output),
                "run_manifest_path": str(metadata_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    result = run_claim_judge(args.jobs, args.outcomes, endpoint=args.endpoint, model=args.model,
                            tokenizer_path=args.tokenizer, api_key_env=args.api_key_env)
    print(json.dumps(result, indent=2))
    if result["unresolved"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
