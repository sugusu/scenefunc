"""Read and reconcile an existing native trial; no generation or SDK replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(repo):
    root = repo / "runs/autodl/20261110_complete_native_runtime"
    source = root / "trial_actual_v2/result.json"
    summary = root / "summary.json"
    raw = source.read_bytes()
    total = json.loads(raw)
    legacy = total["model_ledger"]
    rows = legacy["forwards"]
    if legacy["incomplete_forward_accounting"]:
        raise ValueError("Historical generation accounting is incomplete")
    by_role = {}
    for role in sorted({r["role"] for r in rows}):
        subset = [r for r in rows if r["role"] == role]
        for row in subset:
            if row["output_tokens"] != len(row["output_token_ids"]):
                raise ValueError("Saved token-length disagreement")
            if row["status"] != "returned":
                raise ValueError("Do not silently omit a failed historical request")
        by_role[role] = dict(generate_requests=len(subset),
            input_tokens=sum(r["input_tokens"] for r in subset),
            output_tokens=sum(r["output_tokens"] for r in subset),
            generation_boundary_wall_seconds=math.fsum(r["seconds"] for r in subset))
    claimed = json.loads(summary.read_bytes())["corrected_continuation"]
    request_count = sum(x["generate_requests"] for x in by_role.values())
    input_count = sum(x["input_tokens"] for x in by_role.values())
    output_count = sum(x["output_tokens"] for x in by_role.values())
    if (request_count != claimed["forwards"] or
            output_count != claimed["output_tokens"] or
            output_count != legacy["output_tokens"]):
        raise ValueError("Historical aggregate disagrees with original records")
    if (by_role["native_joint_policy"]["generate_requests"] != claimed["policy_forwards"] or
            by_role["native_tool"]["generate_requests"] != claimed["tool_VLM_forwards"]):
        raise ValueError("Historical role split changed")
    return dict(
        source_files=[dict(path=str(p.relative_to(repo)), bytes=p.stat().st_size,
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in (source, summary)],
        scope="One existing frozen partial native generation. Not a new trial, kernel-forward count, "
              "learned comparison or complete scene success.",
        legacy_field_forwards_means="admitted model.generate requests, not each neural forward within decoding",
        original_status=total["status"], by_role=by_role,
        generate_requests=request_count, input_tokens=input_count, output_tokens=output_count,
        generation_boundary_wall_seconds=math.fsum(r["seconds"] for r in rows),
        native_tool_attempts=total["native_tool_attempts"],
        native_tool_attempts_are_not_model_generations=True,
        old_host_already_counted_policy_and_tool_generations=True,
        shared_old_model_objects=len({r["model_object_id"] for r in rows}),
        all_old_parameters_frozen=all(r["all_parameters_frozen"] for r in rows),
        new_8B_model_or_role_qualification=False,
        new_model_generate_requests=0, optimizer_updates=0,
        original_source_still_matches_before=hashlib.sha256(source.read_bytes()).hexdigest()
                                              == hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep existing result; choose a new path")
    result = audit(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"generate_requests": result["generate_requests"],
                      "native_tool_attempts": result["native_tool_attempts"],
                      "new_model_generate_requests": 0}))
