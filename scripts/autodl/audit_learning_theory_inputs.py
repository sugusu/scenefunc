"""Read the actual prepared corpus and state what its loss can certify.

Standard-library, read-only audit. No model, tokenizer, source-package import,
new labels, optimizer, simulator, or execution of downloaded code. This is an
objective/data audit, not a learning or scene-generation result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def audit(repo: Path) -> dict:
    root = repo / "runs/autodl/20261111_native_phase_batch_bridge/material_v4_2"
    paths = [root / "prospective_native_phase_v4.jsonl.gz",
             root / "prospective_native_baselines_v4.jsonl.gz"]
    before = {str(p.relative_to(repo)): sha(p) for p in paths}
    rows_by_arm = defaultdict(list)
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                weights = row["token_objective_weights"]
                ids = row["target_token_ids"]
                origins = row["source_token_origins"]
                if len(weights) != len(ids) or len(ids) != len(origins):
                    raise ValueError("Source target/weight/origin lengths disagree")
                if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
                    raise ValueError("Nonfinite/negative/empty source weights")
                if not any(w > 0 for w in weights):
                    raise ValueError("Source component has no positive objective mass")
                rows_by_arm[row["arm"]].append(row)

    arms = []
    for arm, rows in sorted(rows_by_arm.items()):
        positive = [w for r in rows for w in r["token_objective_weights"] if w > 0]
        # A loss-measure signature excludes the condition/prompt on purpose.
        # Equal signatures prove coefficient equality, NOT equal conditionals.
        label_measure = []
        for row in rows:
            for token, weight, origin in zip(row["target_token_ids"],
                                              row["token_objective_weights"],
                                              row["source_token_origins"]):
                if weight > 0:
                    label_measure.append([row["unit_id"], token, weight, origin])
        label_measure.sort(key=canonical)
        family_tokens = Counter()
        for row in rows:
            family_tokens[row["family_id"]] += sum(w > 0 for w in row["token_objective_weights"])
        arms.append({
            "arm": arm,
            "components": len(rows),
            "unique_action_units": len({r["unit_id"] for r in rows}),
            "source_families": len(family_tokens),
            "stage_counts": dict(Counter(r["stage"] for r in rows)),
            "positive_target_tokens": len(positive),
            "zero_weight_tokens": sum(w == 0 for r in rows for w in r["token_objective_weights"]),
            "positive_weight_sum": math.fsum(positive),
            "positive_weight_min": min(positive),
            "positive_weight_max": max(positive),
            "max_to_min_positive_weight_ratio": max(positive) / min(positive),
            "family_positive_target_tokens": dict(sorted(family_tokens.items())),
            "label_coefficient_measure_sha256": hashlib.sha256(canonical(label_measure)).hexdigest(),
            "components_marked_training_admitted": sum(r.get("training_admitted") is True for r in rows),
            "components_with_images_encoding_checked_flag": sum(r.get("image_encoding_checked") is True for r in rows),
        })

    matched = [x for x in arms if not x["arm"].endswith(":action_direct")]
    measure_equal = len({x["label_coefficient_measure_sha256"] for x in matched}) == 1
    if not measure_equal:
        raise AssertionError("Prepared matched arms no longer preserve the same source-label coefficients")
    after = {str(p.relative_to(repo)): sha(p) for p in paths}
    if before != after:
        raise RuntimeError("Source corpus changed during audit")
    return {
        "audit_kind": "existing_prepared_loss_measure_and_theorem_assumptions",
        "source_hashes": before,
        "sources_unchanged": True,
        "arms": arms,
        "matched_arms_same_positive_label_coefficient_measure": measure_equal,
        "not_inferred": [
            "Equal source-token coefficients do not mean equal input prefixes or compute",
            "These prepared-file flags are not a replacement for later multimodal input checks",
            "No complete expert trajectory distribution or its terminal utility is identified by these rows",
            "No population excess cross entropy, trajectory KL, or scene-success bound was measured",
            "Weighted target-token CE is not automatically the unweighted trajectory KL chain rule",
            "Training admission, scene success and independent preference labels are not granted by this audit",
        ],
        "model_forwards": 0,
        "model_generations": 0,
        "optimizer_updates": 0,
        "new_scene_trials": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    allowed = repo / "runs/autodl/20261111_book_evolution_mechanisms"
    ram = Path("/dev/shm")
    if not (output.is_relative_to(allowed) or output.is_relative_to(ram)):
        raise ValueError("Write only to this new result directory or RAM")
    if output.exists():
        raise FileExistsError("Preserve existing audit outputs; select a new path")
    result = audit(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "arms": len(result["arms"]),
                      "sources_unchanged": True, "model_forwards": 0,
                      "matched_label_coefficient_measures_equal": True}))


if __name__ == "__main__":
    main()
