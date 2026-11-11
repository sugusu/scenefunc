"""Census the actual official OTC test annotations without model inference.

Image identity is based on the author's SHA256SUMS. Only the separately logged
two-file probe is byte-verified locally. Components are content-overlap blocks,
not inferred room IDs. The held-out benchmark never becomes training data.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

DATA_PIN = "e50788bdbe74644d98e62929cd24218c9110e812"
STAGE = "20261111_critic_conditioning_evidence"
DATA_HASH = "8afbe7f0213849466724f5d7398222c1f9b68ebe48ceb2c66e9201fe6a15e0c1"
SUM_HASH = "fe5f2e1431c7334ce2f8b5106f74ed94653060e15c49165db98759b5e612b37a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_without_reference(row):
    """Official benchmark observation only; never invent an instance purpose."""
    return dict(id=row["id"], labels=list("ABCD"[:row["numimages"]]), images=list(row["images"]),
        instruction="Rank the candidate scenes from best to worst by physical validity, functional correctness, and layout coherence.",
        per_instance_room_purpose=None, evidence_scope="official generic scene-quality test")


def permute_without_leakage(row, order):
    observation = input_without_reference(row)
    old_labels = observation["labels"]
    observation["images"] = [row["images"][i] for i in order]
    old_to_new = {old_labels[index]: old_labels[new_pos] for new_pos, index in enumerate(order)}
    # This reference is returned separately, not embedded in the observation.
    return observation, [old_to_new[label] for label in row["ranking_best_to_worst"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    repo, root, out = a.repo.resolve(), a.source_root.resolve(), a.output.resolve()
    assert out.is_relative_to(repo/"runs/autodl"/STAGE) or out.is_relative_to(Path("/dev/shm"))
    if out.exists(): raise FileExistsError(out)
    start = time.monotonic()
    assert json.loads((root/"otc_info.json").read_text())["sha"] == DATA_PIN
    source = root/"otc/scene_otc_bench.json"
    sums = root/"otc/SHA256SUMS"
    assert sha(source) == DATA_HASH and sha(sums) == SUM_HASH
    declared = {line.split(maxsplit=1)[1].lstrip("*./"): line.split()[0]
                for line in sums.read_text().splitlines() if line.strip()}
    assert declared["scene_otc_bench.json"] == DATA_HASH
    rows = json.loads(source.read_text())
    assert len(rows) == len({row["id"] for row in rows}) == 149
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        x, y = find(x), find(y)
        if x != y: parent[x] = y

    image_paths = []
    preferences = defaultdict(list)
    canonical_questions = defaultdict(list)
    permutation_checks = 0
    for row in rows:
        assert set(row) == {"id", "numimages", "images", "ranking_best_to_worst"}
        n = row["numimages"]
        assert n in (2,4) and len(row["images"]) == n
        assert sorted(row["ranking_best_to_worst"]) == list("ABCD"[:n])
        hashes = [declared[name] for name in row["images"]]
        assert len(set(hashes)) == n
        image_paths.extend(row["images"])
        for value in hashes: union(hashes[0], value)
        ranking = [hashes[ord(label)-65] for label in row["ranking_best_to_worst"]]
        canonical_questions[tuple(sorted(hashes))].append(dict(id=row["id"], ranking=ranking))
        for i, j in itertools.combinations(range(n), 2):
            pair = tuple(sorted((ranking[i], ranking[j])))
            preferences[pair].append(dict(id=row["id"], better=ranking[i], worse=ranking[j]))
        for order in itertools.permutations(range(n)):
            obs, reference = permute_without_leakage(row, order)
            assert "ranking_best_to_worst" not in obs
            assert [declared[obs["images"][ord(label)-65]] for label in reference] == ranking
            permutation_checks += 1

    blocks = defaultdict(list)
    for row in rows: blocks[find(declared[row["images"][0]])].append(row["id"])
    blocks = sorted((sorted(v) for v in blocks.values()), key=lambda values: values[0])
    block_ids = {qid: f"content_block_{i+1:02}" for i, ids in enumerate(blocks) for qid in ids}
    conflicts = [v for v in preferences.values() if len({x["better"] for x in v}) > 1]
    assert not conflicts  # Actual data has no repeated-pair directional contradiction.
    question_rows = [dict(id=row["id"], numimages=row["numimages"], content_block=block_ids[row["id"]],
        has_per_instance_purpose=False, has_raw_geometry=False, has_tool_history=False,
        labels_quarantined_from_observation=True) for row in rows]
    duplicate_questions = [dict(ids=[v["id"] for v in values], same_ranking=len({tuple(v["ranking"]) for v in values}) == 1)
                           for values in canonical_questions.values() if len(values) > 1]
    probe = json.loads((root/"duplicate_probe_result.json").read_text())
    for r in probe["files"]:
        assert r.get("matches_manifest") and sha(root/"otc"/r["path"]) == r["sha256"] == declared[r["path"]]
    assert probe["both_bytes_verified"]
    result = dict(kind="Actual held-out scene-quality benchmark evidence; no model or 3D execution",
        pin=DATA_PIN, annotation_sha256=DATA_HASH, sum_manifest_sha256=SUM_HASH,
        questions=len(rows), question_types=dict(Counter(row["numimages"] for row in rows)),
        unique_image_paths=len(set(image_paths)), author_declared_unique_image_hashes=len({declared[p] for p in image_paths}),
        content_overlap_blocks=len(blocks), questions_per_overlap_block=sorted([len(x) for x in blocks], reverse=True),
        content_blocks_are_not_claimed_room_count=True,
        per_instance_purpose_fields=0, raw_geometry_fields=0, tool_history_fields=0,
        expanded_pair_preferences=sum(len(v) for v in preferences.values()), unique_declared_image_pairs=len(preferences),
        repeated_pair_direction_conflicts=len(conflicts), exact_unordered_duplicate_questions=duplicate_questions,
        label_permutation_checks=permutation_checks, label_permutations_are_not_model_predictions=True,
        byte_verified_duplicate_probe=probe,
        interpretation="Suitable public generic scene-quality control; cannot alone identify purpose-conditioned preference reversal, and rows sharing exact image content are not independent content samples",
        held_out_not_used_for_training=True, author_image_manifest_not_fully_rehashed=True,
        new_model_calls=0, new_paid_calls=0, new_training_updates=0, new_scene_executions=0, seconds=time.monotonic()-start)
    out.mkdir(parents=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"content_blocks.json").write_text(json.dumps([dict(block=f"content_block_{i+1:02}", question_ids=ids) for i,ids in enumerate(blocks)],indent=2)+"\n")
    with (out/"question_review.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(question_rows[0]),lineterminator="\n");w.writeheader();w.writerows(question_rows)
    print(json.dumps(result))


if __name__ == "__main__": main()
