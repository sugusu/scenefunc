"""Export source-bound review records to the pinned official critic/QA formats.

This adapter validates data relationships, not professional judgments. It emits
unqualified episodes; only a separately reviewed manifest may admit training.
"""
import copy
import json
from pathlib import Path

from decision_state import canonical, object_table, read_checkpoint, sha

VIEWS = ("topdown", "south", "east", "west", "north")


def checked_ref(root, ref):
    root = Path(root).resolve()
    path = (root / ref["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Reference escapes experiment root")
    if not path.is_file() or sha(path) != ref["sha256"]:
        raise ValueError("Missing or changed referenced file: " + ref["path"])
    return path


def candidate_observation(root, candidate):
    checkpoint = checked_ref(root, candidate["checkpoint"])
    state = read_checkpoint(checkpoint)
    receipt_path = checked_ref(root, candidate["render_receipt"])
    receipt = json.loads(receipt_path.read_text())
    scene = checked_ref(root, candidate["scene"])
    if scene != checkpoint.parent / "partial_scene.glb":
        raise ValueError("Scene/checkpoint directory mismatch")
    if Path(receipt["source_scene"]).resolve() != scene or receipt["source_sha256"] != sha(scene):
        raise ValueError("Render receipt belongs to another scene")
    if receipt.get("status") != "completed" or receipt.get("render_exit") != 0:
        raise ValueError("Render did not complete")
    expected = receipt["views"]
    if len(expected) != 5 or len(candidate["views"]) != 5:
        raise ValueError("Expected five saved candidate views")
    images = []
    for view, ref, actual in zip(VIEWS, candidate["views"], expected):
        path = checked_ref(root, ref)
        if path != Path(actual["path"]).resolve() or ref["sha256"] != actual["sha256"]:
            raise ValueError("View disagrees with original render receipt")
        if not path.name.endswith("." + view + ".png"):
            raise ValueError("Candidate views are out of order")
        images.append(str(path.relative_to(Path(root).resolve())))
    return {"candidate_id": candidate["candidate_id"], "room_id": state["room_id"],
            "objects": object_table(state), "images": images,
            "native_ID_annotations": receipt["native_ID_annotations"],
            "physics_validated": receipt["physics_validated"]}


def validate_judgment(judgment, candidate_ids):
    verdict = judgment["verdict"]
    groups = judgment.get("ranking_groups", [])
    if verdict not in ("strict_preference", "tie_or_partial_order", "insufficient_evidence"):
        raise ValueError("Unknown review verdict")
    if verdict == "insufficient_evidence":
        if groups or not judgment.get("unknowns"):
            raise ValueError("Insufficient evidence requires unknowns and no ranking")
    else:
        if not groups or any(not isinstance(g, list) or not g for g in groups):
            raise ValueError("Ranking groups must be nonempty lists")
        flattened = [x for group in groups for x in group]
        if sorted(flattened) != sorted(candidate_ids):
            raise ValueError("Ranking must cover each candidate exactly once")
        if verdict == "strict_preference" and any(len(g) != 1 for g in groups):
            raise ValueError("Strict preference cannot encode a tie")
        if verdict == "tie_or_partial_order" and all(len(g) == 1 for g in groups):
            raise ValueError("A tie verdict requires at least one tied group")
    if not judgment.get("claims") or not judgment.get("evidence"):
        raise ValueError("Review needs explicit claims and evidence references")
    evidence_ids = [e["evidence_id"] for e in judgment["evidence"]]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("Duplicate review evidence IDs")
    for claim in judgment["claims"]:
        if claim["candidate_id"] not in candidate_ids or not claim.get("activity"):
            raise ValueError("Review claim has unknown candidate or no activity")
        if not claim.get("object_ids") or not claim.get("assessment") or not claim.get("evidence_ids"):
            raise ValueError("Claim needs objects, assessment and evidence")
        if not set(claim["evidence_ids"]) <= set(evidence_ids):
            raise ValueError("Claim refers to missing evidence")


def export_review(record, root, order=None):
    candidates = record["candidates"]
    ids = [c["candidate_id"] for c in candidates]
    if len(ids) not in (2, 3, 4) or len(set(ids)) != len(ids):
        raise ValueError("Review requires two to four distinct candidate IDs")
    order = ids if order is None else list(order)
    if sorted(order) != sorted(ids):
        raise ValueError("Candidate display order is not a permutation")
    common_start = checked_ref(root, record["comparison_start"])
    for candidate in candidates:
        if checked_ref(root, candidate["comparison_start"]) != common_start:
            raise ValueError("Candidates do not share a recorded comparison start")
    brief_path = checked_ref(root, record["brief_source"])
    if json.loads(brief_path.read_text())["brief"] != record["original_brief"]:
        raise ValueError("Original requirement text changed")
    observations = {c["candidate_id"]: candidate_observation(root, c) for c in candidates}
    if len({o["room_id"] for o in observations.values()}) != 1:
        raise ValueError("Candidates belong to different rooms")
    judgment = record["judgment"]
    validate_judgment(judgment, ids)
    initial_ids = {o["object_id"] for o in object_table(read_checkpoint(common_start))}
    for claim in judgment["claims"]:
        current_ids = {o["object_id"] for o in observations[claim["candidate_id"]]["objects"]}
        if not set(claim["object_ids"]) <= initial_ids | current_ids:
            raise ValueError("Review claim names an object absent from initial and candidate states")
    for evidence in judgment["evidence"]:
        checked_ref(root, evidence["artifact"])
    slots = {cid: "ABCD"[i] for i, cid in enumerate(order)}
    strict = judgment["verdict"] == "strict_preference"
    images, layout = [], []
    for cid in order:
        obs = observations[cid]
        selected = obs["images"][:1] if strict else obs["images"]
        first = len(images)
        images.extend(selected)
        layout.append({"slot": slots[cid], "candidate_id": cid,
            "image_indices_zero_based": list(range(first, len(images))),
            "view_order": ["topdown"] if strict else list(VIEWS),
            "objects": obs["objects"], "native_ID_annotations": obs["native_ID_annotations"],
            "physics_validated": obs["physics_validated"]})
    prompt = (record["original_brief"] + "\n\nReview question:\n" + record["question"] +
              "\n\nCandidate images and current objects (display order):\n" + canonical(layout))
    provenance = {"kind": record["origin"], "review_id": record["review_id"],
                  "project_id": record["project_id"], "family_id": record["family_id"],
                  "training_eligible": False, "qualification": "requires_separate_review",
                  "display_order": order, "candidate_slots": slots}
    # Do not place privileged labels, reasons or evidence annotations in prompt.
    reasoning = canonical({"claims": judgment["claims"], "unknowns": judgment.get("unknowns", [])})
    if strict:
        return {"stage": "critic", "prompt": prompt,
            "images": images, "reasoning": reasoning,
            "answer": "".join(slots[g[0]] for g in judgment["ranking_groups"]),
            "provenance": provenance}
    answer = {"verdict": judgment["verdict"], "ranking_groups": judgment["ranking_groups"],
              "claims": judgment["claims"], "unknowns": judgment.get("unknowns", [])}
    return {"stage": "qa", "system": "Assess only the supplied evidence. Preserve ties and uncertainty.",
        "provenance": provenance,
        "turns": [{"role": "user", "content": prompt, "image": images},
                  {"role": "assistant", "content": "", "answer": canonical(answer)}]}


def decode_review_answer(answer_text, candidate_slots):
    """Consumer contract: uncertainty is a distinct result, never a preference."""
    import re
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", answer_text, flags=re.DOTALL)
    if not match:
        raise ValueError("Missing review answer tag")
    text = match.group(1).strip()
    ids = list(candidate_slots)
    if text.startswith("{"):
        value = json.loads(text)
        # Evidence file references live in the original reviewed record.
        validation = copy.deepcopy(value)
        refs = sorted({ref for claim in value.get("claims", []) for ref in claim.get("evidence_ids", [])})
        validation["evidence"] = [{"evidence_id": ref} for ref in refs]
        validate_judgment(validation, ids)
        return value
    inverse = {slot: cid for cid, slot in candidate_slots.items()}
    if len(inverse) != len(ids) or sorted(text) != sorted(inverse):
        raise ValueError("Malformed or incomplete strict ranking")
    return {"verdict": "strict_preference", "ranking_groups": [[inverse[x]] for x in text]}
