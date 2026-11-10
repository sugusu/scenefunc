#!/usr/bin/env python3
"""Audit existing conditional design supervision without inventing labels.

Enumerates every supplied-object-table match in the fixed24-unit corpus, then
checks what else changed. Reuses the already-published ordinary edit/stop gate.
No model, geometry query, edit, preference annotation or data augmentation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path

from audit_declared_stop_gate import decide


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_observation(row):
    users = [x for x in row["messages"] if x["role"] == "user"]
    if len(users) != 1 or not isinstance(users[0]["content"], list):
        raise ValueError("Unexpected observation schema")
    parts = users[0]["content"]
    text = "\n".join(x.get("text", "") for x in parts if x.get("type") == "text")
    marker = "CURRENT NATIVE STATE:\n"
    if text.count(marker) != 1:
        raise ValueError("State marker must occur exactly once")
    start = text.index(marker) + len(marker)
    state, length = json.JSONDecoder().raw_decode(text[start:])
    # Preserve all non-state text, including original task/history/tool receipts.
    outer = text[:start] + "<STATE>" + text[start + length:]
    return state, outer, [x["image"] for x in parts if x.get("type") == "image"]


def parse_decision(row):
    if row is None:
        return None
    raw = row["target_text"].split("<think>", 1)[1].split("</think>", 1)[0].strip()
    return json.loads(raw)


def audit(source):
    rows = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    actions = [x for x in rows if x["arm"] == "paired_phase_v3" and x["stage"] == "action"]
    decisions = {x["unit_id"]: x for x in rows
                 if x["arm"] == "paired_phase_v3" and x["stage"] == "decision"}
    if len(actions) != 24 or len({x["unit_id"] for x in actions}) != 24:
        raise ValueError("Fixed development corpus changed; review before continuing")
    internal, public = [], []
    negative_flags = []
    family_sources = defaultdict(set)
    for action in actions:
        state, outer, images = parse_observation(action)
        d = parse_decision(decisions.get(action["unit_id"]))
        application = d.get("principle_applicability") if d else None
        source_id = application.get("source_decision_id") if application else None
        if application:
            rejected = any(application.get(k) is False for k in
                           ("applicable", "is_applicable", "applies")) or application.get("status") == "not_applicable"
            if rejected:
                negative_flags.append(action["unit_id"])
                raise ValueError("New explicit rejection record: extend classification after review")
        elif d and action["target_kind"] != "terminal_control":
            raise ValueError("Unrecognized nonterminal decision schema")
        kind = "asserted_source_application" if source_id else "scoped_stop_record" if d else "D_missing"
        # This policy receives only public pre-action state. It never sees the
        # reference call, unit/family ID, rationale target or assistant_prefix.
        gate = decide(state)
        expected = "stop_with_scoped_unknowns" if action["target_kind"] == "terminal_control" else "edit_required"
        row = {
            "unit_id": action["unit_id"], "family_id": action["family_id"],
            "D_kind": kind, "source_decision_id": source_id,
            "supplied_objects_sha256": digest(state["objects"]),
            "public_state_sha256": digest(state),
            "object_count": len(state["objects"]),
            "target_action_sha256": hashlib.sha256(action["target_text"].encode()).hexdigest(),
            "reference_action_kind": action["target_kind"],
            "gate": gate,
            "gate_label_agreement": gate["decision"] == expected if gate["decision"] != "unknown" else None,
            "oracle_assistant_prefix_excluded_from_gate": True,
            "action_reference_excluded_from_gate": True,
            "full_room_success": "unknown",
        }
        internal.append((action, state, outer, images, row))
        public.append(row)
        if source_id:
            family_sources[action["family_id"]].add(source_id)
    pairs = []
    for a, b in combinations(internal, 2):
        ar, sa, oa, ia, ra = a
        br, sb, ob, ib, rb = b
        if ra["supplied_objects_sha256"] != rb["supplied_objects_sha256"]:
            continue
        changed = sorted(k for k in sa.keys() | sb.keys() if sa.get(k) != sb.get(k))
        ma, mb = sa.get("available_measurement"), sb.get("available_measurement")
        narrow = (bool(changed) and set(changed) <= {"current_activity_request", "current_requested_mode"}
                  and oa == ob and ia == ib and ar["messages"][0] == br["messages"][0])
        pairs.append({
            "units": [ra["unit_id"], rb["unit_id"]],
            "supplied_objects_sha256": ra["supplied_objects_sha256"],
            "changed_state_top_level_fields": changed,
            "system_messages_equal": ar["messages"][0] == br["messages"][0],
            "non_state_user_text_equal": oa == ob,
            "ordered_image_references_equal": ia == ib,
            "actual_image_bytes_or_encoding_checked": False,
            "public_measurements_equal": ma == mb if ma is not None and mb is not None else None,
            "condition_only_supplied_observation_pair": narrow,
            "gate_decisions": [ra["gate"]["decision"], rb["gate"]["decision"]],
            "gate_label_agreement": [ra["gate_label_agreement"], rb["gate_label_agreement"]],
            "full_room_success": "unknown",
            "boundary": "Equality is of supplied object descriptions and stated observation references, not restored simulator/asset-version/RNG equality or independent preference evidence",
        })
    kinds = Counter(x["D_kind"] for x in public)
    eligible = [x for x in public if x["gate_label_agreement"] is not None]
    source_to_families = defaultdict(set)
    for family, sources in family_sources.items():
        for source_id in sources:
            source_to_families[source_id].add(family)
    return {
        "source_sha256": file_digest(source),
        "source_units": len(public), "source_families": len({x["family_id"] for x in public}),
        "D_kinds": dict(kinds),
        "family_to_asserted_source_ids": {k: sorted(v) for k, v in family_sources.items()},
        "each_asserted_source_occurs_in_one_family": all(len(v) == 1 for v in source_to_families.values()),
        "explicit_source_inapplicability_labels": len(negative_flags),
        "negative_label_schema_checked": ["applicable=false", "is_applicable=false", "applies=false", "status=not_applicable"],
        "inapplicability_boundary": "STOP is not relabelled as source inapplicability. Only observed schema contents are counted; no inferred negative labels added",
        "all_supplied_object_table_pairs": pairs,
        "pair_search_space": len(public) * (len(public) - 1) // 2,
        "condition_only_pair_count": sum(x["condition_only_supplied_observation_pair"] for x in pairs),
        "ordinary_gate": {"supported": len(eligible), "agreement": sum(x["gate_label_agreement"] for x in eligible),
                          "unknown": len(public) - len(eligible),
                          "boundary": "Reuses an existing development rule and public proxy measurements; not a new independent result or a coordinate/scene-generation policy"},
        "rows": public,
        "interpretation": "Source family and chosen principle remain confounded in these eight asserted applications. Limited condition-sensitive edit/STOP pairs exist, but ordinary public measurements already explain these labels. No impossibility-of-learning claim",
        "new_training_labels": 0, "model_forwards": 0, "new_geometry_queries": 0,
        "new_scene_edits": 0, "API_calls": 0, "optimizer_updates": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = audit(a.source)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["source_units", "D_kinds", "pair_search_space", "condition_only_pair_count", "ordinary_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
