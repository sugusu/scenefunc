"""Audit public annotations and pinned Critic source without loading a model.

The public benchmark is evaluation-only. Source-derived label policies are not
measured frequencies in unavailable training data. Publisher image checksums
are not independent verification of image bytes or room identity.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict, deque
import hashlib
import itertools
import json
from pathlib import Path
import re
import tarfile
import time

from task_conditioned_critic import OriginalTask, with_original_task


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def calls_with_context_status(path: Path):
    root = ast.parse(path.read_text())
    factory = next(n for n in root.body if isinstance(n, ast.FunctionDef) and n.name == "make_reward_critic")
    result = []
    for node in ast.walk(factory):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in {"advantage_from_critic", "advantage_from_critic_with_legality"}:
            continue
        context_pos = 3 if name == "advantage_from_critic" else 4
        opaque = any(isinstance(x, ast.Starred) for x in node.args) or any(k.arg is None for k in node.keywords)
        explicit = len(node.args) > context_pos or any(k.arg == "user_text" for k in node.keywords)
        result.append({"callee": name, "line": node.lineno, "call": ast.unparse(node),
                       "explicit_custom_text_supplied": explicit, "opaque_expansion": opaque})
    if len(result) != 3 or any(x["opaque_expansion"] for x in result):
        raise ValueError("Reviewed source call topology changed; re-read before interpreting")
    return result


def native_default_prompt(source: Path, k: int) -> str:
    """Execute only reviewed literal constants + pure string formatter AST."""
    tree = ast.parse(source.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"DEFAULT_USER_TEXT", "_LABELS"} for t in node.targets):
            ast.literal_eval(node.value)
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "build_default_user_text":
            selected.append(node)
    if len(selected) != 3:
        raise ValueError("Native pure prompt definition changed")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"__builtins__": {"str": str, "int": int}}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["build_default_user_text"](k)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repository", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--archived-requests", type=Path, default=None,
                    help="Optional verified request cache; otherwise stream only the needed two archive blobs")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    bench_file = args.benchmark / "scene_otc_bench.json"
    checksums = {}
    for line in (args.benchmark / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split(None, 1)
        checksums[path.lstrip("*")] = digest
    if checksums.get("scene_otc_bench.json") != sha(bench_file):
        raise ValueError("Public annotation differs from publisher checksum")
    records = json.loads(bench_file.read_text())
    ids, slots, per_question, label_counts = set(), [], [], defaultdict(Counter)
    hash_questions = defaultdict(set)
    directed_edges = defaultdict(list)
    for record in records:
        if set(record) != {"id", "numimages", "images", "ranking_best_to_worst"}:
            raise ValueError("Public schema changed; re-read rather than silently ignoring fields")
        rid, k = record["id"], record["numimages"]
        if rid in ids or k not in (2, 4) or len(record["images"]) != k:
            raise ValueError("Invalid question identity/size")
        ids.add(rid)
        labels = "ABCD"[:k]
        ranking = record["ranking_best_to_worst"]
        if sorted(ranking) != sorted(labels):
            raise ValueError("Reference is not a strict full permutation")
        label_counts[k]["".join(ranking)] += 1
        mapping = {}
        for label, path in zip(labels, record["images"]):
            if path not in checksums:
                raise ValueError("Image path missing from publisher checksum list")
            digest = checksums[path]
            slots.append(digest)
            mapping[label] = digest
            hash_questions[digest].add(rid)
        for better, worse in itertools.combinations(ranking, 2):
            directed_edges[(mapping[better], mapping[worse])].append(rid)
        per_question.append({"id": rid, "candidate_count": k,
                             "original_task_field_present": False,
                             "annotation_schema_and_permutation_valid": True})

    parent = {rid: rid for rid in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for group in hash_questions.values():
        group = sorted(group)
        for rid in group[1:]:
            parent[find(rid)] = find(group[0])
    components = defaultdict(list)
    for rid in ids:
        components[find(rid)].append(rid)
    node_set, outgoing, indegrees = set(slots), defaultdict(set), Counter()
    for before, after in directed_edges:
        outgoing[before].add(after)
        indegrees[after] += 1
    queue, processed = deque(x for x in node_set if not indegrees[x]), 0
    while queue:
        x = queue.popleft(); processed += 1
        for y in outgoing[x]:
            indegrees[y] -= 1
            if not indegrees[y]: queue.append(y)

    train_source = args.source / "src/train_grpo.py"
    routing = calls_with_context_status(train_source)
    # Recover an actual old input, not a fabricated user preference.
    packets_file = args.repository / "runs/autodl/20261111_closed_loop_data/feedback_material_v2/offline_feedback_packets.json"
    packet = next(x for x in json.loads(packets_file.read_text()) if x["id"] == "furniture:0")
    # The old packet references an internal backend request. The independent
    # actor/host boundary uses another archived path and serialization. Resolve
    # its already-audited mapping explicitly, rather than guessing equivalence.
    ownership = json.loads((args.repository / "runs/autodl/20261111_tool_decision_ownership/actual_audit_v1/policy_decisions.json").read_text())
    boundary = next(x for x in ownership if x["packet"] == "furniture:0")
    request_path = boundary["request"]
    archive = args.repository / "runs/autodl/20261110_complete_native_runtime/evidence_solid_v2/actual_scene_evidence.tar.xz"
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    if h.hexdigest() != "82e203bed22abfe371afe672fca139fa3175670b32d84cd7c29640db9d7257a1":
        raise ValueError("Frozen archive changed")
    archive_manifest = None
    with tarfile.open(archive, "r|xz") as t:
        for member in t:
            if member.name == "paths.json":
                archive_manifest = json.load(t.extractfile(member))
                break
    if archive_manifest is None:
        raise ValueError("Missing archive manifest")
    request_hash = next(x["sha256"] for x in archive_manifest["files"] if x["path"] == request_path)
    if args.archived_requests is not None:
        requests = json.loads(args.archived_requests.read_text())
        request_input_mode = "verified_provided_cache"
    else:
        chosen = {x["sha256"]: x for x in archive_manifest["files"]
                  if x["path"] in {request_path, boundary["receipt"]}}
        requests = {}
        with tarfile.open(archive, "r|xz") as t:
            for member in t:
                digest = member.name.removeprefix("blobs/")
                if digest not in chosen:
                    continue
                row = chosen.pop(digest)
                data = t.extractfile(member).read()
                if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("Selected original request/receipt blob changed")
                requests[row["path"]] = json.loads(data)
                if not chosen:
                    break
        if chosen or len(requests) != 2:
            raise ValueError("Original request/receipt unavailable in frozen archive")
        request_input_mode = "direct_two_blob_archive_read_no_RAM_cache_required"
    request = requests[request_path]
    serialized = json.dumps(request).encode()
    if hashlib.sha256(serialized).hexdigest() != request_hash:
        raise ValueError("Archived request cache does not reproduce frozen request bytes")
    if requests[boundary["receipt"]]["result"]["raw"] != packet["raw_proposal"]:
        raise ValueError("Host and saved packet output no longer align")
    system = request["messages"][0]["content"]
    match = re.search(r"^Room Description: (.+)$", system, re.M)
    if not match:
        raise ValueError("Original task cannot be obtained without guessing")
    task = OriginalTask(request_id=request_path, text=match.group(1))
    base_prompt = native_default_prompt(args.source / "src_critic/grpo_advantage.py", 2)
    conditioned = with_original_task(base_prompt, task)
    (args.output / "native_generic_prompt.txt").write_text(base_prompt + "\n")
    (args.output / "ordinary_task_conditioned_prompt.txt").write_text(conditioned + "\n")
    write(args.output / "task_routing_example.json", {
        "source_packet": "furniture:0", "request_archive_path": request_path,
        "original_actor_request_sha256": request_hash,
        "different_internal_backend_request_reference": packet["original_request"],
        "request_boundary_resolution": "prior exact-response ownership mapping plus current host request byte hash and archive hash verification",
        "request_input_mode": request_input_mode,
        "original_task_text_sha256": task.sha256,
        "task_text_preserved_verbatim": task.text == json.loads(conditioned.split("\n")[-2])["original_request"],
        "model_or_ranker_invoked": False, "scene_or_label_produced": False,
    })
    write(args.output / "source_call_routing.json", routing)
    write(args.output / "benchmark_question_coverage.json", per_question)
    write(args.output / "declared_image_sharing_components.json", {
        "basis": "Publisher SHA256SUMS only; image files not downloaded or independently hashed",
        "components": sorted((sorted(x) for x in components.values()), key=lambda x: (-len(x), x)),
        "components_are_independent_rooms": False,
    })
    result = {
        "status": "metadata_source_and_prompt_preparation_complete",
        "benchmark_questions": len(records), "n2_questions": sum(x["numimages"] == 2 for x in records),
        "n4_questions": sum(x["numimages"] == 4 for x in records),
        "annotation_sha256_verified": sha(bench_file),
        "question_original_task_fields": 0,
        "candidate_image_slots": len(slots), "publisher_declared_unique_image_hashes": len(node_set),
        "publisher_declared_repeated_slots": len(slots) - len(node_set),
        "declared_candidate_sharing_components": len(components),
        "declared_largest_component_questions": max(map(len, components.values())),
        "reference_order_histograms": {str(k): dict(v) for k, v in label_counts.items()},
        "posthoc_listed_order_diagnostic": {
            str(k): {"correct": v["ABCD"[:k]], "total": sum(v.values()),
                     "accuracy": v["ABCD"[:k]] / sum(v.values()),
                     "model_used": False,
                     "interpretation": "Release-order label diagnostic, not spatial reasoning or paper-result reproduction"}
            for k, v in label_counts.items()
        },
        "declared_preference_edge_observations": sum(map(len, directed_edges.values())),
        "declared_preference_graph_acyclic": processed == len(node_set),
        "source_local_critic_call_sites": len(routing),
        "source_local_critic_call_sites_passing_custom_text": sum(x["explicit_custom_text_supplied"] for x in routing),
        "original_helper_supports_custom_text": True,
        "request_input_mode": request_input_mode,
        "source_code_is_evidence_of_all_paper_runtime_inputs": False,
        "inspected_input_source_hashes": {
            name: sha(args.source / name) for name in [
                "src/train_grpo.py", "src_critic/grpo_advantage.py",
                "src_critic/build_dataset.py", "src_critic/split_dataset.py",
                "src_critic/generate_image_grounded_reason.py",
                "src_critic/build_multitarget_pairwise.py", "tools/convert_otc_to_sft.py",
            ]
        },
        "training_dataset_acquired": False, "benchmark_used_as_training": False,
        "new_expert_labels": 0, "new_NN_forwards": 0, "new_model_calls": 0,
        "new_training_updates": 0, "new_complete_scene_successes": 0,
        "elapsed_seconds": time.monotonic() - start,
    }
    write(args.output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
