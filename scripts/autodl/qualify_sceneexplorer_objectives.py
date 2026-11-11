"""Run the inspected upstream layout scorer on its unchanged sample metadata.

No mesh generation, trajectory search, model forward or training is performed.
External source files stay in caller-provided storage and are hash checked.
Only the inspected scorer and angular-distance definitions are compiled; the
upstream main module (which can start workers and delete outputs) is not imported.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace as NS


PIN = "7f9e0e542d75b3f7a6d8b819b573a2d30a3179b3"
MAIN_HASH = "c0263062d9b089d8adee9790ee4b7a14c71eb6682d64d255b55d0cc5cc9e1f5c"
SAMPLE_HASH = "504934e14f4672128b1c444f23208d02ef7a119abfd046f62f127600ccced419"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def definition(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def score_components(mods, room):
    """Independent reference for the unchanged input only, not a search method."""
    from shapely.affinity import rotate, translate

    world = [translate(rotate(m.state_list[m.state].poly, -m.rotation,
                              origin=(0, 0), use_radians=True),
                       float(m.p[0]), float(m.p[1])) for m in mods]
    pairs = []
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            if world[i].intersects(world[j]):
                pairs.append(dict(i=i, j=j, first_key=mods[i].key,
                                  second_key=mods[j].key,
                                  intersection_area=float(world[i].intersection(world[j]).area)))
    blocked = {v for pair in pairs for v in (pair["i"], pair["j"])}
    blocking_area = sum(mods[i].state_list[mods[i].state].poly.area for i in blocked)
    return dict(reference_displacement_term=0.0,
                reference_blocking_area=float(blocking_area),
                reference_code_score=float(0.5 * blocking_area),
                paper_equation_2_at_unchanged_input=float(blocking_area),
                blocked_object_indices=sorted(blocked), intersects_pairs=pairs,
                note="Projected annotated polygons; intersects includes boundary contact. Not FCL or mesh penetration.")


def source_findings(root, tree, tree_paths):
    source = ast.parse((root / "sk.py").read_text())
    scorer = definition(source, "calculate_room_type_values")
    lower = [dict(line=n.lineno, expression=ast.unparse(n)) for n in ast.walk(scorer)
             if isinstance(n, ast.Attribute) and n.attr == "replace"
             and isinstance(n.value, ast.Attribute) and n.value.attr == "lower"]
    sub_values = [dict(line=n.lineno, context=type(n.ctx).__name__) for n in ast.walk(scorer)
                  if isinstance(n, ast.Name) and n.id == "sub_room_values"]
    writes = [n for n in ast.walk(scorer) if isinstance(n, ast.Subscript)
              and isinstance(n.value, ast.Name) and n.value.id == "sub_room_values"
              and isinstance(n.ctx, ast.Store)]
    grouping = definition(source, "calculate_tree_data")
    # Keep the code's L1 grouping separate from the manuscript's L-infinity test.
    norm3 = next(n for n in ast.walk(grouping) if isinstance(n, ast.FunctionDef) and n.name == "norm3")
    action = definition(tree, "action_2")
    sizes = [dict(line=n.lineno, statement=ast.unparse(n)) for n in ast.walk(action)
             if isinstance(n, ast.Assign) and any(isinstance(v, ast.Name) and v.id == "main_poly" for v in n.targets)
             and isinstance(n.value, ast.Attribute)]
    score_fields = sorted({n.slice.value for n in ast.walk(scorer)
                           if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                           and n.value.id == "obj" and isinstance(n.slice, ast.Constant)
                           and isinstance(n.slice.value, str)})
    deps = ["static/dataset/dynamic_objects_data.json", "dataset/occurrenceCount/res_ratio.json",
            "dataset/objCatListAliv2.json"]
    return dict(
        functionality_entrypoint=dict(path="sk.py", lines=[scorer.lineno, scorer.end_lineno],
                                      direct_object_keys=score_fields, user_brief_argument=False),
        required_prior_tables=[dict(path=p, present_in_pinned_tree=p in tree_paths,
                                    locally_fetched=(root / p).exists()) for p in deps],
        lower_method_attribute_not_call=lower,
        prior_subvector_references=sub_values,
        prior_subvector_index_writes=len(writes),
        prior_subvector_scope="Initialized to zero, read in the prior bonus, no indexed updates in this function. Static finding, not a run of that branch.",
        grouping=dict(norm3_lines=[norm3.lineno, norm3.end_lineno],
                      code_distance="sum of absolute component differences (L1)",
                      paper_distance="L-infinity in section5", standard_equivalence=False),
        action_size_assignments=sizes,
        findings_scope="Pinned public code only. No patches applied; this does not quantify paper-result errors or imply all paths fail.",
        full_functionality_pipeline_executed=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    start = time.monotonic()
    repo, root, out = a.repo.resolve(), a.source_root.resolve(), a.output.resolve()
    stage = repo / "runs/autodl/20261111_sceneexplorer_objectives"
    if not (out.is_relative_to(stage) or out.is_relative_to(Path("/dev/shm"))):
        raise ValueError("Write this study or RAM only")
    if out.exists():
        raise FileExistsError(out)
    manifest_path = root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["pin"] != PIN:
        raise ValueError("Source revision changed")
    by_path = {v["path"]: v for v in manifest["files"]}
    for name, digest in [("layoutmethods/sceneexplorer/main.py", MAIN_HASH),
                         ("layoutmethods/sceneexplorer/scenes/sample3.json", SAMPLE_HASH)]:
        if by_path[name]["sha256"] != digest:
            raise ValueError("Frozen input changed: " + name)
    for rel, entry in by_path.items():
        if sha(root / rel) != entry["sha256"]:
            raise ValueError("Downloaded file hash mismatch: " + rel)
    from shapely.geometry import Polygon
    from shapely.affinity import rotate, translate
    import shapely
    import numpy as np

    source_path = root / "layoutmethods/sceneexplorer/main.py"
    tree = ast.parse(source_path.read_text())
    selected = [definition(tree, n) for n in ("angle_diff", "evaluate_layout")]
    isolated = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected,
                          type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = dict(pi=math.pi, rotate=rotate, translate=translate)
    exec(compile(isolated, str(source_path), "exec"), namespace)
    scene = json.loads((root / "layoutmethods/sceneexplorer/scenes/sample3.json").read_text())
    room = scene["rooms"][0]
    shape = Polygon(room["roomShape"])
    space = NS(boundary=shape, xl=shape.bounds[0], yl=shape.bounds[1],
               xh=shape.bounds[2], yh=shape.bounds[3])
    mods, inventory = [], []
    for obj in room["objList"]:
        if obj["format"] != "glb":
            continue
        relative = f'layoutmethods/sceneexplorer/object/{obj["modelId"]}_data.json'
        if relative not in by_path:
            raise ValueError("Unmanifested module metadata: " + relative)
        module = json.loads((root / relative).read_text())
        scale = np.array(obj["scale"], dtype="float32")
        states = [NS(name=s["name"], poly=Polygon([(q[0] * scale[0], q[1] * scale[2]) for q in s["poly"]]))
                  for s in module["data"]]
        initial = next(i for i, v in enumerate(states) if v.name == obj["startState"])
        mods.append(NS(name=obj["modelId"], key=obj["key"], state=initial, state_list=states,
                       p=np.array([obj["translate"][0], obj["translate"][2]], dtype="float32"),
                       rotation=obj["orient"] % (2 * math.pi)))
        inventory.append(dict(key=obj["key"], model_id=obj["modelId"], state=obj["startState"],
                              state_names=[v.name for v in states],
                              annotation_area=float(states[initial].poly.area),
                              metadata_sha256=by_path[relative]["sha256"]))
    score = float(namespace["evaluate_layout"](mods, mods, space))
    reference = score_components(mods, room)
    if not math.isclose(score, reference["reference_code_score"], rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError("Independent unchanged-input score differs from original function")
    git_tree = json.loads((root / "tree.json").read_text())
    if git_tree["pin"] != PIN or git_tree["tree"]["truncated"]:
        raise ValueError("Incomplete or changed source tree")
    findings = source_findings(root, tree, {f["path"] for f in git_tree["tree"]["tree"]})
    after = {p: sha(root / p) for p in by_path}
    if after != {p: v["sha256"] for p, v in by_path.items()}:
        raise RuntimeError("Upstream input changed")
    out.mkdir(parents=True)
    result = dict(kind="One unchanged author sample, original projected-layout scorer and independent reference",
                  pin=PIN, input_sha256=SAMPLE_HASH, inventory=inventory,
                  module_count=len(mods), architectural_entries=len(room["objList"]) - len(mods),
                  annotated_state_combinations=math.prod(len(m.state_list) for m in mods),
                  state_combinations_executed=1, original_score=score, reference=reference,
                  modules_or_positions_edited=0, independent_reference_agrees=True,
                  collision_free_full_scene_claim=False, new_scene_generations=0,
                  motion_planning_runs=0, mesh_loads=0, training_updates=0,
                  model_forwards=0, paid_API_calls=0, neural_cache_tests=0,
                  numpy_version=np.__version__, shapely_version=shapely.__version__,
                  source_files_verified=len(by_path), source_files_unchanged=True,
                  seconds=time.monotonic() - start)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (out / "source_findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n")
    with (out / "inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory)
    print(json.dumps({k: v for k, v in result.items() if k not in ("inventory", "reference")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
