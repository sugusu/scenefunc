"""Run original SceneCritic ontology/count code on its 21 public inventories.

Inventories are input annotations, not generated scenes. The ordinary pooled
counter below prevents reuse of an observed instance across alias requirements;
it cannot verify role, appearance, geometry, or complete task success.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PIN="99f287acbfbf3970e90718895eeb10c87462dbf8"
SOURCE_HASH="fc34c607945e43752f65ad37270290c7d5beca69b4b3c82f690a8c3bfcf288dd"
ONTO_HASH="8a400c91f80ca61fbd466bb74b96f6afbc58004073d5e74903151f81de6a0e7d"
STAGE="20261111_symbolic_design_baseline"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_inventory_score(expected, actual, normalize):
    """Ordinary unit-capacity category matching; not a new method."""
    exp,obs=Counter(),Counter()
    for cat,n in expected.items():exp[normalize(cat)]+=n
    for cat,n in actual.items():obs[normalize(cat)]+=n
    categories=sorted(set(exp)|set(obs));matched=sum(min(exp[c],obs[c]) for c in categories)
    missing=sum(max(0,exp[c]-obs[c]) for c in categories)
    extra=sum(max(0,obs[c]-exp[c]) for c in categories)
    denominator=sum(exp.values())+extra
    return dict(matched=matched,missing=missing,extra=extra,
                checked=denominator,score=matched/denominator if denominator else None,
                scope="canonical inventory only; raw subcategory/role/geometry not verified")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--source-root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();repo,root,out=a.repo.resolve(),a.source_root.resolve(),a.output.resolve()
    assert out.is_relative_to(repo/"runs/autodl"/STAGE) or out.is_relative_to(Path("/dev/shm"))
    if out.exists():raise FileExistsError(out)
    start=time.monotonic()
    assert json.loads((root/"tree.json").read_text())["sha"]==PIN
    source=root/"source/SceneCritic/SceneCritic_evaluator.py"
    assert sha(source)==SOURCE_HASH and sha(root/"SceneOnto.json")==ONTO_HASH
    # Confirm only inspected standard-library/NumPy dependencies are imported at
    # module level. main is not invoked; no external runtime is imported.
    allowed={"__future__","json","math","os","re","copy","abc","dataclasses","enum","pathlib","typing","numpy"}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.Import):assert all(x.name.split('.')[0] in allowed for x in node.names)
        elif isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in allowed
    spec=importlib.util.spec_from_file_location("original_scenecritic_baseline",source)
    original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
    ontology=original.load_ontology(str(root/"SceneOnto.json"))
    assert len(ontology)==79
    task_manifest=json.loads((root/"task_manifest.json").read_text())
    for row in task_manifest:assert sha(root/"source"/row["path"])==row["sha256"]
    rows=[];label_rows=[];all_cats=Counter();covered=full_dims=orientation_sample_support=0
    for row in sorted(task_manifest,key=lambda x:x["path"]):
        task=json.loads((root/"source"/row["path"]).read_text())
        cfg=original.build_config_from_task(task);actual=Counter(task["assets"])
        # This is the declared inventory projected to the original checker's
        # category interface. No coordinates or successful layout are invented.
        objects=[SimpleNamespace(obj_id=f"{cat}:{i}",category=original.normalize_category(cat))
                 for cat,n in actual.items() for i in range(n)]
        checked=original.CompletenessVerifier().check(objects,None,ontology,cfg)
        pooled=canonical_inventory_score(task["assets"],actual,original.normalize_category)
        aliases=defaultdict(list)
        for cat,n in task["assets"].items():
            c=original.normalize_category(cat);aliases[c].append(cat);all_cats[c]+=n;e=ontology.get(c)
            dimensions=bool(e and all(v is not None for v in [e.dim_width_min,e.dim_width_max,e.dim_depth_min,e.dim_depth_max,e.dim_height_min,e.dim_height_max]))
            orient=bool(e and (e.orientation_rel.back_to_wall_n_samples>=10 or e.orientation_rel.faces_center_n_samples>=10 or any(v["n_samples"]>=10 for v in e.orientation_rel.faces_pair.values())))
            if e:covered+=n
            if dimensions:full_dims+=n
            if orient:orientation_sample_support+=n
            label_rows.append(dict(task=row["path"],raw_category=cat,count=n,canonical_category=c,
                ontology_entry_present=bool(e),full_dimension_bands_present=dimensions,
                orientation_statistics_have_min_samples=orient,
                sample_support_not_asserted_actual_rule_applicability=True))
        collision={k:v for k,v in aliases.items() if len(v)>1}
        rows.append(dict(task=row["path"],declared_instances=sum(actual.values()),
            original_checked=checked.n_checked,original_passed=checked.n_passed,original_score=checked.score,
            original_domain=cfg.get("domain"),ordinary_pooled_result=pooled,
            alias_requirement_groups=collision,original_violations=[v.message for v in checked.violations],
            actual_scene_generated=False,role_semantics_verified=False))
    assert len(rows)==21 and sum(x["declared_instances"] for x in rows)==217
    nonperfect=[x for x in rows if x["original_score"]!=1.0]
    assert len(nonperfect)==1 and nonperfect[0]["task"].endswith('classroom_2.json')
    assert nonperfect[0]["original_checked"]==20 and nonperfect[0]["original_passed"]==15
    assert all(x["ordinary_pooled_result"]["score"]==1 for x in rows)
    result=dict(kind="Original symbolic ontology loading and declared-inventory qualification, not scene-quality benchmarking",
        code_pin=PIN,source_sha256=SOURCE_HASH,ontology_sha256=ONTO_HASH,ontology_categories=len(ontology),
        public_task_inventories=len(rows),declared_instances=217,
        instances_with_ontology_entry=covered,instances_with_full_dimension_bands=full_dims,
        instances_with_some_orientation_sample_support=orientation_sample_support,
        uncovered_canonical_categories={c:n for c,n in all_cats.items() if c not in ontology},
        original_nonperfect_self_inventory_checks=nonperfect,
        ordinary_counter_explanation="Expected raw labels that collapse into one canonical category are pooled before unit-capacity count matching; this fixes count reuse, not raw-role verification",
        no_ontology_regeneration=True,full_original_evaluator_not_reproduced=True,
        no_source_rule_or_threshold_patched=True,ordinary_correction_not_claimed_novel=True,
        paid_API_calls=0,model_calls=0,optimizer_updates=0,SDK_or_scene_replays=0,seconds=time.monotonic()-start)
    out.mkdir(parents=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"inventory_details.json").write_text(json.dumps(rows,indent=2)+"\n")
    for name,content in [("inventory_review.csv",[{k:v for k,v in r.items() if k not in ['ordinary_pooled_result','original_violations','alias_requirement_groups']} for r in rows]),("ontology_coverage.csv",label_rows)]:
        with (out/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(content[0]),lineterminator="\n");writer.writeheader();writer.writerows(content)
    print(json.dumps(result))


if __name__=="__main__":main()
