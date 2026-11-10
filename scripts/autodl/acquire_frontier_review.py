"""Fetch a pinned, bounded public source/task subset, not assets or models."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/autodl/20261110_frontier_scene_evaluation'
PIN='1462133985213af9bae14344d2b938e18868350b'
EXPLICIT=['README.md','LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','docs/en/scoring.md',
    'docs/en/experiment.md','docs/en/licenses.md','docs/reference/task-index.csv',
    'src/evaluator/l2.py','src/evaluator/l2_quality_scoring.py',
    'src/evaluator/l2_requirements.py','src/evaluator/l2_vlm_scoring.py',
    'src/evaluator/l2_runner.py','src/evaluator/core.py','src/evaluator/continuous_brief_relations.py',
    'src/evaluator/l2_brief_relations.py','src/evaluator/l2_brief_numeric.py',
    'src/evaluator/l2_brief_route.py','src/evaluator/l2_brief_assemblies.py',
    'harness/L2/Brief2Scene/eval_spec.json','harness/L2/Brief2Scene/requirements.json']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--supplement',action='store_true');args=parser.parse_args()
    tree=json.loads((OUT/'repository_index.json').read_text());assert tree['pin']==PIN
    items={r['path']:r for r in tree['files']}
    task_prefix='data/3dbench/tasks/L2/Brief2Scene/'
    paths=EXPLICIT+[p for p in items if p.startswith(task_prefix) and Path(p).name in ('instruction.md','task.toml')]
    assert len([p for p in paths if p.endswith('/instruction.md')])==20
    if args.supplement:
        paths=['benchmark/engine.py','benchmark/aggregation.py',
            'src/evaluator/legacy_detail.py','src/evaluator/legacy_detail_v2.py',
            'src/evaluator/addition_brief_numeric.py',
            'tools/finalize_effort_addition_brief.py','tools/finalize_effort_addition.py',
            'tools/measure_effort_addition_numeric.py','tools/finalize_effort_legacy_case.py',
            'tools/finalize_effort_addition_l2.py',
            'configs/release/addition-config/task_L2_Brief2Scene_013/plan.json',
            'tools/experiment_additions.py']
        paths += [p for p in items if p.startswith(task_prefix) and p.endswith('/attribution/CREDITS')]
    assert all(p in items for p in paths)
    total=sum(items[p]['size'] for p in paths)
    assert total<1024**2 and shutil.disk_usage(OUT).free-total>5*1024**3
    plan={'pin':PIN,'selection':'Fixed scoring follow-up and all task notices.' if args.supplement else 'All 20 Brief2Scene public instructions, without outcome filtering; fixed scoring source subset.',
        'paths':paths,'expected_bytes':total,'inference_calls':0,'asset_downloads':0}
    suffix='_supplement' if args.supplement else ''
    (OUT/('acquisition_plan'+suffix+'.json')).write_text(json.dumps(plan,indent=2)+'\n')
    def fetch(p):
        dst=OUT/'source'/p;url='https://raw.githubusercontent.com/Einsia/Frontier3D/'+PIN+'/'+p
        if dst.exists():raw=dst.read_bytes()
        else:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'scene-research-audit'}),timeout=30) as r:
                raw=r.read(items[p]['size']+1)
        assert len(raw)==items[p]['size'],p
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==items[p]['sha'],p
        if not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
        return {'path':p,'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'git_blob':items[p]['sha']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(fetch,paths))
    (OUT/('source_manifest'+suffix+'.json')).write_text(json.dumps({'pin':PIN,'files':rows},indent=2)+'\n')
    print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'model_calls':0}))


if __name__=='__main__':main()
