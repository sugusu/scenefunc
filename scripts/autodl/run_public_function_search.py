"""Run fixed ordinary search on every distinct initial public EVALFIX task.

Only round-zero public requirements/objects/checks are supplied to the search.
Old proposal text and all recorded later states are excluded from inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

from audit_public_function_preview import PIN, canonical, dump, geometry_of_request
from search_public_function import search


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    protocol=json.loads(args.protocol.read_text());before={str(args.protocol):sha(args.protocol)}
    source=Path(protocol['source_fragments']);before[str(source)]=sha(source)
    for f in ('search_public_function.py','run_public_function_search.py','public_function_preview.py','audit_prospective_guard.py','audit_public_function_preview.py'):
        p=Path(__file__).with_name(f);before[str(p)]=sha(p)
    dump(args.out/'source_hashes_before.json',before)
    grouped={}
    for line in source.read_text().splitlines():
        r=json.loads(line)
        if not r['id'].endswith('__00'):continue
        parent=r['parent'];cell=r['id'].rsplit('__',1)[0]
        path=f'runs/920/P920-EVALFIX-v1_20261107T130916Z/evaluation/{cell}/recorded/round_000/request.json'
        raw=subprocess.check_output(['git','-c','gc.auto=0','show',PIN+':'+path])
        h=hashlib.sha256(raw).hexdigest()
        if h!=r['original_request_sha256']:raise ValueError('Initial request hash mismatch')
        geometry=geometry_of_request(raw)
        key=hashlib.sha256(canonical([parent,geometry]).encode()).hexdigest()
        if key not in grouped:grouped[key]={'parent':parent,'geometry':geometry,'source_cells':[],'request_hashes':{}}
        grouped[key]['source_cells'].append(cell);grouped[key]['request_hashes'][path]=h
    dump(args.out/'initial_public_inputs.json',grouped)
    rows=[]
    for key,item in grouped.items():
        tick=time.monotonic();result=search(item['parent'],item['geometry'],**protocol['search'])
        elapsed=time.monotonic()-tick;result['seconds']=elapsed
        result['source_cells']=item['source_cells'];dump(args.out/(key[:12]+'.json'),result)
        sol=result['solution'];checks=sol['state']['checks'] if sol else item['parent']['checks']
        row={'mode':item['parent']['requested_mode'],'input_sha256':key,'source_cells':item['source_cells'],
             'status':result['status'],'candidate_count':result['candidate_count'],
             'solution_action_count':len(sol['actions']) if sol else 0,
             'remaining_failed_checks':[k for k,v in checks.items() if v is False],
             'unknown_checks':[k for k,v in checks.items() if v is None], 'seconds':elapsed}
        rows.append(row);print(json.dumps(row),flush=True)
    after={p:sha(Path(p)) for p in before};dump(args.out/'source_hashes_after.json',after)
    if before!=after:raise ValueError('Inputs or source changed during run')
    summary={'status':'completed','distinct_initial_tasks':len(rows),
             'historical_trajectories_represented':sum(len(r['source_cells']) for r in rows),
             'independent_room_families':1,'results':rows,'model_calls':0,'native_calls':0,
             'new_scene_generations':0,'training_updates':0,'whole_scene_success':None,
             'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'source_files_unchanged':before==after,
             'scope':'Public-input nominal proposal search. No new native execution, physical validity, original design completeness, generalization or matched model-cost claim.'}
    dump(args.out/'summary.json',summary)


if __name__=='__main__':main()
