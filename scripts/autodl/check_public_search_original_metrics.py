"""Recompute unchanged original SDF-based task checks on actual native states."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args();root=args.run.resolve();out=root/'original_metric_check';out.mkdir(exist_ok=False)
    start=time.monotonic();source=root/'original_evaluator';manifest=json.loads((source/'manifest.json').read_text())
    hashes={}
    for r in manifest['files']:
        p=source/(r['module']+'.py');h=sha(p)
        if h!=r['sha256']:raise ValueError('Original evaluator modified')
        hashes[str(p)]=h
    sys.path.insert(0,str(source))
    from nesting_room_metrics import measure_nesting
    inputs=json.loads((root/'actual_v1/initial_public_inputs.json').read_text());rows=[]
    for key,item in inputs.items():
        mode=item['parent']['requested_mode'];folder=root/'native_v2'/mode
        initial=json.loads((folder/'before_state.json').read_text());after=json.loads((folder/'after_state.json').read_text())
        program=item['parent']['declared_requirements']
        for p in folder.rglob('*'):
            if p.is_file():hashes[str(p)]=sha(p)
        before_result=measure_nesting(initial,initial,folder,folder,program,mode)
        after_result=measure_nesting(after,initial,folder,folder,program,mode)
        dump(out/(mode+'_before.json'),before_result);dump(out/(mode+'_after.json'),after_result)
        previous=item['parent']['checks'];b=before_result['feedback']['checks'];a=after_result['feedback']['checks']
        predicted=json.loads((root/'actual_v1'/(key[:12]+'.json')).read_text())['solution']['state']['checks']
        rows.append({'mode':mode,'initial_checks_match_original_public_input':b==previous,
                     'actual_checks_match_search_prediction':a==predicted,
                     'repaired':[k for k in b if b[k] is False and a[k] is True],
                     'lost':[k for k in b if b[k] is True and a[k] is not True],
                     'remaining_failed':[k for k,v in a.items() if v is False],
                     'unknown_checks':[k for k,v in a.items() if v is None],
                     'requested_activity_status':after_result['review']['requested_activity_status'],
                     'measured_requirements_status':after_result['review']['measured_requirements_status'],
                     'whole_room_success':after_result['review']['whole_room_success']})
    if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('Read-only inputs changed')
    dump(out/'source_hashes.json',hashes)
    dump(out/'summary.json',{'status':'completed','results':rows,'original_evaluator_modified':False,
         'source_inputs_unchanged':True,'new_model_calls':0,'native_calls_in_this_check':0,
         'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'scope':'Original proxy checker with rebuilt original SDFs on actual native move outputs; not an independent definition of human usability, not a physics or full generation result.'})
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
