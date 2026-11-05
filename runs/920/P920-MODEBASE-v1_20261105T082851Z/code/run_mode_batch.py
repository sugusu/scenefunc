"""Run the prospectively frozen activity/knowledge cases once, preserving failures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from native_project import sha256
from run_mode_policy import write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--freeze',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.plan.read_text());freeze=json.loads(a.freeze.read_text())
    assert sha256(a.plan)==freeze['master_plan_sha256']
    assert sha256(Path(__file__))==freeze['batch_script_sha256']
    for row in freeze['files']:assert sha256(Path(row['path']))==row['sha256'],row['path']
    a.out.mkdir(parents=True,exist_ok=False)
    now=lambda:datetime.now(timezone.utc).isoformat()
    result={'status':'running','started_at':now(),'planned':len(cfg['case_plans']),
            'completed':0,'cases':[],'freeze_sha256':sha256(a.freeze),'training_updates':0}
    write(a.out/'RESULT.json',result)
    for name in cfg['case_plans']:
        cp=Path(name);case=json.loads(cp.read_text());cid=case['case_id']
        rt=Path(cfg['runtime_directory']);out=rt/'rollouts'/cid
        result['active_case']=cid;write(a.out/'RESULT.json',result)
        write(Path(cfg['public_directory']).parent/'latest.json',{
            'run_id':cfg['run_id'],'stage':'model_rollout_'+cid,'status':'running',
            'updated_at':now(),'run_directory':cfg['runtime_directory'],
            'status_path':str(Path(cfg['public_directory'])/'STATUS.json')})
        print(now(),'begin frozen case',cid,flush=True)
        command=[sys.executable,str(Path(__file__).with_name('run_stage.py')),
                 '--receipt-dir',str(rt/'execution'/cid),'--timeout',str(cfg['budget']['case_seconds']),
                 '--',sys.executable,str(Path(__file__).with_name('run_mode_policy.py')),
                 '--plan',str(cp),'--phase','rollout','--out',str(out)]
        code=subprocess.call(command)
        row={'case':cid,'stage_exit':code,'output':str(out),'ended_at':now()}
        if (out/'RESULT.json').exists():
            actual=json.loads((out/'RESULT.json').read_text())
            row.update(model_calls=actual['model_calls'],tool_calls=actual['tool_calls'],
                       status=actual['status'],final_condition_pass=actual.get('final_condition_pass'))
        result['cases'].append(row);result['completed']+=1;write(a.out/'RESULT.json',result)
        print(now(),'case ended',cid,'exit',code,flush=True)
    result.update(status='completed' if all(x['stage_exit']==0 for x in result['cases']) else 'completed_with_failures',
                  active_case=None,ended_at=now())
    write(a.out/'RESULT.json',result)
    print(now(),result['status'],flush=True)


if __name__=='__main__':main()
