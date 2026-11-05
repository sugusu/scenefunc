"""Execute the frozen three-arm inference attempt once; never train or retry."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--execution-plan',type=Path,required=True);a=ap.parse_args()
    plan=json.loads(a.plan.read_text());execution=json.loads(a.execution_plan.read_text())
    assert sha(a.plan)==execution['comparison_plan_sha256']
    assert sha(__file__)==execution['batch_script_sha256']
    root=Path(plan['shared_preparation']).parent;public=a.plan.parent;now=lambda:datetime.now(timezone.utc).isoformat()
    if (root/'BATCH.json').exists():raise ValueError('Batch already attempted; no restart')
    prepared=json.loads((root/'prepare/RESULT.json').read_text());assert prepared['status']=='prepared_without_model'
    assert len(prepared['encoding']['images'])==5
    initial=json.loads((root/'INITIAL_AUDIT.json').read_text());assert all(initial['immutable_records_unchanged'].values())
    plans={}
    for arm,item in plan['arms'].items():
        assert sha(item['path'])==item['sha256'];plans[arm]=json.loads(Path(item['path']).read_text())
    # Arm labels, output paths and adapter weights are the only differences.
    allowed={'comparison_arm','public_directory','runtime_directory','public','runtime','adapter'}
    stripped=[{k:v for k,v in q.items() if k not in allowed} for q in plans.values()]
    assert all(x==stripped[0] for x in stripped)
    result={'status':'running','started_at':now(),'training_updates':0,'arms':[],'shared_initial_request_sha256':prepared['request_sha256']}
    def save():
        write(root/'BATCH.json',result)
        write(public/'STATUS.json',{'run_id':plan['run_id'],'question':plan['question'],'status':result['status'],'updated_at':now(),'started_at':result['started_at'],'ended_at':result.get('ended_at'),'configuration_hashes':{'plan':sha(a.plan),'execution_plan':sha(a.execution_plan),'initial_request':prepared['request_sha256']},'completed_planned':{'arms':[sum(x.get('audit_exit_code')==0 for x in result['arms']),3]},'run_directory':str(root),'latest_verifiable_artifact':str(root/'BATCH.json'),'failure_reason':result.get('error'),'next_step':'Audit complete room outputs against nine retained requirements; never infer human or generalization success.'})
        write(public.parent/'latest.json',{'run_id':plan['run_id'],'stage':'three_arm_whole_room_inference','status':result['status'],'updated_at':now(),'run_directory':str(root)})
    save();print(now(),'begin frozen three-arm inference',flush=True)
    try:
        for arm in plan['execution_order']:
            q=plans[arm];row={'arm':arm,'started_at':now()};result['arms'].append(row);save()
            command=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(root/arm/'rollout_execution'),'--timeout',str(q['budget']['rollout_seconds']),'--',sys.executable,str(Path(__file__).with_name('run_catalog_adapter_policy.py')),'--plan',plan['arms'][arm]['path'],'--phase','rollout','--out',str(root/arm/'rollout')]
            code=subprocess.call(command);row['rollout_exit_code']=code;save()
            if code:raise RuntimeError(arm+' rollout failed; remaining arms not started, no retry')
            rollout=json.loads((root/arm/'rollout/RESULT.json').read_text());row.update(model_calls=rollout['model_calls'],tool_calls=rollout['tool_calls'],stop_reason=rollout['stop_reason'])
            command=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(root/arm/'audit_execution'),'--timeout','120','--',execution['geometry_python'],str(Path(__file__).with_name('audit_transfer_room.py')),'--plan',str(public/'AUDIT_PLAN.json'),'--bundle',str(root/arm/'rollout/final_checkpoint'),'--out',str(root/arm/'FINAL_AUDIT.json')]
            code=subprocess.call(command);row['audit_exit_code']=code;row['ended_at']=now();save()
            if code:raise RuntimeError(arm+' audit failed; batch stopped without retry')
        result['status']='three_arm_attempts_completed_pending_requirement_review'
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['ended_at']=now();save();print(now(),result['status'],flush=True)


if __name__=='__main__':main()
