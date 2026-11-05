"""Run exactly two independently initialized arms after explicit user approval."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from decision_pilot_offload import validate
from run_decision_loss import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--approval',type=Path,required=True)
    a=ap.parse_args();cfg=validate(a.plan,'paired',a.approval,False)
    out=Path(cfg['training_output_root']);out.mkdir(parents=True,exist_ok=False)
    now=lambda:datetime.now(timezone.utc).isoformat()
    result={'status':'running','started_at':now(),'completed_arms':0,'successful_arms':0,'planned_arms':2,'arms':[]}
    save(out/'BATCH.json',result)
    for arm in cfg['arm_order']:
        command=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(out/(arm+'_execution')),
            '--timeout',str(cfg['seconds_per_arm']),'--',sys.executable,cfg['official_overlay_entry'],
            '--decision-pilot-plan',str(a.plan.resolve()),'--decision-arm',arm,'--decision-approval',str(a.approval.resolve())]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        code=subprocess.call(command,env=env)
        result['arms'].append({'arm':arm,'exit_code':code,'ended_at':now()})
        result['completed_arms']+=1;result['successful_arms']+=int(code==0);save(out/'BATCH.json',result)
        if code:break
    result.update(status='completed' if result['completed_arms']==2 and all(x['exit_code']==0 for x in result['arms']) else 'failed',ended_at=now())
    save(out/'BATCH.json',result)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
