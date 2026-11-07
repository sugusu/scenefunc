"""Execute the three prospectively frozen inference cells once; stop on first error."""
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);a=p.parse_args();P=a.public.resolve();R=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
prep=read(P/'prepare/PLAN.json');T=Path(prep['prepared_directory']).parent
if read(T/'prepare_receipt/receipt.json').get('exit_code')!=0 or read(T/'prepare/RESULT.json')['status']!='prepared_without_model':raise ValueError('Preparation not accepted')
state=read(P/'STATUS.json');state.update(started_at=read(T/'prepare/RESULT.json')['started_at'],status='running',stage='rollout',cells=[])
for arm in ['frozen_base_knowledge_feedback','action_only','decision_action']:
 state.update(updated_at=now(),active_cell=arm,latest_verifiable_artifact=str(P/arm/'STATUS.json'),next_step='Complete remaining frozen cells; stop on infrastructure error.')
 write(P/'STATUS.json',state);write(R/'runs/920/latest.json',dict(run_id=prep['run_id'],stage='rollout',updated_at=now(),run_directory=str(T),status_file=str(P/'STATUS.json')))
 cmd=[sys.executable,str(R/'scripts/920/run_stage.py'),'--receipt-dir',str(T/(arm+'_receipt')),'--timeout','1500','--',str(R.parent/'envs/scene_designer311_native/bin/python'),str(R/'scripts/920/run_storage_learning_gate.py'),'--plan',str(P/arm/'PLAN.json'),'--out',str(T/arm),'--phase','rollout']
 code=subprocess.call(cmd,cwd=R)
 result=read(T/arm/'RESULT.json') if (T/arm/'RESULT.json').exists() else {}
 state['cells'].append(dict(arm=arm,exit_code=code,result=result,receipt=read(T/(arm+'_receipt')/'receipt.json')))
 state['completed_planned']=[sum(x['result'].get('model_calls',0) for x in state['cells']),18]
 state.update(updated_at=now());write(P/'STATUS.json',state)
 if code:
  state.update(status='failed',failure_reason=result.get('error','Cell subprocess failure'),next_step='Preserve failure; no retry or remaining-cell execution.');break
else:state.update(status='completed_pending_review',failure_reason=None,next_step='Verify final scenes, archive evidence and publish result.')
state.update(ended_at=now(),updated_at=now(),active_cell=None);write(P/'STATUS.json',state)
write(R/'runs/920/latest.json',dict(run_id=prep['run_id'],stage=state['status'],updated_at=now(),run_directory=str(T),status_file=str(P/'STATUS.json')))
raise SystemExit(0 if state['status']=='completed_pending_review' else 1)
