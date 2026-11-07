"""Run the exact approved training then evaluation once; no retries."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, sys

p=Path(__file__).resolve().parent
repo=p.parents[2]
train=json.loads((p/'TRAIN_PLAN.json').read_text())
study=json.loads((p/'STUDY_PLAN.json').read_text())
now=lambda:datetime.now(timezone.utc).isoformat()
record={'status':'running','started_at':now(),'pid':os.getpid(),'stages':[]}
target=p/'EXECUTION.json'
with target.open('x') as f:json.dump(record,f,indent=2)
def save():
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(target)
status=json.loads((p/'STATUS.json').read_text())
status.update(state='approved_execution_starting',training_started=True,execution_started_at=record['started_at'],first_execution_log='Approved batch launched; train first, evaluation only after accepted adapters.',failure_reason=None)
(p/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(record['started_at'],status['first_execution_log'],flush=True)
commands=[('training',[train['model_python'],'-u',str(repo/'scripts/920/train_terminal_edit_study.py'),'--plan',str(p/'TRAIN_PLAN.json'),'--authorization',str(p/'AUTHORIZATION.json')]),('evaluation',[study['native_python'],'-u',str(repo/'scripts/920/run_terminal_edit_evaluation.py'),'--study',str(p/'STUDY_PLAN.json'),'--authorization',str(p/'AUTHORIZATION.json')])]
exit_code=1
try:
    for name,command in commands:
        row={'stage':name,'started_at':now(),'command':command,'log':str(p/(name+'_batch.log'))};record['stages'].append(row);save()
        with Path(row['log']).open('x') as output:
            child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
            row['pid']=child.pid;save();print(now(),'started',name,child.pid,flush=True)
            row['exit_code']=child.wait()
        row['ended_at']=now();save();print(now(),'ended',name,row['exit_code'],flush=True)
        if row['exit_code']:
            raise RuntimeError(name+' failed; no subsequent stage or retry')
    exit_code=0;record['status']='completed'
except BaseException as exc:
    record.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
finally:
    record.update(ended_at=now(),exit_code=exit_code);save()
    print(json.dumps(record),flush=True)
