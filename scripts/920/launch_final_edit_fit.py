"""Single-attempt, approved evaluation launcher with an external wall-time limit."""
import argparse
from pathlib import Path
import subprocess
import sys
from diagnose_final_edit_fit import authorize, now, read, save, sha, validate_plan


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--authorization',type=Path,required=True)
    args=ap.parse_args()
    authorize(args.authorization,args.plan)
    cfg=read(args.plan);validate_plan(cfg)
    runtime=Path(cfg['runtime_directory'])
    # Admission precedes any model import. Never reuse an attempted run.
    runtime.mkdir(parents=True,exist_ok=False)
    receipt=runtime/'process';out=runtime/'diagnostic'
    scripts=Path(__file__).resolve().parent
    cmd=[sys.executable,str(scripts/'run_stage.py'),'--receipt-dir',str(receipt),
         '--timeout',str(cfg['maximum_process_seconds']-5),'--',cfg['model_python'],
         str(scripts/'diagnose_final_edit_fit.py'),'--plan',str(args.plan.resolve()),
         '--out',str(out),'--authorization',str(args.authorization.resolve())]
    pub=Path(cfg['public_directory'])
    save(pub/'EXECUTION.json',dict(status='launching',started_at=now(),command=cmd,
         plan_sha256=sha(args.plan),authorization_sha256=sha(args.authorization),training_updates=0))
    process=subprocess.run(cmd,check=False)
    record=read(receipt/'receipt.json') if (receipt/'receipt.json').exists() else {
        'status':'failed','exit_code':process.returncode,'ended_at':now(),
        'error':'No process receipt; inspect launcher output. Do not retry automatically.'}
    record.update(plan_sha256=sha(args.plan),authorization_sha256=sha(args.authorization),training_updates=0)
    save(pub/'EXECUTION.json',record)
    partial=read(out/'RESULT.json') if (out/'RESULT.json').exists() else {}
    completed=len(partial.get('forwards',[]))+len(partial.get('generations',[]))
    state='completed' if process.returncode==0 and completed==114 else 'failed'
    save(pub/'STATUS.json',dict(run_id=cfg['run_id'],state=state,question=cfg['question'],
        started_at=record.get('started_at'),ended_at=record.get('ended_at'),updated_at=now(),
        actual_exit_code=record['exit_code'],first_log=record.get('first_log'),completed=completed,planned=114,
        configuration_sha256=sha(args.plan),input_sha256=cfg['encoding']['sha256'],
        latest_verifiable_artifact=str(receipt/'receipt.json'),
        failure_reason=None if state=='completed' else partial.get('error','Process failed or timed out; receipt is authoritative.'),
        next_step='Interpret retained outputs; no automatic retry, extension or training.'))
    save(pub.parent/'latest.json',dict(run_id=cfg['run_id'],stage=state,updated_at=now(),run_directory=str(runtime)))
    raise SystemExit(process.returncode if process.returncode else (0 if state=='completed' else 1))


if __name__=='__main__':main()
