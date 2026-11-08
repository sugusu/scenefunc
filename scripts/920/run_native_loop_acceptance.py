"""Run a frozen generation-free native loop acceptance batch exactly once."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_knowledge_policy import write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    a=ap.parse_args();p=json.loads(a.plan.read_text());root=a.plan.parent
    if (root/'NATIVE_RESULT.json').exists():raise ValueError('Refuse batch restart')
    if any(p[k]!=0 for k in ('model_calls','training_updates','scene_edits')):
        raise ValueError('Generation-free acceptance only')
    now=lambda:datetime.now(timezone.utc).isoformat()
    result=dict(status='running',started_at=now(),stages=[],model_calls=0,training_updates=0,scene_edits=0)
    write(root/'NATIVE_RESULT.json',result)
    error=None
    try:
        for item in p['commands']:
            cmd=item['command'];cellplan=Path(cmd[cmd.index('--plan')+1])
            if hashlib.sha256(cellplan.read_bytes()).hexdigest()!=item['plan_sha256']:
                raise ValueError('Frozen cell changed')
            print(now(),item['cell'],item['phase'],flush=True)
            code=subprocess.call([sys.executable,str(Path(__file__).with_name('run_stage.py')),
                '--receipt-dir',item['receipt'],'--timeout',str(item['timeout']),'--',*cmd])
            receipt=json.loads((Path(item['receipt'])/'receipt.json').read_text())
            result['stages'].append(dict(cell=item['cell'],phase=item['phase'],receipt=receipt))
            write(root/'NATIVE_RESULT.json',result)
            status=json.loads((root/'STATUS.json').read_text())
            status.update(status='running' if code==0 else 'failed',completed=1+sum(s['receipt']['exit_code']==0 for s in result['stages']),
                updated_at=now(),latest_verifiable_artifact=item['receipt'],failure_reason=None if code==0 else 'Native stage exit '+str(code))
            write(root/'STATUS.json',status)
            if code:raise RuntimeError('First native error; batch stopped without retry')
        result['status']='completed'
    except BaseException as exc:
        error=exc;result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        result.update(ended_at=now(),elapsed_stage_seconds=sum(s['receipt']['elapsed_seconds'] for s in result['stages']))
        write(root/'NATIVE_RESULT.json',result)
    if error:raise error


if __name__=='__main__':main()
