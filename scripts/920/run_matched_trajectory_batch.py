"""Run a frozen zero-training matched trajectory matrix once; no retries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from native_project import sha256
from run_knowledge_policy import write


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    a = ap.parse_args(); p = json.loads(a.plan.read_text()); root = a.plan.parent
    if (root/'BATCH_RESULT.json').exists():
        raise ValueError('Existing batch cannot be restarted')
    if p['training_updates'] != 0:
        raise ValueError('Zero-training evaluation only')
    plans = []
    for cell in p['cells']:
        for path, digest in cell['frozen_inputs'].items():
            if sha256(Path(path)) != digest:
                raise ValueError('Frozen evaluation input changed: '+path)
        plan = json.loads(Path(cell['plan']).read_text()); plans.append(plan)
    if sum(x['budget']['model_calls'] for x in plans) != p['budget']['model_calls']:
        raise ValueError('Model budget does not match cells')
    if sum(x['budget']['gpu_seconds_per_cell'] for x in plans) != p['budget']['gpu_subprocess_seconds']:
        raise ValueError('GPU budget does not match cells')
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = dict(status='running', started_at=now(), cells=[], model_calls=0,
                  gpu_subprocess_seconds=0., training_updates=0)
    def progress():
        write(root/'BATCH_RESULT.json', result)
        status = json.loads((root/'STATUS.json').read_text())
        status.update(status=result['status'],updated_at=now(),completed=len(result['cells']),
                      latest_verifiable_artifact=str(root/'BATCH_RESULT.json'),failure_reason=result.get('error'))
        write(root/'STATUS.json',status)
        write(root.parent/'latest.json',dict(run_id=root.name,phase=result['status'],
            updated_at=now(),run_directory=str(root)))
    progress()
    try:
        for cell, plan in zip(p['cells'], plans):
            print(now(),cell['cell_id'],flush=True)
            target = Path(cell['output']); public = Path(plan['public_directory'])/'recorded'
            command = [sys.executable,str(Path(__file__).with_name('run_stage.py')),
                '--receipt-dir',cell['receipt'],'--timeout',str(cell['wall_seconds']),
                '--',*cell['command']]
            code = subprocess.call(command)
            report = json.loads((target/'RESULT.json').read_text()) if (target/'RESULT.json').exists() else {}
            result['cells'].append(dict(cell_id=cell['cell_id'],condition=cell['condition'],arm=cell['arm'],
                exit_code=code,result_path=str(target/'RESULT.json'),
                receipt=json.loads((Path(cell['receipt'])/'receipt.json').read_text())))
            result['model_calls'] += report.get('model_calls',0)
            result['gpu_subprocess_seconds'] += report.get('gpu_subprocess_seconds',0.)
            for name in ('RESULT.json','final_state.json','final_audit.json','final_observation.json',
                         'restored_request.json','restored_audit.json'):
                if (target/name).exists():
                    public.mkdir(parents=True,exist_ok=True);shutil.copy2(target/name,public/name)
            for rd in sorted(target.glob('round_*')):
                for name in ('request.json','tool_reply.json','model/raw_output.txt','model/RESULT.json','model/output_ids.json'):
                    if (rd/name).exists():
                        dest=public/rd.name/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(rd/name,dest)
            if (target/'final_observation.json').exists():
                observation=json.loads((target/'final_observation.json').read_text())
                for i,image in enumerate(observation['image_paths']):
                    dest=public/'views'/f'{i}_{Path(image).name}';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(image,dest)
            progress()
            if code or report.get('status')=='failed':
                raise RuntimeError('First infrastructure failure; batch stopped without retry')
            if result['model_calls'] > p['budget']['model_calls'] or result['gpu_subprocess_seconds'] > p['budget']['gpu_subprocess_seconds']:
                raise RuntimeError('Frozen batch resource ceiling reached')
        result['status']='completed_pending_review'
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['ended_at']=now();progress()


if __name__=='__main__':main()
