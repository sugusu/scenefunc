"""Execute the two approved finite adapters, stopping the batch on any error."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from native_project import sha256
from run_decision_loss import save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--authorization', type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.plan.read_text())
    auth = json.loads(a.authorization.read_text())
    if not auth.get('approved') or auth.get('plan_sha256') != sha256(a.plan):
        raise PermissionError('Exact authorized plan required')
    public = Path(cfg['public_directory'])
    root = Path(cfg['training_output_root'])
    root.mkdir(exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = dict(status='running', started_at=now(), arms={}, plan_sha256=sha256(a.plan))
    def progress():
        save(public/'TRAINING_RESULT.json', result)
        s = json.loads((public/'STATUS.json').read_text())
        s.update(status='training_'+result['status'], updated_at=now(),
                 latest_verifiable_artifact=str(public/'TRAINING_RESULT.json'),
                 failure_reason=result.get('error'), started_at=result['started_at'],
                 next_step='Complete fixed two-adapter training, then common twelve-room evaluation; no budget extension.')
        s['completed_planned']['training_arms']=[sum(x.get('exit_code')==0 for x in result['arms'].values()),2]
        save(public/'STATUS.json', s)
        save(public.parent/'latest.json',dict(run_id=cfg['run_id'],stage=s['status'],updated_at=s['updated_at'],run_directory=cfg['runtime_directory']))
    progress()
    print(now(), 'start authorized edit-learning batch', flush=True)
    try:
        for arm in cfg['arm_order']:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'], HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            command = [sys.executable, str(Path(__file__).with_name('run_stage.py')),
                       '--receipt-dir', str(root/(arm+'_receipt')), '--timeout', str(cfg['seconds_per_arm']),
                       '--', cfg['model_python'], str(Path(__file__).with_name('train_edit_learning_study.py')),
                       '--plan', str(a.plan), '--arm', arm, '--authorization', str(a.authorization)]
            code = subprocess.call(command, env=env)
            receipt=json.loads((root/(arm+'_receipt')/'receipt.json').read_text())
            record=json.loads((root/arm/'RUN.json').read_text()) if (root/arm/'RUN.json').exists() else {}
            result['arms'][arm]=dict(exit_code=code, receipt=receipt, training=record)
            progress()
            if code:
                raise RuntimeError(f'{arm} exited {code}; preserve checkpoints, no automatic restart')
        result.update(status='completed', exit_code=0, ended_at=now())
    except BaseException as exc:
        result.update(status='failed', exit_code=1, ended_at=now(), error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        progress()


if __name__ == '__main__':
    main()
