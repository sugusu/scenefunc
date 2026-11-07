"""Prospective 24-example study; reuse the validated Trainer, never old runs.

The public entry point bounds both arms through run_stage. A training child
requires exact-plan authorization; --dry-run never imports model weights.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from decision_pilot_offload import locked
from native_project import sha256
from run_decision_loss import save
from terminal_training_dataset import CandidateEditExamples
from train_edit_learning_study import normalization_check, train_authorized


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--authorization', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--arm', choices=['action_only', 'decision_action'])
    args = ap.parse_args()
    cfg = json.loads(args.plan.read_text())
    if (cfg['purpose'] != 'executed_edit_terminal_decision_study'
            or cfg['updates_per_arm'] != 8 or cfg['seconds_per_arm'] != 7200
            or cfg['arm_order'] != ['action_only', 'decision_action']
            or cfg['examples_per_arm'] != 24 or cfg['families'] != 6):
        raise ValueError('Frozen prospective study scope changed')
    for path, digest in cfg['code_files'].items():
        locked(path, digest)
    enc = json.loads(locked(cfg['encoding']['path'], cfg['encoding']['sha256']).read_text())
    if enc['schema'] != 'scenefunc-edit-plus-terminal-encoding-v1':
        raise ValueError('Wrong encoding contract')
    datasets = {arm: CandidateEditExamples([r for r in enc['records'] if r['arm'] == arm],
        expected_examples=24, expected_families=6) for arm in cfg['arm_order']}
    if args.dry_run:
        if not args.out or args.arm:
            raise ValueError('Dry-run requires --out and checks both arms')
        save(args.out, dict(status='passed', plan_sha256=sha256(args.plan),
            arms={arm: normalization_check(ds, args.out.parent / (arm+'_fixture'))
                  for arm, ds in datasets.items()}, optimizer_updates=0))
        return
    if not args.authorization:
        raise PermissionError('New exact-plan user training approval is required')
    auth = json.loads(args.authorization.read_text())
    if (auth.get('approved') is not True or auth.get('plan_sha256') != sha256(args.plan)
            or not auth.get('user_messages')):
        raise PermissionError('Approval scope/hash mismatch')
    if args.arm:
        for row in cfg['model_weight_files']:
            locked(Path(cfg['model_directory']) / row['path'], row['sha256'])
        for name, digest in cfg['model_configuration_sha256'].items():
            locked(Path(cfg['model_directory']) / name, digest)
        source = Path(cfg['upstream_source']) / 'src'
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location('terminal_official_sft', source / 'train_sft.py')
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        train_authorized(cfg, datasets[args.arm], args.arm, upstream, args.plan)
        return
    root = Path(cfg['training_output_root'])
    root.mkdir(parents=True, exist_ok=False)
    public = Path(cfg['public_directory'])
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = dict(status='running', started_at=now(), plan_sha256=sha256(args.plan), arms={})
    def progress():
        save(public/'TRAINING_RESULT.json', result)
        status = json.loads((public/'STATUS.json').read_text())
        status.update(state='training_'+result['status'], updated_at=now(),
            latest_verifiable_artifact=str(public/'TRAINING_RESULT.json'),
            failure_reason=result.get('error'), training_updates=sum(
                r.get('training', {}).get('optimizer_updates', 0) for r in result['arms'].values()))
        save(public/'STATUS.json', status)
        save(public.parent/'latest.json', dict(run_id=cfg['run_id'], stage=status['state'],
            updated_at=status['updated_at'], run_directory=cfg['runtime_directory']))
    progress()
    try:
        for arm in cfg['arm_order']:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'], HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            command = [sys.executable, str(Path(__file__).with_name('run_stage.py')),
                '--receipt-dir', str(root/(arm+'_receipt')), '--timeout', str(cfg['seconds_per_arm']),
                '--', cfg['model_python'], str(Path(__file__).resolve()), '--plan', str(args.plan),
                '--arm', arm, '--authorization', str(args.authorization)]
            code = subprocess.call(command, env=env)
            report = root/arm/'RUN.json'
            result['arms'][arm] = dict(exit_code=code,
                receipt=json.loads((root/(arm+'_receipt')/'receipt.json').read_text()),
                training=json.loads(report.read_text()) if report.exists() else {})
            progress()
            if code:
                raise RuntimeError(f'{arm} exited {code}; batch stops without retry')
        result.update(status='completed', ended_at=now(), exit_code=0)
    except BaseException as exc:
        result.update(status='failed', ended_at=now(), exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        progress()


if __name__ == '__main__':
    main()
