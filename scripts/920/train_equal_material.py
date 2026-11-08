"""Bounded training entry for immutable equal-material components.

The default operation is a CPU preflight. Training requires a separate exact
plan approval and runs only through the timeout/receipt supervisor. Historical
training entries and weights are not changed.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from conditional_material_dataset import EqualMaterialExamples
from decision_pilot_offload import locked
from native_project import sha256
from run_decision_loss import save
from train_edit_learning_study import normalization_check, train_authorized


def load_plan(path):
    cfg = json.loads(path.read_text())
    if cfg.get('purpose') != 'equal_material_learning_v1':
        raise ValueError('Wrong study contract')
    arms = cfg['arm_order']
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= {'action_only', 'separate', 'paired'}:
        raise ValueError('Invalid or duplicate arms')
    for key, maximum in [('updates_per_arm', 16), ('seconds_per_arm', 7200)]:
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= maximum:
            raise ValueError('Unbounded or unsupported ' + key)
    if not math.isfinite(cfg['learning_rate']) or not 0 < cfg['learning_rate'] <= 0.0002:
        raise ValueError('Unsupported learning rate')
    if cfg['optimizer'] != 'adafactor' or cfg['decision_loss_coefficient'] != 1:
        raise ValueError('Plan differs from the reused Trainer objective')
    if cfg['total_training_seconds'] != len(arms) * cfg['seconds_per_arm']:
        raise ValueError('Total budget mismatch')
    for name, digest in cfg['code_files'].items():
        locked(name, digest)
    for name, digest in cfg['upstream_hashes'].items():
        locked(Path(cfg['upstream_source']) / name, digest)
    material = cfg['material_manifest']
    manifest = json.loads(locked(material['path'], material['sha256']).read_text())
    if manifest['schema'] != 'equal-material-components-v1':
        raise ValueError('Wrong material schema')
    datasets = {}
    for arm in arms:
        rows = [dict(r) for r in manifest['records'] if
                r['arm'] == ('separate' if arm == 'action_only' else arm)
                and (arm != 'action_only' or r['component'] == 'action')]
        for row in rows:
            row['arm'] = arm
        ds = EqualMaterialExamples(rows)
        counts = cfg['components'][arm]
        if (len(ds) != counts['examples'] or len(ds.families) != 6
                or any(sum(r[c + '_tokens'] for r in rows) != counts[c + '_tokens']
                       for c in ('action', 'decision'))):
            raise ValueError('Material counts mismatch: ' + arm)
        datasets[arm] = ds
    # Hash locks bind the contents; this additionally protects intended pairing.
    if {'separate', 'paired'} <= set(arms):
        def targets(ds):
            return {r['id']: (r['family_id'], r['target_sha256'], r['action_tokens'],
                             r['decision_tokens']) for r in ds.records}
        if targets(datasets['separate']) != targets(datasets['paired']):
            raise ValueError('Unequal supervision materials')
    return cfg, datasets


def require_approval(path, approval):
    if json.loads(path.read_text()).get('training_admission') != 'review_ready':
        raise PermissionError('CPU fixtures and incomplete study plans cannot train')
    if approval is None:
        raise PermissionError('Separate exact-plan training approval required')
    auth = json.loads(approval.read_text())
    if (auth.get('approved') is not True or auth.get('plan_sha256') != sha256(path)
            or not auth.get('user_messages')):
        raise PermissionError('Training approval scope mismatch')


def worker(plan, approval, arm):
    """Called by the bounded supervisor, never by the default preflight CLI."""
    require_approval(plan, approval)
    cfg, datasets = load_plan(plan)
    if arm not in datasets:
        raise ValueError('Arm outside approved plan')
    for row in cfg['model_weight_files']:
        locked(Path(cfg['model_directory']) / row['path'], row['sha256'])
    for name, digest in cfg['model_configuration_sha256'].items():
        locked(Path(cfg['model_directory']) / name, digest)
    source = Path(cfg['upstream_source']) / 'src'
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('equal_material_official_sft', source / 'train_sft.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    train_authorized(cfg, datasets[arm], arm, upstream, plan)


def run_batch(cfg, plan, approval):
    require_approval(plan, approval)
    root = Path(cfg['training_output_root'])
    root.mkdir(parents=True, exist_ok=False)
    public = Path(cfg['public_directory'])
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = dict(status='running', started_at=now(), plan_sha256=sha256(plan), arms={})

    def progress():
        save(public / 'TRAINING_RESULT.json', result)
        status = json.loads((public / 'STATUS.json').read_text())
        status.update(state='training_' + result['status'], updated_at=now(),
                      latest_verifiable_artifact=str(public / 'TRAINING_RESULT.json'),
                      failure_reason=result.get('error'),
                      completed=sum(r.get('training', {}).get('optimizer_updates', 0)
                                    for r in result['arms'].values()),
                      planned=len(cfg['arm_order']) * cfg['updates_per_arm'])
        save(public / 'STATUS.json', status)
        save(public.parent / 'latest.json', dict(run_id=cfg['run_id'], stage=status['state'],
             updated_at=status['updated_at'], run_directory=cfg['runtime_directory']))

    progress()
    try:
        for arm in cfg['arm_order']:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'], HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            # Argument values never enter Python source or a shell command.
            bootstrap = ('import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
                         'from train_equal_material import worker; '
                         'worker(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])')
            command = [cfg['model_python'], str(Path(__file__).with_name('run_stage.py')),
                '--receipt-dir', str(root / (arm + '_receipt')),
                '--timeout', str(cfg['seconds_per_arm']), '--', cfg['model_python'], '-c',
                bootstrap, str(Path(__file__).resolve().parent), str(plan.resolve()),
                str(approval.resolve()), arm]
            code = subprocess.call(command, env=env)
            report = root / arm / 'RUN.json'
            receipt_path = root / (arm + '_receipt') / 'receipt.json'
            result['arms'][arm] = dict(exit_code=code,
                receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None,
                training=json.loads(report.read_text()) if report.exists() else {})
            progress()
            if code or result['arms'][arm]['training'].get('status') != 'completed':
                raise RuntimeError(f'{arm} failed or timed out; batch stops without retry')
        result.update(status='completed', ended_at=now(), exit_code=0)
    except BaseException as exc:
        result.update(status='failed', ended_at=now(), exit_code=1,
                      error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        progress()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--authorization', type=Path)
    args = ap.parse_args()
    if args.train:
        require_approval(args.plan, args.authorization)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if not args.out or args.out.exists():
            raise ValueError('CPU preflight needs a new --out file')
    cfg, datasets = load_plan(args.plan)
    if args.train:
        run_batch(cfg, args.plan, args.authorization)
    else:
        save(args.out, dict(status='passed', plan_sha256=sha256(args.plan),
            arms={arm: normalization_check(ds, args.out.parent / (arm + '_fixture'))
                  for arm, ds in datasets.items()}, optimizer_updates=0,
            model_loaded=False, note='CPU Trainer scaling only; no real backward or GPU fit'))


if __name__ == '__main__':
    main()
