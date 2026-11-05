"""Execute a frozen CPU-only integration audit through patched train_sft.py."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = {'started_at': now(), 'status': 'running', 'plan_sha256': sha(args.plan),
              'model_calls': 0, 'optimizer_updates': 0, 'professional_samples_admitted': 0,
              'stages': {}}
    write(out / 'RESULT.json', result)
    print(now(), 'begin official entry CPU integration audit, no training', flush=True)
    root = Path(plan['repo'])
    source = Path(plan['source'])
    scripts = root / 'scripts/920'
    overlay = out / 'overlay'
    try:
        for name, expected in plan['code_sha256'].items():
            assert sha(scripts / name) == expected, name
        for name, expected in plan['source_sha256'].items():
            assert sha(source / name) == expected, name
        assert sha(plan['patch']) == plan['patch_sha256']
        assert sha(plan['manifest']) == plan['manifest_sha256']
        for ref in plan['processor_files']:
            assert sha(Path(plan['model']) / ref['path']) == ref['sha256']
        (overlay / 'src').mkdir(parents=True)
        for name in plan['source_sha256']:
            target = overlay / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        (overlay / 'src/__init__.py').write_text('')
        for name in ('decision_pairing.py', 'decision_entry_audit.py'):
            shutil.copy2(scripts / name, overlay / 'src' / name)
        patch = subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', plan['patch']],
                               cwd=overlay, capture_output=True, text=True)
        (out / 'patch.log').write_text(patch.stdout + patch.stderr)
        if patch.returncode:
            raise RuntimeError('Pinned official entry patch did not apply')
        assert sha(overlay / 'src/train_sft.py') == plan['patched_entry_sha256']
        os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                          PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                          PYTHONPATH=str(overlay))
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        entry = [sys.executable, str(overlay / 'src/train_sft.py')]
        common = ['--model-path', plan['model'], '--repo-root', plan['input_root'],
                  '--max-length', '8192', '--max-pixels', '65536',
                  '--use-cpu', '--no-bf16', '--dataloader-workers', '0']

        def run(name, command, expected=0):
            receipt_dir = out / (name + '_execution')
            rc = subprocess.call([sys.executable, str(scripts / 'run_stage.py'), '--receipt-dir',
                                  str(receipt_dir), '--timeout', '120', '--', *command], cwd=overlay)
            receipt = json.loads((receipt_dir / 'receipt.json').read_text())
            result['stages'][name] = receipt
            write(out / 'RESULT.json', result)
            if rc != expected or receipt['exit_code'] != expected:
                raise RuntimeError(f'Unexpected exit for {name}: {rc}, expected {expected}')
            return (receipt_dir / 'stage.log').read_text()

        arm_reports = {}
        for arm in ('paired', 'separate'):
            target = out / arm / 'BATCH.json'
            command = entry + common + ['--output-dir', str(out / arm / 'output'),
                      '--dry-run', '--decision-manifest', plan['manifest'], '--decision-arm', arm,
                      '--decision-audit-report', str(target)]
            run(arm, command)
            arm_reports[arm] = json.loads(target.read_text())
        # A real legacy CLI invocation confirms the ordinary dataset path still works.
        legacy = run('legacy', entry + common + ['--output-dir', str(out / 'legacy_output'),
                     '--dry-run', '--sft-root', plan['legacy_fixture_directory']])
        # These calls intentionally have a nonexistent model: the entry must reject
        # them before any processor/model loading or output-directory creation.
        bad = ['--model-path', str(out / 'does_not_exist'), '--dataloader-workers', '0']
        rejected = run('training_rejected', entry + bad + ['--output-dir', str(out / 'rejected_output'),
                       '--decision-manifest', plan['manifest'], '--decision-arm', 'paired'], 1)
        orphan = run('arm_without_manifest_rejected', entry + bad + ['--output-dir', str(out / 'orphan_output'),
                     '--dry-run', '--decision-arm', 'paired'], 1)
        paired, separate = arm_reports['paired'], arm_reports['separate']
        checks = {
            'actual_patched_official_entry_matches_lock': sha(overlay / 'src/train_sft.py') == plan['patched_entry_sha256'],
            'two_examples_per_decision': all(v['first_batch_decisions'] == 1 and v['first_batch_examples'] == 2 for v in arm_reports.values()),
            'same_reason_action_target_ids': all(p['target_ids'] == s['target_ids'] for p, s in zip(paired['rows'], separate['rows'])),
            'same_83_targets': paired['total_supervised_tokens'] == separate['total_supervised_tokens'] == 83,
            'reason_targets_once_and_prefix_masked': all(v['rows'][0]['reason_present_in_targets'] and not v['rows'][1]['reason_present_in_targets'] for v in arm_reports.values()),
            'reason_only_in_paired_action_context': paired['rows'][1]['reason_present_in_input'] and not separate['rows'][1]['reason_present_in_input'],
            'ten_images_same_tensors': all(v['image_instances'] == 10 for v in arm_reports.values()) and all(paired['batch_tensor_sha256'][k] == separate['batch_tensor_sha256'][k] for k in ('pixel_values', 'image_grid_thw')),
            'same_padded_shapes': paired['batch_shapes'] == separate['batch_shapes'],
            'no_eval_sentinel': not any(row['evaluation_only_sentinel_present'] for v in arm_reports.values() for row in v['rows']),
            'no_cuda_model_or_optimizer': all(not v['cuda_initialized'] and not v['model_weights_loaded'] and v['optimizer_updates'] == 0 and not v['trainer_instantiated'] for v in arm_reports.values()),
            'legacy_dry_run_still_reaches_dataset': '[dry-run] inspecting one example' in legacy and 'loss tokens (labels != -100):' in legacy,
            'decision_training_rejected_before_io': 'Decision manifests are audit-only' in rejected and not (out / 'rejected_output').exists() and '[init]' not in rejected,
            'orphan_arm_rejected_before_io': 'must be provided together' in orphan and not (out / 'orphan_output').exists() and '[init]' not in orphan,
            'live_upstream_files_unchanged': all(sha(source / name) == expected for name, expected in plan['source_sha256'].items()),
        }
        result.update(checks=checks, arms=arm_reports)
        if not all(checks.values()):
            raise AssertionError([k for k, v in checks.items() if not v])
        result.update(status='passed_official_entry_dry_run_only', exit_code=0,
                      limitations='Actual CLI and batch selection integrated. Trainer/loss/optimizer/LoRA training, compute parity and professional supervision remain unvalidated; decision training path intentionally rejects execution.')
    except BaseException as exc:
        result.update(status='failed', exit_code=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['ended_at'] = now()
        write(out / 'RESULT.json', result)
        print(now(), result['status'], flush=True)


if __name__ == '__main__':
    main()
