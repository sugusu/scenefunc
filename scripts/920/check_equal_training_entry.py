"""CPU contract checks and service-double tests of training batch control."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import train_equal_material as entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg, _ = entry.load_plan(args.plan)
    checks = []
    with tempfile.TemporaryDirectory(prefix='equal-entry-') as tmp:
        root = Path(tmp)

        def rejected(name, fn, expected=(ValueError, PermissionError)):
            try:
                fn()
            except expected:
                checks.append(name)
            else:
                raise AssertionError('Expected rejection: ' + name)

        rejected('CPU_fixture_cannot_train', lambda: entry.require_approval(args.plan, None))
        altered = root / 'altered.json'
        for key, value in [('arm_order', ['paired', 'paired']),
                           ('updates_per_arm', 0), ('seconds_per_arm', 7201),
                           ('learning_rate', float('nan')), ('optimizer', 'adamw'),
                           ('total_training_seconds', 1)]:
            c = copy.deepcopy(cfg); c[key] = value
            # Inject invalid numeric input deliberately; production save rejects
            # NaN before the loader can exercise its own validation.
            altered.write_text(json.dumps(c))
            rejected('reject_' + key, lambda: entry.load_plan(altered))
        c = copy.deepcopy(cfg); c['training_admission'] = 'review_ready'
        entry.save(altered, c)
        rejected('missing_exact_plan_approval', lambda: entry.require_approval(altered, None))
        fake_approval = root / 'synthetic_service_double_only.json'
        entry.save(fake_approval, dict(approved=True, plan_sha256='0' * 64,
                                     user_messages=['SYNTHETIC TEST; NOT USER APPROVAL']))
        rejected('wrong_plan_approval', lambda: entry.require_approval(altered, fake_approval))

        # The actual batch function runs; only the GPU subprocess is replaced.
        # Synthetic approvals remain in this temporary directory and are erased.
        for scenario, exit_codes in [('completed', [0, 0, 0]), ('first_failure', [1]),
                                      ('timeout', [124])]:
            c = copy.deepcopy(cfg)
            c.update(training_admission='review_ready',
                     training_output_root=str(root / scenario / 'training'),
                     public_directory=str(root / scenario / 'public'))
            plan = root / (scenario + '.json'); entry.save(plan, c)
            entry.save(fake_approval, dict(approved=True, plan_sha256=entry.sha256(plan),
                                         user_messages=['SYNTHETIC SERVICE DOUBLE; NOT USER APPROVAL']))
            entry.save(Path(c['public_directory']) / 'STATUS.json', {})
            observed = []

            def fake_call(command, env):
                arm = command[-1]; index = len(observed)
                assert arm == c['arm_order'][index]
                assert int(command[command.index('--timeout') + 1]) == c['seconds_per_arm']
                assert env['CUDA_VISIBLE_DEVICES'] == c['gpu_uuid']
                assert str(plan.resolve()) in command
                observed.append(arm)
                code = exit_codes[index]
                base = Path(c['training_output_root'])
                entry.save(base / (arm + '_receipt') / 'receipt.json',
                           dict(exit_code=code, service_double=True))
                entry.save(base / arm / 'RUN.json',
                           dict(status='completed' if code == 0 else 'failed',
                                optimizer_updates=c['updates_per_arm'] if code == 0 else 0,
                                service_double=True))
                return code

            with patch.object(entry.subprocess, 'call', side_effect=fake_call):
                if scenario == 'completed':
                    entry.run_batch(c, plan, fake_approval)
                else:
                    rejected(scenario + '_stops_batch',
                             lambda: entry.run_batch(c, plan, fake_approval), (RuntimeError,))
            report = json.loads((Path(c['public_directory']) / 'TRAINING_RESULT.json').read_text())
            assert len(observed) == len(exit_codes)
            assert report['status'] == ('completed' if scenario == 'completed' else 'failed')
            rejected(scenario + '_cannot_overwrite_or_restart',
                     lambda: entry.run_batch(c, plan, fake_approval), (FileExistsError,))
            checks.append(scenario + '_receipt_and_budget_preserved')
    if args.out.exists():
        raise FileExistsError(args.out)
    entry.save(args.out, dict(status='passed', checks=checks, optimizer_updates=0,
        model_calls=0, actual_GPU_subprocesses=0, subprocess_tests='service doubles',
        timeout_signal_delivery_tested=False, plan_sha256=entry.sha256(args.plan)))
    print(json.dumps(dict(status='passed', checks=len(checks), optimizer_updates=0)))


if __name__ == '__main__':
    main()
