"""Recompute the published small evidence bundle without ML/geometry dependencies.

This verifies record consistency and scoped arithmetic, not physical validity,
human preference, model quality, or a new generation experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from consequence_contract import exact_admission, interval_dominates, masked_binary_loss


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    repo = Path(__file__).resolve().parents[2]
    checked = {}
    manifest_path = bundle / 'manifest.json'
    if manifest_path.exists():
        for row in read(manifest_path)['files']:
            path = repo / row['path']
            if digest(path) != row['sha256']:
                raise ValueError('Published hash mismatch: ' + row['path'])
        checked['manifest'] = True
    for row in read(bundle / 'source_bundle/source_manifest.json'):
        if digest(bundle / 'source_bundle' / row['name']) != row['sha256']:
            raise ValueError('Source hash mismatch: ' + row['name'])
    args.scratch.mkdir(parents=True, exist_ok=True)
    # Keep the actual derivation, including stdout and stderr, for review.
    run = Path(tempfile.mkdtemp(prefix='consequence_verify_', dir=args.scratch))
    child = subprocess.run(
        [sys.executable, str(Path(__file__).with_name('build_consequence_evidence.py')),
         '--source-bundle', str(bundle / 'source_bundle'), '--output', str(run / 'derived')],
        capture_output=True, text=True, timeout=60,
    )
    (run / 'stdout.txt').write_text(child.stdout)
    (run / 'stderr.txt').write_text(child.stderr)
    if child.returncode:
        raise RuntimeError('Derivation failed; see ' + str(run))
    for name in ('summary.json', 'ordinary_controls.json', 'scoped_records.jsonl',
                 'candidate_pool.json', 'actual_model_attempts.json'):
        checked[name] = (run / 'derived' / name).read_bytes() == (bundle / name).read_bytes()
    if not all(checked.values()):
        raise ValueError('Derived evidence differs: ' + str(checked))

    pool = read(bundle / 'candidate_pool.json')
    required = ['within_room', 'no_new_mesh_intersection', 'finite_joint_witness']
    admissions = [{'candidate': p['id'], **exact_admission(p['scope_values'], required)} for p in pool]
    for candidate, admission in zip(pool, admissions):
        if (admission['scoped_status'] == 'pass') != candidate['eligible_for_strict_planar_goal']:
            raise ValueError('Admission mismatch: ' + candidate['id'])
    records = [json.loads(line) for line in (bundle / 'scoped_records.jsonl').read_text().splitlines()]
    if any(r['mask'] != (r['target'] is not None) or r['whole_room_mask'] for r in records):
        raise ValueError('Unknown labels or whole-room scope changed')
    # Numerical contract checks, explicitly not additional scene experiments.
    loss = masked_binary_loss({'measured': .8, 'missing': .2}, {'measured': True, 'missing': None})
    if not math.isclose(loss['mean_known_bce'], -math.log(.8)) or loss['known'] != 1:
        raise ValueError('Measured-label loss mismatch')
    if masked_binary_loss({'missing': .2}, {'missing': None})['mean_known_bce'] is not None:
        raise ValueError('All-unknown loss must remain absent')
    valid = interval_dominates({'x': (.8, .9)}, {'x': (.1, .2)}, ['x'])
    missing = interval_dominates({'x': None}, {'x': (.1, .2)}, ['x'])
    if valid['dominates'] is not True or valid['calibration_verified'] or missing['dominates'] is not None:
        raise ValueError('Conditional interval contract mismatch')
    result = {
        'status': 'passed', 'checks': checked, 'candidates': len(pool),
        'scoped_pass': sum(a['scoped_status'] == 'pass' for a in admissions),
        'whole_room_certified': sum(a['whole_room_success'] is True for a in admissions),
        'observed_targets': sum(r['mask'] for r in records),
        'unmeasured_targets': sum(not r['mask'] for r in records),
        'model_calls': 0, 'training_updates': 0, 'scene_executions': 0,
        'scope': __doc__.strip(),
    }
    (run / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'run_directory': str(run), **result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
