"""Verify delivery hashes and recompute the archived-data/math audit on CPU.

This shares the audit implementation: it is a reproducibility check, not an
independent physical validation, model evaluation, or training experiment.
"""
import argparse
import json
from pathlib import Path

from audit_learning_targets import (dump, gradient_expectation, material_audit,
                                    read, scope_audit, sha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError('Keep previous receipts; choose a new output')
    root = Path(__file__).resolve().parents[2]
    package = args.package.resolve()
    manifest = read(package / 'manifest.json')
    checked = []
    for row in manifest['files']:
        source = root / row['path']
        if sha(source) != row['sha256']:
            raise ValueError('Changed delivery or input: ' + row['path'])
        checked.append(row['path'])
    material = material_audit(read(package / 'material_manifest.json'),
                              read(package / 'train_plan.json'))
    scope = scope_audit(read(root / manifest['candidate_pool']))
    credit = [gradient_expectation(p, k) for p in (.2, .5, .8) for k in (2, 4)]
    outputs = package / 'calculation'
    if material != read(outputs / 'material_objective.json'):
        raise ValueError('Material calculation differs')
    if scope != read(outputs / 'target_scope.json'):
        raise ValueError('Target calculation differs')
    if credit != read(outputs / 'branch_credit_math.json')['examples']:
        raise ValueError('Mathematical enumeration differs')
    if read(outputs / 'receipt.json')['sources_unchanged'] is not True:
        raise ValueError('Original audit had changed sources')
    result = {'status': 'passed', 'files_verified': checked,
              'recomputed': ['material weights/counts', 'label scopes and ordinary controls',
                             'set-likelihood finite differences', 'branch-credit mathematical moments'],
              'independent_model_or_geometry_reexecution': False,
              'model_calls': 0, 'training_updates': 0, 'new_scene_executions': 0}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    dump(args.receipt, result)
    print(json.dumps({'status': 'passed', 'verified_file_count': len(checked),
                      'geometry_only_correct': scope['geometry_only']['correct'],
                      'known_conjunction_labels': scope['geometry_only']['denominator']}))


if __name__ == '__main__':
    main()
