"""Audit and summarize completed frozen FITDIAG outputs; CPU only, no inference."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from diagnose_final_edit_fit import ARMS, GENERATE_IDS, parse_output


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def aggregate(rows, component='action'):
    families = defaultdict(lambda: {'tokens': 0, 'nll_sum': 0., 'correct': 0})
    for row in rows:
        part = row['components'][component]
        f = families[row['family_id']]
        f['tokens'] += part['tokens']
        f['nll_sum'] += part['nll_sum']
        f['correct'] += part['teacher_forced_correct']
    for f in families.values():
        f['nll'] = f['nll_sum'] / f['tokens']
        f['accuracy'] = f['correct'] / f['tokens']
    return {'families': dict(families), 'equal_family_nll': mean(f['nll'] for f in families.values()),
            'equal_family_accuracy': mean(f['accuracy'] for f in families.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg = read(args.run / 'PLAN.json')
    runtime = Path(cfg['runtime_directory'])
    source = runtime / 'diagnostic/RESULT.json'
    result = read(source)
    receipt = read(runtime / 'process/receipt.json')
    assert result['status'] == receipt['status'] == 'completed'
    assert result['exit_code'] == receipt['exit_code'] == 0
    assert receipt['elapsed_seconds'] <= cfg['maximum_process_seconds']
    assert result['plan_sha256'] == sha(args.run / 'PLAN.json')
    assert result['training_updates'] == result['scene_edits'] == 0
    assert result['adapters_disk_exact'] and result['all_parameters_frozen_unchanged']
    assert len(result['forwards']) == 96 and len(result['generations']) == 18
    records = {(r['arm'], r['setting'], r['id']): r for r in result['forwards']}
    assert len(records) == 96
    for key, row in records.items():
        base = records[('base', key[1], key[2])]
        assert row['tensor_sha256'] == base['tensor_sha256']
        for name, part in row['components'].items():
            assert len(part['token_nll']) == part['tokens'] == base['components'][name]['tokens']
            assert all(math.isfinite(v) for v in part['token_nll'])
            assert math.isclose(sum(part['token_nll']), part['nll_sum'], rel_tol=2e-6, abs_tol=1e-5)
    assert len({g['id'] for g in result['generations']}) == 6
    assert len({(g['arm'], g['id']) for g in result['generations']}) == 18
    details = []
    for g in result['generations']:
        assert g['id'] in GENERATE_IDS and g['arm'] in ARMS
        path = Path(g['raw_output'])
        assert sha(path) == g['raw_output_sha256']
        parsed = parse_output(path.read_text())
        assert parsed == g['parsed']
        assert (parsed == g['expected']) == g['exact_target_structure']
        assert len(g['output_ids']) <= cfg['max_new_tokens']
        assert g['target_not_in_generation_input'] and not g['geometry_or_utility_verified']
        baseline = next(x for x in result['generations'] if x['arm'] == 'base' and x['id'] == g['id'])
        assert g['input_tensor_sha256'] == baseline['input_tensor_sha256']
        assert g['expected'] == baseline['expected']
        details.append({'arm': g['arm'], 'id': g['id'], 'parsed': parsed is not None,
                        'exact_target_structure': g['exact_target_structure'],
                        'same_structure_as_base': parsed is not None and parsed == baseline['parsed'],
                        'same_raw_as_base': g['raw_output_sha256'] == baseline['raw_output_sha256'],
                        'output_tokens': len(g['output_ids']),
                        'output_kind': parsed['kind'] if parsed else None,
                        'tool': parsed['value'].get('name') if parsed else None,
                        'expected': g['expected'], 'actual': parsed,
                        'raw_output_sha256': g['raw_output_sha256']})
    arms = {}
    for arm in ARMS:
        common = [r for r in result['forwards'] if r['arm'] == arm and r['setting'] == 'common']
        oracle = [r for r in result['forwards'] if r['arm'] == arm and r['setting'] == 'oracle']
        assert len(common) == 24 and len(oracle) == 8
        terminal = [r for r in common if r['id'].endswith('_stop')]
        assert len(terminal) == 2
        gen = [g for g in details if g['arm'] == arm]
        arms[arm] = {'common': aggregate(common), 'oracle_action': aggregate(oracle),
                     'oracle_reason': aggregate(oracle, 'decision'),
                     'terminal_nll': {r['id']: r['components']['action']['nll_sum'] / r['components']['action']['tokens'] for r in terminal},
                     'generations': {'total': len(gen), 'parsed': sum(g['parsed'] for g in gen),
                                     'exact_target_structure': sum(g['exact_target_structure'] for g in gen),
                                     'same_structure_as_base': sum(g['same_structure_as_base'] for g in gen),
                                     'same_raw_as_base': sum(g['same_raw_as_base'] for g in gen),
                                     'terminal_targets_with_done_kind': sum(g['output_kind'] == 'done' for g in gen if g['id'].endswith('_stop'))}}
    for arm in ARMS[1:]:
        arms[arm]['common_nll_relative_reduction_vs_base'] = 1 - arms[arm]['common']['equal_family_nll'] / arms['base']['common']['equal_family_nll']
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'verified_completed',
              'source_sha256': sha(source), 'receipt_sha256': sha(runtime / 'process/receipt.json'),
              'elapsed_seconds': receipt['elapsed_seconds'], 'forwards': 96, 'generations': 18,
              'training_updates': 0, 'scene_edits': 0, 'arms': arms, 'generation_details': details,
              'scope': 'Training-state reproduction only. Structure mismatch is not an execution or utility failure; oracle input is not deployment.',
              'room_quality_or_generalization_established': False}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'FINAL_DIAGNOSTIC.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (args.out / 'LIKELIHOODS.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arm', 'setting', 'family', 'id', 'component', 'tokens', 'nll_sum', 'token_nll'])
        for r in result['forwards']:
            for name, part in r['components'].items():
                if part['tokens']:
                    writer.writerow([r['arm'], r['setting'], r['family_id'], r['id'], name, part['tokens'], part['nll_sum'], part['nll_sum'] / part['tokens']])
    print(json.dumps({arm: {'common_nll': arms[arm]['common']['equal_family_nll'],
                           'oracle_action_nll': arms[arm]['oracle_action']['equal_family_nll'],
                           'generation_counts': arms[arm]['generations']} for arm in ARMS}, indent=2))


if __name__ == '__main__':
    main()
