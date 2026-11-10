"""Identify what archived future-state targets mean; never train or run a tool.

This audits behavior-conditioned ambiguity and supervision eligibility. It reuses
the published ordinary predictor receipts rather than predicting the 72 outcomes
again. Empirical within-table risk is descriptive, never held-out performance.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loadl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n')


def empirical_risk(groups):
    """In-sample Bayes-table Brier risk, averaged over observed row/check pairs.

    A lookup table is used only to measure target ambiguity. It is neither a
    fitted neural predictor nor a policy. Unknown checks are excluded explicitly.
    """
    ss = 0.0
    n = 0
    for group in groups.values():
        names = sorted({k for r in group for k, v in r['after'].items()
                        if type(v) is bool})
        for name in names:
            ys = [int(r['after'][name]) for r in group
                  if type(r['after'].get(name)) is bool]
            mean = sum(ys) / len(ys)
            ss += sum((y - mean) ** 2 for y in ys)
            n += len(ys)
    return {'sum_squared_error': ss, 'known_row_check_pairs': n,
            'mse': ss / n if n else None,
            'meaning': 'Empirical descriptive oracle table, not generalization'}


def entropy(values):
    counts = Counter(values)
    return -sum((n / len(values)) * math.log2(n / len(values))
                for n in counts.values())


def subgroup_description(rows, fragments):
    groups, action_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        p = fragments[row['id']]['parent']
        x = {k: p[k] for k in ('objects', 'requested_mode', 'declared_requirements')}
        key = digest(x)
        groups[key].append(row)
        action_groups[(key, digest(row['call']))].append(row)
    return {
        'rows': len(rows), 'x_groups': len(groups), 'x_call_groups': len(action_groups),
        'x_groups_with_multiple_check_vectors': sum(
            len({digest(r['after']) for r in g}) > 1 for g in groups.values()),
        'x_call_groups_with_multiple_check_vectors': sum(
            len({digest(r['after']) for r in g}) > 1 for g in action_groups.values()),
        'repeated_x_call_groups': sum(len(g) > 1 for g in action_groups.values()),
        'rows_in_repeated_x_call_groups': sum(len(g) for g in action_groups.values() if len(g) > 1),
        'empirical_x_only_check_risk': empirical_risk(groups),
        'empirical_x_call_check_risk': empirical_risk(action_groups),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository', type=Path, default=Path('.'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.repository.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    effects = root / 'runs/autodl/20261110_executed_effects/actual_v2'
    control = root / 'runs/autodl/20261110_functional_preview_control/actual_v2'
    inputs = [effects / 'transitions.jsonl', effects / 'source_fragments.jsonl',
              control / 'summary.json', control / 'outcome_comparison.json',
              Path(__file__).resolve()]
    hashes = {str(p.relative_to(root)): file_hash(p) for p in inputs}
    transitions = loadl(inputs[0])
    fragments = {r['id']: r for r in loadl(inputs[1])}
    ordinary = json.loads(inputs[2].read_text())
    outcomes = {r['id']: r for r in json.loads(inputs[3].read_text())}
    if not (len(transitions) == len(fragments) == len(outcomes) == 72):
        raise ValueError('Frozen source membership changed')
    if {r['id'] for r in transitions} != set(fragments) or set(fragments) != set(outcomes):
        raise ValueError('IDs do not match')

    groups = defaultdict(list)
    action_groups = defaultdict(list)
    rows = []
    for t in transitions:
        f, o = fragments[t['id']], outcomes[t['id']]
        if not t['evaluation_only'] or t['training_admission']:
            raise ValueError('Original evaluation/training boundary changed')
        if t['before'] != f['parent']['checks']:
            raise ValueError('Pre-state check receipt mismatch')
        # Exclude outcomes and checks from X, retaining the exact public state,
        # task and declared parameters. No claim that X equals a human belief.
        x = {k: f['parent'][k] for k in
             ('objects', 'requested_mode', 'declared_requirements')}
        xkey, akey = digest(x), digest(t['call'])
        groups[xkey].append(t)
        action_groups[(xkey, akey)].append(t)
        damage = o['actual_primitive_or_check_damage']
        moves = (t['call'] or {}).get('name') == 'move_furniture_tool'
        unknown = [k for k, v in t['after'].items() if type(v) is not bool]
        rows.append({
            'id': t['id'], 'cell': t['cell'], 'condition': t['condition'],
            'x_sha256': xkey, 'action_sha256': akey,
            'call_name': (t['call'] or {}).get('name', 'NO_CALL'),
            'native_called': t['native_called'], 'tool_success': t['tool_success'],
            'state_changed': t['state_changed'],
            'proxy_damage': damage,
            'proxy_repaired_check_count': len(t['delta']['repaired']),
            'proxy_damaged_check_count': len(t['delta']['damaged']),
            'repair_without_recorded_primitive_or_check_damage':
                o['actual_repair_without_primitive_or_check_damage'],
            'after_state_sha256': t['after_state_sha256'],
            'after_checks_sha256': digest(t['after']),
            'unknown_after_checks': unknown,
            'ordinary_predicted_checks': o['predicted_check_comparisons'],
            'ordinary_prediction_mismatch': bool(o['prediction_mismatches']),
            'target_role': ('descriptive_executed_absolute_move' if moves and t['native_called']
                            else 'descriptive_archive_transition'),
            'desired_future_label': False, 'full_scene_success_label': None,
            'training_admission': False, 'evaluation_only': True,
        })

    group_rows = []
    for key, group in sorted(groups.items()):
        actions = {digest(r['call']) for r in group}
        states = {r['after_state_sha256'] for r in group}
        ys = [digest(r['after']) for r in group]
        group_rows.append({
            'x_sha256': key, 'members': [r['id'] for r in group],
            'row_count': len(group), 'distinct_calls_including_no_call': len(actions),
            'distinct_nonnull_calls': len({digest(r['call']) for r in group if r['call']}),
            'distinct_recorded_after_state_hashes': len(states),
            'distinct_after_check_vectors': len(set(ys)),
            'empirical_check_vector_entropy_bits': entropy(ys),
            'hash_limit': 'State hashes can include metadata; check-vector ambiguity is reported separately',
        })
    ambiguities = [g for g in group_rows if g['distinct_after_check_vectors'] > 1]
    contradictions = [
        {'x_sha256': key[0], 'action_sha256': key[1],
         'members': [r['id'] for r in g],
         'distinct_after_check_vectors': len({digest(r['after']) for r in g})}
        for key, g in action_groups.items()
        if len({digest(r['after']) for r in g}) > 1]
    native = [r for r in rows if r['native_called']]
    moves = [r for r in rows if r['native_called'] and r['call_name'] == 'move_furniture_tool']
    result = {
        'status': 'Archived future-target identifiability audit completed',
        'new_analysis': 'Behavior ambiguity, descriptive-vs-desired target eligibility, and operator coverage',
        'decisions': len(rows), 'trajectories': len({r['cell'] for r in rows}),
        'independent_project_families': 1,
        'condition_counts': dict(Counter(r['condition'] for r in rows)),
        'call_counts': dict(Counter(r['call_name'] for r in rows)),
        'native_call_counts': dict(Counter(r['call_name'] for r in native)),
        'current_state_and_task_groups': len(groups),
        'groups_with_multiple_nonnull_actions': sum(g['distinct_nonnull_calls'] > 1 for g in group_rows),
        'groups_with_distinct_after_check_vectors': len(ambiguities),
        'rows_in_ambiguous_check_groups': sum(g['row_count'] for g in ambiguities),
        'conditioned_on_x_and_call_groups': len(action_groups),
        'same_x_call_check_contradictions': contradictions,
        'selection_sensitivity': {
            'all_native_calls': subgroup_description(
                [r for r in transitions if r['native_called']], fragments),
            'native_moves_only': subgroup_description(
                [r for r in transitions if r['native_called'] and
                 (r['call'] or {}).get('name') == 'move_furniture_tool'], fragments),
        },
        'empirical_unconditioned_future_check_risk': empirical_risk(groups),
        'empirical_action_conditioned_future_check_risk': empirical_risk(action_groups),
        'native_calls': len(native), 'native_moves': len(moves),
        'native_calls_with_primitive_or_check_damage': sum(r['proxy_damage'] for r in native),
        'native_moves_with_primitive_or_check_damage': sum(r['proxy_damage'] for r in moves),
        'native_calls_with_repair_and_no_measured_damage': sum(
            r['repair_without_recorded_primitive_or_check_damage'] for r in native),
        'desired_future_labels_available': 0,
        'complete_success_labels_created': 0,
        'training_admitted_rows': 0,
        'new_predictor_runs': 0, 'new_native_actions': 0,
        'model_calls': 0, 'training_updates': 0, 'new_scene_generations': 0,
        'ordinary_control_reused': {
            'source': str(inputs[2].relative_to(root)),
            'comparisons': ordinary['predicted_check_comparisons'],
            'correct': ordinary['correct_predicted_checks'],
            'scope': 'Supported archived proxy checks only; not full geometry/function/human preference',
        },
        'operator_gap': 'No generation, addition, resizing, support reassignment or physical-interaction transitions in this archive',
        'risk_limit': 'These table risks are in-sample descriptive ambiguity, not neural training, Bayes population risk, generalization or causal policy benefit',
        'conditioning_limit': 'X here is exact exposed object records + declared task/parameters, not the complete image/language/dialogue/history input. Different histories can distinguish rows grouped here.',
        'wall_seconds': time.monotonic() - tick,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if result['native_moves'] != 35 or result['native_calls'] != 56:
        raise ValueError('Operator count changed')
    if sum(r['ordinary_predicted_checks'] for r in rows) != ordinary['predicted_check_comparisons']:
        raise ValueError('Reused control membership differs')
    dump(out / 'row_target_roles.json', rows)
    dump(out / 'state_task_groups.json', group_rows)
    dump(out / 'summary.json', result)
    with (out / 'target_roles.csv').open('w', newline='') as stream:
        fields = [k for k in rows[0] if k != 'unknown_after_checks']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    after = {str(p.relative_to(root)): file_hash(p) for p in inputs}
    if after != hashes:
        raise RuntimeError('Read-only inputs changed')
    dump(out / 'input_hashes.json', {'before': hashes, 'after': after, 'unchanged': True})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
