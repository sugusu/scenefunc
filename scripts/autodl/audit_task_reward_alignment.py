"""Evaluate pinned reward functions on archived statistics, without RL/model calls.

The source functions are executed as exact isolated AST definitions. Archived
furniture counts and decision counts are diagnostic inputs, not original GRPO
buffers or a reproduction of the author's Critic/training results.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import time

EXPECTED_REWARD_SHA = '2c4eba6af647abfe7bbb22ecb2d124b14dbc28c7d544caffdb43e3d460b6f479'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def loadl(p):
    return list(map(json.loads, p.read_text().splitlines()))


def reward_functions(source):
    if sha(source) != EXPECTED_REWARD_SHA:
        raise ValueError('Reward source differs from pinned reviewed revision')
    tree = ast.parse(source.read_text())
    names = {'length_reward', 'objcount_reward'}
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    if {x.name for x in nodes} != names:
        raise ValueError('Missing source reward definitions')
    config = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'GRPOTrainConfig')
    defaults = {x.target.id: ast.literal_eval(x.value) for x in config.body
                if isinstance(x, ast.AnnAssign) and isinstance(x.target, ast.Name)
                and x.target.id in ('len_full', 'len_zero', 'obj_full', 'obj_zero')}
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace, defaults, [
        {'name': x.name, 'start_line': x.lineno, 'end_line': x.end_lineno} for x in nodes]


def shaped_path(rows, gamma):
    """Pathwise finite-horizon identity, not an RL optimization experiment."""
    keys = sorted(rows[0]['before'])
    if not keys:
        raise ValueError('No original checks')
    if any(sorted(x['before']) != keys or sorted(x['after']) != keys for x in rows):
        raise ValueError('Check denominator changed')
    phi = lambda checks: sum(checks[k] is True for k in keys) / len(keys)
    potential = [phi(rows[0]['before'])] + [phi(x['after']) for x in rows]
    terminal = rows[-1]['after']
    scoped_success = all(terminal[k] is True for k in keys)
    base_rewards = [0.0] * (len(rows) - 1) + [float(scoped_success)]
    unmasked = sum(gamma ** t * (base_rewards[t] + gamma * potential[t+1] - potential[t])
                   for t in range(len(rows)))
    masked_phi = potential[:-1] + [0.0]
    shaped = [base_rewards[t] + gamma * masked_phi[t+1] - masked_phi[t]
              for t in range(len(rows))]
    observed = sum(gamma ** t * v for t, v in enumerate(shaped))
    base_return = sum(gamma ** t * v for t, v in enumerate(base_rewards))
    expected = base_return - potential[0]
    residual = observed - expected
    if abs(residual) > 1e-12:
        raise ValueError('Telescoping identity failed')
    return {
        'gamma': gamma, 'fixed_scoped_check_count': len(keys),
        'scoped_conjunction_success': scoped_success,
        'full_scene_success': None,
        'original_potential_trace': potential,
        'terminal_masked_potential_trace': masked_phi,
        'masked_shaped_rewards': shaped,
        'discounted_base_return': base_return,
        'discounted_masked_shaped_return': observed,
        'expected_base_minus_initial': expected,
        'identity_residual': residual,
        'without_terminal_mask_extra': unmasked - expected,
        'predicted_extra_if_terminal_not_zero': gamma ** len(rows) * potential[-1],
        'clipped_positive_progress_only': sum(max(0.0, potential[t+1]-potential[t])
                                              for t in range(len(rows))),
        'scope': 'Fixed archived finite paths; terminal is the diagnostic budget end, not certified task completion. No training or convergence evidence.',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository', type=Path, default=Path('.'))
    ap.add_argument('--reward-source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    root, source, out = a.repository.resolve(), a.reward_source.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    effects = root / 'runs/autodl/20261110_executed_effects/actual_v2'
    comparator = root / 'runs/autodl/20261110_functional_preview_control/actual_v2/outcome_comparison.json'
    files = [source, effects/'source_fragments.jsonl', effects/'transitions.jsonl', comparator,
             Path(__file__).resolve()]
    before = {str(p): sha(p) for p in files}
    functions, defaults, definitions = reward_functions(source)
    fragments = {r['id']: r for r in loadl(files[1])}
    transitions = loadl(files[2])
    comparison = {r['id']: r for r in json.loads(comparator.read_text())}
    if {r['id'] for r in transitions} != set(fragments) or set(fragments) != set(comparison):
        raise ValueError('Archive ID sets differ')
    groups = defaultdict(list)
    for row in transitions:
        if not row['evaluation_only'] or row['training_admission']:
            raise ValueError('Evaluation-only boundary changed')
        groups[row['cell']].append(row)
    results, shaped, intermediate = [], [], []
    for cell, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['turn'])
        if [r['turn'] for r in rows] != list(range(len(rows))):
            raise ValueError('Missing or duplicate decision turn')
        for left, right in zip(rows, rows[1:]):
            if left['after'] != right['before'] or left['after_state_sha256'] != right['before_state_sha256']:
                raise ValueError('Broken saved trajectory chain')
        first_objects = fragments[rows[0]['id']]['parent']['objects']
        objects = dict(first_objects)
        for i, row in enumerate(rows):
            parent = fragments[row['id']]['parent']['objects']
            if set(objects) != set(parent):
                raise ValueError('Saved ID continuation differs')
            # This archive contains move/remove/read-only/no-call only. Use the
            # already independently recovered AFTER missing-ID receipts. Never
            # infer missing IDs from names, required count or a preferred label.
            if (row['call'] or {}).get('name') not in (None, 'move_furniture_tool',
                    'remove_furniture_tool', 'check_facing_tool', 'get_current_scene_state'):
                raise ValueError('An unsupported cardinality-changing operator appears')
            objects = dict(parent)
            for missing in comparison[row['id']]['actual_new_missing_ids']:
                if missing not in objects:
                    raise ValueError('Recovered after-ID receipt not a subset of before')
                del objects[missing]
        n = sum(o['object_type'] == 'furniture' for o in objects.values())
        n_initial = sum(o['object_type'] == 'furniture' for o in first_objects.values())
        length = len(rows)
        rlen = functions['length_reward'](length, defaults['len_full'], defaults['len_zero'])
        robj = functions['objcount_reward'](n, defaults['obj_full'], defaults['obj_zero'])
        failed = [k for k, v in rows[-1]['after'].items() if v is False]
        unknown = [k for k, v in rows[-1]['after'].items() if type(v) is not bool]
        results.append({
            'cell': cell, 'condition': rows[0]['condition'], 'recorded_decisions': length,
            'initial_native_furniture_type_count': n_initial,
            'terminal_native_furniture_type_count': n,
            'isolated_length_reward': rlen, 'isolated_object_count_reward': robj,
            'diagnostic_global_sum': rlen+robj,
            'terminal_failed_scoped_checks': failed,
            'terminal_unknown_scoped_checks': unknown,
            'scoped_check_conjunction_success': not failed and not unknown,
            'full_scene_success': None,
            'actual_author_GRPO_buffer_evaluated': False,
            'actual_count_objects_scene_loader_called': False,
            'local_critic_or_total_training_reward_evaluated': False,
            'training_admission': False,
        })
        for gamma in (1.0, .99):
            shaped.append({'cell': cell, **shaped_path(rows, gamma)})
        intermediate.extend(rows)

    # Actual defaults' sensitivity across the observed trajectories. Since
    # every reward is identical, any subset grouped by prompt is also flat.
    values = [r['diagnostic_global_sum'] for r in results]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v-mean)**2 for v in values) / len(values))
    for r in results:
        r['diagnostic_standardized_global_advantage'] = ((r['diagnostic_global_sum']-mean)/(std+1e-8)
                                                        if std > 1e-8 else 0.0)
    summary = {
        'status': 'Pinned reward-function sensitivity audit completed',
        'source_repository': 'ZJU-OmniAI/Scene-Designer',
        'source_pin': 'a0565e50ed7d32c0e09132b9311c0b9b91165405',
        'source_sha256': sha(source), 'source_definitions_executed': definitions,
        'defaults_from_actual_config_AST': defaults,
        'trajectories': len(results), 'decisions': len(intermediate), 'source_families': 1,
        'recorded_decision_count_range': [min(r['recorded_decisions'] for r in results),
                                          max(r['recorded_decisions'] for r in results)],
        'native_furniture_count_range': [min(r['terminal_native_furniture_type_count'] for r in results),
                                         max(r['terminal_native_furniture_type_count'] for r in results)],
        'diagnostic_reward_values': sorted(set(values)),
        'diagnostic_global_std': std,
        'all_diagnostic_global_advantages_zero': all(r['diagnostic_standardized_global_advantage']==0 for r in results),
        'trajectories_with_failed_recorded_check': sum(bool(r['terminal_failed_scoped_checks']) for r in results),
        'full_scene_success_labels_created': 0,
        'shaping_path_identities_checked': len(shaped),
        'shaping_max_abs_identity_residual': max(abs(r['identity_residual']) for r in shaped),
        'new_model_calls': 0, 'paid_API_calls': 0, 'optimizer_updates': 0,
        'new_native_executions': 0, 'new_room_generations': 0,
        'scope': 'Offline diagnostic of exact scalar source functions with archived native furniture-type counts and decision counts. Not the upstream loader, rollout, local Critic, total reward, training gradient, author performance or a new method experiment.',
        'count_bridge_limit': 'Native furniture-type counts are supplied diagnostic covariates. The author helper counts placed_furniture through its own exporter/loader, which was read but not called here; identity with that loader is not assumed.',
        'length_bridge_limit': 'Archived decision count is not claimed equal to original GRPO buffer.traj_len. Counts from zero through the observed six decisions all lie below the default full-score threshold.',
        'source_comment_limit': 'make_reward_critic docstring discusses a different gate default/global legality location; actual module and CLI default the local legality gate ON, global loop sums length and object-count rewards. Do not substitute comments for these branches.',
        'no_claim_that_total_training_signal_is_zero': True,
        'no_claim_of_policy_invariance_for_the_complete_implemented_GRPO_surrogate': True,
        'wall_seconds': time.monotonic()-tick,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    dump(out/'trajectory_reward_rows.json', results)
    dump(out/'finite_horizon_shaping.json', shaped)
    dump(out/'summary.json', summary)
    with (out/'trajectory_reward_rows.csv').open('w', newline='') as f:
        fields = [k for k in results[0] if k not in ('terminal_failed_scoped_checks','terminal_unknown_scoped_checks')]
        w = csv.DictWriter(f, fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader();w.writerows(results)
    after = {str(p): sha(p) for p in files}
    if after != before:
        raise ValueError('Read-only inputs changed')
    dump(out/'input_hashes.json', {'before': before, 'after': after, 'old_inputs_unchanged': True})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
