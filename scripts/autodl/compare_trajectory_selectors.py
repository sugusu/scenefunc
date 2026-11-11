"""Compare ordinary deletion selectors on the completed replay table.

These are post-hoc explanatory baselines, not learned/held-out comparisons.
The source arithmetic check is executed locally, never called a full CTC run.
"""
import argparse
import copy
import importlib
from pathlib import Path
import sys

from audit_recorded_trajectory_compression import (
    ARMS, PILOT, SOURCE, W, dump, read, source_history,
)


def choose(rows, predicate):
    valid = [r for r in rows if predicate(r)]
    return min(valid, key=lambda r: (r['retained_actions'], r['candidate_index']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not (args.output / 'ordinary_selector_results.json').exists()
    sys.path.insert(0, str(SOURCE))
    math_source = importlib.import_module('trajectory_compression.offline_scene')
    unknown = importlib.import_module('trajectory_compression.offline_check').UnknownEffect
    result = read(args.output / 'replay_results.json')
    states = read(args.output / 'state_and_observation_evidence.json')
    initial = read(W / 'paris_A_inference_20261104/frozen_initial.json')
    floor = next(o for o in initial['room_structure'] if o['name'] == 'floor')
    context = {'geometry': {'length': floor['dimensions'][0],
                             'width': floor['dimensions'][2]}}
    rows, arithmetic = [], []
    for arm in ARMS:
        arm_rows = [r for r in result['rows'] if r['arm'] == arm]
        endpoint = choose(arm_rows, lambda r: r['final_world_preserved_exactly'])
        strict = choose(arm_rows, lambda r: r['world_and_response_reusable'])
        _, history = source_history(arm)
        # Freeze future ordinary control: drop only original no-movement
        # interior events. Native replay must still validate the joint subset.
        drop = [i for i, h in enumerate(history)
                if 0 < i < len(history) - 1
                and h['feedback'][0]['response'].get('error_type') == 'no_movement']
        rule = next(r for r in arm_rows if r['dropped_original_model_turns'] == drop)
        rows.append({'arm': arm,
                     'original_steps': len(history),
                     'endpoint_only_selected_steps': endpoint['retained_actions'],
                     'endpoint_only_feedback_compatible': endpoint['all_retained_tool_responses_preserved'],
                     'exact_replay_selector_steps': strict['retained_actions'],
                     'ordinary_original_noop_rule_steps': rule['retained_actions'],
                     'ordinary_rule_passes_native_replay': rule['world_and_response_reusable'],
                     'ordinary_rule_matches_minimum_replay_steps': rule['retained_actions'] == strict['retained_actions'],
                     'task_quality_improved': False,
                     'new_whole_room_success_labels': 0})
        for i, h in enumerate(history):
            before = states[arm]['initial_world'] if i == 0 else states[arm]['original_world_after_each_action'][i-1]
            after = states[arm]['original_world_after_each_action'][i]
            state = math_source.snapshot_state({'objects': {o['object_id']: o for o in before['placed_instances']}})
            expected = math_source.snapshot_state({'objects': {o['object_id']: o for o in after['placed_instances']}})
            event = h['feedback'][0]
            row = {'arm': arm, 'source_model_turn': i,
                   'source_result_success': event['response']['success']}
            try:
                math_source.calculate(state, event['call'], event['response'],
                                      'furniture', context, expected)
                row.update(status='calculated', exact_state_agreement=state == expected)
                assert state == expected
            except unknown as exc:
                row.update(status='unmodelled', reason=str(exc),
                           note='Conservative source behavior, not evidence of a tool/source bug')
            arithmetic.append(row)
    dump(args.output / 'ordinary_selector_results.json', {
        'role': 'Post-hoc explanatory comparison on two neural traces of one development room',
        'no_new_poses_or_tool_permissions': True, 'new_model_calls': 0,
        'new_training_steps': 0, 'new_native_actions': 0,
        'rule_not_general_theorem': 'Repeated failed tools may have side effects in other settings; joint native-state/result validation is required.',
        'rows': rows})
    dump(args.output / 'official_local_arithmetic_results.json', {
        'function': 'Unchanged public offline_scene.calculate, original eight recorded input/action/result triples',
        'full_compressor_runs': 0, 'new_model_calls': 0, 'rows': arithmetic})
    print('ordinary selector comparison completed; original local arithmetic: '
          + str(sum(r['status'] == 'calculated' for r in arithmetic)) + '/8 calculated')


if __name__ == '__main__':
    main()
