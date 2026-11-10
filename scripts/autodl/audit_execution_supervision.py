"""Audit observed action support WITHOUT promoting evaluation data to training.

Local proxy vectors are not full-scene truth. Missing alternatives are unknown,
not negative actions. Exact serialized-call dedup is not geometric equivalence.
No model, trainer, native scene execution, or synthetic labels are involved.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--transitions', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    raw = args.transitions.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line]
    assert rows and len({row['id'] for row in rows}) == len(rows)
    assert all(row['evaluation_only'] and not row['training_admission'] for row in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['parent_group']].append(row)
    output = []
    for key, group in sorted(groups.items()):
        assert len({canonical(row['before']) for row in group}) == 1
        calls = defaultdict(list)
        for row in group:
            calls[canonical(row['call'])].append(row)
        observed = []
        for call, matches in sorted(calls.items()):
            observed.append({
                'call': json.loads(call), 'source_ids': [row['id'] for row in matches],
                'native_replies': sum(row['native_called'] for row in matches),
                'native_successes': sum(row['tool_success'] for row in matches),
                'after_state_sha256': sorted({row['after_state_sha256'] for row in matches}),
                'observed_proxy_vectors': [json.loads(value) for value in sorted({canonical(row['after']) for row in matches})],
                'observed_repairs': sorted({name for row in matches for name in row['delta']['repaired']}),
                'observed_damages': sorted({name for row in matches for name in row['delta']['damaged']}),
                'locally_improving_without_new_measured_harm': any(row['delta']['scoped_improvement_without_new_harm'] for row in matches),
                'full_scene_valid': None, 'training_admission': False,
                'reason': 'Evaluation provenance; proxy coverage is not complete-scene verification'})
        output.append({'parent_group': key, 'condition': group[0]['condition'],
                       'before_state_sha256': group[0]['before_state_sha256'],
                       'observed_actions': observed,
                       'unobserved_actions_status': 'unknown_not_negative',
                       'complete_valid_action_set': None,
                       'complete_valid_trajectory_buffer': None})
    native = [row for row in rows if row['native_called']]
    successful = [row for row in native if row['tool_success']]
    summary = {
        'status': 'observed_support_audited_without_new_labels_or_training',
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'source': str(args.transitions),
        'trajectories': len({row['cell'] for row in rows}),
        'independent_room_families': 1,
        'recorded_decisions': len(rows), 'native_calls': len(native),
        'tool_successes': len(successful),
        'successful_calls_with_measured_damage': sum(bool(row['delta']['damaged']) for row in successful),
        'scoped_improvements_without_new_measured_harm': sum(row['delta']['scoped_improvement_without_new_harm'] for row in rows),
        'observed_context_groups': len(groups),
        'context_groups_with_multiple_nonnull_actions': sum(sum(a['call'] is not None for a in g['observed_actions']) > 1 for g in output),
        'all_recorded_proxy_entries_true': sum(all(v is True for v in row['after'].values()) for row in rows),
        'conjunction_limit': 'Includes threshold sensitivity proxies; not a definition of the full task or proof that full success is zero',
        'full_scene_positive_labels_created': 0, 'training_admitted_rows': 0,
        'MML_valid_buffer_available': False, 'MML_objective_value': None,
        'new_model_calls': 0, 'new_native_calls': 0, 'new_training_updates': 0,
        'seconds': time.monotonic() - started,
        'research_effect': 'Do not treat native success or one local improvement as a positive complete-scene trajectory. Existing same-source supervised comparison stays separate.'}
    args.out.mkdir(exist_ok=False, parents=True)
    (args.out / 'observed_support.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
