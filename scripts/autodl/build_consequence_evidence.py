"""Build scoped consequence records and replay ordinary search controls.

No training, model call, new geometry search, or new scene execution. Existing
geometric previews are explicitly distinct from executed native terminals.
Unmeasured consequences remain null, never fabricated as negative labels.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-base', type=Path)
    ap.add_argument('--source-bundle', type=Path, help='Published small raw records; permits independent stdlib-only recomputation')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if (args.work_base is None) == (args.source_bundle is None):
        ap.error('Choose exactly one of --work-base or --source-bundle')
    w = args.work_base.resolve() if args.work_base else None
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.source_bundle:
        bundle = args.source_bundle.resolve()
        candidates_file, result_file, response_file = [bundle / f for f in ['candidates.json', 'result.json', 'evaluation.json']]
        sources = [candidates_file, result_file, response_file, bundle / 'ordinary_control.py', bundle / 'ordinary_control_protocol.json', bundle / 'strong_control.json']
        initial_audit_file = bundle / 'initial_mesh_audit.json'
        brief_directory = bundle
    else:
        old = w / 'joint_use_feedback_20261104'
        candidates_file = old / 'ordinary_control/B/candidates.json'
        result_file = old / 'ordinary_control/B/result.json'
        response_file = old / 'evaluation.json'
        sources = [candidates_file, result_file, response_file, old / 'ordinary_control.py',
                   old / 'ordinary_control_protocol.json', w / 'joint_use_witness_20261104/strong_control.json']
        brief_directory = w / 'paris_A_inference_20261104'
        initial_audit_file = w / 'paris_local_run_results_20261104/B_after_feedback_audit.json'
    sources += [brief_directory / 'input_A.json', brief_directory / 'input_B.json']
    sources.append(initial_audit_file)
    if args.source_bundle:
        for entry in read(bundle / 'source_manifest.json'):
            if digest(bundle / entry['name']) != entry['sha256']:
                raise ValueError('Source bundle hash mismatch: ' + entry['name'])
    before_hashes = {str(p): digest(p) for p in sources}
    raw = read(candidates_file)
    result = read(result_file)
    model_rows = read(response_file)['rows']
    parent = result['original_checkpoint_sha256']
    initial_audit = read(initial_audit_file)
    assert initial_audit['checkpoint'] == result['original_checkpoint']
    assert len(initial_audit['furniture_surface_intersection_pairs']) == result['before_collisions']
    before_joint = result['before_joint']
    assert before_joint['finite_pool_complete'] is True
    assert before_joint['status'] in ('witness', 'not_found')
    controls = read(sources[5])['rows']
    initial = [r for r in controls if r['arm'] == result['arm'] and
               r['stage'] == 'mesh_feedback_sdpa_kv' and r['mode'] == 'scene_section_free']
    assert len(initial) == 1 and initial[0]['joint'] == before_joint
    before_separate = all(initial[0]['pool_sizes'][k] > 0 for k in ('work', 'dining'))
    pool = []
    base = {'within_room': not initial_audit['out_of_rectangular_floor_objects'], 'no_new_mesh_intersection': True,
            'finite_separate_witness': before_separate,
            'finite_joint_witness': before_joint['status'] == 'witness',
            'whole_room_success': None, 'human_preference': None,
            'storage_can_accept_work_materials': None}
    pool.append({'id': 'keep', 'action': None, 'action_source': 'explicit no-op',
                 'execution_kind': 'fresh native replay recorded separately in 20261110_cpu_continuation',
                 'parent_checkpoint_sha256': parent, 'scope_values': base, 'movement_m': 0.0,
                 'furniture_intersections': result['before_collisions'], 'eligible_for_strict_planar_goal': False})
    for i, row in enumerate(raw):
        reason = row.get('reason')
        bounded = reason != 'outside_rectangular_floor'
        safe = None if not bounded else reason != 'new_mesh_intersection'
        joint = row.get('joint')
        joint_value = None if joint is None else joint['status'] == 'witness'
        if joint is not None:
            assert joint['finite_pool_complete'] is True
        assert row['accepted'] is (bounded and safe is True and joint_value is True)
        values = {'within_room': bounded, 'no_new_mesh_intersection': safe,
                  'finite_separate_witness': True if joint_value is True else None,
                  'finite_joint_witness': joint_value, 'whole_room_success': None,
                  'human_preference': None, 'storage_can_accept_work_materials': None}
        pool.append({'id': f'preview_{i:02d}',
                     'action': {'type': 'mesh_translation_preview', 'object_id': row['object_id'], 'offset_xy_m': row['offset_xy_m']},
                     'action_source': 'frozen ordinary candidate generator; no new proposal in this audit',
                     'execution_kind': 'in-memory geometric preview; not a saved native terminal',
                     'parent_checkpoint_sha256': parent, 'scope_values': values,
                     'movement_m': math.hypot(*row['offset_xy_m']),
                     'furniture_intersections': len(row['furniture_pairs']) if 'furniture_pairs' in row else None,
                     'eligible_for_strict_planar_goal': row['accepted'],
                     'logged_preview_seconds': row.get('seconds'),
                     'evaluation_censoring': reason if joint is None else None})
    assert sum(p['eligible_for_strict_planar_goal'] for p in pool) == result['eligible_candidates']
    scopes = {
        'A': {'brief_source': str(brief_directory / 'input_A.json'), 'label_field': 'finite_separate_witness',
              'limitation': 'Separate zero-thickness rectangles only. Storage and the transition between activities are unmeasured.'},
        'B': {'brief_source': str(brief_directory / 'input_B.json'), 'label_field': 'finite_joint_witness',
              'limitation': 'Disjoint zero-thickness rectangles on the same fixed 1cm/0-or-90-degree pool. Human use and continuous feasibility are unmeasured.'},
    }
    records = []
    for item in pool:
        for condition, scope in scopes.items():
            value = item['scope_values'][scope['label_field']]
            records.append({'record_id': item['id'] + '__' + condition,
                            'project_family': 'paris_development_shell', 'parent_checkpoint_sha256': parent,
                            'candidate_id': item['id'], 'condition': condition, 'action': item['action'],
                            'execution_kind': item['execution_kind'], 'metric': scope['label_field'],
                            'target': value, 'mask': value is not None,
                            'target_origin': ('measured_initial_finite_pool' if item['id'] == 'keep' else
                                              'unmeasured' if value is None else
                                              'logical_implication_from_joint_witness' if condition == 'A' else
                                              'recorded_finite_pool_check'),
                            'provenance': 'recorded geometric preview or measured no-op; not independent human preference',
                            'whole_room_target': None, 'whole_room_mask': False,
                            'usable_for_new_training_now': False})
    with (out / 'scoped_records.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    dump(out / 'candidate_pool.json', pool)
    dump(out / 'conditions.json', scopes)

    # Order functions see proposals only, not outcome labels. Success values are
    # revealed one by one by this archival oracle. This is retrospective replay,
    # not a prospectively timed benchmark or a learned-policy result.
    fixed = list(range(len(pool)))
    minimal_move = sorted(fixed, key=lambda i: (pool[i]['movement_m'], i))
    def replay(order):
        rows, found = [], None
        for k, i in enumerate(order, 1):
            candidate = pool[i]
            if found is None and candidate['eligible_for_strict_planar_goal']:
                found = {'candidate': candidate['id'], 'queries': k, 'movement_m': candidate['movement_m'],
                         'remaining_intersections': candidate['furniture_intersections']}
            rows.append({'queries': k, 'found_scoped_witness_without_new_recorded_geometry_harm': found is not None})
        return {'first_success': found, 'budget_curve': rows}
    n, g = len(pool), sum(p['eligible_for_strict_planar_goal'] for p in pool)
    random_curve = [{'queries': k, 'hit_probability': 1.0 if k > n-g else 1-math.comb(n-g, k)/math.comb(n, k)} for k in range(1, n+1)]
    selected = min((p for p in pool if p['eligible_for_strict_planar_goal']),
                   key=lambda p: (p['furniture_intersections'], p['movement_m'], p['id']))
    control = {'scope': 'Retrospective candidate-oracle replay; proposal order never reads outcomes; quality is limited to recorded proxies',
               'fixed_order': replay(fixed), 'minimal_move_order': replay(minimal_move),
               'uniform_random_without_replacement': random_curve,
               'full_pool_measured_selection': selected['id'],
               'learning_upper_reference': 'The full-pool best is an oracle reference, not a trained predictor',
               'original_search_seconds': result['search_seconds'],
               'timing_limits': '14 outside-bound candidates have no individual timer. Stored run wall time includes setup; ordering replay does not establish wall-clock savings or cross-hardware efficiency.',
               'full_scene_success': None}
    dump(out / 'ordinary_controls.json', control)
    attempts = []
    for row in model_rows:
        attempts.append({'condition': row['condition'], 'before_scene_sha256': row['baseline_scene_sha256'],
                         'after_scene_sha256': row['scene_sha256'], 'scene_unchanged': row['baseline_scene_sha256'] == row['scene_sha256'],
                         'calls': [e['call'] for e in row['tool_events']], 'responses': [e['response'] for e in row['tool_events']],
                         'any_tool_success': any(e['response'].get('success') is True for e in row['tool_events']),
                         'interpretation': 'Unsupported-tool attempt is not a changed-layout preference sample'})
    dump(out / 'actual_model_attempts.json', attempts)
    known = collections.Counter((r['condition'], r['target']) for r in records)
    summary = {'real_project_families': 1, 'candidate_parent_states': 1, 'geometric_previews': len(raw), 'explicit_noop': 1,
               'scoped_rows': len(records), 'ordinary_eligible_previews': g,
               'activity_measured_previews': sum('joint' in r for r in raw),
               'activity_censored_previews': sum('joint' not in r for r in raw),
               'known_positive_A': known['A', True], 'known_negative_A': known['A', False], 'unknown_A': known['A', None],
               'known_positive_B': known['B', True], 'known_negative_B': known['B', False], 'unknown_B': known['B', None],
               'existing_model_attempts': len(attempts), 'successful_tools_in_those_attempts': sum(a['any_tool_success'] for a in attempts),
               'distinct_model_terminal_scene_hashes': len({a['after_scene_sha256'] for a in attempts}),
               'independent_generalization_test_families': 0, 'human_preference_labels': 0,
               'training_authorized': False, 'trainable_full_room_preferences': 0,
               'actual_new_model_calls': 0, 'new_scene_executions': 0,
               'conclusion': 'Enough for a masked-label and search-control development fixture; not enough for learning or generalization claims'}
    dump(out / 'summary.json', summary)
    after_hashes = {str(p): digest(p) for p in sources}
    assert after_hashes == before_hashes
    dump(out / 'sources.json', {'all_unchanged': True, 'hashes': before_hashes})
    print(json.dumps({'summary': summary, 'ordinary_first_hits': {k: control[k]['first_success'] for k in ['fixed_order', 'minimal_move_order']}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
