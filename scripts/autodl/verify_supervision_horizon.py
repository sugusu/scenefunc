"""Verify the saved candidate-filter audit, not a new scene experiment."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

from audit_prospective_guard import predict, world_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--effects', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    fragments = [json.loads(line) for line in (args.effects/'source_fragments.jsonl').read_text().splitlines()]
    transitions = {r['id']: r for r in map(json.loads, (args.effects/'transitions.jsonl').read_text().splitlines())}
    rows = json.loads((args.audit/'guard_v2/rows.json').read_text())
    by_id = {r['id']: r for r in rows}
    assert set(by_id) == set(transitions) == {f['id'] for f in fragments}
    summary = json.loads((args.audit/'guard_v2/summary.json').read_text())
    source_hashes = {}
    for result in [summary, json.loads((args.audit/'segments_v2/summary.json').read_text())]:
        for path, expected in result['source_sha256'].items():
            data = (subprocess.check_output(['git', 'show', '75fc783f86d4fea78714639e110fbe77476ee106:'+path])
                    if path.startswith('runs/920/') else Path(path).read_bytes())
            actual = hashlib.sha256(data).hexdigest()
            assert actual == expected, path
            source_hashes[path] = actual
    prediction_matches = 0
    new_primitive = []
    withdrawn_positive = []
    for f in fragments:
        t = transitions[f['id']]
        r = by_id[f['id']]
        parent_before = copy.deepcopy(f['parent'])
        result = predict(f['parent'], t['call'])
        assert f['parent'] == parent_before, 'Prediction mutated the source'
        assert result == r['prediction'], f['id']
        # Replace all outcome/measurement fields: dispatch still sends only pre-state and call.
        altered = {**f, 'after_measurements': {'corrupted': True}, 'before_measurements': None}
        assert predict(altered['parent'], t['call']) == result
        prediction_matches += 1
        primitive = bool(r['actual_new_overlap_pairs'] or r['actual_new_outside_ids'] or r['actual_new_missing_ids'])
        if primitive and not r['actual_damage']:
            new_primitive.append(r['id'])
        if primitive and r['actual_improve_no_harm']:
            withdrawn_positive.append(r['id'])
    assert len(new_primitive) == summary['hidden_primitive_damage_steps'] == 11
    assert len(withdrawn_positive) == summary['rejected_improve_no_harm'] == 3
    assert all(r['pre_pair_membership_agrees_with_archived_checker'] and r['pre_xy_boundary_agrees_with_archived_checker'] for r in rows)
    moves = [r for r in rows if r['projection_change_agrees_with_actual_record'] is not None]
    assert len(moves) == 35 and all(r['projection_change_agrees_with_actual_record'] for r in moves)
    # One analytical check detects axis/quaternion-order mistakes in the projection.
    obj = {'bbox_min': [-1, -.5, 0], 'bbox_max': [1, .5, .1],
           'transform': {'rotation_wxyz': [math.sqrt(.5), 0, 0, math.sqrt(.5)], 'translation': [2, 3, 0]}}
    lo, hi = world_box(obj)
    assert all(abs(x-y) < 1e-12 for x,y in zip(lo+hi, [1.5, 2, 0, 2.5, 4, .1]))
    segments = json.loads((args.audit/'segments_v2/observed_segments.json').read_text())
    assert len(segments) == 2
    for s in segments:
        assert s['observed_prefix_to_complete_goal'] == 2
        assert not s['hypothetical_per_step_no_regression_filter_accepts_first']
        assert s['two_action_recorded_endpoint_meets_scoped_goal']
        assert s['hard_geometry_clear_at_every_recorded_endpoint']
        assert s['fixed_access_counts_preserved'] and s['original_instances_preserved_count'] == 27
        assert s['oracle_uses_observed_future'] and not s['training_admission']
    out = dict(status='passed', prediction_matches=prediction_matches,
        projection_change_membership_matches=len(moves), analytical_quaternion_check=True,
        source_files_verified=len(source_hashes), source_sha256=source_hashes,
        hidden_primitive_damage_ids=new_primitive,
        coarse_positive_with_new_primitive_damage_ids=withdrawn_positive,
        source_mutations=0, model_calls=0, native_tool_calls=0, training_updates=0,
        limits=['Not independent validation of the original geometry checker.',
                'New pair membership does not detect greater penetration within an existing pair.',
                'Per-instance boundary membership does not measure worsening excursion.',
                'No continuous-motion, human-use, preference or complete-room certificate.',
                'Fixed archived proposals; rejecting one would change later policy inputs.'])
    (args.audit/'validation.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
