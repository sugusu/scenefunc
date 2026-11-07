"""Measure the two fixed researcher references, without native edit calls.

Candidate endpoint states are explicitly labelled proposals. Results must not
be described as execution or model gains. Native bundle hashes are validated.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from native_project import validate_bundle, sha256
from preflight_sourced_trajectory import boxes, inspect
from measure_room_access import measure
from measure_free_floor import rigid_sweep_diagnostic
from replay_sourced_decision import reserved_area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    task = json.loads((a.run/'TASK_INPUTS.json').read_text())
    plan = json.loads((a.run/'FEASIBILITY_PLAN.json').read_text())
    manifest, house, _, _ = validate_bundle(a.bundle)
    initial = house['rooms'][manifest['selected_room_id']]
    room = a.bundle/('room_'+manifest['selected_room_id'])

    def write(name, obj):
        (a.out/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

    def evaluate(state):
        bounds = boxes(state)
        knee = task['tasks']['desk_work']['knee_volume_lo_hi_m']
        knee_hits = []
        for oid,(lo,hi) in bounds.items():
            # Seat itself is outside the declared desk cavity and is checked
            # independently. Tabletop and side supports remain in this test.
            if np.all(np.minimum(hi,knee[1])-np.maximum(lo,knee[0]) > 1e-6):
                knee_hits.append(oid)
        seat = state['objects']['recon_work_stool_0']['transform']['translation']
        region = task['tasks']['desk_work']['seat_target_region_xyxy_m']
        floor = reserved_area(state,room,a.bundle,task['tasks']['clear_floor']['reserved_floor_xyxy_m'])
        floor['scope'] = 'Researcher-requested central activity rectangle, not unfolded-bed capacity.'
        access = measure(state,room,a.bundle,plan['access_settings'])
        geometric = inspect(state)
        return dict(clear_floor=floor,desk_knee_colliders=knee_hits,
            desk_seat_in_declared_working_region=region[0]<=seat[0]<=region[2] and region[1]<=seat[1]<=region[3],
            desk_activity_proxy=not knee_hits and region[0]<=seat[0]<=region[2] and region[1]<=seat[1]<=region[3],
            access=access,geometry=geometric,
            whole_room_human_usability='unknown',native_execution=False)

    result = dict(started_at=datetime.now(timezone.utc).isoformat(),
        bundle_manifest_sha256=sha256(a.bundle/'manifest.json'),
        plan_sha256=sha256(a.run/'FEASIBILITY_PLAN.json'),
        evaluator_sha256=sha256(Path(__file__)),initial=evaluate(initial),candidates={},
        model_calls=0,training_updates=0,native_edits=0)
    write('initial_state.json',initial)
    for condition, commands in plan['references'].items():
        state = copy.deepcopy(initial)
        steps = []
        for command in commands:
            args = command['arguments']
            assert args['yaw'] == 0
            proposed = copy.deepcopy(state)
            proposed['objects'][args['object_id']]['transform'] = {
                'translation':[args['x'],args['y'],0], 'rotation_wxyz':[1,0,0,0]}
            sweep = rigid_sweep_diagnostic(state,proposed,args['object_id'],room,a.bundle)
            steps.append(dict(command=command,sweep=sweep,endpoint=inspect(proposed)))
            state = proposed
        assessment = evaluate(state)
        motion_ok = all(s['sweep']['clear_under_declared_sweep_model'] and
            not s['endpoint']['positive_3d_AABB_pairs'] and not s['endpoint']['outside_declared_wall_inner_faces'] for s in steps)
        result['candidates'][condition] = dict(steps=steps,assessment=assessment,
            trajectory_preflight_passed=motion_ok,
            activity_proxy=assessment['clear_floor']['clear_under_declared_envelope'] if condition=='clear_floor' else assessment['desk_activity_proxy'],
            native_execution=False)
        write(condition+'_proposed_state.json',state)
    result['ended_at'] = datetime.now(timezone.utc).isoformat()
    result['status'] = 'measured_with_rejected_candidates' if any(not c['trajectory_preflight_passed'] for c in result['candidates'].values()) else 'references_geometrically_checked'
    write('RESULT.json',result)
    print(json.dumps({k:{'motion':v['trajectory_preflight_passed'],'activity':v['activity_proxy'],
        'failed_access':{str(s['square_body_width_m']):[k for k,t in s['targets'].items() if not t['reachable_from_entry']] for s in v['assessment']['access']['scenarios']},
        'sweep_obstacles':[s['sweep']['obstacle_overlaps'] for s in v['steps']]} for k,v in result['candidates'].items()}))


if __name__ == '__main__':
    main()
