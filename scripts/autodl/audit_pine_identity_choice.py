"""Check the actual interchangeable-asset ambiguity in one sourced example.

Constructs an alternative static endpoint, not a new native execution, human
preference or motion plan. Never changes the original demonstration files.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess


PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
PREFIX = 'runs/920/P920-PINE-v1_20261106T002523Z/'
PAIRS = [('old_desk_0', 'work_desk_0'), ('old_chair_0', 'dining_chair_1_0')]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def world_signature(state):
    # Drop only instance labels. Every other recorded mesh/scale/frame/pose
    # field is retained, so a match is stronger than matching AABBs or pixels.
    objects = [{k: v for k, v in o.items()
                if k not in ['object_id', 'name', 'description']}
               for o in state['objects'].values()]
    return canonical(dict(room_geometry=state['room_geometry'],
                          objects=sorted(objects, key=canonical)))


def yaw(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def movement(initial, final):
    rows = []
    for oid in ['old_desk_0', 'old_chair_0']:
        before = initial['objects'][oid]['transform']
        after = final['objects'][oid]['transform']
        shift = math.dist(before['translation'], after['translation'])
        rotation = abs((yaw(after['rotation_wxyz'])-
                        yaw(before['rotation_wxyz'])+180) % 360-180)
        rows.append(dict(object_id=oid, straight_endpoint_displacement_m=shift,
                         absolute_endpoint_yaw_change_deg=rotation))
    return dict(objects=rows,
                total_straight_endpoint_displacement_m=sum(x['straight_endpoint_displacement_m'] for x in rows),
                total_absolute_endpoint_yaw_change_deg=sum(x['absolute_endpoint_yaw_change_deg'] for x in rows),
                not_actual_motion_path_cost=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Keep previous audit outputs')
    a.out.mkdir(parents=True)
    sources = []

    def read(name):
        raw = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show', PIN+':'+PREFIX+name])
        sources.append(dict(path=PREFIX+name, bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)

    initial = read('native/initial_state.json')
    final = read('native/final_state.json')
    task = read('TASK_INPUT.json')
    alternative = copy.deepcopy(final)
    shapes = []
    for old, new in PAIRS:
        left, right = final['objects'][old], final['objects'][new]
        skip = ['object_id', 'name', 'description', 'transform']
        left_shape = {k: v for k, v in left.items() if k not in skip}
        right_shape = {k: v for k, v in right.items() if k not in skip}
        assert left_shape == right_shape
        assert left['sdf_path'] == right['sdf_path']
        assert left['geometry_path'] == right['geometry_path']
        alternative['objects'][old]['transform'] = copy.deepcopy(right['transform'])
        alternative['objects'][new]['transform'] = copy.deepcopy(left['transform'])
        shapes.append(dict(pair=[old, new], equal_all_recorded_nonidentity_shape_fields=True,
                           sdf_path=left['sdf_path'], geometry_path=left['geometry_path'],
                           scale_factor=left['scale_factor']))
    assert set(alternative['objects']) == set(final['objects'])
    assert set(initial['objects']) <= set(alternative['objects'])
    assert world_signature(final) == world_signature(alternative)
    outside_pairs = set(final['objects']) - {x for pair in PAIRS for x in pair}
    assert all(final['objects'][x] == alternative['objects'][x] for x in outside_pairs)
    original_move = movement(initial, final)
    alternative_move = movement(initial, alternative)
    assert alternative_move['total_straight_endpoint_displacement_m'] < original_move['total_straight_endpoint_displacement_m']
    assignments = []
    for desk_swap, chair_swap in itertools.product([False, True], repeat=2):
        candidate = copy.deepcopy(final)
        for swap, (old, new) in zip([desk_swap, chair_swap], PAIRS):
            if swap:
                candidate['objects'][old]['transform'] = copy.deepcopy(final['objects'][new]['transform'])
                candidate['objects'][new]['transform'] = copy.deepcopy(final['objects'][old]['transform'])
        same_world = world_signature(candidate) == world_signature(final)
        assert same_world
        cost = movement(initial, candidate)
        assignments.append(dict(desk_swap=desk_swap, chair_swap=chair_swap,
                                same_static_world=True, movement=cost))
    best = min(assignments, key=lambda x: x['movement']['total_straight_endpoint_displacement_m'])
    assert best['desk_swap'] and best['chair_swap']
    result = dict(
        source_pin=PIN, task=task['researcher_authored_brief'], interchangeable_pairs=shapes,
        endpoint_object_id_sets_equal=True, retained_initial_ids_preserved=True,
        world_signature_ignoring_only_instance_labels_equal=True,
        world_signature_sha256=hashlib.sha256(world_signature(final).encode()).hexdigest(),
        other_object_records_unchanged=len(outside_pairs),
        original_retained_instance_movement=original_move,
        swapped_retained_instance_movement=alternative_move,
        ordinary_assignment_control=dict(
            rule='Exhaustive two-by-two same-shape endpoint assignment; ordinary matching',
            endpoint_pose_targets_obtained_posthoc_from_demonstration=True,
            candidate_assignments=assignments,
            selected=best,
            not_independent_generation_or_a_new_search_method=True),
        static_consequence='Any evaluator depending only on the same world '
           'geometry and unordered workstation uses has the same input. '
           'Task identity retention also holds, but identity-specific uses must '
           'not be assumed invariant for other tasks.',
        source_scope='Published Pine has a bespoke window desk and a reused IKEA '
           'desk. Our two identical substituted meshes erase that asset-role '
           'distinction. This does not invalidate the reported real project.',
        label_revision='Original arrangement remains one positive imitation '
           'example. Do not mark this alternative negative or require its '
           'particular old/new role assignment as a unique functional answer.',
        no_claim='No user preference, actual carrying route, stability or '
           'new simulator execution is established by the static permutation.',
        source_receipts=sources, new_model_calls=0, new_native_scene_runs=0,
        new_training=0)
    (a.out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (a.out/'alternative_static_state.json').write_text(json.dumps(alternative, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(world_equal=True,
        original_displacement_m=original_move['total_straight_endpoint_displacement_m'],
        alternative_displacement_m=alternative_move['total_straight_endpoint_displacement_m'],
        original_yaw_deg=original_move['total_absolute_endpoint_yaw_change_deg'],
        alternative_yaw_deg=alternative_move['total_absolute_endpoint_yaw_change_deg'],
        new_scene_executions=0)))


if __name__ == '__main__':
    main()
