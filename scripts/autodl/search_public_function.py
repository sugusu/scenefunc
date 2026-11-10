"""Bounded ordinary proposal search over public endpoint geometry.

This is NOT an Agent, native rollout or whole-scene generator. Candidate poses
come from explicit task regions and a room-relative lattice, never saved model
actions, teacher decisions or future states. Joint candidates can temporarily
lose a functional predicate; acceptance is against the INITIAL state.
"""
import copy
import itertools
import math

import numpy as np

from audit_prospective_guard import world_box, furniture_pairs, outside_xy, rectangular_inner_bounds
from public_function_preview import predict, structural_boxes


def apply_move(parent, call):
    state = copy.deepcopy(parent)
    a = call['arguments']; theta = math.radians(a['yaw'])
    state['objects'][a['object_id']]['transform'] = {
        'translation': [a['x'], a['y'], 0.0],
        'rotation_wxyz': [math.cos(theta/2), 0., 0., math.sin(theta/2)]}
    return state


def yaw_of(o):
    w, x, y, z = o['transform']['rotation_wxyz']
    return math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def moves(parent, geometry):
    """Fixed ordering: explicit target, minimal clearance, then nearest lattice.

    Preserve current yaw. This bounded translation control does not claim
    coverage of rotations, asset changes, other task types or all feasible poses.
    """
    p = parent['declared_requirements']; mode = parent['requested_mode']
    rect = rectangular_inner_bounds(parent['objects'])
    if rect is None: raise ValueError('Rectangular public shell required')
    queues = []
    for oid in sorted(p['movable_ids']):
        o = parent['objects'][oid]
        if o.get('immutable'): raise ValueError('Movable ID is immutable')
        cur = o['transform']['translation']; lo, hi = world_box(o)
        points = []
        if mode == 'desk_work' and oid == p['stool_id']:
            r = p['seat_target_region_xyxy_m']
            points.append(((r[0]+r[2])/2, (r[1]+r[3])/2))
        forbidden = []
        if mode == 'clear_floor': forbidden.append(p['reserved_floor_xyxy_m'])
        elif mode == 'desk_work':
            a, b = p['knee_volume_lo_hi_m']
            if hi[2] > a[2] and lo[2] < b[2]: forbidden.append([a[0], a[1], b[0], b[1]])
        else: raise ValueError('Unsupported task')
        for r in forbidden:
            if min(hi[0], r[2]) > max(lo[0], r[0]) and min(hi[1], r[3]) > max(lo[1], r[1]):
                points.extend([(cur[0]+r[0]-hi[0]-.02, cur[1]),
                               (cur[0]+r[2]-lo[0]+.02, cur[1]),
                               (cur[0], cur[1]+r[1]-hi[1]-.02),
                               (cur[0], cur[1]+r[3]-lo[1]+.02)])
        xs = np.linspace(rect[0]-(lo[0]-cur[0])+.02, rect[2]-(hi[0]-cur[0])-.02, 9)
        ys = np.linspace(rect[1]-(lo[1]-cur[1])+.02, rect[3]-(hi[1]-cur[1])-.02, 9)
        lattice = sorted(itertools.product(xs, ys), key=lambda q: ((q[0]-cur[0])**2+(q[1]-cur[1])**2, q))
        points.extend(lattice); seen = set(); calls = []
        for x, y in points:
            key = (round(float(x), 9), round(float(y), 9))
            if key in seen or math.dist(key, cur[:2]) < 1e-8: continue
            seen.add(key)
            calls.append({'name':'move_furniture_tool', 'arguments':{
                'object_id':oid, 'x':key[0], 'y':key[1], 'yaw':yaw_of(o)}})
        queues.append(calls)
    # Round-robin objects: neither a known good object nor an entire lattice
    # gets privileged access to the finite evaluation budget.
    for batch in itertools.zip_longest(*queues):
        for call in batch:
            if call is not None: yield call


def primitive(state, geometry):
    objects = state['objects']
    pairs = {tuple(sorted([p['a'], p['b']])) for p in furniture_pairs(objects)}
    wall_pairs = set()
    for oid, o in objects.items():
        if o['object_type'] != 'furniture': continue
        lo, hi = map(np.array, world_box(o))
        for wid, wl, wh in structural_boxes(geometry, float(hi[2])):
            if np.all(np.minimum(hi, wh)-np.maximum(lo, wl) > 1e-6): wall_pairs.add((oid, wid))
    outside = set(outside_xy(objects, rectangular_inner_bounds(objects)))
    return {'furniture_pairs':pairs, 'wall_pairs':wall_pairs, 'outside':outside}


def activity_keys(mode):
    return (['central_activity_floor_clear'] if mode == 'clear_floor' else
            ['desk_knee_volume_clear', 'work_stool_in_declared_region', 'desk_work_patch_supported_and_unoccupied'])


def task_residual(state):
    """Only a beam tie-break, not a replacement for frozen boolean acceptance."""
    p = state['declared_requirements']; mode = state['requested_mode']
    if mode == 'desk_work':
        x, y, _ = state['objects'][p['stool_id']]['transform']['translation']
        r = p['seat_target_region_xyxy_m']
        return math.hypot(max(r[0]-x, 0, x-r[2]), max(r[1]-y, 0, y-r[3]))
    r = p['reserved_floor_xyxy_m']; area = 0.
    for oid in p['movable_ids']:
        lo, hi = world_box(state['objects'][oid])
        area += max(0, min(hi[0],r[2])-max(lo[0],r[0])) * max(0,min(hi[1],r[3])-max(lo[1],r[1]))
    return area


def search(initial, geometry, *, candidate_budget=128, root_budget=64, beam_width=4, max_depth=2):
    if max_depth != 2 or root_budget > candidate_budget: raise ValueError('Frozen depth/budget contract')
    root_primitive = primitive(initial, geometry)
    root_true = {k for k,v in initial['checks'].items() if v is True}
    records = []; beam = []; seen = set(); solution = None
    def evaluate(parent, path, call):
        nonlocal solution
        state = apply_move(parent, call)
        key = tuple((oid, tuple(state['objects'][oid]['transform']['translation']))
                    for oid in sorted(initial['declared_requirements']['movable_ids']))
        if key in seen: return None
        seen.add(key)
        preview = predict(parent, call, geometry)
        if 'predicted_checks' not in preview: raise ValueError('Preview did not cover generated move')
        state['checks'] = preview['predicted_checks']
        actual = primitive(state, geometry)
        new = {k: sorted(actual[k]-root_primitive[k]) for k in actual}
        lost = sorted(k for k in root_true if state['checks'].get(k) is not True)
        task_ok = all(state['checks'].get(k) is True for k in activity_keys(initial['requested_mode']))
        complete = task_ok and not lost and not any(new.values())
        actions = path+[call]
        record = {'candidate':len(records)+1, 'depth':len(actions), 'actions':actions,
                  'checks':state['checks'], 'new_primitive_violations':new,
                  'lost_initial_true_checks':lost, 'activity_pass':task_ok,
                  'scoped_endpoint_pass':complete, 'preview_status':preview['status'],
                  'task_residual':task_residual(state)}
        records.append(record)
        motion = sum(math.dist(initial['objects'][oid]['transform']['translation'],
                               state['objects'][oid]['transform']['translation'])
                     for oid in initial['declared_requirements']['movable_ids'])
        score = (sum(map(len,new.values())), len(lost),
                 sum(state['checks'].get(k) is not True for k in activity_keys(initial['requested_mode'])),
                 task_residual(state), motion, record['candidate'])
        if complete: solution = {'actions':actions, 'state':state, 'candidate':record['candidate']}
        return (score,state,actions)
    for call in moves(initial,geometry):
        if len(records) >= root_budget or solution: break
        item = evaluate(initial,[],call)
        if item: beam.append(item)
    if not solution:
        beam = sorted(beam,key=lambda x:x[0])[:beam_width]
        streams = [iter(moves(state,geometry)) for _,state,_ in beam]
        while len(records) < candidate_budget and streams and not solution:
            advanced = False
            for i, stream in enumerate(streams):
                if len(records) >= candidate_budget or solution: break
                try: call = next(stream)
                except StopIteration: continue
                advanced = True
                evaluate(beam[i][1],beam[i][2],call)
            if not advanced: break
    return {'solution':solution,'evaluations':records,'candidate_count':len(records),
            'generated_unique_states':len(seen), 'status':'nominal_endpoint_found' if solution else 'bounded_search_exhausted',
            'whole_scene_success':None, 'native_calls':0, 'model_calls':0}
