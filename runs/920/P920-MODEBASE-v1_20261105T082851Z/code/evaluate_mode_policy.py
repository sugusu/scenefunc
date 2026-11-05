"""Evaluate explicit activity constraints, preserving geometry and identity failures."""
import copy
import itertools
import numpy as np
from measure_table_modes import table_modes
from measure_free_floor import footprint
from preflight_sourced_trajectory import inspect
from inspect_native_mesh_contacts import world_transform


def evaluate(state, initial, room, root, plan):
    required = set(initial['objects'])
    present = set(state['objects'])
    geometry = inspect(state)
    immutable = [k for k in required & present if k not in plan['movable_ids']
                 and state['objects'][k] != initial['objects'][k]]
    altered = []
    for k in required & present:
        a, b = copy.deepcopy(initial['objects'][k]), copy.deepcopy(state['objects'][k])
        a.pop('transform'); b.pop('transform')
        if a != b: altered.append(k)
    identity_ok = required == present and not immutable and not altered
    result = {'inventory_preserved': required == present, 'missing_ids': sorted(required-present),
              'extra_ids': sorted(present-required), 'fixed_objects_changed': immutable,
              'nontransform_records_changed': altered, 'identity_ok': identity_ok,
              'geometry': geometry, 'measurement': None, 'condition_pass': False}
    if not set(plan['movable_ids']).issubset(present):
        result['measurement_error'] = 'Required table or seat deleted; mode cannot pass.'
        return result
    # Evaluate every assignment rather than enforce the author's previous target coordinates.
    # Four fixed named seats are matched to the two opposite long edges of both tables.
    spec = plan['measurement_settings']
    slots = [(t, side) for t in spec['table_ids'] for side in (-1, 1)]
    chairs = plan['chair_ids']
    valid = {}; details = {}
    for chair_id, (table_id, side) in itertools.product(chairs, slots):
        chair, table = state['objects'][chair_id], state['objects'][table_id]
        tm = world_transform(table); inverse = np.linalg.inv(tm)
        center = inverse @ np.r_[world_transform(chair)[:3, 3], 1]
        # A rotated square stool has a different projection onto the table normal.
        points = np.array([[x, y, 0, 1] for x, y in footprint(chair).exterior.coords]) @ inverse.T
        edge = table['bbox_max'][1] if side > 0 else table['bbox_min'][1]
        chair_edge = points[:, 1].min() if side > 0 else points[:, 1].max()
        gap = float(side * (chair_edge-edge))
        width, depth = spec['place_rectangle_width_depth_m']
        lo, hi = table['bbox_min'], table['bbox_max']
        within = lo[0] <= center[0]-width/2 and center[0]+width/2 <= hi[0] and depth <= hi[1]-lo[1]
        valid[(chair_id, table_id, side)] = bool(within and .05 <= gap <= .4)
        details[f'{chair_id}:{table_id}:{side}'] = {'gap_m': gap, 'place_on_table': bool(within)}
    assignments = []
    for order in itertools.permutations(chairs):
        rows = [{'chair_id': c, 'table_id': t, 'side_y': side} for c, (t, side) in zip(order, slots)]
        count = sum(valid[(x['chair_id'], x['table_id'], x['side_y'])] for x in rows)
        work_rows = [x for x in rows if x['chair_id'] in spec['work_chair_ids']]
        work_ok = len({x['table_id'] for x in work_rows}) == 2 and all(valid[(x['chair_id'], x['table_id'], x['side_y'])] for x in work_rows)
        assignments.append((count == 4 if plan['condition']=='hosting' else work_ok, count, rows))
    # Deterministic assignment from current geometry only, not a reference solution.
    _, _, chosen = max(assignments, key=lambda x: (x[0], x[1]))
    local = dict(spec, seat_assignments=chosen)
    m = table_modes(state, room, root, local)
    m['seat_assignment_candidates'] = details
    m['selected_seat_assignments'] = chosen
    # Override the older fixed-seat approximation using the exact projected stool bounds.
    gap = m['table_geometry']['footprint_gap_m']
    m['mode_geometry']['separate_work'] = bool(gap >= spec['separate_min_gap_m'] and any(
        len({x['table_id'] for x in rows if x['chair_id'] in spec['work_chair_ids']}) == 2 and all(
            valid[(x['chair_id'], x['table_id'], x['side_y'])] for x in rows if x['chair_id'] in spec['work_chair_ids'])
        for _, _, rows in assignments))
    m['mode_geometry']['joined_four_place_hosting'] = bool(m['table_geometry']['joined_under_declared_threshold'] and
        any(all(valid[(x['chair_id'], x['table_id'], x['side_y'])] for x in rows) for _, _, rows in assignments))
    # Use table-normal standing targets, independent of the stool's yaw.
    from measure_room_access import measure
    seat_access = []
    for width in spec['access_settings']['square_body_widths_m']:
        points = {}
        for row in chosen:
            c = state['objects'][row['chair_id']]; t = state['objects'][row['table_id']]
            direction = world_transform(t)[:2, 1] * row['side_y']
            coords = np.array(footprint(c).exterior.coords)
            center = world_transform(c)[:2, 3]
            radius = max((coords-center) @ direction)
            points[row['chair_id']] = (center+direction*(radius+width/2+.05)).tolist()
        cfg = dict(spec['access_settings'], square_body_widths_m=[width], approach_points_xy_m=points)
        seat_access.append(measure(state, room, root, cfg)['scenarios'][0])
    m['seat_approaches'] = seat_access
    needed = chairs if plan['condition']=='hosting' else spec['work_chair_ids']
    mode = 'joined_four_place_hosting' if plan['condition']=='hosting' else 'separate_work'
    access_ok = all(all(t['reachable_from_entry'] for t in s['targets'].values()) for s in m['fixed_access']['scenarios'])
    seat_ok = all(all(s['targets'][k]['reachable_from_entry'] for k in needed) for s in seat_access)
    geometry_ok = not geometry['outside_declared_wall_inner_faces'] and not geometry['positive_3d_AABB_pairs']
    result.update(measurement=m, geometry_ok=geometry_ok, fixed_access_ok=access_ok,
                  required_seat_approaches_ok=seat_ok,
                  condition_pass=bool(identity_ok and geometry_ok and access_ok and seat_ok and m['mode_geometry'][mode]))
    return result


def feedback(result):
    m = result['measurement']
    return {'identity_ok':result['identity_ok'], 'geometry':result['geometry'],
            'mode_geometry':m['mode_geometry'] if m else None,
            'table_geometry':m['table_geometry'] if m else None,
            'seat_alignment_candidates':m['seat_assignment_candidates'] if m else None,
            'fixed_unreachable':[{ 'width_m':s['square_body_width_m'], 'ids':[k for k,v in s['targets'].items() if not v['reachable_from_entry']]}
                for s in m['fixed_access']['scenarios']] if m else None,
            'condition_pass_under_declared_proxy':result['condition_pass'],
            'limits':'Researcher geometry and access proxies; not comfort, real wheel operation or expert preference. No target coordinates supplied.'}
