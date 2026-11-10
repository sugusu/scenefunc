"""Current-state checks for known rectangular table/seat proxies.

No reference pose or action is consumed. Missing instances fail; unsupported
replacement geometry remains unknown. These are declared proxies, not comfort.
"""
import itertools

import numpy as np
from shapely.geometry import Polygon

from build_activity_completion_reviews import aggregate
from inspect_native_mesh_contacts import world_transform
from measure_free_floor import footprint
from measure_room_access import measure
from preflight_sourced_trajectory import inspect


MODES = ('separate_work', 'joined_four_place_hosting')


def nonpose(obj):
    return {k: v for k, v in obj.items() if k != 'transform'}


def checked_access(state, room, root, settings):
    result = measure(state, room, root, settings)
    g = state['room_geometry']
    for scenario in result['scenarios']:
        for target in scenario['targets'].values():
            x, y = target['requested_point_m']
            inside = -g['length']/2 <= x <= g['length']/2 and -g['width']/2 <= y <= g['width']/2
            target['requested_point_inside_room'] = inside
            # The generic grid routine samples its nearest cell. An external
            # target must not inherit that boundary cell's reachability.
            if not inside:
                target.update(reachable_from_entry=False, target_cell_clear=False, grid_route_length_m=None)
    return result


def measure_endpoint(state, initial, room, root, program, mode):
    if mode not in MODES:
        raise ValueError('Unsupported requested activity')
    if set(program) != {'table_ids', 'work_chair_ids', 'seat_assignments', 'joined_max_gap_m',
                        'separate_min_gap_m', 'place_rectangle_width_depth_m', 'access_settings', 'movable_ids'}:
        raise ValueError('Program must contain declared requirements only, without reference actions or endpoint metadata')
    objects, original = state['objects'], initial['objects']
    tables = program['table_ids']
    seats = program['seat_assignments']
    movable = set(program['movable_ids'])
    widths = program['access_settings']['square_body_widths_m']
    needed = program['work_chair_ids'] if mode == MODES[0] else [s['chair_id'] for s in seats]
    if (len(tables) != 2 or len(set(tables)) != 2 or not set(tables) <= original.keys()
            or len({s['chair_id'] for s in seats}) != len(seats)
            or not set(needed) <= {s['chair_id'] for s in seats}
            or not movable <= original.keys() or len(set(widths)) != len(widths) or not widths):
        raise ValueError('Invalid frozen object/requirement allocation')
    for s in seats:
        if s['table_id'] not in tables or s['chair_id'] not in original or s['side_y'] not in (-1, 1):
            raise ValueError('Invalid frozen seat assignment')
    geometry = inspect(state)
    fixed = checked_access(state, room, root, program['access_settings'])
    support = {}
    for oid in set(tables) | {s['chair_id'] for s in seats}:
        if oid not in objects:
            support[oid] = 'missing'
        elif nonpose(objects[oid]) != nonpose(original[oid]):
            support[oid] = 'changed_geometry_or_asset'
        elif not np.allclose(world_transform(objects[oid])[:3, 2], [0, 0, 1], atol=1e-6, rtol=0):
            support[oid] = 'non_upright_pose'
        else:
            support[oid] = 'supported_known_proxy'

    def unavailable(ids):
        return False if any(support[i] == 'missing' for i in ids) else None

    table_geometry = None
    if all(support[t] == 'supported_known_proxy' for t in tables):
        a, b = (objects[t] for t in tables)
        am, bm = world_transform(a), world_transform(b)
        offset = np.linalg.inv(am) @ np.r_[bm[:3, 3], 1]
        gap = float(footprint(a).distance(footprint(b)))
        top_difference = float(abs(am[2, 3] + a['bbox_max'][2] - bm[2, 3] - b['bbox_max'][2]))
        dot = float(np.dot(am[:2, 0], bm[:2, 0]))
        table_geometry = dict(footprint_gap_m=gap, relative_center_offset_m=offset[:3].tolist(),
            parallel_axis_dot=dot, top_height_difference_m=top_difference,
            joined=bool(gap <= program['joined_max_gap_m'] and abs(offset[1]) <= .03
                        and dot > .999 and top_difference <= .01))
    alignment, places, seat_points = {}, {}, {w: {} for w in widths}
    for seat in seats:
        cid, tid, side = seat['chair_id'], seat['table_id'], seat['side_y']
        if any(support[i] != 'supported_known_proxy' for i in (cid, tid)):
            alignment[cid] = dict(pass_=unavailable((cid, tid)), evidence='; '.join(support[i] for i in (cid, tid)))
            continue
        chair, table = objects[cid], objects[tid]
        tm, cm = world_transform(table), world_transform(chair)
        local = np.linalg.inv(tm) @ np.r_[cm[:3, 3], 1]
        corners = np.array(list(itertools.product(*zip(chair['bbox_min'], chair['bbox_max']))))
        local_corners = np.c_[corners, np.ones(len(corners))] @ (np.linalg.inv(tm) @ cm).T
        lo, hi = table['bbox_min'], table['bbox_max']
        edge = hi[1] if side > 0 else lo[1]
        near = local_corners[:, 1].min() if side > 0 else local_corners[:, 1].max()
        far = local_corners[:, 1].max() if side > 0 else local_corners[:, 1].min()
        gap = float(side * (near - edge))
        w, d = program['place_rectangle_width_depth_m']
        x, y = float(local[0]), edge - side*d/2
        corners2 = [[x-w/2, y-d/2], [x+w/2, y-d/2], [x+w/2, y+d/2], [x-w/2, y+d/2]]
        within = all(lo[0]-1e-6 <= q[0] <= hi[0]+1e-6 and lo[1]-1e-6 <= q[1] <= hi[1]+1e-6 for q in corners2)
        places[cid] = Polygon([(tm @ np.array([*q, 0, 1]))[:2] for q in corners2])
        alignment[cid] = dict(pass_=bool(within and .05 <= gap <= .4),
            table_id=tid, chair_envelope_to_table_edge_gap_m=gap, place_rectangle_on_table=within)
        for width in widths:
            # Use the table's outward direction and the rotated chair extent,
            # rather than assuming every chair's local Y equals table Y.
            seat_points[width][cid] = (tm @ np.array([x, far+side*(width/2+.05), 0, 1]))[:2].tolist()
    seat_access = []
    for width in widths:
        settings = dict(program['access_settings'], square_body_widths_m=[width], approach_points_xy_m=seat_points[width])
        seat_access.append(checked_access(state, room, root, settings)['scenarios'][0])
    overlap = None if len(places) != len(seats) else any(
        a.intersection(b).area > 1e-8 for a, b in itertools.combinations(places.values(), 2))
    rows = []

    def add(name, value, ids, evidence):
        rows.append(dict(requirement=name, **{'pass': value}, object_ids=ids, evidence=evidence))

    add('all_original_instances_and_assets', set(objects) == set(original) and all(
        nonpose(objects[o]) == nonpose(original[o]) for o in original.keys() & objects.keys()), sorted(original), 'native records excluding pose')
    add('fixed_objects_unchanged', all(objects.get(o) == original[o] for o in original.keys()-movable),
        sorted(original.keys()-movable), 'complete fixed native records')
    add('room_geometry_unchanged', state['room_geometry'] == initial['room_geometry'], [], 'room geometry records')
    add('no_recorded_envelope_overlap_or_boundary_violation',
        not geometry['positive_3d_AABB_pairs'] and not geometry['outside_declared_wall_inner_faces'], sorted(objects), 'current conservative AABB geometry')
    for scenario in fixed['scenarios']:
        for name, target in scenario['targets'].items():
            add(f'access_{name}_width_{scenario["square_body_width_m"]}', target['reachable_from_entry'], [], 'current fixed_access measurement')
    if mode == MODES[0]:
        add('separate_table_gap', unavailable(tables) if table_geometry is None else
            table_geometry['footprint_gap_m'] >= program['separate_min_gap_m'], tables, 'current table footprints')
    else:
        add('joined_table_geometry', unavailable(tables) if table_geometry is None else table_geometry['joined'], tables, 'current table gap, axes, height, alignment')
        add('nonoverlapping_declared_place_rectangles', None if overlap is None else not overlap, needed, 'current table-frame place polygons')
    for oid in needed:
        add('assigned_seat_alignment_'+oid, alignment[oid]['pass_'], [oid], 'current table-frame projected chair envelope and place rectangle')
        for scenario in seat_access:
            target = scenario['targets'].get(oid)
            add(f'seat_access_{oid}_width_{scenario["square_body_width_m"]}',
                target['reachable_from_entry'] if target else alignment[oid]['pass_'], [oid], 'current outward approach; unsupported geometry unknown')
    review = dict(requested_mode=mode, checks=rows, measured_requirements_status=aggregate(rows),
        whole_room_success='unknown', professional_preference='not_assessed')
    measurements = dict(fixed_access=fixed, seat_access=seat_access, seat_alignment=alignment,
        table_geometry=table_geometry, place_rectangles_overlap=overlap, geometry=geometry, measurement_support=support)
    return dict(review=review, measurements=measurements,
        feedback=dict(requested_mode=mode, declared_requirements=program,
            checks={r['requirement']:r['pass'] for r in rows},
            table_geometry=table_geometry, seat_alignment=alignment,
            place_rectangles_overlap=overlap, geometry=geometry, measurement_support=support,
            scope='Current declared proxy measurements, no reference actions or target scene; comfort and whole-room quality unknown.'))
