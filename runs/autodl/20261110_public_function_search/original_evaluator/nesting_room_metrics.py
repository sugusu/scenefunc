"""Current-state nesting-task proxies; no reference layout/action lookup."""
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from build_activity_completion_reviews import aggregate
from conditional_room_metrics import checked_access, nonpose
from inspect_native_mesh_contacts import world_transform
from measure_free_floor import footprint
from measure_room_access import obstacles
from preflight_sourced_trajectory import boxes, inspect


def measure_nesting(state, initial, room, root, program, mode):
    keys = {'movable_ids', 'desk_id', 'stool_id', 'reserved_floor_xyxy_m',
            'desk_top_work_patch_xyxy_m', 'knee_volume_lo_hi_m', 'seat_target_region_xyxy_m', 'access_settings'}
    if set(program) != keys or mode not in ('clear_floor', 'desk_work'):
        raise ValueError('Only frozen requirement fields and supported modes are allowed')
    original, objects = initial['objects'], state['objects']
    movable = set(program['movable_ids']); desk = program['desk_id']; stool = program['stool_id']
    if not movable <= original.keys() or desk not in original or stool not in movable:
        raise ValueError('Invalid task identities')
    widths = program['access_settings']['square_body_widths_m']
    if not widths or len(set(widths)) != len(widths):
        raise ValueError('Invalid body scenarios')
    bounds = boxes(state); geometry = inspect(state)
    access = checked_access(state, room, root, program['access_settings'])
    region = box(*program['reserved_floor_xyxy_m']); hits = []
    for obstacle in obstacles(state, room, root, program['access_settings']['assumed_body_height_m']):
        lo, hi = obstacle['lo'], obstacle['hi']
        area = region.intersection(box(lo[0], lo[1], hi[0], hi[1]))
        if area.area > 1e-9: hits.append((obstacle['id'], area))
    occupied = float(unary_union([area for _, area in hits]).area)
    knee = program['knee_volume_lo_hi_m']
    knee_hits = [oid for oid, (lo, hi) in bounds.items()
                 if np.all(np.minimum(hi, knee[1])-np.maximum(lo, knee[0]) > 1e-6)]
    support = {}
    for oid in (desk, stool):
        if oid not in objects: support[oid] = 'missing'
        elif nonpose(objects[oid]) != nonpose(original[oid]): support[oid] = 'changed_geometry_or_asset'
        elif not np.allclose(world_transform(objects[oid])[:3, 2], [0, 0, 1], atol=1e-6, rtol=0):
            support[oid] = 'non_upright'
        else: support[oid] = 'supported_known_proxy'
    def unsupported(oid): return False if support[oid] == 'missing' else None
    seat_ok = unsupported(stool)
    if support[stool] == 'supported_known_proxy':
        x, y, _ = objects[stool]['transform']['translation']; seat = program['seat_target_region_xyxy_m']
        seat_ok = bool(seat[0] <= x <= seat[2] and seat[1] <= y <= seat[3])
    patch = box(*program['desk_top_work_patch_xyxy_m']); surface_ok = unsupported(desk); surface_hits = []
    if support[desk] == 'supported_known_proxy':
        top = bounds[desk][1][2]
        for oid, (lo, hi) in bounds.items():
            if oid != desk and hi[2] > top+1e-6 and lo[2] < top+.5 and patch.intersection(box(lo[0],lo[1],hi[0],hi[1])).area > 1e-9:
                surface_hits.append(oid)
        surface_ok = bool(footprint(objects[desk]).buffer(1e-8).covers(patch) and not surface_hits)
    activity = [('central_activity_floor_clear', occupied < 1e-9)] if mode == 'clear_floor' else [
        ('desk_work_patch_supported_and_unoccupied', surface_ok), ('desk_knee_volume_clear', not knee_hits),
        ('work_stool_in_declared_region', seat_ok)]
    checks = []
    def add(name, value, ids, evidence):
        checks.append(dict(requirement=name, **{'pass':value}, object_ids=ids, evidence=evidence))
    add('all_original_instances_and_assets', set(objects)==set(original) and all(
        nonpose(objects[o])==nonpose(original[o]) for o in objects.keys() & original.keys()), sorted(original), 'native nonpose fields')
    add('fixed_objects_unchanged', all(objects.get(o)==original[o] for o in original.keys()-movable), sorted(original.keys()-movable), 'native records')
    add('room_geometry_unchanged', state['room_geometry']==initial['room_geometry'], [], 'room records')
    add('no_recorded_envelope_overlap_or_boundary_violation', not geometry['positive_3d_AABB_pairs'] and not geometry['outside_declared_wall_inner_faces'], sorted(objects), 'current AABB geometry')
    for scenario in access['scenarios']:
        for name, target in scenario['targets'].items():
            add(f'access_{name}_width_{scenario["square_body_width_m"]}', target['reachable_from_entry'], [], 'absolute current proxy access; pre-existing failures retained')
    for name, value in activity: add(name, value, [desk,stool] if mode=='desk_work' else [], 'current declared activity geometry')
    activity_status = aggregate([{'pass':v} for _,v in activity])
    review = dict(requested_mode=mode, checks=checks, measured_requirements_status=aggregate(checks),
        requested_activity_status=activity_status, whole_room_success='unknown', professional_preference='not_assessed')
    measurements = dict(geometry=geometry, access=access, central_area_m2=region.area,
        occupied_activity_floor_m2=occupied, central_colliders=[x for x,_ in hits],
        knee_colliders=knee_hits, seat_ready=seat_ok, work_patch_available=surface_ok,
        work_patch_colliders=surface_hits, measurement_support=support)
    return dict(review=review, measurements=measurements, feedback=dict(requested_mode=mode,
        declared_requirements=program, checks={c['requirement']:c['pass'] for c in checks},
        measurements=measurements, scope='Current declared proxies; overall failure and unknown human usability must remain visible. No reference poses, actions or original project solution.'))
