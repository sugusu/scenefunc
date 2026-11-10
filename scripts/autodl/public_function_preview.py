"""Ordinary endpoint lookahead using public geometry and explicit task regions.

No model, hidden target pose, after-state or historical outcome enters predict.
This is a deliberately bounded strong control: axis-aligned wall slabs with
declared hypothetically-open doorways, upright furniture, and the existing
rectangular activity/access proxies. It neither verifies human use nor plans
continuous physical motion. Unsupported inputs raise rather than pass.
"""
from collections import deque
import copy
import math

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from audit_prospective_guard import (world_box, furniture_pairs, outside_xy,
                                     rectangular_inner_bounds, predict as box_guard)


def structural_boxes(geometry, height):
    """Split declared wall slabs at fully traversable door openings.

    Window sill walls still occupy their full XY footprint at body height.
    Openings must have explicit geometry and a unique containing wall. This
    uses no SDF paths, source-specific object IDs, or evaluator after-state.
    """
    walls = geometry['wall_objects']
    if len(walls) != geometry['wall_count']:
        raise ValueError('Incomplete public wall list')
    assignments = {w['object_id']: [] for w in walls}
    for opening in geometry['recorded_openings']:
        if (opening['opening_type'] != 'door' or opening['sill_height'] > .05
                or opening['sill_height'] + opening['height'] < height):
            continue
        start, end = np.array(opening['wall_start']), np.array(opening['wall_end'])
        delta = abs(end-start)
        varying = np.flatnonzero(delta > 1e-8)
        if len(varying) != 1:
            raise ValueError('Only axis-aligned declared doors are supported')
        axis = int(varying[0]); normal = 1-axis
        if not math.isclose(float(delta[axis]), opening['width'], abs_tol=1e-8):
            raise ValueError('Door extent conflicts with declared width')
        matches = []
        for w in walls:
            lo, hi = np.array(w['bounds_min']), np.array(w['bounds_max'])
            mid = (lo+hi)/2
            if (hi[normal]-lo[normal] < (hi[axis]-lo[axis])/20
                    and abs(start[normal]-mid[normal]) < 1e-8
                    and min(start[axis], end[axis]) >= lo[axis]-1e-8
                    and max(start[axis], end[axis]) <= hi[axis]+1e-8):
                matches.append(w['object_id'])
        if len(matches) != 1:
            raise ValueError('Door lacks one unambiguous public wall binding')
        assignments[matches[0]].append((axis, sorted([float(start[axis]), float(end[axis])])))
    result = []
    for w in walls:
        lo, hi = np.array(w['bounds_min'], float), np.array(w['bounds_max'], float)
        if hi[2] <= .05 or lo[2] >= height:
            continue
        axis = int(np.argmax(hi[:2]-lo[:2])); normal = 1-axis
        if hi[normal]-lo[normal] >= (hi[axis]-lo[axis])/20:
            raise ValueError('Wall is not a supported axis-aligned slab')
        cuts = sorted(assignments[w['object_id']], key=lambda v: v[1])
        cursor = float(lo[axis])
        for cut_axis, (left, right) in cuts:
            if cut_axis != axis or left < cursor-1e-8:
                raise ValueError('Overlapping or misaligned doorway intervals')
            if left > cursor:
                a, b = lo.copy(), hi.copy(); a[axis], b[axis] = cursor, left
                result.append((w['object_id'], a, b))
            cursor = right
        if cursor < hi[axis]:
            a = lo.copy(); a[axis] = cursor
            result.append((w['object_id'], a, hi))
    return result


def obstacles(objects, geometry, height):
    result = structural_boxes(geometry, height)
    for oid, obj in objects.items():
        if obj['object_type'] != 'furniture':
            continue
        lo, hi = map(np.array, world_box(obj))
        if hi[2] > .05 and lo[2] < height:
            result.append((oid, lo, hi))
    return result


def access(objects, geometry, settings):
    """Original 4-neighbour grid convention, using the PUBLIC shell instead of SDF."""
    dx = settings['grid_m']; length, width = geometry['recorded_length'], geometry['recorded_width']
    if dx <= 0 or not np.allclose([length/dx, width/dx], np.round([length/dx, width/dx]), atol=1e-8):
        raise ValueError('Grid must partition the rectangle')
    xs = -length/2 + (np.arange(round(length/dx))+.5)*dx
    ys = -width/2 + (np.arange(round(width/dx))+.5)*dx
    X, Y = np.meshgrid(xs, ys)
    boxes = obstacles(objects, geometry, settings['assumed_body_height_m'])
    results = {}
    for body in settings['square_body_widths_m']:
        if body <= 0: raise ValueError('Invalid body width')
        radius = body/2; pad = radius+dx/2
        blocked = (abs(X)>length/2-pad+1e-9) | (abs(Y)>width/2-pad+1e-9)
        for _, lo, hi in boxes:
            blocked |= ((X>lo[0]-pad-1e-9)&(X<hi[0]+pad+1e-9)
                        &(Y>lo[1]-pad-1e-9)&(Y<hi[1]+pad+1e-9))
        def cell(point):
            return int(np.argmin(abs(ys-point[1]))), int(np.argmin(abs(xs-point[0])))
        start = (np.array(settings['entry_center_xy_m'], float)
                 +np.array(settings['entry_inward_normal_xy'])*(radius+2*dx))
        if not (-length/2 <= start[0] <= length/2 and -width/2 <= start[1] <= width/2):
            raise ValueError('Entry sampling point outside room')
        sy, sx = cell(start); reached = np.zeros_like(blocked); queue = deque()
        if not blocked[sy, sx]: reached[sy, sx] = True; queue.append((sy, sx))
        while queue:
            y, x = queue.popleft()
            for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0 <= ny < len(ys) and 0 <= nx < len(xs) and not blocked[ny,nx] and not reached[ny,nx]:
                    reached[ny,nx] = True; queue.append((ny,nx))
        for name, point in settings['approach_points_xy_m'].items():
            inside = -length/2 <= point[0] <= length/2 and -width/2 <= point[1] <= width/2
            results[f'access_{name}_width_{body}'] = bool(inside and reached[cell(point)])
    return results


def upright_footprint(obj):
    w, x, y, z = obj['transform']['rotation_wxyz']
    norm = math.sqrt(w*w+x*x+y*y+z*z); w,x,y,z = [v/norm for v in (w,x,y,z)]
    if not np.allclose([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)], [0,0,1], atol=1e-6, rtol=0):
        raise ValueError('Non-upright task furniture')
    rot = np.array([[1-2*(y*y+z*z),2*(x*y-z*w)], [2*(x*y+z*w),1-2*(x*x+z*z)]])
    lo, hi = obj['bbox_min'], obj['bbox_max']
    pts = np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
    return Polygon(pts@rot.T + np.array(obj['transform']['translation'][:2]))


def functional_checks(objects, geometry, program, mode):
    checks = access(objects, geometry, program['access_settings'])
    bounds = {oid: tuple(map(np.array, world_box(o))) for oid,o in objects.items() if o['object_type']=='furniture'}
    if mode == 'clear_floor':
        region = box(*program['reserved_floor_xyxy_m'])
        occupied = [region.intersection(box(lo[0],lo[1],hi[0],hi[1]))
                    for _,lo,hi in obstacles(objects,geometry,program['access_settings']['assumed_body_height_m'])]
        checks['central_activity_floor_clear'] = unary_union(occupied).area < 1e-9
    elif mode == 'desk_work':
        knee = program['knee_volume_lo_hi_m']; desk, stool = program['desk_id'], program['stool_id']
        checks['desk_knee_volume_clear'] = not any(np.all(np.minimum(hi,knee[1])-np.maximum(lo,knee[0]) > 1e-6)
                                                   for lo,hi in bounds.values())
        if stool not in objects:
            checks['work_stool_in_declared_region'] = False
        else:
            upright_footprint(objects[stool])
            x,y,_ = objects[stool]['transform']['translation']; r = program['seat_target_region_xyxy_m']
            checks['work_stool_in_declared_region'] = r[0] <= x <= r[2] and r[1] <= y <= r[3]
        if desk not in bounds:
            checks['desk_work_patch_supported_and_unoccupied'] = False
        else:
            patch = box(*program['desk_top_work_patch_xyxy_m']); top = bounds[desk][1][2]
            hits = [oid for oid,(lo,hi) in bounds.items() if oid != desk and hi[2]>top+1e-6 and lo[2]<top+.5
                    and patch.intersection(box(lo[0],lo[1],hi[0],hi[1])).area>1e-9]
            checks['desk_work_patch_supported_and_unoccupied'] = upright_footprint(objects[desk]).buffer(1e-8).covers(patch) and not hits
    else:
        raise ValueError('Unsupported declared activity')
    return {k:bool(v) for k,v in checks.items()}


def predict(parent, call, geometry):
    """Return partial prediction, keeping unsupported/invalid cases explicit."""
    gate = box_guard(parent, call)
    if call is None:
        return {'status':'unknown','reason':'unparsed_call','box_guard':gate}
    if call['name'] in ('get_current_scene_state','check_facing_tool'):
        return {'status':'allow','reason':'read_only','box_guard':gate,
                'predicted_checks':dict(parent['checks']), 'predicted_lost_checks':[]}
    if call['name'] != 'move_furniture_tool' or 'projected_overlap_pairs' not in gate:
        return {'status':gate['status'], 'reason':gate['reason'],'box_guard':gate}
    objects = copy.deepcopy(parent['objects']); a = call['arguments']; angle = math.radians(a['yaw'])
    objects[a['object_id']]['transform'] = {'translation':[a['x'],a['y'],0.0],
                                           'rotation_wxyz':[math.cos(angle/2),0,0,math.sin(angle/2)]}
    checks = functional_checks(objects,geometry,parent['declared_requirements'],parent['requested_mode'])
    # Move preserves identities, assets and all non-target fixed records. Carry
    # the pre-existing values, not a fabricated assertion that all are correct.
    for k in ('all_original_instances_and_assets','fixed_objects_unchanged','room_geometry_unchanged'):
        checks[k] = parent['checks'][k]
    checks['no_recorded_envelope_overlap_or_boundary_violation'] = (
        not gate['projected_overlap_pairs'] and not gate['projected_outside_xy_ids'])
    lost = [k for k,v in checks.items() if parent['checks'].get(k) is True and v is False]
    unsupported = sorted(set(parent['checks'])-set(checks))
    status = 'reject' if gate['status']=='reject' or lost else 'unknown' if unsupported or gate['status']=='unknown' else 'allow'
    return {'status':status,'reason':'ordinary_public_geometry_preview','box_guard':gate,
            'predicted_checks':checks,'predicted_lost_checks':lost,'unsupported_checks':unsupported,
            'endpoint_only':True,'motion_or_human_use_verified':False}
