"""Mode-specific table geometry, with explicit limits on human-use inference."""
import numpy as np
from shapely.geometry import Polygon
from measure_room_access import measure
from measure_free_floor import footprint, largest_floor_rectangle
from inspect_native_mesh_contacts import world_transform


def table_modes(state, room, root, plan):
    settings = plan['access_settings']
    a, b = [state['objects'][x] for x in plan['table_ids']]
    am, bm = [world_transform(x) for x in (a, b)]
    distance = footprint(a).distance(footprint(b))
    yaw_dot = float(np.dot(am[:2, 0], bm[:2, 0]))
    # A common-coordinate seam is meaningful only for these near-parallel,
    # frozen rectangular tabletop substitutes, not arbitrary detected tables.
    offset = np.linalg.inv(am) @ np.r_[bm[:3, 3], 1]
    top_difference = abs(am[2, 3] + a['bbox_max'][2] - bm[2, 3] - b['bbox_max'][2])
    joined = (distance <= plan['joined_max_gap_m'] and abs(offset[1]) <= .03
              and yaw_dot > .999 and top_difference <= .01)
    table_areas = {}
    for oid in plan['table_ids']:
        obj = state['objects'][oid]
        lo, hi = obj['bbox_min'], obj['bbox_max']
        table_areas[oid] = float((hi[0]-lo[0])*(hi[1]-lo[1]))
    alignment = {}
    place_polygons = []
    for seat in plan['seat_assignments']:
        t = state['objects'][seat['table_id']]
        chair = state['objects'][seat['chair_id']]
        matrix = world_transform(t)
        local = np.linalg.inv(matrix) @ np.r_[world_transform(chair)[:3, 3], 1]
        lo, hi = t['bbox_min'], t['bbox_max']
        side = seat['side_y']
        edge = hi[1] if side > 0 else lo[1]
        chair_half = (chair['bbox_max'][1]-chair['bbox_min'][1])/2
        edge_gap = float(side*(local[1]-edge)-chair_half)
        w, d = plan['place_rectangle_width_depth_m']
        x = float(local[0]); y = edge-side*d/2
        local_corners = [[x-w/2,y-d/2], [x+w/2,y-d/2],
                         [x+w/2,y+d/2], [x-w/2,y+d/2]]
        within = all(lo[0]-1e-6 <= q[0] <= hi[0]+1e-6 and lo[1]-1e-6 <= q[1] <= hi[1]+1e-6
                     for q in local_corners)
        poly = Polygon([(matrix @ np.array([*q,0,1]))[:2] for q in local_corners])
        place_polygons.append(poly)
        alignment[seat['chair_id']] = {'table_id':seat['table_id'], 'local_chair_xy_m':local[:2].tolist(),
            'chair_envelope_to_table_edge_gap_m':edge_gap, 'place_rectangle_on_table':within,
            'aligned_under_declared_proxy':bool(within and .05 <= edge_gap <= .4)}
    place_overlap = any(x.intersection(y).area > 1e-8 for i,x in enumerate(place_polygons)
                        for y in place_polygons[i+1:])
    approaches=[]
    for width in settings['square_body_widths_m']:
        points={}
        for seat in plan['seat_assignments']:
            obj=state['objects'][seat['chair_id']];m=world_transform(obj)
            side=seat['side_y'];edge=obj['bbox_max'][1] if side>0 else obj['bbox_min'][1]
            points[seat['chair_id']]=(m @ np.array([0,edge+side*(width/2+.05),0,1]))[:2].tolist()
        cfg=dict(settings, square_body_widths_m=[width], approach_points_xy_m=points)
        approaches.append(measure(state,room,root,cfg)['scenarios'][0])
    work_ids=plan['work_chair_ids']
    work_ok=(distance >= plan['separate_min_gap_m'] and
             all(alignment[x]['aligned_under_declared_proxy'] for x in work_ids))
    host_ok=joined and all(x['aligned_under_declared_proxy'] for x in alignment.values()) and not place_overlap
    return {'fixed_access':measure(state,room,root,settings),
        'floor_rectangle':largest_floor_rectangle(state,room,root,settings),
        'table_geometry':{'footprint_gap_m':distance,'parallel_axis_dot':yaw_dot,
            'relative_center_offset_m':offset[:3].tolist(),'top_height_difference_m':float(top_difference),
            'top_rectangular_areas_m2':table_areas,'joined_under_declared_threshold':bool(joined)},
        'seat_alignment':alignment,'place_rectangles_overlap':bool(place_overlap),
        'seat_approaches':approaches,
        'mode_geometry':{'separate_work':bool(work_ok),'joined_four_place_hosting':bool(host_ok)},
        'limits':['Known generated rectangular tabletops, not a semantic estimate from an arbitrary mesh.',
            'Place rectangles and chair-edge gaps are researcher assumptions, not ergonomic or meal-use standards.',
            'Standing approaches do not validate sitting, knees, comfort or simultaneous people.',
            'Joined geometry does not validate wheel brakes, structural connection, load or human pushing.',
            'A mode-specific geometric pass is not full room usability or a professional preference label.']}
