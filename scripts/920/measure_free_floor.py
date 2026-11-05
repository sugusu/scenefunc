"""Largest empty grid rectangle and conservative rigid-table sweep diagnostics."""
import itertools
import math
import numpy as np
from shapely.geometry import Polygon, box
from inspect_native_mesh_contacts import world_transform
from measure_room_access import obstacles


def largest_floor_rectangle(native,room,root,settings):
    step=settings['grid_m'];g=native['room_geometry'];lx,ly=g['length'],g['width']
    xs=-lx/2+(np.arange(round(lx/step))+0.5)*step
    ys=-ly/2+(np.arange(round(ly/step))+0.5)*step
    X,Y=np.meshgrid(xs,ys);blocked=np.zeros(X.shape,dtype=bool)
    for obstacle in obstacles(native,room,root,settings['assumed_body_height_m']):
        lo,hi=obstacle['lo'],obstacle['hi'];pad=step/2
        blocked|=(X>lo[0]-pad-1e-9)&(X<hi[0]+pad+1e-9)&(Y>lo[1]-pad-1e-9)&(Y<hi[1]+pad+1e-9)
    heights=np.zeros(len(xs),dtype=int);best=(0,0,0,0,0)
    for row in range(len(ys)):
        heights=np.where(blocked[row],0,heights+1)
        stack=[]
        for col,height in enumerate([*heights,0]):
            start=col
            while stack and stack[-1][1]>height:
                left,h=stack.pop();area=int(h*(col-left))
                if area>best[0]:best=(area,left,col,row-int(h)+1,row+1)
                start=left
            if not stack or stack[-1][1]<height:stack.append((start,int(height)))
    area,left,right,bottom,top=best
    if area and blocked[bottom:top,left:right].any():
        raise ValueError('Reported clear rectangle contains occupied cells')
    bounds=[-lx/2+left*step,-ly/2+bottom*step,-lx/2+right*step,-ly/2+top*step]
    return {'largest_grid_empty_rectangle_m2':area*step*step,'bounds_xyxy_m':bounds,
            'width_depth_m':[(right-left)*step,(top-bottom)*step],
            'scope':'Empty axis-aligned floor rectangle under conservative AABBs and assumed clear height, anywhere in room; not an activity pass or reachable human area.'}


def footprint(obj):
    low,high=obj['bbox_min'],obj['bbox_max'];matrix=world_transform(obj)
    corners=np.array([[low[0],low[1],0,1],[high[0],low[1],0,1],[high[0],high[1],0,1],[low[0],high[1],0,1]])
    return Polygon((corners@matrix.T)[:,:2])


def sweep_diagnostic(before,after,oid,room,root):
    first,last=before['objects'][oid],after['objects'][oid]
    matrices=[world_transform(x) for x in (first,last)]
    angles=[math.atan2(x[1,0],x[0,0]) for x in matrices]
    delta=(angles[1]-angles[0]+math.pi)%(2*math.pi)-math.pi
    if abs(delta)>math.radians(1):
        raise ValueError('Only near-constant yaw translation is in this frozen sweep contract')
    radius=max(math.hypot(x,y) for x,y in itertools.product(*zip(first['bbox_min'][:2],first['bbox_max'][:2])))
    # Convex endpoint hull is exact for constant orientation and straight
    # translation. This bound covers small linearly interpolated yaw drift.
    margin=radius*abs(delta)/2+1e-6
    swept=footprint(first).union(footprint(last)).convex_hull.buffer(margin/math.cos(math.pi/64),resolution=16)
    upper=max(first['bbox_max'][2]+first['transform']['translation'][2],last['bbox_max'][2]+last['transform']['translation'][2])
    hits=[]
    for obstacle in obstacles(before,room,root,upper):
        if obstacle['id']==oid:continue
        lo,hi=obstacle['lo'],obstacle['hi']
        intersection=swept.intersection(box(lo[0],lo[1],hi[0],hi[1]))
        if intersection.area>1e-9:hits.append({'obstacle_id':obstacle['id'],'planar_overlap_m2':intersection.area})
    g=before['room_geometry'];inside=box(-g['length']/2,-g['width']/2,g['length']/2,g['width']/2).covers(swept)
    return {'within_room':bool(inside),'obstacle_overlaps':hits,'yaw_change_degrees':math.degrees(delta),
            'rotation_bound_margin_m':margin,'clear_under_declared_sweep_model':bool(inside and not hits),
            'scope':'Conservative box-footprint straight translation with small yaw interpolation and hypothetically open door leaves; tools teleport, handling effort/mass/friction/support dynamics and person movement unverified.'}


def rigid_sweep_diagnostic(before, after, oid, room, root):
    """Conservative arbitrary-yaw sweep, preserving the old translation metric."""
    first, last = before['objects'][oid], after['objects'][oid]
    matrices = [world_transform(x) for x in (first, last)]
    angles = [math.atan2(x[1, 0], x[0, 0]) for x in matrices]
    delta = (angles[1]-angles[0]+math.pi) % (2*math.pi)-math.pi
    if abs(delta) <= math.radians(1):
        return sweep_diagnostic(before, after, oid, room, root)
    from shapely.geometry import Point
    radius = max(math.hypot(x, y) for x, y in itertools.product(
        *zip(first['bbox_min'][:2], first['bbox_max'][:2])))
    centers = [Point(m[0, 3], m[1, 3]) for m in matrices]
    # Circumscribed polygon covers the complete disk at every linearly
    # interpolated center. This can reject valid motions but never uses a
    # sampled orientation as proof that the entire rotation was clear.
    swept = centers[0].union(centers[1]).convex_hull.buffer(
        (radius+1e-6)/math.cos(math.pi/128), resolution=32)
    upper = max(o['bbox_max'][2]+o['transform']['translation'][2] for o in (first, last))
    hits = []
    for obstacle in obstacles(before, room, root, upper):
        if obstacle['id'] == oid:
            continue
        lo, hi = obstacle['lo'], obstacle['hi']
        area = swept.intersection(box(lo[0], lo[1], hi[0], hi[1])).area
        if area > 1e-9:
            hits.append({'obstacle_id':obstacle['id'], 'planar_overlap_m2':area})
    g = before['room_geometry']
    inside = box(-g['length']/2, -g['width']/2, g['length']/2, g['width']/2).covers(swept)
    return {'within_room':bool(inside), 'obstacle_overlaps':hits,
        'yaw_change_degrees':math.degrees(delta), 'rotation_radius_m':radius,
        'clear_under_declared_sweep_model':bool(inside and not hits),
        'scope':'Full circumscribed rotating-footprint disk along linear center path; conservative geometry only. Tools teleport; mass, handling and human movement unverified.'}
