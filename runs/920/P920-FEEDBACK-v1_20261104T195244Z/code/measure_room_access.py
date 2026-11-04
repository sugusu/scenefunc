"""Conservative, explicitly conditional planar access in a native room.

Axis-aligned square body widths and height are research scenarios, not resident
measurements or legal standards. Door leaves are assumed open, without hinges
or sweep validation. No positive human-usability claim follows from a route.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from inspect_native_mesh_contacts import sdf_visuals, world_transform
from native_project import sha256, validate_bundle


def obstacles(native,room,root,height):
    boxes=[]
    for oid,obj in native['objects'].items():
        if obj['sdf_path'] is None or obj['object_type']=='thin_covering':
            continue
        corners=np.array(list(itertools.product(*zip(obj['bbox_min'],obj['bbox_max']))))
        corners=trimesh.transform_points(corners,world_transform(obj))
        lo,hi=corners.min(0),corners.max(0)
        if hi[2]>0.05 and lo[2]<height:
            boxes.append({'id':oid,'lo':lo.tolist(),'hi':hi.tolist()})
    for name,mesh in sdf_visuals(room/native['room_geometry']['sdf_path'],root,{}):
        if name in ('floor','ceiling') or 'door' in name:
            continue
        lo,hi=mesh.bounds
        if hi[2]>0.05 and lo[2]<height:
            boxes.append({'id':'structure:'+name,'lo':lo.tolist(),'hi':hi.tolist()})
    return boxes


def measure(native,room,root,settings):
    dx=settings['grid_m'];height=settings['assumed_body_height_m']
    geom=native['room_geometry'];length,width=geom['length'],geom['width']
    if not np.allclose([length/dx,width/dx],np.round([length/dx,width/dx]),atol=1e-8):
        raise ValueError('Grid must exactly partition the declared rectangle')
    xs=-length/2+(np.arange(round(length/dx))+0.5)*dx
    ys=-width/2+(np.arange(round(width/dx))+0.5)*dx
    X,Y=np.meshgrid(xs,ys)
    boxes=obstacles(native,room,root,height)
    scenarios=[]
    for body in settings['square_body_widths_m']:
        radius=body/2;pad=radius+dx/2
        blocked=(np.abs(X)>length/2-pad+1e-9)|(np.abs(Y)>width/2-pad+1e-9)
        for box in boxes:
            lo,hi=np.array(box['lo']),np.array(box['hi'])
            blocked|=((X>lo[0]-pad-1e-9)&(X<hi[0]+pad+1e-9)&
                      (Y>lo[1]-pad-1e-9)&(Y<hi[1]+pad+1e-9))
        start=np.array(settings['entry_center_xy_m'],float)+np.array(settings['entry_inward_normal_xy'])*(radius+2*dx)
        def cell(point):
            return (int(np.argmin(abs(ys-point[1]))),int(np.argmin(abs(xs-point[0]))))
        start_cell=cell(start)
        distance=np.full(blocked.shape,-1,dtype=int)
        queue=deque()
        if not blocked[start_cell]:
            distance[start_cell]=0;queue.append(start_cell)
        while queue:
            y,x=queue.popleft()
            for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=ny<len(ys) and 0<=nx<len(xs) and not blocked[ny,nx] and distance[ny,nx]<0:
                    distance[ny,nx]=distance[y,x]+1;queue.append((ny,nx))
        targets={}
        for name,point in settings['approach_points_xy_m'].items():
            y,x=cell(point);reachable=distance[y,x]>=0
            targets[name]={'requested_point_m':point,'sampled_grid_point_m':[float(xs[x]),float(ys[y])],
                'target_cell_clear':bool(not blocked[y,x]),'reachable_from_entry':bool(reachable),
                'grid_route_length_m':float(distance[y,x]*dx) if reachable else None}
        scenarios.append({'square_body_width_m':body,'entry_grid_point_m':[float(xs[start_cell[1]]),float(ys[start_cell[0]])],
            'entry_cell_clear':bool(not blocked[start_cell]),'reachable_floor_center_area_m2':int((distance>=0).sum())*dx*dx,
            'targets':targets})
    return {'settings':settings,'obstacle_count':len(boxes),'scenarios':scenarios,
            'door_assumption':'Both proxy door leaves omitted as hypothetically fully open; hinge/sweep unverified.',
            'limits':['Full-height AABB footprints are conservative, including gaps within furniture.',
                      '2 cm discretization and square body scenarios; not a human or accessibility model.',
                      'A failed conservative route is a warning, not proof no human route exists.',
                      'A successful route does not verify sitting, opening storage, reaching, turning or comfort.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());root=Path(plan['bundle'])
    print(datetime.now(timezone.utc).isoformat(),'begin conditional whole-room access measurement',flush=True)
    if sha256(root/'manifest.json')!=plan['manifest_sha256']:
        raise ValueError('Frozen input changed')
    manifest,state,_,_=validate_bundle(root);rid=manifest['selected_room_id']
    result=measure(state['rooms'][rid],root/('room_'+rid),root,plan['settings'])
    result.update(status='conditional_access_measured',plan_sha256=sha256(a.plan),input_manifest_sha256=plan['manifest_sha256'],
                  model_calls=0,scene_edits=0,professional_labels=0)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({str(s['square_body_width_m']):{n:t['reachable_from_entry'] for n,t in s['targets'].items()}
                      for s in result['scenarios']}),flush=True)


if __name__=='__main__':
    main()
