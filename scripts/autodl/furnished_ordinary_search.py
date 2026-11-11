"""Frozen ordinary proxy search, same move-only native action budget.

This finite AABB proxy is not exact mesh optimization or complete search. Every
proxy candidate is reported separately from native calls and model compute.
"""
import argparse
import copy
import itertools
import math
import os
from pathlib import Path
import time
import numpy as np

from cpu_continuation import read, dump
from furnished_pair_pilot import initial, observed, payload


def yaw(obj):
    w,x,y,z = obj['transform']['rotation_wxyz']
    return math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def bounds(obj):
    angle=math.radians(yaw(obj))
    c,s=math.cos(angle),math.sin(angle)
    rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    pts=np.array(list(itertools.product(*zip(obj['bbox_min'],obj['bbox_max']))))
    pts=pts@rot.T+np.array(obj['transform']['translation'])
    return np.array([pts.min(0),pts.max(0)])


def area_intersect(a,b):
    return float(np.maximum(0,np.minimum(a[1,:2],b[1,:2])-np.maximum(a[0,:2],b[0,:2])).prod())


def score(objects,half,edit_size):
    bs={o['object_id']:bounds(o) for o in objects}
    outside=sum(bool(np.any(b[0,:2]<-half-1e-5) or np.any(b[1,:2]>half+1e-5)) for b in bs.values())
    overlap=0.
    for a,b in itertools.combinations(bs.values(),2):
        if min(a[1,2],b[1,2])-max(a[0,2],b[0,2])>1e-5:
            overlap+=area_intersect(a,b)
    blocked=0.
    for obj in objects:
        if obj['name'] not in ['desk','office_desk','dining_table']: continue
        bb=bs[obj['object_id']];z=bb[1,2]+1e-4
        for oid,b in bs.items():
            if oid!=obj['object_id'] and b[0,2]<=z<=b[1,2]:
                blocked+=area_intersect(bb,b)
    return [outside,round(overlap,10),round(blocked,10),edit_size]


def propose(cp):
    state=payload(cp)
    objects=state['placed_instances']
    shell=read(state['frozen_shell']['path'])
    floor=next(o for o in shell['room_structure'] if o['name']=='floor')
    half=np.array([floor['dimensions'][0],floor['dimensions'][2]])/2
    rows=[dict(action=None,score=score(objects,half,0),candidate_kind='stay')]
    for obj in sorted(objects,key=lambda o:o['object_id']):
        if obj.get('immutable'): continue
        x,y,_=obj['transform']['translation'];a=yaw(obj)
        poses=[]
        for delta in [.25,.5,1.]:
            for dx,dy in [(delta,0),(-delta,0),(0,delta),(0,-delta)]:
                poses.append((x+dx,y+dy,a,delta))
        poses.extend([(x,y,a-90,90),(x,y,a+90,90)])
        for xx,yy,aa,size in poses:
            cloned=copy.deepcopy(objects)
            modified=next(o for o in cloned if o['object_id']==obj['object_id'])
            modified['transform']['translation'][:2]=[xx,yy]
            modified['transform']['rotation_wxyz']=[math.cos(math.radians(aa)/2),0,0,math.sin(math.radians(aa)/2)]
            action=dict(name='move_furniture_tool',arguments=dict(object_id=obj['object_id'],x=xx,y=yy,yaw=aa))
            rows.append(dict(action=action,score=score(cloned,half,size),candidate_kind='single_move'))
    best=min(range(len(rows)),key=lambda i:rows[i]['score'])
    return rows,best


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    arm=root/'B2';arm.mkdir(exist_ok=False)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      PYTHONPATH='/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps')
    state=initial();start=time.monotonic();total=0
    try:
        for step in range(4):
            turn=arm/f'turn_{step:02d}';turn.mkdir()
            t=time.monotonic();rows,best=propose(Path(state['checkpoint']));total+=len(rows)
            dump(turn/'proxy_candidates.json',dict(rows=rows,selected_index=best,seconds=time.monotonic()-t,
                    proxy_only=True,finite_not_complete=True))
            action=rows[best]['action']
            if action is None:
                state['status']='ordinary_no_proxy_improvement'
                break
            state.update(observed(root,'B2',step,state,[action]));state['turns']=step+1
            state['history'].append(dict(action=action,feedback=read(turn/'tool_responses.json')))
            state['status']='four_attempt_cap'
    except Exception as e:
        state.update(status='execution_failure_no_retry',error=f'{type(e).__name__}: {e}')
    state.update(proxy_candidates=total,cpu_pipeline_wall_seconds=time.monotonic()-start,model_calls=0)
    dump(arm/'state.json',state);print(state['status'],total,flush=True)


if __name__=='__main__':main()
