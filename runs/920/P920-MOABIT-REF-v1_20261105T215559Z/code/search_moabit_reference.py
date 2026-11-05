"""One bounded geometry-only feasibility search; never a professional demonstration.

Continuous locations and four cardinal rotations; exact catalogue footprints.
The dining assembly and approach squares are declared research assumptions.
A solver timeout or infeasible restricted model is not an impossibility result
for the original design brief. No models, training, or native edits here.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def rotate(rect, angle):
    r=math.radians(angle); m=np.array([[round(math.cos(r)), -round(math.sin(r))], [round(math.sin(r)),round(math.cos(r))]])
    q=np.array(list(itertools.product([rect[0],rect[2]],[rect[1],rect[3]])))@m.T
    return [*q.min(0),*q.max(0)]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    p=json.loads(a.plan.read_text());a.out.mkdir(exist_ok=False,parents=True)
    assert sha(Path(__file__))==p['code_sha256']
    bundle=Path(p['bundle']);assert sha(bundle/'manifest.json')==p['manifest_sha256']
    cat=json.loads((bundle/'catalog.json').read_text());byname={e['template']['name']:e['template'] for e in cat['entries']}
    print(datetime.now(timezone.utc).isoformat(),'begin frozen bounded geometric feasibility search',flush=True)
    groups=[]
    for role in p['main_roles']:
        if role=='dining_assembly':
            t=byname['dining_table'];c=byname['dining_chair_1']
            parts=[('dining_table',0.,0.,0),('dining_chair_1',0.,-.77,0),('dining_chair_1',0.,.77,180)]
            accesses=[['daily_seat_side',[c['bbox_max'][0]+.02,-1.02,c['bbox_max'][0]+.52,-.52]],
                      ['guest_seat_side',[c['bbox_max'][0]+.02,.52,c['bbox_max'][0]+.52,1.02]]]
        else:
            t=byname[role];parts=[(role,0.,0.,0)]
            if role=='queen_bed': accesses=[['bed_side',[t['bbox_max'][0]+.02,-.25,t['bbox_max'][0]+.52,.25]]]
            else: accesses=[[role+'_front',[-.25,t['bbox_max'][1]+.02,.25,t['bbox_max'][1]+.52]]]
        physical=[]
        for name,x,y,yaw in parts:
            obj=byname[name]; rect=rotate([*obj['bbox_min'][:2],*obj['bbox_max'][:2]],yaw)
            physical.append([rect[0]+x,rect[1]+y,rect[2]+x,rect[3]+y])
        groups.append(dict(role=role,parts=parts,physical=physical,accesses=accesses))
    nvars=0; lower=[];upper=[];integer=[];rows=[];los=[];his=[]
    def var(lo,hi,kind=0):
        nonlocal nvars
        k=nvars;nvars+=1;lower.append(lo);upper.append(hi);integer.append(kind);return k
    def add(row,lo=-np.inf,hi=np.inf):rows.append(row);los.append(lo);his.append(hi)
    for g in groups:
        g['xy']=[var(-1.75,1.75),var(-3,3)];g['rot']=[var(0,1,1) for _ in range(4)]
        add({k:1 for k in g['rot']},1,1)
    def expr(g,rect,side):
        return {g['xy'][side%2]:1,**{k:rotate(rect,angle)[side] for k,angle in zip(g['rot'],[0,90,180,270])}}
    def sub(x,y):
        r=x.copy()
        for k,v in y.items():r[k]=r.get(k,0)-v
        return r
    def separate(g,rect,h,other,gap):
        z=[var(0,1,1) for _ in range(4)];add({k:1 for k in z},1,np.inf)
        for k,(side,opposite) in enumerate([(2,0),(3,1),(0,2),(1,3)]):
            ex=expr(g,rect,side)
            if h is None:
                rhs=other[opposite]
            else:
                ex=sub(ex,expr(h,other,opposite));rhs=0
            if k>=2:ex={j:-v for j,v in ex.items()};rhs=-rhs
            ex[z[k]]=20;add(ex,hi=20+rhs-gap)
    for g in groups:
        allrect=g['physical']+[r for _,r in g['accesses']]
        for rect in allrect:
            for side,bound in enumerate([-1.75,-3,1.75,3]):
                ex=expr(g,rect,side)
                add(ex,lo=bound+.01 if side<2 else -np.inf,hi=bound-.01 if side>=2 else np.inf)
            for fixed in p['main_keepouts']:separate(g,rect,None,fixed['rectangle'],.01)
    for g,h in itertools.combinations(groups,2):
        for r,s in itertools.product(g['physical'],h['physical']):separate(g,r,h,s,.01)
        for r,(_,s) in itertools.product(g['physical'],h['accesses']):separate(g,r,h,s,0)
        for (_,r),s in itertools.product(g['accesses'],h['physical']):separate(g,r,h,s,0)
    ii=[];jj=[];vv=[]
    for i,r in enumerate(rows):
        for j,v in r.items():ii.append(i);jj.append(j);vv.append(v)
    matrix=coo_matrix((vv,(ii,jj)),shape=(len(rows),nvars)).tocsc()
    write(a.out/'SEARCH_MODEL.json',dict(groups=groups,variables=nvars,integer_variables=sum(integer),constraints=len(rows),keepouts=p['main_keepouts']))
    result=milp(np.zeros(nvars),integrality=integer,bounds=Bounds(lower,upper),constraints=LinearConstraint(matrix,los,his),
                options={'time_limit':p['budget']['solver_seconds'],'node_limit':p['budget']['nodes'],'presolve':True,'disp':True})
    summary=dict(status='restricted_search_no_incumbent',solver_status=int(result.status),solver_message=result.message,
        started_plan_sha256=sha(a.plan),model_calls=0,training_updates=0,native_actions=0,
        interpretation='Restricted catalogue geometry only; not full functional, professional, learned, or original-project feasibility.')
    if result.x is not None:
        x=result.x;res=matrix@x
        residual=max(float(np.max(np.array(los)-res)),float(np.max(res-np.array(his))),float(np.max(np.array(lower)-x)),float(np.max(x-np.array(upper))),0.)
        integrality_error=max(abs(x[k]-round(x[k])) for k in range(nvars) if integer[k])
        assert residual<1e-5 and integrality_error<1e-5,(residual,integrality_error)
        commands=[];anchors={};placements=[];counts={}
        for g in groups:
            cx,cy=(float(x[k]) for k in g['xy']);yaw=90*int(np.argmax(x[g['rot']]));rr=math.radians(yaw)
            rot=np.array([[round(math.cos(rr)),-round(math.sin(rr))],[round(math.sin(rr)),round(math.cos(rr))]])
            placements.append(dict(group=g['role'],x=cx,y=cy,yaw=yaw))
            for name,dx,dy,localyaw in g['parts']:
                px,py=np.array([cx,cy])+rot@np.array([dx,dy]);idx=counts.get(name,0);counts[name]=idx+1
                commands.append(dict(role=name,planned_id=name+'_'+str(idx),command=dict(name='add_furniture_to_scene_tool',arguments=dict(asset_id=byname[name]['object_id'],x=float(px),y=float(py),yaw=(yaw+localyaw)%360))))
            for name,rect in g['accesses']:
                point=np.array([cx,cy])+rot@np.array([(rect[0]+rect[2])/2,(rect[1]+rect[3])/2]);anchors[name]=point.tolist()
        for item in p['bathroom_placements']:
            name=item['role'];commands.append(dict(role=name,planned_id=name+'_0',command=dict(name='add_furniture_to_scene_tool',arguments=dict(asset_id=byname[name]['object_id'],**{k:item[k] for k in ['x','y','yaw']}))))
            obj=byname[name];point=rotate([0,obj['bbox_max'][1]+.27,0,obj['bbox_max'][1]+.27],item['yaw'])
            anchors[name+'_front']=[point[0]+item['x'],point[1]+item['y']]
        anchors['bath_door_inside']=[.55,-2.1]
        summary.update(status='restricted_geometry_incumbent_found',constraint_residual=residual,integrality_error=integrality_error,commands=commands,approach_points_xy_m=anchors,placements=placements)
        write(a.out/'SOLUTION_VECTOR.json',x.tolist())
    write(a.out/'RESULT.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['commands','placements','approach_points_xy_m']}),flush=True)


if __name__=='__main__':main()
