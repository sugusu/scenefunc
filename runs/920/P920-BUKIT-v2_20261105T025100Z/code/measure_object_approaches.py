"""State-relative sampled approach strips; not sitting, reaching or operability."""
import argparse,itertools,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from native_project import sha256
from inspect_native_mesh_contacts import world_transform
from measure_room_access import measure


def locations(state,body,plan):
    radius=body/2;clearance=plan['clearance_m'];n=plan['line_samples'];out={}
    def side(oid,edge):
        obj=state['objects'][oid];lo=np.array(obj['bbox_min']);hi=np.array(obj['bbox_max']);axis=0 if edge in ('+Y','-Y') else 1;normal=1-axis
        start,end=lo[axis]+radius,hi[axis]-radius
        if start>end:start=end=(lo[axis]+hi[axis])/2
        fixed=(hi[normal]+radius+clearance) if edge[0]=='+' else (lo[normal]-radius-clearance)
        points=np.zeros((n,4));points[:,3]=1;points[:,axis]=np.linspace(start,end,n);points[:,normal]=fixed
        return [list(p) for p in dict.fromkeys(tuple(map(float,p)) for p in (points@world_transform(obj).T)[:,:2])]
    for name,spec in plan['object_groups'].items():out[name]=sum((side(spec['object_id'],edge) for edge in spec['local_sides']),[])
    for name,opening in plan['windows'].items():
        x,y,width=opening;start,end=x-width/2+radius,x+width/2-radius
        out[name]=[[float(v),y-radius-clearance] for v in np.linspace(start,end,n)] if start<=end else []
    x0,y0,x1,y1=plan['reserved_bounds_xyxy_m'];out['reserved_floor_centers']=[[float(x),float(y)] for x,y in itertools.product(np.linspace(x0+radius+clearance,x1-radius-clearance,5),np.linspace(y0+radius+clearance,y1-radius-clearance,4))]
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text())
    print(datetime.now(timezone.utc).isoformat(),'begin state-relative approach strip measurements',flush=True)
    definition=plan['definition'];out={}
    for label,spec in plan['states'].items():
        f=Path(spec['path'])
        if sha256(f)!=spec['sha256']:raise ValueError('Input state changed')
        state=json.loads(f.read_text());rows=[]
        for body in definition['body_widths_m']:
            groups=locations(state,body,definition);points={name+'__'+str(i):xy for name,xylist in groups.items() for i,xy in enumerate(xylist)}
            settings={**definition['routing'],'square_body_widths_m':[body],'approach_points_xy_m':points}
            measured=measure(state,Path(spec['room_directory']),Path(spec['root_directory']),settings)['scenarios'][0]
            summary={}
            for name,xylist in groups.items():
                entries=[measured['targets'][name+'__'+str(i)] for i in range(len(xylist))];reachable=sum(x['reachable_from_entry'] for x in entries)
                summary[name]={'reachable_samples':reachable,'total_samples':len(entries),'reachable_fraction':reachable/len(entries) if entries else None,'any_reachable':reachable>0,'all_reachable':bool(entries) and reachable==len(entries),'samples':entries}
            rows.append({'body_width_m':body,'groups':summary})
        out[label]=rows
    result={'status':'sampled_object_approaches_measured','created_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha256(a.plan),'definition':definition,'states':out,'scope':'Additive diagnostic beside legacy fixed points. Any reachable sample is not an activity pass; all samples passing is still not actual ergonomic/door/drawer/seating validation.','model_calls':0,'tool_calls':0}
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({label:[{'body':x['body_width_m'],'any_reachable_groups':[k for k,v in x['groups'].items() if v['any_reachable']]} for x in rows] for label,rows in out.items()}),flush=True)


if __name__=='__main__':main()
