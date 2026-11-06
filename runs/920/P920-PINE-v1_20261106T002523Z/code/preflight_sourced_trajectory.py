"""Check a prospectively fixed trajectory without calling native edit tools."""
import argparse, copy, itertools, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from native_project import sha256, validate_bundle
from inspect_native_mesh_contacts import world_transform
from measure_free_floor import rigid_sweep_diagnostic


def boxes(state):
    out={}
    for oid,obj in state['objects'].items():
        if obj['sdf_path'] is None:continue
        p=np.array(list(itertools.product(*zip(obj['bbox_min'],obj['bbox_max']))))
        p=np.c_[p,np.ones(len(p))]@world_transform(obj).T
        out[oid]=(p[:,:3].min(0),p[:,:3].max(0))
    return out


def inspect(state):
    geom=state['room_geometry'];margin=geom['wall_thickness']/2
    lo=np.array([-geom['length']/2+margin,-geom['width']/2+margin,0]);hi=np.array([geom['length']/2-margin,geom['width']/2-margin,geom['wall_height']])
    env=boxes(state);outside=[oid for oid,(a,b) in env.items() if np.any(a<lo-1e-8) or np.any(b>hi+1e-8)]
    overlaps=[]
    for (a,(al,ah)),(b,(bl,bh)) in itertools.combinations(env.items(),2):
        delta=np.minimum(ah,bh)-np.maximum(al,bl)
        if np.all(delta>1e-6):overlaps.append({'a':a,'b':b,'overlap_extents_m':delta.tolist()})
    return {'outside_declared_wall_inner_faces':outside,'positive_3d_AABB_pairs':overlaps,'scope':'Conservative box geometry, not physical stability or use.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text())
    print(datetime.now(timezone.utc).isoformat(),'begin one fixed trajectory preflight',flush=True)
    for name,digest in plan['scripts'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Frozen script changed')
    root=Path(plan['bundle'])
    if sha256(root/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Bundle changed')
    manifest,state,_,_=validate_bundle(root);rid=manifest['selected_room_id'];current=state['rooms'][rid];room=root/('room_'+rid);initial=inspect(current);steps=[]
    for i,command in enumerate(plan['tool_calls'],1):
        args=command['arguments'];oid=args['object_id'];assert not current['objects'][oid]['immutable'];after=copy.deepcopy(current);angle=math.radians(args['yaw'])
        after['objects'][oid]['transform']={'translation':[args['x'],args['y'],0],'rotation_wxyz':[math.cos(angle/2),0,0,math.sin(angle/2)]}
        steps.append({'index':i,'command':command,'sweep':rigid_sweep_diagnostic(current,after,oid,room,root),'endpoint':inspect(after)});current=after
    passed=not initial['outside_declared_wall_inner_faces'] and not initial['positive_3d_AABB_pairs'] and all(s['sweep']['clear_under_declared_sweep_model'] and not s['endpoint']['outside_declared_wall_inner_faces'] and not s['endpoint']['positive_3d_AABB_pairs'] for s in steps)
    result={'status':'preflight_passed' if passed else 'preflight_rejected','created_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha256(a.plan),'initial':initial,'steps':steps,'native_tool_calls':0,'candidate_sequences':1,'no_native_state_mutated':True,'next':'Execute same frozen sequence once.' if passed else 'Do not execute this sequence; preserve failed geometric evidence.'}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'step_count':len(steps),'invalid_steps':[s['index'] for s in steps if not s['sweep']['clear_under_declared_sweep_model'] or s['endpoint']['outside_declared_wall_inner_faces'] or s['endpoint']['positive_3d_AABB_pairs']]}),flush=True)
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
