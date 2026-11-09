"""Fixed ordinary one-object translation search on the same real rooms.

This is a strong easy counterexample control, not a learned/new algorithm.
"""
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parent
W=ROOT.parent
OLD=W/'paris_local_run_results_20261104'
M=W/'joint_use_witness_20261104'
sys.path.insert(0,str(M))
from measure_joint_use import section_region, candidates, TASKS, measure, digest
from strong_pair_control import strong_solve
from evaluate import strong_measure, payload, changes


def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def canonical(pairs):return {tuple(sorted(p)) for p in pairs}


def run(arm,protocol):
    start=time.monotonic();out=ROOT/'ordinary_control'/arm;out.mkdir(parents=True,exist_ok=False)
    cp=OLD/f'{arm}_mesh_feedback_sdpa_kv/tools/checkpoint.json';state=payload(cp)
    oldaudit=read(OLD/f'{arm}_after_feedback_audit.json')
    measured=next(r for r in read(M/'measurements.json')['rows'] if r['arm']==arm and r['stage']=='mesh_feedback_sdpa_kv')
    assert measured['scene_sha256']==digest(cp.parent/'partial_scene.glb')
    before=strong_measure(measured)
    proposed=[];log=[];selected=None
    if before['status']!='witness':
        scene=trimesh.load(cp.parent/'partial_scene.glb',force='scene');meshes={}
        furniture=trimesh.collision.CollisionManager();architecture=trimesh.collision.CollisionManager()
        objects={o['object_id']:o for o in state['placed_instances']}
        for node in scene.graph.nodes_geometry:
            tf,gid=scene.graph[node];mesh=scene.geometry[gid].copy();mesh.apply_transform(tf);meshes[node]=mesh
            (furniture if node in objects else architecture).add_object(node,mesh)
        shell=read(state['frozen_shell']['path']);floor=next(o for o in shell['room_structure'] if o['name']=='floor')
        half=np.array([floor['dimensions'][0],floor['dimensions'][2]])/2
        oldpairs=canonical(oldaudit['furniture_surface_intersection_pairs'])
        oldshell=canonical(oldaudit['architecture_surface_intersection_pairs']['walls_doors_windows'])
        target_ids=sorted({b['object_id'] for s in measured['surfaces'] for b in s['blockers']} & objects.keys())
        # Translation leaves Z sections unchanged except for the same XY offset.
        sections={}
        for s in measured['surfaces']:
            sections[s['object_id']]={oid:section_region(mesh,s['z_m']+1e-4) for oid,mesh in meshes.items()}
        offsets=protocol['offsets_m']
        for oid,dx,dy in itertools.product(target_ids,offsets,offsets):
            if dx==dy==0:continue
            trial={'object_id':oid,'offset_xy_m':[dx,dy]};t=time.monotonic()
            bounds=meshes[oid].bounds.copy();bounds[:,:2]+=np.array([dx,dy])
            if np.any(np.maximum(-half-bounds[0,:2],bounds[1,:2]-half)>1e-5):
                trial.update(accepted=False,reason='outside_rectangular_floor');log.append(trial);continue
            tf=np.eye(4);tf[:2,3]=[dx,dy];furniture.set_transform(oid,tf)
            _,pairs=furniture.in_collision_internal(return_names=True)
            _,sp=furniture.in_collision_other(architecture,return_names=True)
            pairs=canonical(pairs)
            shellpairs=canonical((a,b) for a,b in sp if not b.startswith(('floor_','ceiling_')))
            furniture.set_transform(oid,np.eye(4))
            trial.update(furniture_pairs=sorted(pairs),new_furniture_pairs=sorted(pairs-oldpairs),new_shell_pairs=sorted(shellpairs-oldshell))
            if pairs-oldpairs or shellpairs-oldshell:
                trial.update(accepted=False,reason='new_mesh_intersection',seconds=time.monotonic()-t);log.append(trial);continue
            pools={t:[] for t in TASKS}
            for s in measured['surfaces']:
                sid=s['object_id'];poly=shape(s['surface'])
                if sid==oid:poly=translate(poly,dx,dy)
                blocked=[];unknown=[]
                for other,mesh in meshes.items():
                    if other==sid:continue
                    b=mesh.bounds.copy()
                    if other==oid:b[:,:2]+=np.array([dx,dy])
                    if not box(*b[:,:2].reshape(-1)).intersects(poly):continue
                    region,uncertain=sections[sid][other]
                    if other==oid:region=translate(region,dx,dy)
                    if uncertain:unknown.append(other)
                    blocked.append(region.intersection(poly))
                if unknown:continue
                free=poly.difference(unary_union(blocked) if blocked else Polygon())
                for task,dims in TASKS.items():pools[task].extend(candidates(free,dims,sid,s['z_m']))
            joint=strong_solve(pools['work'],pools['dining'])
            trial.update(joint=joint,accepted=joint['status']=='witness',seconds=time.monotonic()-t)
            log.append(trial)
        eligible=[r for r in log if r['accepted']]
        if eligible:
            selected=min(eligible,key=lambda r:(len(r['furniture_pairs']),math.hypot(*r['offset_xy_m']),r['object_id'],r['offset_xy_m']))
            obj=objects[selected['object_id']];q=obj['transform']['rotation_wxyz']
            yaw=math.degrees(math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3])))
            x,y,_=obj['transform']['translation'];dx,dy=selected['offset_xy_m']
            proposed=[{'name':'move_furniture_tool','arguments':{'object_id':obj['object_id'],'x':x+dx,'y':y+dy,'yaw':yaw}}]
    dump(out/'candidates.json',log);dump(out/'parsed_tool_calls.json',proposed)
    search_seconds=time.monotonic()-start
    command=[sys.executable,str(OLD/'dispatch_corrected.py'),'--calls',str(out/'parsed_tool_calls.json'),
             '--resume',str(cp),'--output',str(out/'tools')]
    t=time.monotonic();p=subprocess.run(command,capture_output=True,text=True,timeout=300)
    (out/'execution.stdout').write_text(p.stdout);(out/'execution.stderr').write_text(p.stderr);assert p.returncode==0,p.stderr
    execution_seconds=time.monotonic()-t
    p=subprocess.run([sys.executable,str(OLD/'audit_scene.py'),'--checkpoint',str(out/'tools/checkpoint.json'),
                      '--out',str(out/'audit.json')],capture_output=True,text=True)
    (out/'audit.stdout').write_text(p.stdout);(out/'audit.stderr').write_text(p.stderr);assert p.returncode==0,p.stderr
    after=measure(out/'tools/checkpoint.json');joint=strong_measure(after)
    dump(out/'functional_measure.json',after);dump(out/'joint_control.json',joint)
    diff=changes(state,payload(out/'tools/checkpoint.json'));dump(out/'state_changes.json',diff)
    audit=read(out/'audit.json')
    assert not diff['removed'] and not diff['added']
    if selected:
        assert joint['status']=='witness'
        assert not (canonical(audit['furniture_surface_intersection_pairs'])-canonical(oldaudit['furniture_surface_intersection_pairs']))
        assert not audit['out_of_rectangular_floor_objects']
        assert not (canonical(audit['architecture_surface_intersection_pairs']['walls_doors_windows'])-canonical(oldaudit['architecture_surface_intersection_pairs']['walls_doors_windows']))
    result={'arm':arm,'protocol_sha256':digest(ROOT/'ordinary_control_protocol.json'),'original_checkpoint':str(cp),
            'original_checkpoint_sha256':digest(cp),'before_joint':before,'after_joint':joint,'selected':selected,
            'candidates':len(log),'eligible_candidates':sum(r['accepted'] for r in log),'edits':len(proposed),
            'search_seconds':search_seconds,'execution_seconds':execution_seconds,'total_seconds':time.monotonic()-start,
            'before_collisions':len(oldaudit['furniture_surface_intersection_pairs']),
            'after_collisions':len(audit['furniture_surface_intersection_pairs']),
            'objects':audit['placed_furniture'],'boundary_objects':len(audit['out_of_rectangular_floor_objects']),
            'new_model_calls':0,'full_scene_success':None}
    dump(out/'result.json',result);print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    protocol=read(ROOT/'ordinary_control_protocol.json')
    rows=[run(arm,protocol) for arm in ['A','B']]
    dump(ROOT/'ordinary_control/results.json',rows)
