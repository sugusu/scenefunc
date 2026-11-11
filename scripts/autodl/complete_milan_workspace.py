"""Ordinary bounded catalog/geometry completion of a furnished Milan input.

Keeps all original objects unchanged. Chooses two new instances from the same
public catalog and searches their joint placement. Does not consume any old
east/west solution, designer placement, model response, or partial failed scene.
No model, training, renderer or paid service is called.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import types

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name, p):
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--native-source', type=Path, required=True)
    a = ap.parse_args(); root = a.root.resolve(); scratch = a.scratch.resolve()
    if (root/'completion_protocol.json').exists():
        raise FileExistsError('No implicit scene-search rerun')
    start = time.monotonic()
    sys.path.insert(0, str(Path('runs/autodl/20261110_public_function_search/original_evaluator').resolve()))
    sys.path.insert(0, str(a.native_source.resolve()))
    from native_project import validate_bundle, import_project
    from inspect_native_mesh_contacts import sdf_visuals, world_transform
    from measure_room_access import measure, obstacles
    from scenesmith.agent_utils.room import SceneObject
    import fcl
    from trimesh.collision import mesh_to_BVH
    bundle = scratch/'bundle'; manifest, house, runtime, task = validate_bundle(bundle)
    rid = manifest['selected_room_id']; initial = house['rooms'][rid]; room = bundle/('room_'+rid)
    meta = root/'source/runs/920/P920-MILAN-BASE-v1_20261105T225639Z/EVALUATION_PLAN.json'
    spec = read(meta); catalog = read(bundle/'catalog.json')['entries']
    assert sha(bundle/'manifest.json') == spec['initial_manifest_sha256']
    fixed_targets = {k:v['initial_xy'] for k,v in spec['role_approaches'].items()}
    protocol = dict(source_split='Previously exposed Milan development task, not held-out',
        starting_inventory='13 real HSSD furniture instances plus17 architectural records',
        immutable_start=True, preserve_every_initial_object_and_pose=True,
        action_space='Add one catalog table and one catalog chair; no scaling, deletion or movement of originals',
        role_inputs='Frozen public structured task and catalog names, not automatic raw-brief interpretation',
        independent_agent_comparison=False, model_calls=0, training_updates=0,
        candidate_cap=2000, exact_final_candidate_cap=12,
        lattice_m=.5, yaw_degrees=[0,90,180,270], chair_front_gap_m=.15,
        asset_order='Existing dining-height table before low coffee table, same dining chair capability; ordinary design heuristic, not a preference label',
        pose_order='Increasing distance to current sofa, then deterministic pose tuple',
        full_checks='All14 original role approaches at3 body widths; at least one work-chair approach per width; .4x.3m real-top witness, independent meal surface; mesh intersections; at least1 declared sofa ray',
        search_stop='First candidate passing conditional checks, or fixed cap. No quality-dependent retry.',
        original_solutions_read=False, real_use_acceptance=False,
        note='Additional instances are research-generated IDs, not historical owner furniture.')
    dump(root/'completion_protocol.json', protocol)
    before = {str(p):sha(p) for p in bundle.rglob('*') if p.is_file()}
    dump(root/'completion_input_hashes_before.json', before)
    cache = {}; local_mesh = {}
    for e in catalog:
        t = e['template']; key = t['sdf_path']
        local_mesh[key] = trimesh.util.concatenate([m for _,m in sdf_visuals(room/key,bundle,cache)])
    structures = [(n,m) for n,m in sdf_visuals(room/initial['room_geometry']['sdf_path'],bundle,cache)
                  if n not in ('floor','ceiling')]
    def bbox(obj):
        p = np.array(list(itertools.product(*zip(obj['bbox_min'],obj['bbox_max']))))
        p = trimesh.transform_points(p, world_transform(obj))
        return np.array([p.min(0),p.max(0)])
    def overlap(a,b,eps=1e-6):
        return bool(np.all(np.minimum(a[1],b[1])-np.maximum(a[0],b[0])>eps))
    originals = {i:o for i,o in initial['objects'].items() if o['object_type']=='furniture'}
    fixed_bboxes = [bbox(o) for o in originals.values()]+[m.bounds for _,m in structures]
    tables = [e for e in catalog if e['template']['name'] in spec['table_asset_names']]
    tables.sort(key=lambda e:(e['template']['name']!='dining_table',e['template']['name']))
    chairs = [e for e in catalog if e['template']['name'] in spec['chair_asset_names']]
    assert len(tables)==2 and len(chairs)==1
    width,depth = spec['material_patch_m']; capabilities=[]
    for e in tables:
        mesh=local_mesh[e['template']['sdf_path']];top=float(mesh.bounds[1,2]);tri=mesh.triangles
        sel=(mesh.face_normals[:,2]>.99)&(tri[:,:,2].min(1)>=top-.03)
        surface=unary_union([Polygon(t[:,:2]) for t in tri[sel] if Polygon(t[:,:2]).area>1e-10])
        x,y=mesh.bounds[:,:2].mean(0)
        patch=Polygon([(x+dx,y+dy) for dx,dy in [(-width/2,-depth/2),(width/2,-depth/2),(width/2,depth/2),(-width/2,depth/2)]])
        capabilities.append(dict(asset_id=e['binding']['asset_id'],name=e['template']['name'],
                                 top_height_m=top,material_patch_covered=bool(surface.buffer(1e-8).covers(patch))))
    dump(root/'catalog_workspace_capabilities.json',capabilities)
    sofa=initial['objects'][spec['sofa_id']]['transform']['translation'][:2]
    def instantiate(entry,oid,x,y,yaw):
        t=copy.deepcopy(entry['template']);theta=math.radians(yaw)
        t['object_id']=oid;t['transform']={'translation':[float(x),float(y),0.],
            'rotation_wxyz':[math.cos(theta/2),0.,0.,math.sin(theta/2)]}
        return t
    proposals=[]
    for ti,e in enumerate(tables):
        if not capabilities[ti]['material_patch_covered']:continue
        ch=chairs[0];template=e['template'];seat=ch['template']
        delta_y=template['bbox_min'][1]-seat['bbox_max'][1]-.15
        for x,y,yaw in itertools.product(np.arange(-4.5,4.51,.5),np.arange(-3.,3.01,.5),[0,90,180,270]):
            theta=math.radians(yaw);rot=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
            cx,cy=np.array([x,y])+rot@np.array([0.,delta_y])
            desk=instantiate(e,'ordinary_work_table_0',x,y,yaw)
            chair=instantiate(ch,'ordinary_work_chair_0',cx,cy,yaw)
            proposals.append(dict(id=len(proposals),table_index=ti,table=desk,chair=chair,
                sort=[ti,float(np.linalg.norm(np.array([x,y])-sofa)),float(x),float(y),yaw]))
    assert len(proposals)<=protocol['candidate_cap']
    proposals.sort(key=lambda p:p['sort'])
    logs=[];eligible=[];half=np.array([initial['room_geometry']['length'],initial['room_geometry']['width']])/2
    for p in proposals:
        bs=[bbox(p['table']),bbox(p['chair'])]
        inside=all(np.all(b[0,:2]>=-half-1e-6) and np.all(b[1,:2]<=half+1e-6) for b in bs)
        clear=inside and not overlap(*bs) and not any(overlap(b,old) for b in bs for old in fixed_bboxes)
        logs.append(dict(candidate_id=p['id'],table_asset=tables[p['table_index']]['binding']['asset_id'],
                         table_pose=p['table']['transform'],chair_pose=p['chair']['transform'],
                         inside=bool(inside),conservative_envelopes_clear=bool(clear)))
        if clear:eligible.append(p)
    dump(root/'cheap_joint_candidates.json',logs)
    # Compile the exact recorded ray helper alone; the module's old command
    # harness has unrelated filesystem dependencies and is never executed.
    fsrc=root/'source/scripts/920/measure_family_choice.py'
    fn=next(n for n in ast.parse(fsrc.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='hit_segment')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(fsrc),'exec'),ns)
    hit_segment=ns['hit_segment']
    helper=types.ModuleType('measure_family_choice');helper.hit_segment=hit_segment
    sys.modules['measure_family_choice']=helper
    bvh={k:mesh_to_BVH(m) for k,m in local_mesh.items()}
    structural={n:fcl.CollisionObject(mesh_to_BVH(m)) for n,m in structures}
    def contacts(s):
        bodies={i:fcl.CollisionObject(bvh[o['sdf_path']],fcl.Transform(world_transform(o)[:3,:3],world_transform(o)[:3,3]))
                for i,o in s['objects'].items() if o['object_type']=='furniture'}
        bbs={i:bbox(s['objects'][i]) for i in bodies};pairs=[]
        for left,right in itertools.combinations(sorted(bodies),2):
            if not overlap(bbs[left],bbs[right],1e-4):continue
            res=fcl.CollisionResult();fcl.collide(bodies[left],bodies[right],fcl.CollisionRequest(),res)
            if res.is_collision:pairs.append([left,right])
        walls=[]
        for oid,body in bodies.items():
            for name,mesh in structures:
                if not overlap(bbs[oid],mesh.bounds,1e-4):continue
                res=fcl.CollisionResult();fcl.collide(body,structural[name],fcl.CollisionRequest(),res)
                if res.is_collision:walls.append([oid,name])
        return dict(furniture_pairs=pairs,structure_pairs=walls)
    original_contacts=contacts(initial);dump(root/'initial_contacts.json',original_contacts)
    original_access=measure(initial,room,bundle,dict(spec['access'],approach_points_xy_m=fixed_targets))
    dump(root/'initial_access.json',original_access)
    def all_access(m):
        return all(s['entry_cell_clear'] and all(t['reachable_from_entry'] for t in s['targets'].values()) for s in m['scenarios'])
    exact=[];chosen=None
    for p in eligible[:protocol['exact_final_candidate_cap']]:
        t=time.monotonic();state=copy.deepcopy(initial)
        for key in ['table','chair']:state['objects'][p[key]['object_id']]=p[key]
        access=measure(state,room,bundle,dict(spec['access'],approach_points_xy_m=fixed_targets))
        chair=p['chair'];cm=world_transform(chair)
        approaches={k:(cm@np.array([*xy,0.,1.]))[:2].tolist() for k,xy in spec['chair_approach_offsets_m'].items()}
        seat_access=measure(state,room,bundle,dict(spec['access'],approach_points_xy_m=approaches))
        seat_ok=all(s['entry_cell_clear'] and any(t['reachable_from_entry'] for t in s['targets'].values()) for s in seat_access['scenarios'])
        blocks=obstacles(state,room,bundle,1.8);sm=world_transform(state['objects'][spec['sofa_id']]);rays=[]
        for z,dx in itertools.product(spec['ray_heights_m'],spec['sofa_offsets_x_m']):
            origin=np.array([cm[0,3],cm[1,3],z]);target=(sm@np.array([dx,0,z,1.]))[:3]
            hits=[b['id'] for b in blocks if hit_segment(origin,target,np.array(b['lo']),np.array(b['hi']))]
            rays.append(dict(clear=not hits,blockers=hits))
        contact=contacts(state)
        newf=[x for x in contact['furniture_pairs'] if x not in original_contacts['furniture_pairs']]
        neww=[x for x in contact['structure_pairs'] if x not in original_contacts['structure_pairs']]
        accepted=all_access(access) and seat_ok and any(x['clear'] for x in rays) and not newf and not neww
        record=dict(candidate_id=p['id'],accepted=bool(accepted),original_role_access_pass=all_access(access),
                    work_seat_access_pass=bool(seat_ok),clear_sight_rays=sum(x['clear'] for x in rays),
                    total_sight_rays=len(rays),new_furniture_pairs=newf,new_structure_pairs=neww,
                    exact_check_wall_seconds=time.monotonic()-t)
        exact.append(record);dump(root/'exact_candidates.json',exact)
        if accepted:
            chosen=dict(proposal=p,state=state,access=access,seat_access=seat_access,rays=rays,contact=contact)
            break
    result=dict(status='bounded_no_conditional_solution' if chosen is None else 'ordinary_conditional_workspace_completed',
        proposed_candidates=len(proposals),cheap_envelope_clear_candidates=len(eligible),
        exact_candidates=len(exact),new_instances=0,native_object_additions=0,model_calls=0,
        training_updates=0,new_whole_scene_generations=0,original_historical_house=False,
        automatic_raw_brief_understanding=False,full_scene_success=None)
    if chosen:
        completed=scratch/'completed_bundle';shutil.copytree(bundle,completed)
        project=import_project(completed,scratch/'completion_native_import')
        for key in ['table','chair']:
            obj=chosen['proposal'][key]
            project.room.add_object(SceneObject.from_dict(obj,scene_dir=project.room.scene_dir))
            entry=(tables[chosen['proposal']['table_index']] if key=='table' else chairs[0])
            runtime['rooms'][rid]['asset_bindings'][obj['object_id']]={**entry['binding'],
                'registry_id':entry['template']['object_id'],'identity_provenance':'ordinary_generated_task_instance'}
        actual=project.room.to_state_dict();assert all(actual['objects'][i]==o for i,o in initial['objects'].items())
        house['rooms'][rid]=actual
        dump(completed/'house_state.json',house);dump(completed/'runtime.json',runtime)
        newmanifest=copy.deepcopy(manifest)
        for path in ['house_state.json','runtime.json']:
            newmanifest['files'][path]={'bytes':(completed/path).stat().st_size,'sha256':sha(completed/path)}
        newmanifest['checkpoint_parent']={'manifest_sha256':sha(bundle/'manifest.json')}
        dump(completed/'manifest.json',newmanifest);validate_bundle(completed)
        dump(root/'completed_room_state.json',actual);dump(root/'completed_contacts.json',chosen['contact'])
        dump(root/'completed_role_access.json',chosen['access']);dump(root/'completed_seat_access.json',chosen['seat_access'])
        dump(root/'completed_sight_rays.json',chosen['rays'])
        # Execute the original evaluator once on this actual final native state.
        # Only filesystem relocation/code-path guard is adapted, all numerical
        # predicates and frozen thresholds remain the original evaluator.
        em=module('frozen_milan_evaluator',root/'source/scripts/920/evaluate_milan_policy.py')
        adapted=copy.deepcopy(spec);adapted['code_sha256']={}
        adapted['initial_bundle']=str(bundle);dump(root/'relocated_evaluation_plan.json',adapted)
        oldargs=sys.argv;sys.argv=['frozen_evaluator','--plan',str(root/'relocated_evaluation_plan.json'),
            '--bundle',str(completed),'--out',str(root/'original_evaluator_result.json')]
        try:em.main()
        finally:sys.argv=oldargs
        ev=read(root/'original_evaluator_result.json')
        result.update(selected_candidate_id=chosen['proposal']['id'],new_instances=2,native_object_additions=2,
            furniture_instances=sum(o['object_type']=='furniture' for o in actual['objects'].values()),
            all_original_objects_exact=True,all_original_assets_exact=True,
            original_required_role_access=[dict(width=s['square_body_width_m'],passed=sum(t['reachable_from_entry'] for t in s['targets'].values()),total=len(s['targets'])) for s in chosen['access']['scenarios']],
            dedicated_workspace_geometry_verified=ev['dedicated_work_geometry_found'],
            final_furniture_intersections=len(chosen['contact']['furniture_pairs']),
            final_structure_intersections=len(chosen['contact']['structure_pairs']),
            completed_native_bundle=str(completed),final_objects=len(actual['objects']),
            native_add_is_not_official_HTTP_tool_call=True)
    after={p:sha(p) for p in before};assert before==after
    dump(root/'completion_input_hashes_after.json',after)
    result.update(wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_inputs_unchanged=True,local_algorithm='ordinary finite CSP and catalog role matching')
    dump(root/'completion_result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
