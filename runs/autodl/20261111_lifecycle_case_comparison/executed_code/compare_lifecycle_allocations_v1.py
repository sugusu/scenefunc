"""Actual CPU/native whole-home completion, with a shared bounded selector.

The case arm is a hand-compiled proposal prior, not a neural Agent. Target
family final plans/actions are excluded. Failure and identical policies remain
valid outcomes. Geometry and approach witnesses are explicitly conditional.
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
from shapely.geometry import Polygon


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()


def pose(x,y,yaw):
    h=math.radians(yaw)/2
    return dict(translation=[float(x),float(y),0.],rotation_wxyz=[math.cos(h),0.,0.,math.sin(h)])


def new_object(template, oid, xy, yaw):
    obj=copy.deepcopy(template);obj.update(object_id=oid,name=oid,transform=pose(*xy,yaw))
    return obj


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--scratch',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();scratch=a.scratch.resolve()
    if (root/'summary.json').exists() or (root/'search_started.json').exists():
        raise FileExistsError('No automatic overwrite or quality-dependent rerun')
    started=time.monotonic();protocol=read(root/'protocol.json')
    write(root/'search_started.json',dict(unix=time.time(),protocol_sha256=digest(protocol)))
    sys.path.insert(0,str(Path('runs/autodl/20261110_native_state_runtime/source').resolve()))
    sys.path.insert(0,str(Path('runs/autodl/20261110_public_function_search/original_evaluator').resolve()))
    from native_project import import_project,validate_bundle,sha256
    from inspect_native_mesh_contacts import sdf_visuals,world_transform
    from measure_room_access import measure
    from trimesh.collision import mesh_to_BVH
    import trimesh
    import fcl
    bundle=scratch/'initial/bundle'
    manifest,house,_,_=validate_bundle(bundle)
    rid=manifest['selected_room_id'];initial=house['rooms'][rid];roomdir=bundle/('room_'+rid)
    entries={e['template']['name']:e for e in read(bundle/'catalog.json')['entries']}
    assert len(initial['objects'])==29 and 'former_child_bed_0' in initial['objects']
    # Public historical evaluation settings are read as task conventions only,
    # before search. No original final pose or tool action is read.
    import subprocess
    pin='75fc783f86d4fea78714639e110fbe77476ee106'
    path='runs/920/P920-LIFECYCLE-v1_20261105T221829Z/GEOMETRY_PLAN.json'
    raw=subprocess.check_output(['git','-c','gc.auto=0','show',pin+':'+path])
    (root/'source/geometry_measurement_conventions.json').write_bytes(raw)
    old=read(root/'source/geometry_measurement_conventions.json')
    settings=dict(protocol['access']);settings.pop('preservation')
    settings.update(entry_center_xy_m=old['entry']['interior_boundary_xy_m'],
                    entry_inward_normal_xy=old['entry']['inward_normal_xy'],
                    approach_points_xy_m=old['fixed_approach_points_xy_m'])
    write(root/'common_task_measurement.json',dict(settings=settings,
        source_role='Historical research proxy endpoints, not original resident measurements or final desk coordinates',
        office_bounds=[2.,0.,5.,2.5],desk_min_area_m2=0.60,desk_height_range_m=[0.65,0.85],
        use_capacity='One exclusive table/chair pair in the convertible room; old three dining seats unchanged',
        destination_of_removed_bed='unknown, explicitly recorded loss of capacity'))
    # Build geometry once; exact FCL results remain independently evaluated in
    # each arm. No cross-arm result memoization or extra case-only tools.
    meshes={};bvhs={};cache={}
    def mesh_for(obj):
        key=obj['sdf_path']
        if key not in meshes:
            meshes[key]=trimesh.util.concatenate([m for _,m in sdf_visuals(roomdir/key,bundle,cache)])
            bvhs[key]=mesh_to_BVH(meshes[key])
        return meshes[key]
    for obj in initial['objects'].values():
        if obj['sdf_path']:mesh_for(obj)
    for name in ['dining_table','dining_chair_1']:mesh_for(entries[name]['template'])
    structure=[(n,m) for n,m in sdf_visuals(roomdir/initial['room_geometry']['sdf_path'],bundle,cache)
               if n not in ['floor','ceiling'] and 'window' not in n]
    wall_bvhs={n:mesh_to_BVH(m) for n,m in structure}
    def bounds(o):
        pts=np.array(list(itertools.product(*zip(o['bbox_min'],o['bbox_max']))))
        pts=trimesh.transform_points(pts,world_transform(o))
        return np.array([pts.min(0),pts.max(0)])
    def poly(o):
        b=bounds(o);return Polygon([[b[0,0],b[0,1]],[b[1,0],b[0,1]],
                                   [b[1,0],b[1,1]],[b[0,0],b[1,1]]])
    def contacts(s):
        obs={oid:o for oid,o in s['objects'].items() if o['sdf_path']}
        boxes={i:bounds(o) for i,o in obs.items()};bodies={}
        for i,o in obs.items():
            mesh_for(o);t=world_transform(o)
            bodies[i]=fcl.CollisionObject(bvhs[o['sdf_path']],fcl.Transform(t[:3,:3],t[:3,3]))
        pairs=[];walls=[]
        def overlap(b,c):
            return bool(np.all(np.minimum(b[1],c[1])-np.maximum(b[0],c[0])>protocol['contact_threshold_m']))
        for i,j in itertools.combinations(sorted(obs),2):
            if overlap(boxes[i],boxes[j]):
                r=fcl.CollisionResult();fcl.collide(bodies[i],bodies[j],fcl.CollisionRequest(),r)
                if r.is_collision:pairs.append([i,j])
        for oid,body in bodies.items():
            for name,m in structure:
                if overlap(boxes[oid],m.bounds):
                    r=fcl.CollisionResult();fcl.collide(body,fcl.CollisionObject(wall_bvhs[name]),fcl.CollisionRequest(),r)
                    if r.is_collision:walls.append([oid,name])
        return dict(furniture_pairs=pairs,structure_pairs=walls)
    init_access=measure(initial,roomdir,bundle,settings);init_contacts=contacts(initial)
    write(root/'initial_access.json',init_access);write(root/'initial_contacts.json',init_contacts)
    write(root/'initial_state.json',initial)
    def access_regressions(after):
        regression=[]
        for old_s,new_s in zip(init_access['scenarios'],after['scenarios']):
            if old_s['entry_cell_clear'] and not new_s['entry_cell_clear']:
                regression.append([old_s['square_body_width_m'],'entry'])
            for name,t in old_s['targets'].items():
                if t['reachable_from_entry'] and not new_s['targets'][name]['reachable_from_entry']:
                    regression.append([old_s['square_body_width_m'],name])
        return regression
    # Only 2 role allocations are possible with the frozen table/seat template
    # vocabulary. Reusing an occupied family pair is available to BOTH arms,
    # but cannot satisfy the retained dining identities/exclusive office goal.
    table=entries['dining_table']['template'];chair=entries['dining_chair_1']['template']
    half=np.array(table['bbox_max'])[:2];chalf=np.array(chair['bbox_max'])[:2]
    pool=[]
    for remove,allocation,yaw,side in itertools.product([False,True],['add_catalog_pair','reuse_family_pair'],[0,90,180,270],[-1,1]):
        rad=math.radians(yaw);R=np.array([[math.cos(rad),-math.sin(rad)],[math.sin(rad),math.cos(rad)]])
        extent=np.abs(R)@half
        xs=np.arange(2.+extent[0]+.10,5.-extent[0]-.10+1e-8,.30)
        ys=np.arange(0.+extent[1]+.10,2.5-extent[1]-.10+1e-8,.30)
        for x,y in itertools.product(xs,ys):
            txy=np.array([x,y]);cxy=txy+R@np.array([0.,side*(half[1]+chalf[1]+.15)])
            pool.append(dict(candidate_id=len(pool),remove_unneeded_bed=remove,allocation=allocation,
                table_xy=txy.tolist(),chair_xy=cxy.tolist(),table_yaw=yaw,
                chair_yaw=(yaw+(180 if side>0 else 0))%360,
                approach_xy=(cxy+R@np.array([0.,side*(chalf[1]+.35)])).tolist()))
    def apply(row):
        s=copy.deepcopy(initial)
        if row['remove_unneeded_bed']:del s['objects']['former_child_bed_0']
        if row['allocation']=='add_catalog_pair':
            s['objects']['new_work_table_0']=new_object(table,'new_work_table_0',row['table_xy'],row['table_yaw'])
            s['objects']['new_work_chair_0']=new_object(chair,'new_work_chair_0',row['chair_xy'],row['chair_yaw'])
        else:
            for oid,point,yaw in [('family_table_0',row['table_xy'],row['table_yaw']),
                                 ('family_chair_n_0',row['chair_xy'],row['chair_yaw'])]:
                s['objects'][oid]['transform']=pose(*point,yaw)
        return s
    # Cheap task-aware factors do not read exact route/mesh test outputs.
    for row in pool:
        s=apply(row);add=row['allocation']=='add_catalog_pair'
        ids=['new_work_table_0','new_work_chair_0'] if add else ['family_table_0','family_chair_n_0']
        violations=int(not add);outside=0;overlaps=0
        for oid in ids:
            b=bounds(s['objects'][oid]);outside+=int(not(np.all(b[0,:2]>=[2.,0.]) and np.all(b[1,:2]<=[5.,2.5])))
            p=poly(s['objects'][oid])
            for j,o in s['objects'].items():
                if j in ids or not o['sdf_path']:continue
                overlaps+=int(p.intersection(poly(o)).area>1e-8)
        violations+=outside
        # Both source cards are common information. A source-compatible prior
        # penalizes consuming a continuing family resource; TM8's blanket
        # bedroom-retention transfer is explicitly rejected by task conditions.
        penalty=int(row['allocation']=='reuse_family_pair')
        operations=int(row['remove_unneeded_bed'])+2
        row.update(cheap_task_violations=violations,cheap_overlap_count=overlaps,
            common_order_key=[violations,overlaps,operations,row['candidate_id']],
            case_compatibility_penalty=penalty,
            case_order_key=[violations,overlaps,penalty,operations,row['candidate_id']])
    write(root/'common_candidates.json',pool)
    write(root/'case_applicability_trace.json',dict(pine=dict(partial=True,
        transfer='Separate new office surface so occupied family dining resource is not consumed.',
        not_transferred=['two-worker count','exact L shape','window coordinates']),
        tm8=dict(blanket_preserve_all_bedrooms_applicable=False,
        reason='Task explicitly releases one former bed and does not require guest sleeping.'),
        target_case_read=False,independently_learned=False,expected_redundancy='Occupied resources already protected by common hard task factors'))
    rows_by_arm={};results=[]
    for arm in protocol['arms']:
        t0=time.monotonic();native=import_project(bundle,scratch/arm/'native')
        key='common_order_key' if arm.startswith('B0') else 'case_order_key'
        order=sorted(pool,key=lambda r:r[key]);rows_by_arm[arm]=[r['candidate_id'] for r in order]
        ledger=[];chosen=None
        for rank,row in enumerate(order):
            if len(ledger)>=protocol['common_budget']['native_candidate_cap'] or time.monotonic()-t0>protocol['common_budget']['wall_seconds_cap_per_arm']:
                break
            # Cheap hard requirements are equal and prior-independent. The
            # geometrically overlapping ones remain in the trace; they are
            # not forced through native/FCL just to weaken ordinary search.
            if row['cheap_task_violations'] or row['cheap_overlap_count']:continue
            s=apply(row);native.room.restore_from_state_dict(s);actual=native.room.to_state_dict()
            assert set(s['objects'])==set(actual['objects'])
            for oid,o in s['objects'].items():
                assert np.max(np.abs(world_transform(o)-world_transform(actual['objects'][oid])))<1e-12
            settings_new=copy.deepcopy(settings);settings_new['approach_points_xy_m']['new_office_seat']=row['approach_xy']
            access=measure(actual,roomdir,bundle,settings_new);geom=contacts(actual)
            regressions=access_regressions(access)
            office=all(scn['targets']['new_office_seat']['reachable_from_entry'] for scn in access['scenarios'])
            new_pairs=[p for p in geom['furniture_pairs'] if p not in init_contacts['furniture_pairs']]
            new_walls=[p for p in geom['structure_pairs'] if p not in init_contacts['structure_pairs']]
            newids=['new_work_table_0','new_work_chair_0']
            ground={i:bool(abs(bounds(actual['objects'][i])[0,2])<=protocol['ground_tolerance_m']) for i in newids}
            unchanged=all(actual['objects'][i]==o for i,o in initial['objects'].items() if i!='former_child_bed_0')
            accepted=bool(office and not regressions and not new_pairs and not new_walls and all(ground.values()) and unchanged)
            entry=dict(candidate_id=row['candidate_id'],rank=rank,native_evaluation=len(ledger)+1,
                accepted=accepted,office_access_all_widths=office,access_regressions=regressions,
                new_furniture_intersections=new_pairs,new_structure_intersections=new_walls,
                ground=ground,preserved_objects_unchanged=unchanged)
            ledger.append(entry)
            write(root/arm/f'candidate_{row["candidate_id"]:04d}.json',dict(result=entry,access=access,contacts=geom))
            if accepted:chosen=(row,actual,access,geom);break
        write(root/arm/'candidate_evaluations.json',ledger)
        final=chosen[1] if chosen else initial
        # Explicit native commit/roundtrip. An unaccepted candidate never
        # replaces the initial state. Addition/removal is an atomic endpoint;
        # carrying/disposal paths are NOT claimed to be demonstrated.
        native.room.restore_from_state_dict(final)
        final=native.room.to_state_dict();write(root/arm/'final_state.json',final)
        unknowns=['Actual desk suitability/equipment/power','Sitting/reach/comfort','Transport and destination of removed bed',
                  'Privacy/door operation/utility function','Original surveyed-home validity','Daylight and human preference']
        record=dict(arm=arm,proposals=len(pool),cheap_admissible_candidates=sum(not r['cheap_task_violations'] and not r['cheap_overlap_count'] for r in pool),
            native_candidates=len(ledger),first_valid_native_evaluation=next((r['native_evaluation'] for r in ledger if r['accepted']),None),
            first_valid_stream_rank=next((r['rank'] for r in ledger if r['accepted']),None),
            selected_candidate_id=chosen[0]['candidate_id'] if chosen else None,
            office_conditional_geometry=bool(chosen),original_object_count=len(initial['objects']),
            final_object_count=len(final['objects']),retained_unchanged_object_count=sum(i in final['objects'] and final['objects'][i]==o for i,o in initial['objects'].items()),
            required_instance_identity_preserved=all(i in final['objects'] for i in initial['objects'] if i!='former_child_bed_0'),
            additions=2 if chosen else 0,permitted_removals=int('former_child_bed_0' not in final['objects']),
            fixed_access_regressions=0 if chosen else None,
            remaining_furniture_intersections=len(chosen[3]['furniture_pairs'] if chosen else init_contacts['furniture_pairs']),
            remaining_structure_intersections=len(chosen[3]['structure_pairs'] if chosen else init_contacts['structure_pairs']),
            final_state_digest=digest(final),wall_seconds=time.monotonic()-t0,
            model_calls=0,training_updates=0,full_scene_generation_success=None,unknowns=unknowns)
        results.append(record);write(root/arm/'result.json',record);print(json.dumps(record),flush=True)
    write(root/'proposal_order_comparison.json',dict(orders=rows_by_arm,
        identical=rows_by_arm[protocol['arms'][0]]==rows_by_arm[protocol['arms'][1]],
        case_condition_is_already_a_common_task_factor=True,
        identical_backend=True,learned_agent_comparison=False))
    result=dict(arms=results,wall_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        same_final_state=results[0]['final_state_digest']==results[1]['final_state_digest'],
        target_final_solution_used=False,from_empty_scene_generation=False,
        actual_native_completion_experiments=2,independent_projects=1,new_model_calls=0,paid_API_calls=0,training_updates=0,
        all_declared_capabilities='Native state reconstruction/addition/removal/serialization; mesh/planar route evidence, not real human use',
        full_scene_generation_success=None)
    write(root/'summary.json',result)
    fields=[k for k in results[0] if k!='unknowns']
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in results)
    print(json.dumps(dict(status='completed_bounded_comparison',same_end=result['same_final_state'],seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':
    main()
