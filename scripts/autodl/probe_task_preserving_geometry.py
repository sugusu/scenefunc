"""CPU task-preserving geometry probes on one furnished native source family.

Creates two explicit research worlds, never edits the source bundle. A baked
XY-scaled derivative of one real table supplies a necessary-area UNSAT witness;
the other world supplies actual simultaneous surface-packing witnesses. This is
ordinary labelled program disambiguation, not a new learning algorithm, human
preference study, model failure, or two independent room-generation successes.
"""
import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import resource
import shutil
import socket
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

R=Path(__file__).resolve().parents[2]
W=R.parent.parent
NATIVE=R/'runs/autodl/20261110_native_state_runtime/source'
HELPERS=R/'runs/autodl/20261110_public_function_search/original_evaluator'


def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--source-bundle',type=Path,required=True)
    ap.add_argument('--scratch',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();source=a.source_bundle.resolve();scratch=a.scratch.resolve()
    if root.exists() or scratch.exists():raise FileExistsError('Fresh probe outputs only; no implicit retry')
    if shutil.disk_usage(R).free < 5*1024**3+3*1024**2:raise RuntimeError('Keep5GiB reserve; world geometry belongs in RAM')
    root.mkdir(parents=True);scratch.mkdir(parents=True);started=time.monotonic()
    def offline(*_,**__):raise RuntimeError('Offline geometry probe forbids network')
    socket.socket.connect=offline;socket.create_connection=offline
    sys.path[:0]=[str(NATIVE),str(HELPERS),str(W/'joint_use_witness_20261104')]
    from native_project import validate_bundle,import_project
    from inspect_native_mesh_contacts import sdf_visuals,world_transform
    from measure_joint_use import candidates
    solver=module('same_goal_ordinary_solver',R/'scripts/autodl/activity_resource_program.py')
    manifest,house,runtime,original_task=validate_bundle(source)
    rid=manifest['selected_room_id'];initial=house['rooms'][rid];table=initial['objects']['meal_table_0']
    room=source/('room_'+rid)
    before={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
    raw=trimesh.util.concatenate([m for _,m in sdf_visuals(room/table['sdf_path'],source,{})])
    assert np.allclose(raw.bounds,np.array([table['bbox_min'],table['bbox_max']]),atol=1e-6,rtol=0)
    # Fixed prospectively, before packing either world. Preserve table height.
    scale=.65
    task_text=(original_task['design_request']+'\n\n研究者补充的同任务机制检查（不是原住户数据）：'
        '住户工作时需要40×40厘米的资料区域；资料用餐时仍留在同一工作面。'
        '用餐时住户与一位客人各需35×25厘米的独立用餐区域，住户此时不另占工作席位。'
        '检查现有meal_table_0能否在两个活动阶段支持这些区域；同时占用的区域不得重叠。'
        '可使用0或90度的平面矩形区域；区域仅代表声明的研究代理，不能据此认定实际人体使用合格。'
        '这里只询问共享桌面的可行性，共享不行不等于整场任务不可能，可另用工作面。')
    contract=dict(work_m=[.4,.4],meal_m=[.35,.25],work_people=['resident'],
        meal_people=['resident','guest'],material_persists_during_meal=True,
        resident_working_during_meal=False,sharing_seat_between_modes_allowed=True,
        table_id='meal_table_0',source='Research-authored explicit contract, not automatic language interpretation',
        quality_scope='Planar reservation feasibility only; no ergonomic or subjective label')
    task=copy.deepcopy(original_task);task['design_request']=task_text
    task['same_task_probe_contract']=contract
    task['request_authorship']='Research transfer plus disclosed fixed activity proxy; same text in both worlds'
    derived_id='research_dining_table_xy065'
    asset_rel='assets/'+derived_id+'/geometry.glb'
    sdf_rel='objects/'+derived_id+'/object.sdf'
    dump(root/'protocol.json',dict(source_manifest_sha256=sha(source/'manifest.json'),
        task_text_sha256=hashlib.sha256(task_text.encode()).hexdigest(),task_text=task_text,
        contract=contract,world_xy_scales=[1.,scale],scale_fixed_before_packing=True,
        world_intervention='New research initial world with one baked XY-scale table derivative; not an allowed policy edit or original HSSD asset',
        same_task_same_parameters=True,other29objects_unchanged=True,
        sample_status='One previously exposed development source family, two controlled worlds; not independent test cases',
        hypotheses=['required_persistence','omit_persistence','force_separate_tables'],
        ground_truth='Explicit contract plus positive geometric witness or sound necessary-area certificate; never unlabelled majority',
        source_task_score_unchanged=True,model_calls=0,training_updates=0,paid_API_calls=0,
        no_quality_dependent_world_or_task_resampling=True))
    derived=raw.copy();derived.vertices=derived.vertices*np.array([scale,scale,1.])
    # SDF visuals interpret glTF Y-up. Bake the actual local Z-up visual into it.
    to_yup=np.eye(4);to_yup[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
    gltf=derived.copy();gltf.apply_transform(to_yup)
    asset_bytes=trimesh.Scene(gltf).export(file_type='glb')
    sdf=ET.Element('sdf',version='1.7');model=ET.SubElement(sdf,'model',name=derived_id)
    link=ET.SubElement(model,'link',name='body');visual=ET.SubElement(link,'visual',name='research_scaled_source')
    geom=ET.SubElement(visual,'geometry');mesh=ET.SubElement(geom,'mesh')
    ET.SubElement(mesh,'uri').text='../../'+asset_rel
    center=derived.bounds.mean(0);ext=derived.extents
    col=ET.SubElement(link,'collision',name='research_envelope')
    ET.SubElement(col,'pose').text=' '.join(map(str,[*center,0.,0.,0.]))
    box=ET.SubElement(ET.SubElement(col,'geometry'),'box');ET.SubElement(box,'size').text=' '.join(map(str,ext))
    sdf_bytes=ET.tostring(sdf,encoding='utf-8',xml_declaration=True)
    rows=[];world_states={}
    for world_id,factor in [('original_table',1.),('smaller_table',scale)]:
        t=time.monotonic();dest=scratch/world_id;shutil.copytree(source,dest)
        h=copy.deepcopy(house);rt=copy.deepcopy(runtime);m=copy.deepcopy(manifest)
        catalog=read(source/'catalog.json')
        base=dest/('room_'+rid)
        for rel,blob in [(asset_rel,asset_bytes),(sdf_rel,sdf_bytes)]:
            p=base/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
            key=p.relative_to(dest).as_posix();m['files'][key]=dict(bytes=len(blob),sha256=sha(p))
        template=copy.deepcopy(table)
        template.update(object_id='catalog_'+derived_id,name=derived_id,geometry_path=asset_rel,sdf_path=sdf_rel,
            transform=dict(translation=[0.,0.,0.],rotation_wxyz=[1.,0.,0.,0.]),
            bbox_min=derived.bounds[0].tolist(),bbox_max=derived.bounds[1].tolist(),
            description='Research-derived baked XY-scale geometry. No claim of real ergonomic suitability.',
            metadata=dict(asset_source='research_derived_hssd_geometry',parent_asset='5d70cc76c99acee4513ec5f7fad497d2baebca9e',
                operation='baked_xy_scale',xy_scale=scale,source_local_visual_bounds=raw.bounds.tolist()))
        catalog['entries'].append(dict(template=template,binding=dict(asset_id=derived_id,
            identity_provenance='research_derived_geometry_not_original_catalog_asset'),
            limitations=['No material-fidelity or ergonomic certification; triangles and bounding boxes only.']))
        if factor!=1.:
            replacement=copy.deepcopy(template);replacement.update(object_id='meal_table_0',name=table['name'],transform=table['transform'])
            h['rooms'][rid]['objects']['meal_table_0']=replacement
            rt['rooms'][rid]['asset_bindings']['meal_table_0']=dict(asset_id=derived_id,
                registry_id=template['object_id'],identity_provenance='explicit_research_counterfactual_initial_geometry')
        for filename,value in [('house_state.json',h),('runtime.json',rt),('catalog.json',catalog),('task.json',task)]:
            dump(dest/filename,value);m['files'][filename]=dict(bytes=(dest/filename).stat().st_size,sha256=sha(dest/filename))
        m['checkpoint_parent']=dict(manifest_sha256=sha(source/'manifest.json'))
        m['project']['provenance']=str(m['project']['provenance'])+'; declared same-task geometry probe'
        dump(dest/'manifest.json',m);validate_bundle(dest)
        imported=import_project(dest,scratch/(world_id+'_native_import'))
        state=imported.room.to_state_dict();assert state==h['rooms'][rid]
        assert len(state['objects'])==30
        assert all(state['objects'][oid]==obj for oid,obj in initial['objects'].items() if oid!='meal_table_0')
        o=state['objects']['meal_table_0'];local=trimesh.util.concatenate([x for _,x in sdf_visuals(base/o['sdf_path'],dest,{})])
        # Serialized GLB float32 roundoff is measured, not silently asserted zero.
        expected_local=raw if factor==1. else derived
        max_bounds_error=float(np.abs(local.bounds-expected_local.bounds).max())
        assert max_bounds_error<=1e-6
        actual=local.copy();actual.apply_transform(world_transform(o));top_z=float(actual.bounds[1,2])
        tri=actual.triangles;keep=(actual.face_normals[:,2]>.99)&(tri[:,:,2].min(1)>=top_z-.03)
        top=unary_union([Polygon(t[:,:2]) for t in tri[keep] if Polygon(t[:,:2]).area>1e-10])
        assert top.is_valid and not top.is_empty
        work=candidates(top,tuple(contract['work_m']),'meal_table_0',top_z)
        meal=candidates(top,tuple(contract['meal_m']),'meal_table_0',top_z)
        inverse=np.linalg.inv(world_transform(o))
        n=[x for x in meal if (inverse@np.array([*x['center_xy'],top_z,1]))[1]>=0]
        s=[x for x in meal if (inverse@np.array([*x['center_xy'],top_z,1]))[1]<=0]
        anchor=dict(surface_id='unused_auxiliary',polygon=Polygon([(100,100),(101,100),(101,101),(100,101)]))
        mealpair=solver.three_rectangles([anchor],n,s)
        assert work and mealpair['status']=='witness','Individual modes must remain feasible; not a vacuous globally broken world'
        required=float(np.prod(contract['work_m'])+2*np.prod(contract['meal_m']))
        if required>top.area:
            shared=dict(status='certified_infeasible_for_disjoint_planar_reservations',value=False,
                certificate=dict(kind='necessary_area_bound',required_m2=required,available_m2=float(top.area),
                                 slack_m2=float(top.area)-required),
                continuous_positions_and_angles_also_impossible=True)
        else:
            witness=solver.three_rectangles(work,n,s)
            assert witness['status']=='witness','No resampling or finite-search-negative relabeling'
            shared=dict(status='witness',value=True,certificate=witness,
                continuous_positions_and_angles_also_impossible=False)
        bounds=actual.bounds;old_bounds=raw.bounds+np.array(table['transform']['translation'])
        assert np.all(bounds[0]>=old_bounds[0]-1e-6) and np.all(bounds[1]<=old_bounds[1]+1e-6)
        world_states[world_id]=state
        dump(root/(world_id+'_surface.json'),dict(polygon=mapping(top),world_bounds_m=bounds.tolist(),top_z=top_z,
            area_m2=float(top.area),derived_asset_GLTF_roundoff_max_bounds_m=max_bounds_error))
        dump(root/(world_id+'_world_metadata.json'),dict(manifest=m,house_state=h,runtime=rt,catalog=catalog,task=task,
            source_bundle=str(source),RAM_bundle=str(dest),new_resources=[dict(relative=('room_'+rid+'/'+rel),sha256=sha(base/rel),bytes=(base/rel).stat().st_size) for rel in [asset_rel,sdf_rel]],
            reconstruct='Run this script using the original licensed source; generated resource hashes are checked in this record. No copyrighted mesh copied into Git deliverable.'))
        row=dict(world=world_id,task_text_sha256=hashlib.sha256(task_text.encode()).hexdigest(),xy_scale=factor,
            objects=30,furniture=13,unmodified_other_objects=29,table_height_m=top_z,
            source_mesh_triangles=len(raw.faces),world_mesh_triangles=len(actual.faces),
            top_area_m2=float(top.area),reserved_area_m2=required,
            work_alone_witness={k:(mapping(v) if k=='polygon' else v) for k,v in work[0].items()},
            two_meals_without_work_reservation_witness=mealpair,
            correct_shared_feasibility=shared,
            hypothesis_outputs=dict(required_persistence=shared['value'],
                omit_persistence=True,force_separate_tables=False),
            individual_activity_modes_feasible=True,table_envelope_subset_of_initial=True,
            raw_NL_parsed_by_model=False,real_human_acceptance=None,
            native_state_roundtrip_exact=True,CPU_seconds=time.monotonic()-t)
        rows.append(row);dump(root/'world_results.json',dict(status='running',rows=rows))
    assert len({r['task_text_sha256'] for r in rows})==1
    assert [r['correct_shared_feasibility']['value'] for r in rows]==[True,False]
    assert world_states['original_table']==initial
    # No learned classifier or claim of identifying a general program class.
    signatures={h:[r['hypothesis_outputs'][h] for r in rows] for h in rows[0]['hypothesis_outputs']}
    gold=[r['correct_shared_feasibility']['value'] for r in rows]
    survivors={}
    for subset in [[],[0],[1],[0,1]]:
        survivors[str(subset)]=[h for h,v in signatures.items() if all(v[i]==gold[i] for i in subset)]
    assert survivors['[0, 1]']==['required_persistence']
    dump(root/'labelled_program_disambiguation.json',dict(hypotheses=signatures,gold=gold,survivors=survivors,
        all_worlds_have_identical_task=True,single_source_family=True,
        ordinary_method='Labelled consistency filtering on explicitly authored finite hypotheses; known semantic-parsing prior, not a new learner',
        unlabeled_disagreement_is_not_label_authority=True,
        general_program_identifiability=False,policy_accuracy_not_evaluated=True))
    assert before=={p:sha(p) for p in before}
    result=dict(status='same_task_geometry_probe_completed',rows=rows,
        wall_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_files_unchanged=len(before),worlds=2,independent_source_families=1,
        new_raw_text_whole_scene_generations=0,model_calls=0,model_forwards=0,training_updates=0,paid_API_calls=0,
        altered_original_scores=False,full_scene_functional_success=None,
        limitations=['Controlled research task and derivative geometry; not a real user study.',
          'Planar region reservations are not complete dining ergonomics or material objects.',
          'No new native five-view render, neural judgement, critic training or generalization result.',
          'Scaling initial research geometry is not a policy permission to resize fixed user furniture.',
          'Final world envelope containment does not validate all human functions or exact photometric materials.'])
    dump(root/'world_results.json',result)
    dump(root/'source_before_after.json',dict(unchanged=True,files=before))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
