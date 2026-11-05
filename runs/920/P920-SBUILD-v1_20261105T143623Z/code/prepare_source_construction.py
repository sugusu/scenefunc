"""Derive a new construction task from a registered training-source reconstruction.

Installed fixtures remain explicit inputs. Source placements are supervised
targets only; no historical trajectory or professional reason is invented.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

import trimesh
from inspect_native_mesh_contacts import sdf_visuals
from native_project import sha256, validate_bundle


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    for name,h in plan['code_sha256'].items():
        assert sha256(Path(__file__).parent/name)==h,name
    for item in plan['inputs'].values():
        assert sha256(Path(item['path']))==item['sha256'],item['path']
    registry=json.loads(Path(plan['inputs']['registry']['path']).read_text())
    project=next(x for x in registry['projects'] if x['family_id']==plan['family_id'])
    assert project['allocation']=='train' and plan['family_id'] not in registry['sealed_test_projects']
    source=Path(plan['inputs']['base_manifest']['path']).parent
    manifest,state,runtime,original_task=validate_bundle(source)
    rid=manifest['selected_room_id'];room=state['rooms'][rid]
    target=json.loads(Path(plan['inputs']['target_state']['path']).read_text())
    assert target['room_geometry']==room['room_geometry']
    selected=plan['movable_source_ids'];assert len(set(selected))==9
    assert all(room['objects'][oid]['object_type']=='furniture' for oid in selected)
    args.out.mkdir(parents=True,exist_ok=False);bundle=args.out/'bundle'
    shutil.copytree(source,bundle);roomdir=bundle/('room_'+rid)
    entries=[];commands=[];mapping=[]
    for index,oid in enumerate(selected,1):
        old=room['objects'].pop(oid);binding=runtime['rooms'][rid]['asset_bindings'].pop(oid)
        template=copy.deepcopy(old);reg='catalog_construct_'+old['name']
        assert abs(old['bbox_min'][2])<1e-9 and abs(target['objects'][oid]['transform']['translation'][2])<1e-9
        if template['geometry_path'] is None:
            meshes=[m for _,m in sdf_visuals(roomdir/template['sdf_path'],bundle,{})]
            mesh=trimesh.util.concatenate(meshes)
            path=roomdir/'assets'/reg/'proxy_geometry.glb';path.parent.mkdir(parents=True)
            mesh.export(path)
            template['geometry_path']=str(path.relative_to(roomdir))
        template.update(object_id=reg,immutable=False,placement_info=None,support_surfaces=[],
                        transform={'translation':[0.0,0.0,0.0],'rotation_wxyz':[1.0,0.0,0.0,0.0]})
        template['metadata']={**template['metadata'],'catalog_registry_id':reg,
                              'source_object_id':oid,'authorship':'researcher_reconstruction'}
        entries.append({'template':template,'binding':{**binding,
            'identity_provenance':'new_instance_from_reconstructed_source_asset_not_historical_identity'},
            'limitations':['Original procedural proxies remain proxies; a derived GLB is not a real product.',
                           'Construction order and coordinates are researcher labels, not original architect actions.']})
        pose=target['objects'][oid]['transform'];w,x,y,z=pose['rotation_wxyz']
        yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        if oid in plan['target_pose_overrides']:
            px,py,yaw=plan['target_pose_overrides'][oid]
        else: px,py=pose['translation'][:2]
        command={'name':'add_furniture_to_scene_tool','arguments':{'asset_id':reg,'x':px,'y':py,'yaw':yaw}}
        commands.append({'role':old['name'],'planned_id':old['name']+'_0','source_object_id':oid,'command':command})
        mapping.append({'step':index,'source_id':oid,'registry_id':reg,'expected_native_name':old['name'],
                        'source_asset_binding':binding,'geometry_kind':'procedural_proxy' if old['geometry_path'] is None else 'cached_mesh_substitute'})
    assert sum(x['object_type']=='furniture' for x in room['objects'].values())==6
    assert all(x['immutable'] for x in room['objects'].values())
    manifest['project']={**manifest['project'],'project_id':plan['run_id'],
                         'provenance':'new_construction_task_from_training_source_reconstruction'}
    manifest['inputs']['asset_catalog']='catalog.json'
    task={'design_request':plan['design_request'],'original_request':None,'request_provenance':'adapted',
          'unknowns':original_task.get('unknowns',[])+plan['unknowns'],
          'request_authorship':'researcher_construction_task_not_original_resident_brief'}
    write(bundle/manifest['inputs']['native_state'],state)
    write(bundle/manifest['inputs']['runtime_metadata'],runtime)
    write(bundle/manifest['inputs']['task'],task)
    write(bundle/'catalog.json',{'schema':'scenefunc-native-catalog-v1','entries':entries})
    write(bundle/'construction_provenance.json',{'build_plan_sha256':sha256(args.plan),
        'source_family':plan['family_id'],'source_to_new_instance_mapping':mapping,
        'original_task_retained_for_provenance_only':original_task,'historical_initial_state':False,
        'installed_fixtures_count':6,'movable_furniture_initial_count':0,'training_authorized':False})
    manifest['files']={str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':sha256(p)}
        for p in sorted(bundle.rglob('*')) if p.is_file() and p!=bundle/'manifest.json'}
    write(bundle/'manifest.json',manifest);validate_bundle(bundle)
    write(args.out/'COMMANDS.json',commands);write(args.out/'IDENTITY_MAPPING.json',mapping)
    write(args.out/'RESULT.json',{'status':'construction_input_prepared','model_calls':0,'training_updates':0,
        'manifest_sha256':sha256(bundle/'manifest.json'),'catalog_entries':len(entries),
        'initial_installed_fixtures':6,'initial_movable_furniture':0,'planned_additions':len(commands),
        'source_to_plan_identity_preservation_claim':False,'plan_sha256':sha256(args.plan)})
    print('Prepared training-source construction: 6 fixed fixtures, 9 addable objects',flush=True)


if __name__=='__main__':main()
