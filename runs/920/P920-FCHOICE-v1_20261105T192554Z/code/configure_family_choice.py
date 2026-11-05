"""Make one portable catalogue and lock two independent native candidate runs."""
import argparse
import copy
import json
from pathlib import Path
import trimesh
from inspect_native_mesh_contacts import sdf_visuals
from native_project import sha256, validate_bundle


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]; public = repo/'runs/920'/a.run
    runtime = repo.parent/'runtime'/a.run; bundle = runtime/'reference/bundle'
    manifest,house,metadata,_ = validate_bundle(bundle)
    rid = manifest['selected_room_id']; state = house['rooms'][rid]; room = bundle/('room_'+rid)
    entries = []
    for oid,obj in state['objects'].items():
        if obj['object_type'] != 'furniture': continue
        template = copy.deepcopy(obj); reg = 'family_catalog_'+obj['name']
        if template['geometry_path'] is None:
            mesh = trimesh.util.concatenate([m for _,m in sdf_visuals(room/obj['sdf_path'],bundle,{})])
            path = room/'assets'/reg/'proxy.glb'; path.parent.mkdir(parents=True); mesh.export(path)
            template['geometry_path'] = str(path.relative_to(room))
        template.update(object_id=reg,transform={'translation':[0,0,0],'rotation_wxyz':[1,0,0,0]},
            support_surfaces=[],placement_info=None)
        template['metadata'] = {**template['metadata'],'catalog_registry_id':reg}
        entries.append(dict(template=template,binding=metadata['rooms'][rid]['asset_bindings'][oid],
            limitations=['Replacement templates do not authorise removing the declared existing instances.',
                         'Proxy furniture does not establish capacity or human use.']))
    assert len(entries) == 8
    write(bundle/'catalog.json',dict(schema='scenefunc-native-catalog-v1',entries=entries))
    manifest['inputs']['asset_catalog'] = 'catalog.json'
    manifest['files'] = {str(p.relative_to(bundle)):dict(bytes=p.stat().st_size,sha256=sha256(p))
        for p in sorted(bundle.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write(bundle/'manifest.json',manifest); validate_bundle(bundle)
    choice = json.loads((public/'CHOICE_PLAN.json').read_text())
    common = json.loads((repo/'runs/920/P920-PAPER-v2_20261105T182424Z/EXECUTION_PLAN.json').read_text())
    for label,pose in choice['candidates'].items():
        out = runtime/label; out.mkdir()
        measurement = copy.deepcopy(choice['measurement'])
        measurement['approach_points_xy_m']['storage_front'] = choice['storage_approach_by_candidate'][label]
        plan = dict(run_id=a.run,candidate=label,bundle=str(bundle),manifest_sha256=sha256(bundle/'manifest.json'),
            designer=common['designer'],scenesmith=common['scenesmith'],
            choice_plan_sha256=sha256(public/'CHOICE_PLAN.json'),
            code_sha256={name:sha256(Path(__file__).parent/name) for name in [
                'construct_and_switch_room.py','catalog_project_session.py','native_project.py',
                'native_checkpoint.py','measure_room_access.py','preflight_sourced_trajectory.py']},
            commands=[dict(role='retained_storage_placement',command=dict(name='move_furniture_tool',
                arguments=dict(object_id='recon_tall_storage_0',**pose)))],
            preflight_result=str(out/'preflight/RESULT.json'),reference_measurement=measurement,
            budget=dict(tool_calls=1,observations=2,native_seconds=360,model_calls=0,training_updates=0))
        write(public/(label+'_PLAN.json'),plan)
    write(public/'INPUT_LOCK.json',dict(manifest_sha256=sha256(bundle/'manifest.json'),bundle=str(bundle),
        source_state_sha256=sha256(bundle/'house_state.json'),furniture_ids=[k for k,v in state['objects'].items() if v['object_type']=='furniture'],
        script_sha256=sha256(Path(__file__))))
    print('Prepared two independent candidates from one eight-object initial state; no native edits.')


if __name__ == '__main__': main()
