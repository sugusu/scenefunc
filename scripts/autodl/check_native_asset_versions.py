"""Actual scene/registry helper mutation and persisted-version recovery, no model."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from native_trial_evidence import restore_solid_evidence,relocate_restored_paths
from native_asset_version_store import NativeAssetVersionStore


def differences(first,second,path=''):
    if isinstance(first,dict) and isinstance(second,dict):
        return [item for k in sorted(set(first)|set(second)) for item in differences(first.get(k),second.get(k),path+'/'+str(k))]
    if isinstance(first,list) and isinstance(second,list) and len(first)==len(second):
        return [item for i,(a,b) in enumerate(zip(first,second)) for item in differences(a,b,path+'/'+str(i))]
    if first==second:return []
    return [dict(path=path,before=first,after=second)]


def main(repo,stage,out_name):
    out=stage/out_name;out.mkdir(parents=True,exist_ok=False)
    prior=repo/'runs/autodl/20261110_complete_native_runtime';overlay=prior/'runtime'
    native=Path(json.loads((prior/'runtime_overlay_manifest.json').read_text())['native_root'])
    for path in (overlay,overlay/'src'):sys.path.insert(0,str(path))
    os.environ.update(SCENESMITH_ROOT=str(native),PYTHONDONTWRITEBYTECODE='1',OPENAI_AGENTS_DISABLE_TRACING='1')
    blocked=[]
    def guard(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen'):
            blocked.append(event);raise RuntimeError('No models, workers or network in version storage probe')
    sys.addaudithook(guard)
    tick=time.monotonic();result=dict(status='started',model_forwards=0,training_updates=0,paid_API_calls=0,
                                    actual_native_helper_calls=0,new_room_success_labels=0)
    try:
        importlib.import_module('src.runtime')
        from scenesmith.agent_utils.room import RoomScene
        from scenesmith.agent_utils.asset_registry import AssetRegistry
        from scenesmith.agent_utils.rescale_helpers import rescale_object_common
        with tempfile.TemporaryDirectory(prefix='native_asset_versions_',dir='/dev/shm') as scratch:
            root=Path(scratch)/'restored';proof=restore_solid_evidence(prior/'evidence_solid_v2/receipt.json',root)
            relocate_restored_paths(root,proof['original_root'])
            scene=RoomScene(room_geometry=None,scene_dir=root/'scene/room_bedroom')
            scene.restore_from_state_dict(json.loads((root/'last_actual_room_state.json').read_text()))
            scene.restore_from_state_dict(scene.to_state_dict())
            reg_path=scene.scene_dir/'generated_assets/furniture/asset_registry.json'
            registry=AssetRegistry(auto_save_path=reg_path);registry.load_from_file(reg_path)
            groups={}
            for obj in scene.objects.values():
                if obj.object_type.value=='furniture' and obj.sdf_path:groups.setdefault(obj.sdf_path,[]).append(str(obj.object_id))
            sdf,ids=next((p,x) for p,x in sorted(groups.items(),key=lambda x:str(x[0])) if len(x)>1)
            selected=ids[0];store=NativeAssetVersionStore(root,root/'asset_versions')
            def observe():
                return dict(logical_scale=scene.get_object(selected).scale_factor,
                            SDF_scales=sorted({x.text for x in ET.parse(sdf).getroot().findall('.//geometry/mesh/scale')}),
                            SDF_sha256=hashlib.sha256(sdf.read_bytes()).hexdigest())
            initial=observe();v0=store.capture(scene,registry,label='original_parent')
            replies=[]
            for index in range(2):
                response=rescale_object_common(scene=scene,object_id=selected,scale_factor=.9,
                                               object_type_name='furniture',asset_registry=registry)
                result['actual_native_helper_calls']+=1;replies.append(response)
                # Native helper returns structured output; inspect rather than
                # interpreting an exception-free invocation as success.
                success=response.get('success') if isinstance(response,dict) else getattr(response,'success',None)
                assert success is True,repr(response)
                if index==0:
                    middle=observe();v1=store.capture(scene,registry,label='after_first_actual_rescale')
                else:
                    final=observe();v2=store.capture(scene,registry,label='after_second_actual_rescale')
            roundtrips=[]
            for version,expected_observation in [(v1,middle),(v0,initial)]:
                record=store.restore(version,scene,registry)
                assert observe()==expected_observation
                for item in record['files']:
                    assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256']
                registry.save_to_file(root/'registry_roundtrip_check.json')
                registry_agrees=json.loads((root/'registry_roundtrip_check.json').read_text())==json.loads((root/record['registry_path']).read_text())
                scene_actual=scene.to_state_dict();exact_scene=scene_actual==record['scene_state']
                reg_actual=json.loads((root/'registry_roundtrip_check.json').read_text())
                reg_stored=json.loads((root/record['registry_path']).read_text())
                roundtrips.append(dict(version=version,file_bytes_exact=True,
                    scene_roundtrip_exact=exact_scene,registry_roundtrip_exact=registry_agrees,
                    scene_differences=differences(record['scene_state'],scene_actual),
                    registry_differences=differences(reg_stored,reg_actual)))
            canonical_parent=scene.to_state_dict()
            store.restore(v0,scene,registry)
            repeated_parent_equal=canonical_parent==scene.to_state_dict()
            versions={p.name:json.loads(p.read_text()) for p in (store.store/'versions').glob('*.json')}
            # Persist small mutable file versions and records, not duplicate meshes.
            import shutil
            shutil.copytree(store.store,out/'stored_versions')
            result.update(status='file_versions_recovered_native_roundtrip_characterized',selected_object=selected,
                shared_asset_instances=ids,version_ids=[v0,v1,v2],initial=initial,middle=middle,after_second=final,
                recovered_intermediate_file_bytes_exact=True,recovered_parent_file_bytes_exact=True,
                selected_asset_logical_scale_recovered_exact=True,
                native_scene_registry_roundtrip_all_exact=all(x['scene_roundtrip_exact'] and x['registry_roundtrip_exact'] for x in roundtrips),
                repeated_same_serialized_parent_load_equal=repeated_parent_equal,
                restored_version_checks=roundtrips,
                stored_versions=len(versions),stored_file_blobs=len(list((store.store/'blobs').iterdir())),
                source_archive_SHA256=proof['archive_sha256'],
                scope='Original helper and actual registry/SDF/SceneJSON only, no SDK dispatcher, workers, rendering caches or RNG restoration qualification.')
    except BaseException as exc:
        import traceback
        result.update(status='failed_no_automatic_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    result.update(seconds=time.monotonic()-tick,blocked_operations=blocked)
    (out/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='traceback'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--out-name',default='asset_version_store_v1')
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve(),a.out_name)
