"""Real source helper: JSON-only vs byte-aware rollback on a new scene copy.

No model, optimizer, rendering, coordinate search or old-file mutation. Uses
the public helper's explicit registry=None mode; it does not qualify registry,
RNG, process-cache or whole SDK rollback.
"""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from native_trial_evidence import restore_solid_evidence,relocate_restored_paths,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    stage=a.stage.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    overlay=stage/'runtime';native=Path(json.loads((stage/'runtime_overlay_manifest.json').read_text())['native_root'])
    for d in (overlay,overlay/'src'):sys.path.insert(0,str(d))
    os.environ.update(SCENESMITH_ROOT=str(native),PYTHONDONTWRITEBYTECODE='1',OPENAI_AGENTS_DISABLE_TRACING='1')
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen'):
            raise RuntimeError('No service or model processes in rollback diagnostic')
    sys.addaudithook(audit)
    started=time.monotonic();result=dict(status='started',model_calls=0,optimizer_updates=0,
                                        original_scene_modified=False,method_advantage_claimed=False)
    try:
        importlib.import_module('src.runtime')
        from scenesmith.agent_utils.room import RoomScene
        from scenesmith.agent_utils.rescale_helpers import rescale_object_common
        with tempfile.TemporaryDirectory(prefix='native_asset_rollback_',dir='/dev/shm') as scratch:
            root=Path(scratch)/'restored';proof=restore_solid_evidence(stage/'evidence_solid_v2/receipt.json',root)
            relocate_restored_paths(root,proof['original_root'])
            scene=RoomScene(room_geometry=None,scene_dir=root/'scene/room_bedroom')
            scene.restore_from_state_dict(json.loads((root/'last_actual_room_state.json').read_text()))
            snapshot=copy.deepcopy(scene.to_state_dict())
            groups={}
            for obj in scene.objects.values():
                if obj.object_type.value=='furniture' and obj.sdf_path:groups.setdefault(obj.sdf_path,[]).append(str(obj.object_id))
            shared=next((p,ids) for p,ids in sorted(groups.items(),key=lambda x:str(x[0])) if len(ids)>1)
            sdf,ids=shared;chosen=ids[0];original=sdf.read_bytes();initial_sha=sha(sdf)
            def measure():
                obj=scene.get_object(chosen)
                scales=[e.text for e in ET.parse(sdf).getroot().findall('.//geometry/mesh/scale')]
                return dict(scene_scale_factor=obj.scale_factor,mesh_scale_values=sorted(set(scales)),SDF_SHA256=sha(sdf),
                            serialized_scene_equals_parent=scene.to_state_dict()==snapshot)
            initial=measure()
            first=rescale_object_common(scene,chosen,.9,'furniture',asset_registry=None);assert first.success
            after_A=measure();scene.restore_from_state_dict(copy.deepcopy(snapshot));after_JSON_reset=measure()
            second=rescale_object_common(scene,chosen,.9,'furniture',asset_registry=None);assert second.success
            B_after_A=measure()
            # Ordinary file-aware reset, only of this explicitly mutated SDF.
            sdf.write_bytes(original);scene.restore_from_state_dict(copy.deepcopy(snapshot))
            independent=rescale_object_common(scene,chosen,.9,'furniture',asset_registry=None);assert independent.success
            B_from_clean=measure()
            assert after_JSON_reset['serialized_scene_equals_parent'] and after_JSON_reset['SDF_SHA256']!=initial_sha
            assert B_after_A['scene_scale_factor']==B_from_clean['scene_scale_factor']
            assert B_after_A['mesh_scale_values']!=B_from_clean['mesh_scale_values']
            sdf.write_bytes(original);scene.restore_from_state_dict(copy.deepcopy(snapshot));restored=measure()
            assert restored==initial
            source=Path(importlib.import_module('scenesmith.agent_utils.rescale_helpers').__file__)
            result.update(status='same_JSON_different_asset_state_reproduced',selected_object=chosen,shared_object_ids=ids,
                          explicit_registry_mode=None,scale_action=.9,initial=initial,after_first_action=after_A,
                          JSON_only_reset=after_JSON_reset,second_action_after_JSON_only=B_after_A,
                          same_action_after_byte_reset=B_from_clean,finally_restored=restored,
                          native_helper_source=str(source),native_helper_SHA256=sha(source),
                          copied_asset_bytes_restored=True,whole_SDK_registry_and_cache_rollback_not_tested=True,
                          original_scene_archive_SHA256=proof['archive_sha256'])
    except BaseException as exc:
        import traceback
        result.update(status='diagnostic_failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-started
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
