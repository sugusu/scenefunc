"""Characterize one saved parent using owned restored files, without workers."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

from native_trial_evidence import restore_solid_evidence,relocate_restored_paths
from check_native_asset_versions import differences


def main(repo,stage):
    out=stage/'move_parent_metadata_v2';out.mkdir(exist_ok=False)
    prior=repo/'runs/autodl/20261110_complete_native_runtime'
    for p in (prior/'runtime',prior/'runtime/src'):sys.path.insert(0,str(p))
    native=Path(json.loads((prior/'runtime_overlay_manifest.json').read_text())['native_root'])
    os.environ['SCENESMITH_ROOT']=str(native)
    blocked=[]
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen'):
            blocked.append(event);raise RuntimeError('No workers, tools or network in metadata diagnosis')
    sys.addaudithook(audit)
    result=dict(status='started',SDK_calls=0,VLM_forwards=0,training_updates=0)
    try:
        importlib.import_module('src.runtime')
        from scenesmith.agent_utils.room import RoomScene
        with tempfile.TemporaryDirectory(prefix='native_parent_metadata_',dir='/dev/shm') as scratch:
            root=Path(scratch)/'restored'
            proof=restore_solid_evidence(prior/'evidence_solid_v2/receipt.json',root)
            relocate_restored_paths(root,proof['original_root'])
            parent=json.loads((stage/'move_SDK_v1/frozen_parent.json').read_text())['scene']
            oldroot=str(Path(parent['room_geometry']['sdf_path']).parents[2])
            def relocate(x):
                if isinstance(x,str):return x.replace(oldroot,str(root))
                if isinstance(x,list):return [relocate(v) for v in x]
                if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
                return x
            before=relocate(parent)
            scene=RoomScene(room_geometry=None,scene_dir=root/'scene/room_bedroom');rows=[]
            for step in range(3):
                scene.restore_from_state_dict(before);after=scene.to_state_dict()
                rows.append(dict(step=step+1,differences=differences(before,after),before=before,after=after))
                before=after
            result.update(status='metadata_only_roundtrip_compared',roundtrips=rows,
                source_archive_sha256=proof['archive_sha256'],path_relocation_only=True,
                parent_old_owned_root=oldroot,original_parent_modified=False)
    except BaseException as exc:
        import traceback
        result.update(status='failed_no_automatic_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    result['blocked_operations']=blocked
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='roundtrips'}))
    if 'roundtrips' in result:print(json.dumps([dict(step=r['step'],differences=r['differences']) for r in result['roundtrips']]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve())
