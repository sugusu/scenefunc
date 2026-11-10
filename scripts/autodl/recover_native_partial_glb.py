"""Export one exact archived partial-scene blend and verify even cleanup failures.

No model calls, changed layout or repeated candidates. A nonzero child exit
remains a process failure even if its GLB is independently readable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time

from native_trial_evidence import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    stage=a.stage.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((stage/'evidence_solid_v2/receipt.json').read_text())
    archive=Path(receipt['path']);archive=archive if archive.is_absolute() else Path(__file__).resolve().parents[2]/archive
    assert sha(archive)==receipt['sha256']
    started=time.monotonic();result=dict(status='started',LLM_forwards=0,API_calls=0,training_steps=0,
                                       complete_scene_success=None,new_scene_generation=False)
    try:
        with tempfile.TemporaryDirectory(prefix='native_partial_glb_',dir='/dev/shm') as scratch:
            root=Path(scratch);blob_hash=None;mapping=None
            with tarfile.open(archive,'r|xz') as tar:
                for item in tar:
                    data=tar.extractfile(item).read()
                    if item.name=='paths.json':
                        mapping={r['path']:r for r in json.loads(data)['files']}
                        key='scene/room_bedroom/rollout_wall/step_001.blend'
                        blob_hash=mapping[key]['sha256']
                        # No scene placement occurred after this observation.
                        assert mapping['last_actual_room_state.json']['sha256']==mapping['scene/room_bedroom/rollout_wall/state_00.json']['sha256']
                    if blob_hash and item.name=='blobs/'+blob_hash:
                        assert hashlib.sha256(data).hexdigest()==blob_hash
                        blend=root/'exact_native_partial.blend';blend.write_bytes(data);break
            assert blob_hash is not None and blend.is_file()
            glb=root/'partial_scene.glb';script=Path(__file__).with_name('export_complete_trial_glb.py')
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            proc=subprocess.run([sys.executable,str(script),str(blend),str(glb)],env=env,capture_output=True,text=True,timeout=90)
            (out/'exporter.log').write_text(proc.stdout+'\n'+proc.stderr)
            result.update(exporter_exit_code=proc.returncode,exporter_process_passed=proc.returncode==0,
                          source_blend_SHA256=blob_hash,source_observation='rollout_wall/step_001',
                          state_equivalence='last_actual_room_state == rollout_wall/state_00 byte hashes',
                          exporter_source_SHA256=sha(script))
            if not glb.is_file():raise RuntimeError('Exporter produced no GLB; no alternate layout/export retry')
            raw=glb.read_bytes();magic,version,size=struct.unpack_from('<4sII',raw)
            assert magic==b'glTF' and version==2 and size==len(raw)
            json_length,json_type=struct.unpack_from('<I4s',raw,12);assert json_type==b'JSON'
            document=json.loads(raw[20:20+json_length].rstrip(b' \x00'))
            assert all('uri' not in x for x in document.get('buffers',[]))
            assert all('uri' not in x or x['uri'].startswith('data:') for x in document.get('images',[]))
            import trimesh
            loaded=trimesh.load_scene(glb)
            assert loaded.geometry and all(len(g.vertices)>0 for g in loaded.geometry.values())
            if shutil.disk_usage(out).free-len(raw)<5*1024**3+4*1024**2:raise RuntimeError('Reserve insufficient for GLB persistence')
            shutil.copyfile(glb,out/'partial_scene.glb');assert sha(out/'partial_scene.glb')==sha(glb)
            result.update(status='partial_artifact_cold_verified_exporter_exit_separately_reported',
                          GLB_bytes=len(raw),GLB_sha256=sha(glb),cold_geometry_count=len(loaded.geometry),
                          GLB_nodes=len(document.get('nodes',[])),self_contained_buffer_and_image_uris=True,
                          exported_names_not_independent_object_semantics=True,
                          original_model_trial_remains_incomplete=True)
    except BaseException as exc:
        import traceback
        result.update(status='artifact_recovery_failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    result.update(seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
