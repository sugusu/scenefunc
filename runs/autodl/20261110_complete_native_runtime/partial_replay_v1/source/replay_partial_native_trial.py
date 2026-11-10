"""Cold-replay and inspect the exact partial scene; no policy or tool actions.

Restores all source bytes before explicit path-only relocation in a new RAM
copy. Actual Drake contacts, native five-view export, and a cold GLB read do
not certify task semantics or turn an interrupted rollout into a success.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    stage = a.stage.resolve(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    overlay = stage/'runtime'; native = Path(json.loads((stage/'runtime_overlay_manifest.json').read_text())['native_root'])
    receipt = json.loads((stage/'trial_actual_v2/archive_receipt.json').read_text())
    source = Path(receipt['path']); assert sha(source) == receipt['sha256']
    for path in (overlay, overlay/'src', repo/'scripts/autodl'):
        sys.path.insert(0, str(path))
    os.environ.update(SCENESMITH_ROOT=str(native), OPENAI_AGENTS_DISABLE_TRACING='1',
                      PYTHONDONTWRITEBYTECODE='1', SCENE_RENDER_THREADS='2', SCENE_RENDER_SAMPLES='8',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NO_PROXY='127.0.0.1,localhost,::1')
    os.environ['PYTHONPATH'] = str(overlay)+':'+str(repo/'scripts/autodl')+':'+os.environ.get('PYTHONPATH','')
    owned = []; ports = set(); blocked = []; original_popen = subprocess.Popen
    admitted = {(overlay/'scripts/export_scene_worker.py').resolve(),
                (overlay/'tools/render_five_views.py').resolve(),
                (repo/'scripts/autodl/export_complete_trial_glb.py').resolve()}
    class Popen(original_popen):
        def __init__(self, *args, **kwargs):
            kwargs['start_new_session'] = True
            super().__init__(*args, **kwargs); owned.append(self)
    subprocess.Popen = Popen
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            blocked.append(event); raise RuntimeError('Cold replay does not call any service')
        if event == 'subprocess.Popen':
            cmd = args[1]
            if not isinstance(cmd, list) or len(cmd)<2 or Path(cmd[1]).resolve() not in admitted:
                blocked.append(event); raise RuntimeError('Unadmitted replay child process')
    sys.addaudithook(audit)
    started = time.monotonic(); result = dict(status='started', LLM_forwards=0, CLIP_forwards=0,
          paid_API_calls=0, optimizer_updates=0, new_policy_actions=0, full_scene_success=None,
          original_trial_status='failed_no_restart')
    cwd = Path.cwd()
    try:
        import numpy as np
        from scenesmith.agent_utils.room import RoomScene
        from scenesmith.agent_utils.drake_utils import create_drake_plant_and_scene_graph_from_scene
        from scenesmith.agent_utils.physics_validation import _get_object_info_from_geometry_id
        from pydrake.all import DiagramBuilder
        with tempfile.TemporaryDirectory(prefix='native_partial_replay_', dir='/dev/shm') as scratch:
            root = Path(scratch)
            with zipfile.ZipFile(source) as z:
                paths = json.loads(z.read('paths.json'))
                for row in paths['files']:
                    rel = Path(row['path']); assert not rel.is_absolute() and '..' not in rel.parts
                    data = z.read('blobs/'+row['sha256']); assert hashlib.sha256(data).hexdigest()==row['sha256']
                    target = root/rel; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
            # The original archive remains immutable. Only exact path prefixes
            # in text artifacts in this newly restored copy are rewritten.
            relocation = []
            old = paths['original_root'].encode(); new = str(root).encode()
            text_suffixes = {'.json','.gltf','.sdf','.yaml','.yml','.xml','.mtl'}
            for path in root.rglob('*'):
                if not path.is_file() or path.suffix not in text_suffixes:
                    continue
                data = path.read_bytes()
                if old in data:
                    path.write_bytes(data.replace(old,new))
                    relocation.append(dict(path=str(path.relative_to(root)), before_sha256=hashlib.sha256(data).hexdigest(), after_sha256=sha(path)))
            dump(out/'path_relocation.json',dict(original_root=paths['original_root'],restored_root=str(root),
                                               files=relocation, original_bytes_verified=len(paths['files'])))
            state = json.loads((root/'last_actual_room_state.json').read_text())
            scene = RoomScene(room_geometry=None,scene_dir=root/'scene/room_bedroom')
            scene.restore_from_state_dict(state); scene.room_type='bedroom'
            actual = scene.to_state_dict()
            assert set(actual['objects']) == set(state['objects'])
            for key, before in state['objects'].items():
                after=actual['objects'][key]
                assert before['object_type']==after['object_type']
                for field in ('translation','rotation_wxyz'):
                    assert np.allclose(before['transform'][field],after['transform'][field],rtol=0,atol=1e-12)
                assert before['scale_factor']==after['scale_factor']
                for field in ('bbox_min','bbox_max'):
                    if before[field] is not None:assert np.allclose(before[field],after[field],rtol=0,atol=1e-12)
            dump(out/'actual_room_state.json',actual)
            builder=DiagramBuilder()
            plant,graph=create_drake_plant_and_scene_graph_from_scene(scene,builder=builder,weld_furniture=False,
                                                                     free_mounted_objects_for_collision=True)
            diagram=builder.Build(); context=diagram.CreateDefaultContext()
            query=graph.get_query_output_port().Eval(graph.GetMyContextFromRoot(context)); inspector=query.inspector()
            def info(gid):
                shape=inspector.GetShape(gid); pose=query.GetPoseInWorld(inspector.GetFrameId(gid)).multiply(inspector.GetPoseInFrame(gid))
                shape_info={'type':type(shape).__name__}
                for name in ('width','depth','height','radius','length','scale','filename'):
                    method=getattr(shape,name,None)
                    if callable(method):
                        try:shape_info[name]=str(method()) if name=='filename' else method()
                        except (TypeError,RuntimeError):pass
                return dict(id=gid.get_value(),name=inspector.GetName(gid),shape=shape_info,
                            X_WG=pose.GetAsMatrix4().tolist(),
                            native_object=_get_object_info_from_geometry_id(geometry_id=gid,scene=scene,query_object=query))
            contacts=[]
            for pair in query.ComputeSignedDistancePairwiseClosestPoints(max_distance=0.0):
                contacts.append(dict(a=info(pair.id_A),b=info(pair.id_B),signed_distance=pair.distance,
                                     p_ACa=pair.p_ACa.tolist(),p_BCb=pair.p_BCb.tolist()))
            dump(out/'actual_drake_contacts.json',contacts)
            result.update(objects=len(scene.objects),furniture=sum(x.object_type.value=='furniture' for x in scene.objects.values()),
                          raw_nonpositive_distance_pairs=len(contacts),native_drake_bodies=plant.num_bodies(),
                          original_ids_poses_scales_and_boxes_preserved=True)
            os.chdir(native)
            render=importlib.import_module('src.render_observation')
            views=render.render_scene_views(SimpleNamespace(scene=scene),scene,root/'cold_views',0)
            (out/'views').mkdir()
            for path in (root/'cold_views').glob('*'):
                if path.suffix=='.png' or path.name.endswith('.meta.json'):
                    shutil.copyfile(path,out/'views'/path.name)
            blend=root/'cold_views/step_000.blend'; glb=root/'cold_views/partial_scene.glb'
            proc=subprocess.run([sys.executable,str(repo/'scripts/autodl/export_complete_trial_glb.py'),str(blend),str(glb)],
                                capture_output=True,text=True,timeout=120)
            (out/'glb_export.log').write_text(proc.stdout+'\n'+proc.stderr)
            assert proc.returncode==0 and glb.is_file()
            import trimesh
            loaded=trimesh.load_scene(glb)
            if shutil.disk_usage(out).free-glb.stat().st_size < 5*1024**3+4*1024**2:
                raise RuntimeError('Partial GLB persistence would breach reserve; no result replaced')
            shutil.copyfile(glb,out/'partial_scene.glb')
            result.update(status='partial_scene_cold_exported_and_drake_inspected',views=len(views),
                          partial_GLB_bytes=glb.stat().st_size,partial_GLB_sha256=sha(glb),cold_GLB_geometries=len(loaded.geometry),
                          no_scene_action_or_geometry_fix_applied=True)
    except BaseException as exc:
        import traceback
        result.update(status='cold_replay_failed_no_scene_edits',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        os.chdir(cwd);subprocess.Popen=original_popen
        for child in reversed(owned):
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            if child.poll() is None:
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        result.update(seconds=time.monotonic()-started,blocked_operations=blocked,
                      owned_processes=[dict(pid=x.pid,exit_code=x.poll()) for x in owned],
                      original_archive_unchanged=sha(source)==receipt['sha256'])
        dump(out/'result.json',result); print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
