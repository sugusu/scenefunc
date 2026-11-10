"""Zero-model native geometry/floor handover from the actual model's layout.

No coordinate repair, opening insertion, asset generation or policy retry.
Normal floor constructor and original finalizer/derive_room_scene are used.
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
import time
import traceback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')


def main(stage):
    stage=stage.resolve();out=stage/'actual_floor_geometry_v1';out.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    meta=json.loads((stage/'materialization.json').read_text())
    runtime=Path(meta['runtime_root']);native=runtime/'native_source'
    source=stage/'native_joint_actual_v2/layout_after.json'
    source_sha=sha(source)
    assert shutil.disk_usage(stage).free>5*1024**3+12*1024**2
    dump(out/'protocol.json',dict(source=str(source),source_sha256=source_sha,
        model_calls=0,API_calls=0,training_steps=0,maximum_seconds=120,
        source_dimensions_not_edited=True,geometry_only=True,full_scene_success=None))
    for p in (runtime,runtime/'src',repo/'scripts/autodl'):sys.path.insert(0,str(p))
    os.environ['SCENESMITH_ROOT']=str(native)
    os.environ['OPENAI_AGENTS_DISABLE_TRACING']='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    owned=[];blocked=[];original=subprocess.Popen
    class Popen(original):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);owned.append(self)
    subprocess.Popen=Popen
    blender=native/'scenesmith/agent_utils/blender/standalone_server.py'
    def audit(event,args):
        if event=='subprocess.Popen':
            cmd=args[1];valid=(not owned and isinstance(cmd,list) and len(cmd)>1 and Path(cmd[1]).resolve()==blender.resolve())
        elif event=='socket.connect':
            addr=args[1];valid=isinstance(addr,tuple) and addr[0] in ('127.0.0.1','::1') and 8000<=addr[1]<=8050
        elif event=='socket.getaddrinfo':valid=args[0] in ('127.0.0.1','localhost','::1')
        else:return
        if not valid:blocked.append(event);raise RuntimeError('Unadmitted external/model/process operation')
    sys.addaudithook(audit)
    def deadline(signum,frame):raise TimeoutError('Single native floor finalization exceeded120s')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(120)
    result=dict(status='started',model_calls=0,paid_API_calls=0,training_steps=0,
        new_complete_scenes=0,full_scene_success=None)
    started=time.monotonic();agent=None;client=None;cwd=Path.cwd()
    try:
        module=importlib.import_module('src.agent_rollout_decoupled')
        from scenesmith.agent_utils.house import HouseLayout
        from scenesmith.agent_utils.vlm_service import VLMService
        from scenesmith.agent_utils.drake_utils import create_drake_plant_and_scene_graph_from_scene
        from remote_native_control import RemoteNativeClient,bind_remote_native_runtime
        client=RemoteNativeClient(0,checkpoint='/root/autodl-tmp/agent_method_preparation_20261027/work/critic_public_runtime_20261104/checkpoint')
        data=json.loads(source.read_text())
        with bind_remote_native_runtime(VLMService,client):
            agent,tools,_,materials=module.build_floor_plan_agent_tools('scene_designer_local',data['house_prompt'],out/'scene')
            assert materials is None
            agent.layout=HouseLayout.from_dict(data,house_dir=out/'scene')
            assert len(agent.layout.placed_rooms)==1 and not agent.layout.room_geometries
            # Original native generator defaults are relative to repository CWD.
            # Run within the new source copy; never rewrite old material/source paths.
            os.chdir(native)
            env=module.ToolEnvironment(stage='floor_plan',room_prompt=data['house_prompt'],
                agent=agent,scene=agent.layout,tool_map=tools,steps_dir=out/'steps')
            env.finalize_stage()
            scene=module.derive_room_scene(agent,agent.layout.placed_rooms[0].room_id)
            snapshot={str(k):v.to_dict(scene_dir=scene.scene_dir) for k,v in scene.objects.items()}
            dump(out/'derived_architecture_objects.json',snapshot)
            directive=scene.to_drake_directive();(out/'derived_scene_directive.yaml').write_text(directive)
            plant,scene_graph=create_drake_plant_and_scene_graph_from_scene(scene)
            dump(out/'layout_with_geometry.json',agent.layout.to_dict(scene_dir=out/'scene'))
            result.update(status='original_floor_finalizer_and_room_handover_returned',
                native_room_geometry_count=len(agent.layout.room_geometries),
                actual_room_object_count=len(scene.objects),room_id=scene.room_id,
                native_object_types=[o['object_type'] for o in snapshot.values()],
                room_length_m=scene.room_geometry.length,room_width_m=scene.room_geometry.width,
                height_m=scene.room_geometry.wall_height,original_task=scene.text_description,
                drake_plant_finalized=plant.is_finalized(),drake_num_bodies=plant.num_bodies(),
                room_geometry_sdf=str(scene.room_geometry.sdf_path),
                original_policy_layout_unchanged=sha(source)==source_sha,
                no_doors_or_windows_invented=True,furniture_generated=0,
                source_dimensions_reused_without_repair=True,new_scene_rendered=False)
            assert result['original_policy_layout_unchanged']
            assert result['drake_plant_finalized'] and result['actual_room_object_count']==4
        assert not blocked
    except BaseException as exc:
        result.update(status='failed_no_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        os.chdir(cwd)
        if agent is not None:agent.cleanup()
        if client is not None:client.close()
        for child in owned:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=8)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        signal.alarm(0);subprocess.Popen=original
        result.update(seconds=time.monotonic()-started,blocked_operations=blocked,
            owned_processes=[dict(pid=p.pid,exit_code=p.poll()) for p in owned])
        dump(out/'result.json',result)
        files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p))
            for p in out.rglob('*') if p.is_file()]
        dump(out/'file_manifest.json',dict(files=files,total_bytes=sum(r['bytes'] for r in files)))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    main(p.parse_args().stage)
