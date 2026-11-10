"""Normal native session + original CLIP service + furniture + real five views.

Zero LLM/CLIP-query forwards. Replay the actual model-generated floor, without
teacher coordinates or new room selection. All heavy scratch lives in RAM;
persist observation PNGs, exact state and lifecycle proof on the data disk.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')


def main(stage,attempt):
    stage=stage.resolve();out=stage/('services_actual_'+attempt);out.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    previous=repo/'runs/autodl/20261110_remote_native_controller'
    overlay=json.loads((stage/'runtime_overlay_manifest.json').read_text())
    root=stage/'runtime';native=Path(overlay['native_root'])
    assert shutil.disk_usage(stage).free>5*1024**3+8*1024**2
    for p in (root,root/'src',repo/'scripts/autodl'):sys.path.insert(0,str(p))
    os.environ.update(SCENESMITH_ROOT=str(native),OPENAI_AGENTS_DISABLE_TRACING='1',
        PYTHONDONTWRITEBYTECODE='1',SCENE_RENDER_THREADS='2',SCENE_RENDER_SAMPLES='8',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NO_PROXY='127.0.0.1,localhost,::1')
    inherited=os.environ.get('PYTHONPATH','')
    os.environ['PYTHONPATH']=str(root)+':'+str(repo/'scripts/autodl')+':'+inherited
    original_popen=subprocess.Popen;owned=[];blocked=[];allowed_ports={7001}
    allowed_scripts={
        (native/'scenesmith/agent_utils/blender/standalone_server.py').resolve(),
        (native/'scenesmith/agent_utils/convex_decomposition_server/standalone_server.py').resolve(),
        (root/'scripts/export_scene_worker.py').resolve(),
        (root/'tools/render_five_views.py').resolve()}
    class Popen(original_popen):
        def __init__(self,*args,**kwargs):
            kwargs['start_new_session']=True
            super().__init__(*args,**kwargs);owned.append(self)
            cmd=args[0] if args else kwargs.get('args',[])
            if '--port' in cmd:allowed_ports.add(int(cmd[cmd.index('--port')+1]))
    subprocess.Popen=Popen
    def audit(event,args):
        if event=='socket.connect':
            addr=args[1];ok=isinstance(addr,tuple) and addr[0] in ('127.0.0.1','::1') and addr[1] in allowed_ports
        elif event=='socket.getaddrinfo':ok=args[0] in ('127.0.0.1','localhost','::1')
        elif event=='subprocess.Popen':
            cmd=args[1];ok=isinstance(cmd,list) and len(cmd)>1 and Path(cmd[1]).resolve() in allowed_scripts
        else:return
        if not ok:blocked.append(dict(event=event,argument=str(args[1])));raise RuntimeError('Unadmitted external service/process')
    sys.addaudithook(audit)
    def deadline(signum,frame):raise TimeoutError('Single service/observation qualification 360-second cap')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(360)
    started=time.monotonic();session=None;env=None;client=None;cwd=Path.cwd()
    result=dict(status='started',LLM_forwards=0,CLIP_query_forwards=0,paid_API_calls=0,
        training_steps=0,new_complete_scenes=0,full_scene_success=None)
    try:
        native_module=importlib.import_module('src.agent_rollout_decoupled')
        from scenesmith.agent_utils.asset_manager import AssetManager
        from scenesmith.agent_utils.house import HouseLayout
        from scenesmith.agent_utils.vlm_service import VLMService
        from remote_native_control import RemoteNativeClient,bind_remote_native_runtime
        from retrieval_image_binding import bind_retrieval_without_images
        from owned_native_retrieval import owned_hssd_backend,bounded_native_collision_threads
        inventory=repo/'runs/autodl/20261110_sdk_asset_qualification/native_data_v2/summary.json'
        weights=Path('/dev/shm/original_dfn_clip_20261110_2m9t72h2/open_clip_pytorch_model.bin')
        config=Path('/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/model_cache/open_clip_config.json')
        client=RemoteNativeClient(0,checkpoint='/root/autodl-tmp/agent_method_preparation_20261027/work/critic_public_runtime_20261104/checkpoint')
        with tempfile.TemporaryDirectory(prefix='native_whole_services_',dir='/dev/shm') as scratch:
            base=Path(scratch);bundle=base/'bundle';restored=bundle/'actual_floor_geometry_v1'
            with zipfile.ZipFile(previous/'native_floor_geometry.zip') as z:
                archive=json.loads(z.read('paths.json'))
                for row in archive['files']:
                    rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
                    data=z.read('blobs/'+row['sha256']);assert hashlib.sha256(data).hexdigest()==row['sha256']
                    target=restored/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            shutil.copytree(previous/'runtime/native_source/materials/Wood094_1K-JPG',
                bundle/'runtime/native_source/materials/Wood094_1K-JPG')
            original_layout=json.loads((previous/'actual_floor_geometry_v1/layout_with_geometry.json').read_text())
            with ExitStack() as scope:
                binding=scope.enter_context(bind_remote_native_runtime(VLMService,client))
                scope.enter_context(bind_retrieval_without_images(AssetManager,receipt=out/'image_binding.json'))
                scope.enter_context(bounded_native_collision_threads(receipt=out/'collision_thread_binding.json',max_threads=2))
                retrieval=scope.enter_context(owned_hssd_backend(native_module,receipt=out/'hssd_service.json',
                    weights=weights,config=config,inventory=inventory))
                session=native_module.RolloutSession(base/'normal_session','scene_designer_local',original_layout['house_prompt'])
                os.chdir(native);session.start()
                session.fp_agent.layout=HouseLayout.from_dict(original_layout,house_dir=restored/'scene')
                assert session.handover_after_floor_plan('bedroom')
                assert session.room_summary and '4.8m wide' in session.room_summary
                logger=native_module.ConsoleLogger(output_dir=session.scene.scene_dir)
                env=session.make_roomscene_env('furniture',None,logger)
                before=json.dumps(session.scene.to_state_dict(),sort_keys=True)
                obs=env.observation()
                assert json.dumps(session.scene.to_state_dict(),sort_keys=True)==before
                assert len(obs['image_paths'])==5
                (out/'views').mkdir()
                images=[]
                from PIL import Image
                for original in obs['image_paths']:
                    source=Path(original);target=out/'views'/source.name;shutil.copyfile(source,target)
                    meta=Path(str(source)+'.meta.json')
                    if meta.exists():shutil.copyfile(meta,Path(str(target)+'.meta.json'))
                    with Image.open(target) as image:size=list(image.size)
                    images.append(dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,size=size))
                dump(out/'actual_room_state.json',session.scene.to_state_dict())
                dump(out/'native_observation_text.json',dict(stage=obs['stage'],system=obs['system'],
                    room_prompt=obs['room_prompt'],memory_str=obs['memory_str'],original_image_paths=obs['image_paths']))
                result.update(status='normal_native_services_furniture_tools_and_five_views_returned',
                    actual_native_session_class=type(session).__module__+'.'+type(session).__name__,
                    actual_furniture_agent_class=type(env.agent).__module__+'.'+type(env.agent).__name__,
                    actual_furniture_tools=sorted(env.tool_map),actual_room_objects=len(session.scene.objects),
                    room_summary=session.room_summary,images=images,
                    model_service_instances_prepared=binding['vlm_instances'],
                    image_generation_service_absent=env.agent.asset_manager.image_generator is None,
                    LLM_requests_sent=False,CLIP_query_forwards=len(retrieval['encoder_requests']),
                    whole_runtime_ready=False,remaining='Actual policy/tool requests, mounted stages, surface workflow and final task evaluation')
                assert result['CLIP_query_forwards']==0
                env.cleanup();env=None;session.close();session=None
    except BaseException as exc:
        result.update(status='failed_no_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        os.chdir(cwd)
        if env is not None:env.cleanup()
        if session is not None:session.close()
        if client is not None:client.close()
        for child in reversed(owned):
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        subprocess.Popen=original_popen;signal.alarm(0)
        result.update(seconds=time.monotonic()-started,parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            owned_processes=[dict(pid=p.pid,exit_code=p.poll()) for p in owned],blocked_operations=blocked,
            heavy_scratch_in_RAM_removed=True,old_results_modified=False)
        dump(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('actual_furniture_tools','images')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--attempt',choices=('v1','v2'),required=True)
    args=p.parse_args();main(args.stage,args.attempt)
