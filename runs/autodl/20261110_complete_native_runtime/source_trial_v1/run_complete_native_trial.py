"""One actual bounded complete-room continuation with shared frozen local VLM.

No API, training, injected objects, threshold edits, sample change or hidden
restart. All stages are attempted with original native constructors/tools;
surface objects use the already documented common local workflow adaptation.
"""
import argparse
import asyncio
from contextlib import ExitStack
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')


async def main(stage):
    stage=stage.resolve();out=stage/'trial_actual_v1';out.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[2];protocol=json.loads((stage/'trial_protocol.json').read_text())
    for p,h in protocol['code_sha256'].items():assert sha(p)==h,p
    for r in protocol['weight_sha256_verified']:
        stat=Path(r['path']).stat();assert (stat.st_size,stat.st_mtime_ns,stat.st_ino)==(r['bytes'],r['mtime_ns'],r['inode'])
    for k in ('initial_floor_state','initial_floor_proposal','initial_floor_tool_results'):
        assert sha(protocol[k])==protocol[k+'_sha256']
    root=stage/'runtime';meta=json.loads((stage/'runtime_overlay_manifest.json').read_text());native=Path(meta['native_root'])
    for p in (root,root/'src',repo/'scripts/autodl'):sys.path.insert(0,str(p))
    os.environ.update(SCENESMITH_ROOT=str(native),OPENAI_AGENTS_DISABLE_TRACING='1',
        PYTHONDONTWRITEBYTECODE='1',SCENE_RENDER_THREADS='2',SCENE_RENDER_SAMPLES='8',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NO_PROXY='127.0.0.1,localhost,::1')
    os.environ['PYTHONPATH']=str(root)+':'+str(repo/'scripts/autodl')+':'+os.environ.get('PYTHONPATH','')
    owned=[];blocked=[];allowed_ports=set();original_popen=subprocess.Popen
    model_script=repo/'scripts/autodl/complete_native_model_host.py'
    glb_script=repo/'scripts/autodl/export_complete_trial_glb.py'
    scripts={model_script.resolve(),glb_script.resolve(),
        (native/'scenesmith/agent_utils/blender/standalone_server.py').resolve(),
        (native/'scenesmith/agent_utils/convex_decomposition_server/standalone_server.py').resolve(),
        (root/'scripts/export_scene_worker.py').resolve(),(root/'tools/render_five_views.py').resolve()}
    class Popen(original_popen):
        def __init__(self,*args,**kwargs):
            kwargs['start_new_session']=True;super().__init__(*args,**kwargs);owned.append(self)
            cmd=args[0] if args else kwargs.get('args',[])
            if '--port' in cmd:allowed_ports.add(int(cmd[cmd.index('--port')+1]))
    subprocess.Popen=Popen
    def audit(event,args):
        if event=='socket.connect':
            addr=args[1];ok=isinstance(addr,tuple) and addr[0] in ('127.0.0.1','::1') and addr[1] in allowed_ports
        elif event=='socket.getaddrinfo':ok=args[0] in ('127.0.0.1','localhost','::1')
        elif event=='subprocess.Popen':
            cmd=args[1];ok=isinstance(cmd,list) and len(cmd)>1 and Path(cmd[1]).resolve() in scripts
        else:return
        if not ok:blocked.append(dict(event=event,argument=str(args[1])));raise RuntimeError('External/unadmitted trial operation')
    sys.addaudithook(audit)
    def deadline(signum,frame):raise TimeoutError('Fixed complete-room parent deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(protocol['parent_seconds'])
    result=dict(status='started',stages=[],paid_API_calls=0,training_steps=0,
        complete_scene_success=None,learned_method_comparison=False,
        continuation_not_independent_from_empty=True)
    dump(out/'live_trial.json',dict(pid=os.getpid(),status='running',protocol_sha256=sha(stage/'trial_protocol.json')))
    started=time.monotonic();session=None;env=None;client=None;actor=None;actor_log=None;cwd=Path.cwd()
    scratch=tempfile.TemporaryDirectory(prefix='complete_native_trial_',dir='/dev/shm')
    work=Path(scratch.name);tool_count=0;last_agent=None
    try:
        native_module=importlib.import_module('src.agent_rollout_decoupled')
        from scenesmith.agent_utils.asset_manager import AssetManager
        from scenesmith.agent_utils.house import HouseLayout
        from scenesmith.agent_utils.vlm_service import VLMService
        from remote_native_control import RemoteNativeClient,bind_remote_native_runtime
        from owned_native_retrieval import owned_hssd_backend,bounded_native_collision_threads
        from retrieval_image_binding import bind_retrieval_without_images
        from remote_surface_workflow import RemoteSurfaceWorkflow
        from native_manipuland_runner import NativeStageReceipt,run_native_manipulands
        from task_grounded_completion import bind_original_task_to_selection
        class Client(RemoteNativeClient):
            def __init__(self,*a,**k):super().__init__(*a,**k);self.rows=[]
            def create_completion(self,*a,**k):
                row=dict(status='started');self.rows.append(row)
                try:
                    value=super().create_completion(*a,**k);row['status']='returned';return value
                except BaseException as exc:row.update(status='failed',error=str(exc));raise
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        client=Client(port,checkpoint=protocol['checkpoint'],timeout=200)
        child_env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        child_env.pop('PYTHONPATH',None)
        model_python='/root/autodl-tmp/agent_method_preparation_20261027/work/v100_runtime_20261104/runtime/bin/python'
        actor_log=(out/'actor.log').open('w')
        actor=subprocess.Popen([model_python,str(model_script),'--protocol',str(stage/'trial_protocol.json'),
            '--out',str(work/'actor'),'--port',str(port)],stdout=actor_log,stderr=subprocess.STDOUT,env=child_env)
        dump(out/'model_actor_process.json',dict(pid=actor.pid,port=port))
        tick=time.monotonic()
        while True:
            if actor.poll() is not None:raise RuntimeError('Actor stopped before readiness, no restart')
            try:
                if client.session.get(client.base+'/health',timeout=.5).json()['status']=='ready':break
            except Exception:pass
            if time.monotonic()-tick>40:raise TimeoutError('Actor readiness')
            time.sleep(.1)
        async def drive(current,cap):
            nonlocal tool_count
            turns=[]
            for n in range(cap):
                obs=current.observation()
                proposal=policy.generate(obs)
                row=dict(turn=n,observation=obs,proposal=proposal,status='proposed');turns.append(row)
                dump(current.steps_dir/'trace.json',turns)
                calls=proposal['tool_calls']
                if not calls:row['status']='native_policy_stop';dump(current.steps_dir/'trace.json',turns);return 'native_policy_stop'
                if tool_count+len(calls)>protocol['max_tool_calls']:raise RuntimeError('Cumulative tool budget exhausted')
                tool_count+=len(calls);row['results']=await current.step(calls);row['status']='native_tools_returned'
                if current.stage!='floor_plan':dump(current.steps_dir/f'state_{n:02d}.json',current.scene.to_state_dict())
                else:dump(current.steps_dir/f'layout_{n:02d}.json',session.fp_agent.layout.to_dict())
                dump(current.steps_dir/'trace.json',turns)
            return 'stage_turn_cap_reached'
        with ExitStack() as scope:
            scope.enter_context(bind_remote_native_runtime(VLMService,client))
            scope.enter_context(bind_retrieval_without_images(AssetManager,receipt=out/'image_binding.json'))
            scope.enter_context(bounded_native_collision_threads(receipt=out/'collision_threads.json',max_threads=2))
            retrieval=scope.enter_context(owned_hssd_backend(native_module,receipt=out/'hssd_service.json',
                weights='/dev/shm/original_dfn_clip_20261110_2m9t72h2/open_clip_pytorch_model.bin',
                config='/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/model_cache/open_clip_config.json',
                inventory=repo/'runs/autodl/20261110_sdk_asset_qualification/native_data_v2/summary.json',allowed_ports=allowed_ports))
            session=native_module.RolloutSession(work/'scene','scene_designer_local',protocol['task'])
            os.chdir(native);session.start()
            session.fp_agent.layout=HouseLayout.from_dict(json.loads(Path(protocol['initial_floor_state']).read_text()),house_dir=work/'scene')
            policy=native_module.ModelPolicy(protocol['checkpoint'],max_new_tokens=1024)
            env=session.make_floor_plan_env()
            prior=json.loads(Path(protocol['initial_floor_proposal']).read_text())['tool_calls'][0]
            prior_result=json.loads(Path(protocol['initial_floor_tool_results']).read_text())[0]['output']
            env._handler(env.mem,prior['name'],prior['arguments'],prior_result);env.step_idx=1
            for name,cap in protocol['stage_policy_turns'].items():
                tick=time.monotonic();entry=dict(stage=name,status='started');result['stages'].append(entry);dump(out/'progress.json',result)
                if name!='floor_plan':
                    logger=native_module.ConsoleLogger(output_dir=session.scene.scene_dir)
                    env=session.make_roomscene_env(name,None,logger)
                entry['stop_reason']=await drive(env,cap)
                env.finalize_stage();entry.update(status='native_stage_finalizer_returned',seconds=time.monotonic()-tick)
                if name=='floor_plan':
                    if not session.handover_after_floor_plan(None):raise RuntimeError('Actual floor handover failed')
                    session.fp_agent.cleanup()
                else:last_agent=env.agent;env.cleanup();env=None
                dump(out/'progress.json',result)
            # Common surface stage is executed; suggestions are not user truth.
            from scenesmith.manipuland_agents.stateful_manipuland_agent import StatefulManipulandAgent
            logger=native_module.ConsoleLogger(output_dir=session.scene.scene_dir)
            surface=native_module.BaseExperiment.build_manipuland_agent(cfg_dict=session.cfg_dict,
                compatible_agents={'stateful_manipuland_agent':StatefulManipulandAgent},logger=logger,render_gpu_id=None)
            tick=time.monotonic();entry=dict(stage='manipuland',status='started');result['stages'].append(entry)
            try:
                workflow=RemoteSurfaceWorkflow(client,out=work/'surface_workflow',requested_model=surface.cfg.openai.model,
                    original_user_request=protocol['task'],max_steps_per_parent=protocol['surface_steps_per_parent'],
                    max_tool_calls=protocol['surface_tool_cap'],max_seconds=protocol['surface_seconds'])
                receipt=NativeStageReceipt(work/'surface_stage.json');selection=[]
                with bind_original_task_to_selection(surface.scene_analyzer,protocol['task'],audit=selection):
                    session.scene=await run_native_manipulands(surface,session.scene,workflow=workflow,receipt=receipt)
                entry.update(status=receipt.data['status'],seconds=time.monotonic()-tick);last_agent=surface
                dump(work/'surface_selection_task_binding.json',selection)
            finally:surface.cleanup()
            final=work/'final';final.mkdir()
            dump(final/'actual_room_state.json',session.scene.to_state_dict())
            native_module._save_roomscene_blend(last_agent,final/'scene.blend')
            subprocess.run([sys.executable,str(glb_script),str(final/'scene.blend'),str(final/'scene.glb')],check=True,capture_output=True,text=True,timeout=90)
            import trimesh
            geometry=trimesh.load_scene(final/'scene.glb')
            result.update(status='all_native_stages_attempted_final_scene_exported',
                actual_final_objects=len(session.scene.objects),final_GLB_bytes=(final/'scene.glb').stat().st_size,
                cold_GLB_geometries=len(geometry.geometry),CLIP_query_forwards=len(retrieval['encoder_requests']),
                task_semantics_independently_verified=False,full_scene_success=None)
            session.close();session=None
    except BaseException as exc:
        result.update(status='failed_no_restart',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        if session is not None and session.fp_agent is not None:
            dump(work/'last_actual_floor_layout.json',session.fp_agent.layout.to_dict())
        if session is not None and session.scene is not None:
            dump(work/'last_actual_room_state.json',session.scene.to_state_dict())
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
        if actor_log:actor_log.close()
        signal.alarm(0);subprocess.Popen=original_popen
        ledger=work/'actor/ledger.json'
        if ledger.exists():result['model_ledger']=json.loads(ledger.read_text())
        result.update(seconds=time.monotonic()-started,native_tool_attempts=tool_count,
            owned_processes=[dict(pid=p.pid,exit_code=p.poll()) for p in owned],blocked_operations=blocked)
        dump(out/'result.json',result)
        # Retain all generated/request bytes losslessly once, no old-file edits.
        blobs={};files=[]
        for p in work.rglob('*'):
            if p.is_file():
                data=p.read_bytes();h=hashlib.sha256(data).hexdigest();blobs.setdefault(h,data)
                files.append(dict(path=str(p.relative_to(work)),sha256=h,bytes=len(data)))
        memory=io.BytesIO()
        with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            z.writestr('paths.json',json.dumps(dict(original_root=str(work),files=files)))
            for h,data in blobs.items():z.writestr('blobs/'+h,data)
        data=memory.getvalue()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            assert z.testzip() is None
            for r in files:assert hashlib.sha256(z.read('blobs/'+r['sha256'])).hexdigest()==r['sha256']
        target=out/'actual_scene_evidence.zip'
        if shutil.disk_usage(out).free-len(data)<5*1024**3+2*1024**2:
            target=Path('/dev/shm/complete_native_trial_verified_evidence.zip')
            if target.exists():raise FileExistsError('Refusing to overwrite retained RAM evidence')
        target.write_bytes(data)
        dump(out/'archive_receipt.json',dict(path=str(target),sha256=sha(target),bytes=len(data),files=len(files),
            persisted_on_data_disk=target.parent==out,lossless_verified=True,
            original_absolute_paths_not_relocated=True,companion_original_materials_still_required=True))
        scratch.cleanup()
        dump(out/'live_trial.json',dict(pid=os.getpid(),status='terminal',outcome=result['status']))
        dump(stage/'TRIAL_CLOSED.json',dict(attempt='trial_actual_v1',unused_allowance_carries=False))
    print(json.dumps({k:v for k,v in result.items() if k!='model_ledger'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    asyncio.run(main(p.parse_args().stage))
