"""Zero-model normal native SDK branch comparison using the actual saved room."""
import argparse
import asyncio
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET

from native_trial_evidence import restore_solid_evidence,relocate_restored_paths,sha
from native_edit_snapshot import NativeEditSnapshot


def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')


async def main(stage,out):
    stage=stage.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2];prior=repo/'runs/autodl/20261110_complete_native_runtime'
    overlay=prior/'runtime';native=Path(json.loads((prior/'runtime_overlay_manifest.json').read_text())['native_root'])
    for p in (overlay,overlay/'src',repo/'scripts/autodl'):sys.path.insert(0,str(p))
    os.environ.update(SCENESMITH_ROOT=str(native),OPENAI_AGENTS_DISABLE_TRACING='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    original=subprocess.Popen;owned=[];ports=set();blocked=[];client=None;agent=None;env=None;cwd=Path.cwd()
    admitted={(native/'scenesmith/agent_utils/blender/standalone_server.py').resolve(),
              (native/'scenesmith/agent_utils/convex_decomposition_server/standalone_server.py').resolve()}
    class Popen(original):
        def __init__(self,*args,**kwargs):
            kwargs['start_new_session']=True;super().__init__(*args,**kwargs);owned.append(self)
            cmd=args[0]
            if '--port' in cmd:ports.add(int(cmd[cmd.index('--port')+1]))
    subprocess.Popen=Popen
    def audit(event,args):
        if event=='subprocess.Popen':
            cmd=args[1];ok=isinstance(cmd,list) and len(cmd)>1 and Path(cmd[1]).resolve() in admitted
        elif event=='socket.connect':
            addr=args[1];ok=isinstance(addr,tuple) and addr[0] in ('127.0.0.1','::1') and addr[1] in ports
        elif event=='socket.getaddrinfo':ok=args[0] in ('localhost','127.0.0.1','::1')
        else:return
        if not ok:blocked.append(event);raise RuntimeError('Unadmitted network/model/subprocess')
    sys.addaudithook(audit)
    def deadline(sig,frame):raise TimeoutError('Fixed240s native SDK state probe')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(240)
    tick=time.monotonic();results=[];calls=[]
    result=dict(status='started',model_forwards=0,paid_API_calls=0,training_updates=0,new_generated_rooms=0,
                whole_scene_success=None,not_a_learned_method=True)
    dump(out/'SDK_PROTOCOL.json',dict(max_native_actions=7,scale_A=.9,scale_B=.95,actions_not_selected_by_quality=True,
         no_asset_generation=True,constructor='normal BaseExperiment.build_furniture_agent',
         operation='original rescale_furniture_tool via original ToolEnvironment.step',
         source_pin='e888297a8b0a2f3acbb8634edf9def59cceab836',
         code_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('native_edit_snapshot.py')]}))
    try:
        module=importlib.import_module('src.agent_rollout_decoupled')
        from scenesmith.agent_utils.room import RoomScene
        from scenesmith.agent_utils.asset_manager import AssetManager
        from scenesmith.agent_utils.vlm_service import VLMService
        from remote_native_control import RemoteNativeClient,bind_remote_native_runtime
        from retrieval_image_binding import bind_retrieval_without_images
        from owned_native_retrieval import bounded_native_collision_threads
        class NoModelClient(RemoteNativeClient):
            def __init__(self):super().__init__(0,checkpoint='no_model_process_permitted');self.attempts=0
            def create_completion(self,*a,**k):self.attempts+=1;raise RuntimeError('Model calls forbidden in state probe')
        client=NoModelClient()
        with tempfile.TemporaryDirectory(prefix='native_sdk_branches_',dir='/dev/shm') as scratch:
            work=Path(scratch);root=work/'restored'
            proof=restore_solid_evidence(prior/'evidence_solid_v2/receipt.json',root)
            relocate_restored_paths(root,proof['original_root'])
            scene=RoomScene(room_geometry=None,scene_dir=root/'scene/room_bedroom')
            scene.restore_from_state_dict(json.loads((root/'last_actual_room_state.json').read_text()))
            with ExitStack() as scope:
                scope.enter_context(bind_remote_native_runtime(VLMService,client))
                scope.enter_context(bind_retrieval_without_images(AssetManager,receipt=out/'image_binding.json'))
                scope.enter_context(bounded_native_collision_threads(receipt=out/'collision_threads.json',max_threads=2))
                os.chdir(native)
                cfg,cfg_dict=module.compose_full_cfg('scene_designer_local',scene.scene_dir)
                logger=module.ConsoleLogger(output_dir=scene.scene_dir)
                agent=module.BaseExperiment.build_furniture_agent(cfg_dict=cfg_dict,
                    compatible_agents=module.COMPATIBLE_FURNITURE_AGENTS,logger=logger,render_gpu_id=None)
                agent.scene=scene
                registry=agent.asset_manager.registry
                registry_path=scene.scene_dir/'generated_assets/furniture/asset_registry.json'
                registry.load_from_file(registry_path)
                tool_map={t.name:t for t in agent._create_designer_tools()}
                env=module.ToolEnvironment(stage='furniture',room_prompt=scene.text_description,agent=agent,scene=scene,
                    tool_map=tool_map,steps_dir=work/'diagnostic_steps',skip_render=True)
                history=json.loads((prior/'review_traces/furniture_trace.json').read_text())
                for turn in history:
                    for call,res in zip(turn['proposal']['tool_calls'],turn.get('results',[])):
                        env._handler(env.mem,call['name'],call.get('arguments') or {},res['output'])
                env.step_idx=len(history)
                groups={}
                for obj in scene.objects.values():
                    if obj.object_type.value=='furniture' and obj.sdf_path:groups.setdefault(obj.sdf_path,[]).append(str(obj.object_id))
                sdf,ids=next((p,x) for p,x in sorted(groups.items(),key=lambda x:str(x[0])) if len(x)>1)
                chosen=ids[0]
                snap=NativeEditSnapshot(env,owned_root=root,snapshot_dir=work/'snapshot')
                def measured(label):
                    obj=scene.get_object(chosen)
                    assets=[dict(id=str(a.object_id),scale=a.scale_factor,bbox_min=a.bbox_min.tolist(),bbox_max=a.bbox_max.tolist())
                            for a in registry.list_all() if a.sdf_path==sdf]
                    return dict(label=label,scene_scale=obj.scale_factor,registry_assets=assets,
                        SDF_scales=sorted(set(e.text for e in ET.parse(sdf).getroot().findall('.//geometry/mesh/scale'))),
                        files=snap.file_fingerprint(),scene=scene.to_state_dict())
                async def action(factor):
                    if len(calls)>=7:raise RuntimeError('Probe tool budget exhausted')
                    call=dict(name='rescale_furniture_tool',arguments=dict(object_id=chosen,scale_factor=factor))
                    snap.check_action(call);calls.append(call)
                    reply=await env.step([call]);results.append(reply);dump(out/'actual_calls.json',dict(calls=calls,replies=results))
                    assert len(reply)==1 and isinstance(reply[0]['output'],dict) and reply[0]['output']['success']
                snap.restore();initial=measured('initial')
                await action(.95);clean_B=measured('B_clean')
                snap.restore();await action(.9);clean_A=measured('A_clean');snap.restore(restore_assets=False)
                json_only_parent=measured('JSON_only_parent');await action(.95);leaked_B=measured('B_after_JSON_only')
                snap.restore();await action(.9);snap.restore();await action(.95);restored_B=measured('B_after_full_restore')
                snap.restore();await action(.95);snap.restore();await action(.9);restored_A=measured('A_after_full_restore')
                snap.restore();final=measured('final_parent')
                equal=lambda a,b:all(a[k]==b[k] for k in ('scene_scale','registry_assets','SDF_scales','files','scene'))
                assert leaked_B['scene_scale']==clean_B['scene_scale'] and leaked_B['SDF_scales']!=clean_B['SDF_scales']
                assert leaked_B['registry_assets']!=clean_B['registry_assets']
                assert equal(restored_A,clean_A) and equal(restored_B,clean_B) and equal(final,initial)
                assert json_only_parent['scene']==initial['scene'] and json_only_parent['files']!=initial['files']
                dump(out/'measured_branches.json',dict(initial=initial,clean_A=clean_A,clean_B=clean_B,
                    JSON_only_parent=json_only_parent,leaked_B=leaked_B,restored_A=restored_A,restored_B=restored_B,final=final))
                result.update(status='normal_native_SDK_registry_file_branch_control_executed',
                    actual_native_tool_calls=len(calls),selected_object=chosen,shared_object_ids=ids,
                    registry_templates=registry.size(),all_native_tool_names=sorted(tool_map),
                    original_scene_and_registry_aliases_preserved=(id(scene),id(registry))==snap.identities,
                    JSON_only_same_scene_but_different_files_and_registry=True,
                    byte_and_registry_restore_order_invariant_for_A_and_B=True,
                    exact_parent_restored=True,original_source_scene_archive_sha256=proof['archive_sha256'],
                    qualified_operations=sorted(snap.admitted_tools),
                    unqualified=['asset acquisition','worker or GPU state','RNG changes','add/move/snap/remove full branch coverage'],
                    asset_retrieval_or_geometry_search_requests=0)
                agent.cleanup();agent=None;env=None
    except BaseException as exc:
        result.update(status='failed_no_automatic_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        os.chdir(cwd)
        if agent is not None:agent.cleanup()
        if client is not None:result['model_requests_attempted']=client.attempts;client.close()
        for p in reversed(owned):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
        subprocess.Popen=original;signal.alarm(0)
        result.update(seconds=time.monotonic()-tick,owned_processes=[dict(pid=p.pid,exit_code=p.poll()) for p in owned],blocked_operations=blocked)
        dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();asyncio.run(main(a.stage,a.out))
