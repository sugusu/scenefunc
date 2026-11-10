"""Normal native SDK move branches at the unchanged loop guard boundary.

Controlled execution-copy probe only; no model, new assets, scene quality score,
teacher preference, training or coordinate-repair search.
"""
import argparse
import asyncio
import copy
from contextlib import ExitStack
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from native_trial_evidence import restore_solid_evidence,relocate_restored_paths,sha
from native_edit_snapshot import NativeEditSnapshot
from native_move_snapshot import NativeMoveSnapshot
from check_native_asset_versions import differences


def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')


async def main(repo,stage,out_name):
    out=stage/out_name;out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((stage/'PROTOCOL.json').read_text())
    prior=repo/'runs/autodl/20261110_complete_native_runtime';overlay=prior/'runtime'
    native=Path(json.loads((prior/'runtime_overlay_manifest.json').read_text())['native_root'])
    for p in (overlay,overlay/'src',repo/'scripts/autodl'):sys.path.insert(0,str(p))
    os.environ.update(SCENESMITH_ROOT=str(native),OPENAI_AGENTS_DISABLE_TRACING='1',
                      PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    original=subprocess.Popen;owned=[];ports=set();blocked=[];client=None;agent=None;cwd=Path.cwd()
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
        if not ok:blocked.append(event);raise RuntimeError('Unadmitted network/model/process')
    sys.addaudithook(audit)
    def deadline(sig,frame):raise TimeoutError('Fixed native SDK branch probe deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(protocol['max_seconds'])
    tick=time.monotonic();calls=[]
    result=dict(status='started',local_VLM_forwards=0,paid_API_calls=0,training_updates=0,
                new_generated_rooms=0,whole_scene_success=None,not_a_learned_method=True)
    try:
        module=importlib.import_module('src.agent_rollout_decoupled')
        from pydrake.all import RollPitchYaw
        from scenesmith.agent_utils.room import RoomScene
        from scenesmith.agent_utils.asset_manager import AssetManager
        from scenesmith.agent_utils.vlm_service import VLMService
        from remote_native_control import RemoteNativeClient,bind_remote_native_runtime
        from retrieval_image_binding import bind_retrieval_without_images
        from owned_native_retrieval import bounded_native_collision_threads
        class NoModel(RemoteNativeClient):
            def __init__(self):super().__init__(0,checkpoint='no_model_permitted');self.attempts=0
            def create_completion(self,*a,**k):self.attempts+=1;raise RuntimeError('Model requests forbidden')
        client=NoModel()
        with tempfile.TemporaryDirectory(prefix='native_move_branches_',dir='/dev/shm') as scratch:
            work=Path(scratch);root=work/'restored'
            proof=restore_solid_evidence(prior/'evidence_solid_v2/receipt.json',root)
            relocate_restored_paths(root,proof['original_root'])
            scene=RoomScene(room_geometry=None,scene_dir=root/'scene/room_bedroom')
            scene.restore_from_state_dict(json.loads((root/'last_actual_room_state.json').read_text()))
            original_loaded_state=scene.to_state_dict()
            scene.restore_from_state_dict(original_loaded_state)
            canonical_loaded_state=scene.to_state_dict()
            dump(out/'parent_native_canonicalization.json',dict(
                applied_to_owned_copy_only=True,original_archive_changed=False,
                differences=differences(original_loaded_state,canonical_loaded_state)))
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
                registry.load_from_file(scene.scene_dir/'generated_assets/furniture/asset_registry.json')
                env=module.ToolEnvironment(stage='furniture',room_prompt=scene.text_description,agent=agent,scene=scene,
                    tool_map={t.name:t for t in agent._create_designer_tools()},steps_dir=work/'steps',skip_render=True)
                # Reconstitute documented public tool memory only. The closure
                # history begins at actual constructor state, not invented past.
                history=json.loads((prior/'review_traces/furniture_trace.json').read_text())
                for turn in history:
                    for call,res in zip(turn['proposal']['tool_calls'],turn.get('results',[])):
                        env._handler(env.mem,call['name'],call.get('arguments') or {},res['output'])
                env.step_idx=len(history)
                groups={}
                for obj in scene.objects.values():
                    if obj.object_type.value=='furniture' and obj.sdf_path:
                        groups.setdefault(obj.sdf_path,[]).append(str(obj.object_id))
                sdf,ids=next((p,x) for p,x in sorted(groups.items(),key=lambda x:str(x[0])) if len(x)>1)
                selected=ids[0];obj=scene.get_object(selected);position=obj.transform.translation()
                yaw=math.degrees(RollPitchYaw(obj.transform.rotation()).yaw_angle())
                snapshot=NativeMoveSnapshot(env,owned_root=root,snapshot_dir=work/'snapshot')
                assert snapshot.settings[0] and snapshot.settings[1]==protocol['unchanged_loop_limit_required']==25
                assert not snapshot.history
                parent=dict(scene=scene.to_state_dict(),memory=copy.deepcopy(env.mem),files=snapshot.file_fingerprint())
                dump(out/'frozen_parent.json',parent)
                def observe():
                    return dict(scene=scene.to_state_dict(),memory=copy.deepcopy(env.mem),
                        files=snapshot.file_fingerprint(),loop_history=list(snapshot.detector.recent_calls),
                        original_scene_registry_aliases=(id(scene),id(registry))==snapshot.identities)
                async def move(label,delta):
                    if len(calls)>=protocol['max_actual_SDK_calls']:raise RuntimeError('SDK action cap reached')
                    call=dict(name='move_furniture_tool',arguments=dict(object_id=selected,
                        x=float(position[0]+delta),y=float(position[1]),yaw=float(yaw)))
                    snapshot.check_action(call)
                    before=observe();reply=await env.step([call]);after=observe()
                    record=dict(index=len(calls)+1,label=label,call=call,reply=reply,
                                before=before,after=after)
                    calls.append(record);dump(out/'actual_calls.json',calls)
                    assert len(reply)==1 and isinstance(reply[0]['output'],dict)
                    return record,reply[0]['output']
                # Same physical/logical parent, intentionally omitting guard
                # restoration exactly as the previous limited snapshot did.
                for index in range(25):
                    NativeEditSnapshot.restore(snapshot)
                    assert scene.to_state_dict()==parent['scene'] and env.mem==parent['memory']
                    row,reply=await move('A_parent_except_guard_'+str(index+1),.05)
                    assert reply['success'] is True
                NativeEditSnapshot.restore(snapshot)
                row,blocked_reply=await move('A_boundary_parent_except_guard',.05)
                assert blocked_reply['success'] is False and blocked_reply['error_type']=='loop_detected'
                assert row['before']['scene']==row['after']['scene']==parent['scene']
                snapshot.restore();A,reply=await move('A_all_qualified_parent_state_restored',.05)
                assert reply['success'] is True
                snapshot.restore();B,reply=await move('B_clean',.10);assert reply['success'] is True
                snapshot.restore();A2,reply=await move('A_after_B_reset',.05);assert reply['success'] is True
                snapshot.restore();B2,reply=await move('B_after_A_reset',.10);assert reply['success'] is True
                assert A['after']==A2['after'] and B['after']==B2['after']
                snapshot.restore();final=observe()
                assert final['scene']==parent['scene'] and final['memory']==parent['memory']
                assert final['files']==parent['files'] and final['loop_history']==snapshot.history
                dump(out/'final_parent.json',final)
                result.update(status='actual_SDK_move_guard_history_and_branch_restore_qualified',
                    actual_native_tool_calls=len(calls),same_parent_first25_succeeded=True,
                    same_parent_26th_blocked_by_unchanged_guard=True,
                    after_exact_parent_guard_restore_A_succeeded=True,
                    repeated_A_B_outcomes_equal_with_parent_restoration=True,
                    original_guard_enabled=snapshot.settings[0],original_guard_limit=snapshot.settings[1],
                    original_guard_window=snapshot.settings[2],guard_limit_or_noise_changed=False,
                    snapshot_qualified_operations=sorted(snapshot.admitted_tools),
                    selected_object=selected,source_archive_sha256=proof['archive_sha256'],
                    scope='Move/rescale on existing registered furniture only; no acquisition/add/remove/snap/worker/RNG or complete-scene success.',
                    no_old_rollout_failure_frequency_or_learning_gain_inferred=True)
                agent.cleanup();agent=None
    except BaseException as exc:
        import traceback
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
        result.update(seconds=time.monotonic()-tick,actual_native_tool_calls=len(calls),
            blocked_operations=blocked,owned_processes=[dict(pid=p.pid,exit_code=p.poll()) for p in owned])
        dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--out-name',default='move_SDK_v1')
    a=p.parse_args();asyncio.run(main(a.repo.resolve(),a.stage.resolve(),a.out_name))
