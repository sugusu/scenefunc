"""Replay the actual failed model door call through normal native SDK tools.

Zero LLM calls; do not count the replay/rollback as a new generation result.
"""
import argparse
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


async def main(stage,attempt):
    stage=stage.resolve();out=stage/('resume_binding_actual_'+attempt);out.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[2];root=stage/'runtime'
    native=Path(json.loads((stage/'runtime_overlay_manifest.json').read_text())['native_root'])
    for p in (root,root/'src',repo/'scripts/autodl'):sys.path.insert(0,str(p))
    os.environ.update(SCENESMITH_ROOT=str(native),OPENAI_AGENTS_DISABLE_TRACING='1',PYTHONDONTWRITEBYTECODE='1')
    original=subprocess.Popen;owned=[];ports=set();blocked=[]
    allowed=native/'scenesmith/agent_utils/blender/standalone_server.py'
    class Popen(original):
        def __init__(self,*args,**kwargs):
            kwargs['start_new_session']=True;super().__init__(*args,**kwargs);owned.append(self)
            cmd=args[0];ports.add(int(cmd[cmd.index('--port')+1]))
    subprocess.Popen=Popen
    def audit(event,args):
        if event=='subprocess.Popen':
            cmd=args[1];ok=isinstance(cmd,list) and len(cmd)>1 and Path(cmd[1]).resolve()==allowed.resolve()
        elif event=='socket.connect':
            addr=args[1];ok=isinstance(addr,tuple) and addr[0] in ('127.0.0.1','::1') and addr[1] in ports
        else:return
        if not ok:blocked.append(event);raise RuntimeError('No model/external call in layout binding replay')
    sys.addaudithook(audit)
    started=time.monotonic();agent=None;client=None
    result=dict(status='started',model_forwards=0,API_calls=0,new_generation_runs=0)
    try:
        module=importlib.import_module('src.agent_rollout_decoupled')
        from scenesmith.agent_utils.vlm_service import VLMService
        from remote_native_control import RemoteNativeClient,bind_remote_native_runtime
        from restore_native_floor_binding import restore_floor_in_place
        initial=repo/'runs/autodl/20261110_remote_native_controller/native_joint_actual_v2/layout_after.json'
        state=json.loads(initial.read_text())
        current=json.loads((stage/'trial_actual_v1/observed_runtime_root.json').read_text())
        ram=Path(current['root']);trace=ram/'scene/rollout_floor_plan/trace.json'
        if not trace.exists():
            # The archived original trace is recovered by the trial analysis.
            trace=stage/'trial_actual_v1/floor_trace_retained.json'
        history=json.loads(trace.read_text());call=history[0]['proposal']['tool_calls'][0]
        assert call['name']=='add_door' and history[0]['results'][0]['output']['success'] is False
        client=RemoteNativeClient(0,checkpoint='/root/autodl-tmp/agent_method_preparation_20261027/work/critic_public_runtime_20261104/checkpoint')
        with bind_remote_native_runtime(VLMService,client):
            agent,tools,_,server=module.build_floor_plan_agent_tools('scene_designer_local',state['house_prompt'],out/'scene')
            assert server is None
            before=id(agent.layout);binding=restore_floor_in_place(agent,state,house_dir=out/'scene')
            restored=agent.layout.to_dict();assert len(restored['placed_rooms'])==1 and not restored['doors']
            reply=json.loads(await module.dispatch(tools,call['name'],call['arguments']))
            assert reply['success'] is True and len(agent.layout.doors)==1 and id(agent.layout)==before
            updated=agent.layout.to_dict()
            restore_floor_in_place(agent,state,house_dir=out/'scene')
            assert agent.layout.to_dict()==restored and id(agent.layout)==before
            result.update(status='actual_native_model_action_replay_updates_shared_layout_and_rolls_back',
                source_initial_state=str(initial),source_initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
                actual_original_model_call=call,native_reply=reply,doors_before=0,doors_after=1,doors_after_rollback=0,
                binding=binding,restore_exact_state_equal=True,tool_map_replaced=False,
                replay_only_not_policy_improvement=True)
            (out/'layout_after_action.json').write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n')
    except BaseException as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
    finally:
        if agent is not None:agent.cleanup()
        if client is not None:client.close()
        for child in owned:
            if child.poll() is None:child.terminate();child.wait(timeout=10)
        subprocess.Popen=original
        result.update(seconds=time.monotonic()-started,blocked_operations=blocked,
            owned_processes=[dict(pid=p.pid,exit_code=p.poll()) for p in owned])
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--attempt',choices=('v1','v2'),required=True)
    args=p.parse_args();asyncio.run(main(args.stage,args.attempt))
