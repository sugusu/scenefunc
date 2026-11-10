"""One real original-contract floor policy -> SDK tool step from an empty layout.

Not a complete generation experiment or an additional D/A prompt attempt.
The task is the original integration's default, not a benchmark or new label.
"""
import argparse
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')


def prepare(stage):
    repo = Path(__file__).resolve().parents[2]
    prior = json.loads((repo / 'runs/autodl/20261110_shared_model_runtime/protocol.json').read_text())
    for row in prior['weight_sha256_verified']:
        stat = Path(row['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (row['bytes'], row['mtime_ns'], row['inode'])
    material = json.loads((stage / 'materialization.json').read_text())
    runtime = Path(material['runtime_root'])
    files = [repo/'scripts/autodl'/f for f in ('native_joint_model_host.py',
        'run_native_joint_admission.py', 'remote_native_control.py')]
    files += list(runtime.rglob('*.py'))
    files += list((runtime/'prompts').iterdir())
    files += [Path(material['new_configuration']),
        Path('/root/autodl-tmp/agent_method_preparation_20261027/work/paris_local_run_results_20261104/local_vlm_sdpa.py')]
    value = dict(status='frozen_before_inference', checkpoint=prior['checkpoint'],
        max_model_forwards=1, max_total_output_tokens=768, max_native_tool_calls=4,
        max_parent_seconds=300, max_actor_seconds=240, retry=False,
        task_source='Unmodified Designer.loop.DEFAULT_HOUSE_PROMPT in qualified original integration',
        inference_contract='Unmodified native joint messages and parser, no D/A stage suffix or answer prefix',
        previous_phase_budget_reopened=False, training_steps=0, paid_API_calls=0,
        source_scope='Empty-layout controller/tool admission, not full scene generation or method comparison',
        weight_sha256_verified=prior['weight_sha256_verified'],
        weight_hash_scope='Previous full SHA verification; unchanged size/mtime/inode checked now',
        code_sha256={str(f.resolve()):sha(f) for f in files})
    path = stage/'native_joint_protocol.json'
    if path.exists():
        raise FileExistsError('Existing protocol is immutable')
    dump(path,value)
    return path


async def run(stage):
    repo = Path(__file__).resolve().parents[2]
    path = stage/'native_joint_protocol.json'
    protocol = json.loads(path.read_text())
    for f,h in protocol['code_sha256'].items():
        assert sha(f) == h, f
    assert json.loads((stage/'actual_v6/result.json').read_text())['status'] == 'whole_native_module_and_floor_tools_constructed'
    assert shutil.disk_usage(stage).free > 5*1024**3 + 8*1024**2
    out = stage/'native_joint_actual_v1'; out.mkdir(exist_ok=False)
    runtime = Path(json.loads((stage/'materialization.json').read_text())['runtime_root'])
    native = runtime/'native_source'
    for f in (runtime, runtime/'src', repo/'scripts/autodl'):
        sys.path.insert(0,str(f))
    os.environ['SCENESMITH_ROOT']=str(native)
    os.environ['OPENAI_AGENTS_DISABLE_TRACING']='1'
    original = subprocess.Popen
    owned=[]; allowed=[]; blocked=[]
    class OwnedPopen(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);owned.append(self)
    subprocess.Popen=OwnedPopen
    actor=None;client=None;agent=None;log=None
    started=time.monotonic()
    result=dict(status='started', paid_API_calls=0, training_steps=0,
        complete_scenes_generated=0, complete_scene_success=None,
        scope=protocol['source_scope'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    def deadline(signum, frame):
        raise TimeoutError('Native joint qualification parent 300-second deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(300)
    actor_script=repo/'scripts/autodl/native_joint_model_host.py'
    blender_script=native/'scenesmith/agent_utils/blender/standalone_server.py'
    def audit(event,args):
        if event=='socket.connect':
            addr=args[1]
            valid=isinstance(addr,tuple) and addr[0] in ('127.0.0.1','::1') and (addr[1]==port or 8000<=addr[1]<=8050)
        elif event=='subprocess.Popen':
            cmd=args[1]
            valid=isinstance(cmd,list) and len(cmd)>1 and Path(cmd[1]).resolve() in (actor_script.resolve(),blender_script.resolve()) and len(owned)<2
        elif event=='socket.getaddrinfo':
            valid=args[0] in ('127.0.0.1','localhost','::1')
        else:return
        if not valid:
            blocked.append(event);raise RuntimeError('External/unadmitted operation in native joint qualification')
        allowed.append(event)
    sys.addaudithook(audit)
    try:
        module=importlib.import_module('src.agent_rollout_decoupled')
        from Designer.loop import DEFAULT_HOUSE_PROMPT
        from scenesmith.agent_utils.vlm_service import VLMService
        from remote_native_control import RemoteNativeClient,bind_remote_native_runtime
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
        env.pop('PYTHONPATH',None)
        model_python='/root/autodl-tmp/agent_method_preparation_20261027/work/v100_runtime_20261104/runtime/bin/python'
        log=(out/'actor.log').open('w')
        actor=subprocess.Popen([model_python,str(actor_script),'--protocol',str(path.resolve()),
            '--out',str((out/'actor').resolve()),'--port',str(port)],
            stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        dump(out/'actor_process.json',dict(pid=actor.pid,port=port))
        client=RemoteNativeClient(port,checkpoint=protocol['checkpoint'],timeout=210)
        tick=time.monotonic()
        while True:
            if actor.poll() is not None:raise RuntimeError('Actor exited before readiness; no restart')
            try:
                if client.session.get(client.base+'/health',timeout=.5).json()['status']=='ready':break
            except Exception:pass
            if time.monotonic()-tick>40:raise TimeoutError('Actor readiness; no restart')
            time.sleep(.1)
        with bind_remote_native_runtime(VLMService,client):
            agent,tools,_,materials=module.build_floor_plan_agent_tools('scene_designer_local',DEFAULT_HOUSE_PROMPT,out/'floor_plan')
            assert materials is None and not agent.layout.rooms
            scene=module.ToolEnvironment(stage='floor_plan',room_prompt=DEFAULT_HOUSE_PROMPT,
                agent=agent,scene=agent.layout,tool_map=tools,steps_dir=out/'steps')
            obs=scene.observation();dump(out/'native_observation.json',obs)
            before=agent.layout.to_dict();dump(out/'layout_before.json',before)
            policy=module.ModelPolicy(protocol['checkpoint'],max_new_tokens=768)
            proposal=policy.generate(obs);dump(out/'native_policy_proposal.json',proposal)
            calls=proposal['tool_calls']
            if len(calls)>4:raise ValueError('Entire proposed batch exceeds fixed four-tool cap; no subset picking')
            results=await scene.step(calls) if calls else []
            dump(out/'native_tool_results.json',results)
            after=agent.layout.to_dict();dump(out/'layout_after.json',after)
            result.update(status='native_joint_policy_returned_and_original_tool_step_processed',
                task=DEFAULT_HOUSE_PROMPT, proposed_calls=len(calls), tool_results=results,
                room_count_before=len(before.get('rooms',{})),room_count_after=len(after.get('rooms',{})),
                layout_changed=before!=after, remote_policy_class=type(policy).__module__+'.'+type(policy).__name__,
                controller_tools_unchanged=True, phase_format_validity_not_inferred=True,
                observation_sha256=sha(out/'native_observation.json'),
                whole_runtime_ready=False, whole_scene_quality_verified=False)
        assert not blocked
        assert all(sha(f)==h for f,h in protocol['code_sha256'].items())
    except BaseException as exc:
        result.update(status='failed_no_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        if agent is not None:agent.cleanup()
        if client is not None:client.close()
        for child in reversed(owned):
            if child.poll() is None:
                if child is actor:os.killpg(child.pid,signal.SIGTERM)
                else:child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        if log:log.close()
        signal.alarm(0);subprocess.Popen=original
        result.update(seconds=time.monotonic()-started,blocked_operations=blocked,
            allowed_operations=allowed,owned_processes=[dict(pid=c.pid,exit_code=c.poll()) for c in owned])
        ledger=out/'actor/ledger.json'
        if ledger.exists():result['model_ledger']=json.loads(ledger.read_text())
        dump(out/'result.json',result)
        dump(stage/'NATIVE_JOINT_CLOSED.json',dict(attempt='native_joint_actual_v1',unused_allowance_carries=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('tool_results','model_ledger','allowed_operations')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.prepare:print(prepare(args.stage.resolve()))
    else:asyncio.run(run(args.stage.resolve()))
