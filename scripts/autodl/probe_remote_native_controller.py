"""Import whole adapted native controller; construct real floor tools/materials.

No model service/startup, fabricated scene geometry, coordinates or tool reply.
External networking is blocked. The explicit --allow-native-blender option
admits one owned native renderer subprocess and its loopback health checks.
This does not run generation or inference.
"""
import argparse
import ast
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import socket
import signal
import subprocess
import sys
import time
import traceback


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def named_ast(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n,
        (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
        include_attributes=False)


async def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    materialization = json.loads((args.stage / 'materialization.json').read_text())
    assert materialization['status'] == 'materialized'
    runtime = Path(materialization['runtime_root'])
    native_root = Path(materialization['native_root'])
    before = {r['source']: sha(r['source']) for r in materialization['copied_sources']}
    assert all(before[r['source']] == r['sha256'] for r in materialization['copied_sources'])
    patched = {str(Path(r['path']).resolve()): r['after_sha256'] for r in materialization['patches']}
    recovery = args.stage / 'source_recovery_v2.json'
    if recovery.exists():
        for row in json.loads(recovery.read_text())['patches']:
            assert sha(row['preserved_failed_copy']) == row['v1_sha256']
            patched[str(Path(row['destination']).resolve())] = row['after_sha256']
    recovery = args.stage / 'source_recovery_v3.json'
    if recovery.exists():
        for row in json.loads(recovery.read_text())['patches']:
            assert sha(row['preserved_failed_copy']) == row['v2_sha256']
            patched[str(Path(row['destination']).resolve())] = row['after_sha256']
    for row in materialization['copied_sources']:
        assert sha(row['destination']) == patched.get(str(Path(row['destination']).resolve()), row['sha256'])
    for row in materialization['fetched']:
        assert row['status'] == 'fetched' and sha(row['destination']) == row['sha256']
    repo = Path(__file__).resolve().parents[2]
    for path in (runtime, runtime / 'src', repo / 'scripts/autodl'):
        sys.path.insert(0, str(path))
    os.environ['SCENESMITH_ROOT'] = str(native_root)
    os.environ['OPENAI_AGENTS_DISABLE_TRACING'] = '1'
    blocked = []
    allowed = []
    owned = []
    native_blender = native_root / 'scenesmith/agent_utils/blender/standalone_server.py'
    original_popen = subprocess.Popen
    class OwnedPopen(original_popen):
        # Preserve Popen[bytes] annotations used by the actual MCP dependency.
        def __init__(self, *pargs, **kwargs):
            super().__init__(*pargs, **kwargs)
            owned.append(self)
    subprocess.Popen = OwnedPopen
    def audit(event, arguments):
        permitted = False
        if args.allow_native_blender:
            if event == 'subprocess.Popen':
                cmd = arguments[1]
                permitted = (not owned and isinstance(cmd, list) and len(cmd) >= 8 and
                    cmd[0] == sys.executable and Path(cmd[1]).resolve() == native_blender.resolve()
                    and cmd[2:4] == ['--host', '127.0.0.1'] and cmd[4] == '--port'
                    and 8000 <= int(cmd[5]) <= 8050)
            elif event == 'socket.connect':
                addr = arguments[1]
                permitted = (isinstance(addr, tuple) and addr[0] in ('127.0.0.1', '::1')
                    and 8000 <= addr[1] <= 8050)
        if event in ('socket.connect', 'subprocess.Popen') and not permitted:
            blocked.append(event)
            raise RuntimeError('Unadmitted network or process startup in native controller probe')
        if permitted:
            allowed.append({'event': event, 'argument': str(arguments[1])})
    sys.addaudithook(audit)
    result = {'status': 'started', 'model_calls': 0, 'services_started': 0,
              'paid_API_calls': 0, 'new_rooms_generated': 0,
              'scope': 'Whole-module import, normal native floor-agent/tool construction and local materials only'}
    started = time.monotonic()
    agent = None; client = None
    def timeout(signum, frame):
        raise TimeoutError('Native controller qualification 120-second deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        module = importlib.import_module('src.agent_rollout_decoupled')
        from scenesmith.agent_utils.vlm_service import VLMService
        from remote_native_control import RemoteNativeClient, bind_remote_native_runtime
        # Port0 is deliberately not a running service. The constructor may
        # prepare a client, but any unexpected actual request is blocked.
        client = RemoteNativeClient(0, checkpoint=
            '/root/autodl-tmp/agent_method_preparation_20261027/work/critic_public_runtime_20261104/checkpoint')
        original_init = VLMService.__init__
        original_source = Path('/root/autodl-tmp/agent_method_preparation_20261027/work/native_execution_inventory_20261104/source')
        preserved = {}
        for filename, names in [('src/agent_rollout_decoupled.py',
                                 ['ToolEnvironment', 'RolloutSession', 'driver_loop', 'main_async']),
                                ('Designer/agent_rollout.py',
                                 ['parse_assistant_text', 'render_system', 'build_messages', 'render_topdown'])]:
            for name in names:
                equal = named_ast(original_source / filename, name) == named_ast(runtime / filename, name)
                assert equal
                preserved[f'{filename}:{name}'] = equal
        prompt = 'CPU construction/material lookup only; no scene layout has been generated.'
        with bind_remote_native_runtime(VLMService, client) as binding:
            agent, tools, logger, materials_server = module.build_floor_plan_agent_tools(
                config_name='scene_designer_local', house_prompt=prompt,
                scene_dir=args.out / 'normal_native_floor_plan')
            assert materials_server is None
            assert not agent.cfg.materials.use_retrieval_server
            policy = module.ModelPolicy(client.checkpoint, max_new_tokens=512)
            assert not hasattr(policy, 'model') and not hasattr(policy, 'processor')
            lookups = []
            if 'get_material' not in tools:
                raise ValueError('Native floor tool catalogue lacks get_material')
            schema = tools['get_material'].params_json_schema
            required = schema.get('required', [])
            assert required == ['description'], schema
            for description in ['wood floor', 'white plaster wall']:
                raw = await module.dispatch(tools, 'get_material', {'description': description})
                lookups.append({'description': description, 'raw_native_reply': raw})
                decoded = json.loads(raw)
                if not decoded.get('success'):
                    raise RuntimeError('Native material lookup did not succeed: ' + raw)
            result.update(status='whole_native_module_and_floor_tools_constructed',
                controller_module_file=module.__file__, controller_class=module.ToolEnvironment.__module__,
                actual_floor_agent_type=type(agent).__module__ + '.' + type(agent).__name__,
                actual_sdk_tool_types=sorted({type(t).__module__ + '.' + type(t).__name__ for t in tools.values()}),
                native_floor_tool_names=sorted(tools), native_material_lookups=lookups,
                original_tool_loop_AST_preserved=preserved,
                in_process_model_constructed=False, transformers_loaded='transformers' in sys.modules,
                remote_policy_class=type(policy).__module__ + '.' + type(policy).__name__,
                actual_pending_local_VLM_instances=binding['vlm_instances'],
                floor_material_mode='Native two-material local fallback; not semantic materials retrieval',
                whole_runtime_ready=False, full_scene_success=None)
        assert VLMService.__init__ is original_init
        assert not binding['active']
        assert not blocked
        assert all(sha(p) == h for p, h in before.items())
        dump(args.out / 'original_source_integrity.json', before)
    except BaseException as exc:
        result.update(status='failed_no_model_or_service_start', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc())
        raise
    finally:
        if agent is not None and callable(getattr(agent, 'cleanup', None)):
            agent.cleanup()
        if client is not None:
            client.close()
        for child in owned:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=5)
        subprocess.Popen = original_popen
        signal.alarm(0)
        result['services_started'] = len(owned)
        result['owned_renderer_processes'] = [{'pid': c.pid, 'returncode': c.poll()} for c in owned]
        result['allowed_operations'] = allowed
        if result['status'] == 'failed_no_model_or_service_start' and owned:
            result['status'] = 'failed_without_model_call_owned_renderer_reaped'
        result.update(seconds=time.monotonic() - started, blocked_operations=blocked,
            parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        dump(args.out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('native_material_lookups', 'original_tool_loop_AST_preserved')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-native-blender', action='store_true')
    asyncio.run(main(p.parse_args()))
