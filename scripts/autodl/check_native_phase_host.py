"""CPU integration fixture: real native remove/add and fresh mesh renders.

The policy backend emits predetermined fixture text, NOT model inference.
Native ToolEnvironment.step and memory handlers are loaded from pinned source
without booting SDK services. The existing read-only native-kernel dispatcher
runs in its own geometry environment. This is not a design-quality experiment.
"""
import argparse
import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from phase_host import (NativeRoomPort, StageBudget, canonical, digest,
                        furniture_driver, run_phase_stage, write)
from phase_policy import PhasePolicy


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Pinned source definitions missing')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])),
                 str(path), 'exec'), namespace)


def process(command, folder, label, timeout=600):
    tick = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',
               PYTHONPATH='/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps')
    with (folder / (label + '.stdout')).open('w') as out, (folder / (label + '.stderr')).open('w') as err:
        try:
            run = subprocess.run(command, stdout=out, stderr=err, env=env, timeout=timeout)
            record = {'exit_code': run.returncode, 'timeout': False}
        except subprocess.TimeoutExpired:
            record = {'exit_code': None, 'timeout': True}
    record.update(command=command, elapsed_seconds=time.monotonic()-tick)
    write(folder / (label + '_receipt.json'), record)
    if record['exit_code'] != 0:
        raise RuntimeError(label + ' failed; no automatic retry')


class Tokenizer:
    def __init__(self, file):
        from tokenizers import Tokenizer as FastTokenizer
        self.impl = FastTokenizer.from_file(str(file))

    def convert_tokens_to_ids(self, token):
        return self.impl.token_to_id(token)

    def decode(self, ids, skip_special_tokens=False):
        return self.impl.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode(self, text):
        return self.impl.encode(text, add_special_tokens=False).ids


class ScriptedBackend:
    """Only a test double. It never calls a language model or predicts an action."""
    def __init__(self, tokenizer, actions):
        self.processor = SimpleNamespace(tokenizer=tokenizer)
        self.eos_ids = [tokenizer.convert_tokens_to_ids('<|im_end|>')]
        self.actions = actions
        self.action_index = 0
        self.observations = []

    def generate(self, messages, images, *, stage, decision, token_limit, **kwargs):
        self.observations.append({'stage': stage, 'messages_sha256': digest(messages),
                                  'decision_supplied': decision is not None,
                                  'image_sizes': [list(i.size) for i in images]})
        if stage == 'decision':
            text = '<think>Predetermined CPU integration fixture, not a model decision.</think>'
        else:
            call = self.actions[self.action_index]
            self.action_index += 1
            text = (('<tool_call>' + canonical(call) + '</tool_call>') if call else '') + '<|im_end|>'
        ids = self.processor.tokenizer.encode(text)
        assert len(ids) <= token_limit
        return {'output_ids': ids, 'input_tokens': len(self.processor.tokenizer.encode(canonical(messages))),
                'scope': 'scripted token fixture; input count is text-only, NOT multimodal inference cost'}


async def main(args):
    w, out = args.work_base.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    if shutil.disk_usage(out).free < 5*1024**3 + 70*1024**2:
        raise RuntimeError('Two native scene checkpoints need 70 MiB above the 5 GiB reserve')
    old = w / 'paris_local_run_results_20261104'
    original = old / 'B_mesh_feedback_sdpa_kv/tools/checkpoint.json'
    prior = None
    initial = original
    if args.resume_fixture:
        parent = args.resume_fixture.resolve()
        prior = read(parent/'native_stage/phase_host/result.json')
        if prior['status'] != 'host_failure' or prior['native_calls'] != 1 or len(prior['rounds']) != 1:
            raise ValueError('Resume only the recorded one-action render interruption')
        initial = parent/'native_000/tools/checkpoint.json'
        if not args.initial_views or read(args.initial_views)['status'] != 'completed':
            raise ValueError('Fresh completed observation of the actual interrupted state required')
    payload = read(initial)['payload']
    native_source = w / 'native_execution_inventory_20261104/source/src/agent_rollout_decoupled.py'
    memory_source = w / 'memory_replay_source_20261103/src/memory.py'
    schema_source = w / 'professional_data_ingest_20261104/native_src/stage_schemas.py'
    source_files = [initial, initial.parent/'partial_scene.glb', native_source,
                    memory_source, schema_source, old/'dispatch_corrected.py', old/'render_state.py']
    source_files += [Path(x) for x in payload['geometry_hashes']]
    hashes = {str(p): sha(p) for p in source_files}
    write(out/'source_hashes_before.json', hashes)
    mem = load_module(memory_source, 'fixture_memory')
    schemas = load_module(schema_source, 'fixture_schemas').FURNITURE_SCHEMAS
    geometry_python = '/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python'
    render_python = '/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python'
    checkpoint = initial
    dispatch_index = 0
    render_records = []

    class ActualCheckpointScene:
        def to_state_dict(self):
            # Actual native serializations; no synthetic boxes/edited coordinates.
            p = read(checkpoint)['payload']
            return {'room_id': p['room_id'], 'objects': p['placed_instances'],
                    'registered_assets': p['asset_templates'],
                    'geometry_hashes': p['geometry_hashes']}

    async def native_dispatch(tool_map, name, arguments):
        nonlocal checkpoint, dispatch_index
        stage = out / f'native_{dispatch_index:03d}'
        stage.mkdir()
        write(stage/'calls.json', [{'name': name, 'arguments': arguments}])
        process([geometry_python, str(old/'dispatch_corrected.py'), '--calls', str(stage/'calls.json'),
                 '--resume', str(checkpoint), '--output', str(stage/'tools')], stage, 'dispatch', 300)
        checkpoint = stage/'tools/checkpoint.json'
        dispatch_index += 1
        replies = read(stage/'tools/tool_responses.json')
        assert len(replies) == 1
        return canonical(replies[0]['response'])

    namespace = {'json': json, 'sys': sys, 'traceback': traceback, 'Path': Path,
                 '_mem': mem, 'dispatch': native_dispatch}
    selected(native_source, ['ToolEnvironment'], namespace)
    NativeEnvironment = namespace['ToolEnvironment']

    class FixtureEnvironment(NativeEnvironment):
        def observation(self):
            if checkpoint == original:
                views = read(old/'B_after_feedback_views/receipt.json')
            elif checkpoint == initial and args.initial_views:
                views = read(args.initial_views)
            else:
                view_dir = checkpoint.parent.parent/'views'
                process([sys.executable, str(Path(__file__).with_name('render_native_sequential.py')),
                         '--source', str(old/'render_state.py'), '--source-sha256', sha(old/'render_state.py'),
                         '--scene', str(checkpoint.parent/'partial_scene.glb'), '--out', str(view_dir),
                         '--render-python', render_python], checkpoint.parent.parent, 'render', 1800)
                views = read(view_dir/'receipt.json')
            assert views['source_sha256'] == sha(checkpoint.parent/'partial_scene.glb')
            assert all(sha(v['path']) == v['sha256'] for v in views['views'])
            record = {'step': self.step_idx, 'checkpoint': str(checkpoint),
                      'state_sha256': digest(self.scene.to_state_dict()),
                      'source_scene_sha256': views['source_sha256'], 'views': views['views']}
            render_records.append(record)
            messages = [{'role': 'system', 'content': 'CPU fixture only: exercise native remove/add and termination. No design-quality claim.'},
                        {'role': 'user', 'content': [{'type': 'image', 'image': v['path']} for v in views['views']] +
                         [{'type': 'text', 'text': mem.render_memory('furniture', self.mem)}]}]
            return {'messages': messages, 'step_idx': self.step_idx}

    env = FixtureEnvironment.__new__(FixtureEnvironment)
    env.scene = ActualCheckpointScene()
    env.stage = 'furniture'
    env.steps_dir = out/'native_stage'
    env.steps_dir.mkdir()
    env.step_idx = 0
    env.mem = mem.empty_memory('furniture')
    env._handler = mem.HANDLERS['furniture']
    # Exact source argument types converted to JSON Schema solely for this
    # direct-kernel fixture. Production uses LIVE SDK schemas without this step.
    allowed = ('remove_furniture_tool', 'add_furniture_to_scene_tool')
    type_names = {str: 'string', float: 'number', int: 'number'}
    env.tool_map = {}
    for name in allowed:
        spec = schemas[name]
        props = {}
        for key, value in {**spec['required'], **spec['optional']}.items():
            props[key] = {'type': 'number' if isinstance(value, tuple) else type_names[value]}
        env.tool_map[name] = SimpleNamespace(params_json_schema={'type': 'object', 'properties': props,
                                                'required': list(spec['required']), 'additionalProperties': False})
    # Replay only already recorded tool effects to seed the native memory.
    for entry in payload['tool_history']:
        env._handler(env.mem, entry['call']['name'], entry['call']['arguments'], entry['response'])
    baseline_ids = {o['object_id'] for o in payload['placed_instances']}
    def acceptance(e):
        ids = {o['object_id'] for o in e.scene.to_state_dict()['objects']}
        return {'state_sha256': digest(e.scene.to_state_dict()), 'full_scene_success': None,
                'native_object_count': len(ids), 'initial_object_count': len(baseline_ids),
                'removed_original_ids': sorted(baseline_ids-ids), 'new_ids': sorted(ids-baseline_ids),
                'unverified': ['semantic identity after re-addition', 'physical validity', 'original user requirements', 'function and aesthetics'],
                'scope': 'identity/count accounting only; not a scene acceptance oracle'}
    def images(messages):
        from PIL import Image
        paths = [p['image'] for m in messages if isinstance(m.get('content'), list)
                 for p in m['content'] if p.get('type') == 'image']
        return [Image.open(p).convert('RGB') for p in paths]
    tokenizer = Tokenizer(w/'inference_autodl_20261103/processor_only/tokenizer.json')
    actions = [{'name': 'remove_furniture_tool', 'arguments': {'object_id': 'office_chair_0'}},
               {'name': 'add_furniture_to_scene_tool', 'arguments': {'asset_id': 'office_chair_2', 'x': .25, 'y': .55, 'yaw': 0}}, None]
    if prior:
        previous = prior['rounds'][0]['execution']['call']['name']
        expected = 'add_furniture_to_scene_tool' if args.resume_step == 'after_add' else 'remove_furniture_tool'
        if previous != expected:
            raise ValueError('Explicit resume stage does not match actual completed action')
        actions = actions[2:] if args.resume_step == 'after_add' else actions[1:]
    backend = ScriptedBackend(tokenizer, actions)
    policy = PhasePolicy(backend, arm='paired_phase_v3', total_output_tokens=512,
                         decision_output_tokens=128, action_output_tokens=384, total_seconds=30)
    budget = StageBudget(len(actions), 2*len(actions), max(1,len(actions)-1), 512*len(actions), 3000)
    delegated = []
    async def original_driver(**kwargs):
        delegated.append(kwargs['env'].stage)
        return ['untouched_common_base_stage']
    module = SimpleNamespace(driver_loop=original_driver)
    with furniture_driver(module, policy=policy, budget=budget, allowed_tools=allowed,
                          load_images=images, acceptance=acceptance):
        for stage in ('floor_plan', 'wall', 'ceiling'):
            assert await module.driver_loop(env=SimpleNamespace(stage=stage), policy=object(), max_steps=1) == ['untouched_common_base_stage']
        rows = await module.driver_loop(env=env, policy=object(), max_steps=len(actions))
    assert module.driver_loop is original_driver
    result = read(env.steps_dir/'phase_host/result.json')
    assert result['status'] == 'model_stop_proposed'
    assert result['generation_attempts'] == 2*len(actions) and result['native_calls'] == len(actions)-1
    assert len(result['rounds']) == len(actions) and result['full_scene_success'] is None
    for a, b in zip(result['rounds'], result['rounds'][1:]):
        assert a['execution']['after_sha256'] == b['before_sha256']
    assert all(r['execution']['tool_results'][0]['output']['success'] for r in result['rounds'][:-1])
    counts = [len(read(Path(v['checkpoint']))['payload']['placed_instances']) for v in render_records]
    expected_counts = [9] if prior and args.resume_step == 'after_add' else ([8, 9] if prior else [9, 8, 9])
    assert counts == expected_counts
    history_text = canonical(read(env.steps_dir/f'phase_host/turn_{len(actions)-1:03d}/request.json'))
    assert 'Predetermined CPU integration fixture' not in history_text
    write(out/'observations.json', render_records)
    write(out/'fixture_backend_receipts.json', backend.observations)
    after = {str(p): sha(p) for p in source_files}
    write(out/'source_hashes_after.json', after)
    assert hashes == after
    write(out/'summary.json', {'status': 'passed_native_integration_fixture',
        'actual_native_calls': len(actions)-1, 'scripted_generation_attempts': 2*len(actions), 'real_model_calls': 0,
        'training_updates': 0, 'new_generated_rooms': 0, 'paid_api_calls': 0,
        'object_counts': counts, 'full_scene_success': None,
        'fresh_five_view_observations': len(actions), 'new_rendered_states': len(actions)-1,
        'resumed_fixture': str(args.resume_fixture) if prior else None,
        'inherited_counts_not_new_calls': {'native_calls': prior['native_calls'],
            'scripted_generation_attempts': prior['generation_attempts']} if prior else {},
        'native_step_and_memory_body_unchanged': True,
        'sdk_dispatch_scope': 'Existing direct native-kernel subprocess bridge; full SDK server not started',
        'non_furniture_stages_delegate_unchanged': delegated,
        'source_hashes_unchanged': len(hashes), 'elapsed_seconds': time.monotonic()-start,
        'peak_parent_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'not_proven': ['free policy generation', 'trained-policy transfer to the native prompt',
                       'full four-stage execution', 'whole-room design improvement']})
    print(canonical(read(out/'summary.json')), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume-fixture', type=Path)
    ap.add_argument('--resume-step', choices=('after_remove','after_add'), default='after_remove')
    ap.add_argument('--initial-views', type=Path)
    asyncio.run(main(ap.parse_args()))
