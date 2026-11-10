"""Execute pinned reward-call paths up to real CPU image/text encoding.

Deliberately stops before model.generate. No rankings/rewards are fabricated.
Legal flags and buffers below are transport fixtures, not physical assessments.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
import time
import types

from critic_task_context import bind_factory_import, make_task_conditioned_reward

WORK = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
TRAIN = WORK / 'context_credit_source_review_20261103/sources/train_grpo.py'
ADV = WORK / 'critic_scope_bridge_20261104/sources/src_critic/grpo_advantage.py'
CHECKPOINT = WORK / 'critic_public_runtime_20261104/checkpoint'
EXPECTED = {
    TRAIN: '2c4eba6af647abfe7bbb22ecb2d124b14dbc28c7d544caffdb43e3d460b6f479',
    ADV: '9243154a236f61db27f2dc673c6a77381f02c1c3b4b9906ba16eacf4d637afe6'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def native_factory():
    tree = ast.parse(TRAIN.read_text())
    fn_names = {'_count_valid_tool_calls', '_format_score', 'make_reward_critic'}
    var_names = {'_THINK_RE', '_TOOL_CALL_RE', '_TOOL_CALL_BODY_RE',
                 'STRICT_FORMAT_REWARD', 'CRITIC_LEGALITY_GATE', 'LEGALITY_FLOOR'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in fn_names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in var_names for t in node.targets):
            nodes.append(node)
    assert len(nodes) == len(fn_names) + len(var_names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {'Path': Path, 're': re, 'json': json, 'sys': sys}
    exec(compile(module, str(TRAIN), 'exec'), scope)
    factory = scope['make_reward_critic']
    calls = []
    for node in ast.walk(next(n for n in nodes if isinstance(n, ast.FunctionDef)
                              and n.name == 'make_reward_critic')):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
                'advantage_from_critic', 'advantage_from_critic_with_legality'):
            calls.append({'line': node.lineno, 'function': node.func.id,
                          'explicit_user_text': any(k.arg == 'user_text' for k in node.keywords),
                          'positional_arguments': len(node.args)})
    return factory, scope, sorted(calls, key=lambda x: x['line'])


class StopBeforeInference(BaseException):
    pass


class NoModel:
    accesses = 0

    def __getattr__(self, name):
        self.accesses += 1
        raise AssertionError('No model weights or inference allowed: ' + name)


def closure(fn, name):
    return dict(zip(fn.__code__.co_freevars,
                    [c.cell_contents for c in fn.__closure__ or ()]))[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    inputs = []
    for p, expected in EXPECTED.items():
        assert sha(p) == expected, p
        inputs.append({'path': str(p), 'sha256': expected})

    # Reject network instead of allowing processor configuration to fall back.
    attempts = []
    def reject_network(*args, **kwargs):
        attempts.append('rejected')
        raise RuntimeError('Local-only input capture; network disabled')
    socket.socket.connect = reject_network
    socket.create_connection = reject_network

    spec = importlib.util.spec_from_file_location('captured_native_advantage', ADV)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    factory, scope, source_calls = native_factory()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
    # Explicit common CPU capture budget, not a claim of author training config.
    processor.image_processor.max_pixels = 262144
    processor.image_processor.min_pixels = 3136

    previous_request = WORK / 'critic_scope_bridge_20261104/requests/task_forward/request.json'
    prior = read(previous_request)
    image_paths = [Path(c['image']) for c in prior[0]['content'] if c['type'] == 'image']
    assert len(image_paths) == 2
    task_source = WORK / 'paris_A_inference_20261104/request_B.json'
    original_task = read(task_source)[1]['content'][-1]['text'].split('## Memory')[0].strip()
    for path in [previous_request, task_source] + image_paths:
        inputs.append({'path': str(path), 'sha256': sha(path)})
    checkpoint_files = []
    for p in CHECKPOINT.iterdir():
        if p.is_file() and not p.name.endswith('.safetensors'):
            checkpoint_files.append({'path': str(p), 'sha256': sha(p)})

    # Three branches here reuse TWO existing real scene views: A,B,A.
    # This is solely a transport fixture, not three generated states or labels.
    dirs = []
    for i, source in enumerate([image_paths[0], image_paths[1], image_paths[0]]):
        p = args.output / f'branch_{i}'
        p.mkdir()
        (p / '_topdown.png').symlink_to(source)
        dirs.append(str(p))
    parent = args.output / 'parent'
    parent.mkdir()
    (parent / '_topdown.png').symlink_to(image_paths[0])
    model = NoModel()
    rows = []
    original_builder = native._build_chat_inputs
    active = {}

    def capture(processor_arg, paths, text):
        formatted, images = original_builder(processor_arg, paths, text)
        encoded = processor_arg(text=[formatted], images=images,
                                return_tensors='pt', padding=True)
        name = active['name']
        (args.output / f'{name}.prompt.txt').write_text(text)
        (args.output / f'{name}.formatted.txt').write_text(formatted)
        row = {'name': name, 'image_paths': list(map(str, paths)),
               'image_sha256': [sha(p) for p in paths],
               'image_count': len(paths), 'labels': list('ABCD'[:len(paths)]),
               'contains_original_task': (original_task in text or
                                         json.dumps(original_task, ensure_ascii=False) in text),
               'task_transport': ('exact JSON string' if
                   json.dumps(original_task, ensure_ascii=False) in text else
                   'raw string' if original_task in text else 'absent'),
               'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'formatted_sha256': hashlib.sha256(formatted.encode()).hexdigest(),
               'input_shapes': {k: list(v.shape) for k, v in encoded.items()},
               'input_ids_sha256': hashlib.sha256(encoded['input_ids'].numpy().tobytes()).hexdigest(),
               'image_grid_thw': encoded['image_grid_thw'].tolist(),
               'model_called': False, 'reward_computed': False}
        assert len(images) == len(paths) == len(row['image_grid_thw'])
        assert native.build_default_user_text(len(paths)) in text
        rows.append(row)
        raise StopBeforeInference()

    def legality_fixture(scene_dirs):
        assert len(scene_dirs) == 3
        return [True, False, True], {'scope': 'transport fixture, not geometry'}

    native._build_chat_inputs = capture
    try:
        for conditioned in (False, True):
            for path_kind in ('siblings', 'legal_subset', 'parent_child'):
                scope['CRITIC_LEGALITY_GATE'] = path_kind == 'legal_subset'
                audit = []
                # Fixture substitution ONLY for native physical flag queries.
                facade = types.SimpleNamespace(**{k: v for k, v in vars(native).items()
                                                  if not k.startswith('__')})
                facade.legality_flags_from_scenes = legality_fixture
                if conditioned:
                    reward = make_task_conditioned_reward(
                        factory, facade, model, processor, audit=audit)
                else:
                    reward = bind_factory_import(factory, facade)(model, processor)
                active['name'] = ('task_' if conditioned else 'native_') + path_kind
                buffer = types.SimpleNamespace(prompt=original_task, branch_groups=[
                    types.SimpleNamespace(completions=['<think>inspect</think>'] * 3,
                        branch_scene_dirs=dirs, parent_scene_dir=str(parent), step_idx=0)])
                caught = False
                try:
                    if path_kind == 'parent_child':
                        inner = reward.native_reward if conditioned else reward
                        improvement = closure(closure(inner, '_score_buffer'), '_improvement')
                        token = reward.task_scope.set(original_task) if conditioned else None
                        try:
                            improvement(str(parent), [str(Path(dirs[1]) / '_topdown.png')], [0], 1)
                        finally:
                            if conditioned:
                                reward.task_scope.reset(token)
                    else:
                        reward(buffer)
                except StopBeforeInference:
                    caught = True
                assert caught, 'Expected to stop at actual encoding, before model call'
                assert rows[-1]['contains_original_task'] is conditioned
                assert rows[-1]['image_count'] == (3 if path_kind == 'siblings' else 2)
                rows[-1]['scope'] = ('Isolated native _improvement closure; not a completed reward pass'
                                    if path_kind == 'parent_child' else
                                    'Native reward-factory path interrupted before first Critic inference')
                rows[-1]['legal_flags_are_fixture'] = path_kind == 'legal_subset'
                if conditioned:
                    assert reward.task_scope.get() is None
                    assert len(audit) == 1
                    rows[-1]['adapter_audit'] = audit
        bad = types.SimpleNamespace(prompt='', branch_groups=[])
        missing_task_rejected = False
        try:
            reward(bad)
        except ValueError:
            missing_task_rejected = True
        assert missing_task_rejected
    finally:
        native._build_chat_inputs = original_builder
    for item in inputs + checkpoint_files:
        assert sha(item['path']) == item['sha256']
    assert model.accesses == 0 and not attempts
    previous_results = WORK / 'critic_scope_bridge_20261104/model_results.json'
    dump(args.output / 'capture.json', {
        'source_pin': 'a0565e50ed7d32c0e09132b9311c0b9b91165405',
        'source_call_sites': source_calls, 'rows': rows,
        'inputs_unchanged': inputs, 'processor_files_unchanged': checkpoint_files,
        'model_attribute_accesses': model.accesses, 'network_attempts': len(attempts),
        'model_loaded': False, 'model_calls': 0, 'training_steps': 0,
        'new_generated_scenes': 0, 'missing_task_rejected': missing_task_rejected,
        'previous_context_probe_reused_only': {'path': str(previous_results),
            'sha256': sha(previous_results), 'new_inference': False},
        'limits': ['Synthetic buffers and legal flags; no physical-gate execution',
                   'Three-branch transport uses two real images with one duplicated',
                   'CPU processor encoding only, no Critic answers or preferences',
                   'Factory globals snapshotted at binding; rebind after flag changes',
                   'No statement about paper training runs or overall model capability'],
        'seconds': time.monotonic() - start})
    print(json.dumps({'captures': len(rows), 'model_calls': 0,
                      'output': str(args.output), 'seconds': time.monotonic() - start}))


if __name__ == '__main__':
    main()
