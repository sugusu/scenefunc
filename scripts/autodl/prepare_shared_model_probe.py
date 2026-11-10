"""Freeze a new four-forward local runtime protocol; encode before loading weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import socket
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=Path, required=True)
    args = p.parse_args()
    args.stage.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    work = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
    source = repo / 'runs/autodl/20261110_storage_observation_recovery/material_v4/components.jsonl'
    records = [json.loads(l) for l in source.read_text().splitlines()]
    component = next(r for r in records if r['unit_id'] == 'paper_table_modes_build_07'
                     and r['arm'] == 'separate_phase_v3' and r['stage'] == 'action')
    from phase_objective import PHASE_MODE
    messages = copy.deepcopy(component['messages'])
    assert messages[0]['content'].endswith(PHASE_MODE['action'])
    messages[0]['content'] = messages[0]['content'][:-len(PHASE_MODE['action'])]
    assert all(PHASE_MODE[s] not in messages[0]['content'] for s in PHASE_MODE)
    bindings = json.loads((source.parent / 'observation_bindings.json').read_text())
    binding = next(r for r in bindings if r['unit_id'] == component['unit_id'])
    paths = [c['image'] for m in messages if isinstance(m['content'], list)
             for c in m['content'] if c.get('type') == 'image']
    assert paths == [r['path'] for r in binding['images']] and len(paths) == 5
    images = [{'path': p, 'sha256': sha(p)} for p in paths]
    assert images == binding['images']
    state_path = repo / 'runs/autodl/20261110_native_observation_export/export_v2/actual_room_state.json'
    state = json.loads(state_path.read_text())
    state_hash = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert state_hash == binding['runtime_state_sha256']
    payload = {'messages': messages, 'images': images, 'state_sha256': state_hash}
    # Neither original target tokens/text nor any teacher decision are copied.
    dump(args.stage / 'current_observation.json', payload)
    tool_source = repo / 'runs/autodl/20261110_live_manager_request/actual_v1/model_actor/call_01_request_local_only.json'
    # Reuse a real native input, NEVER its old completion. This is a new local forward.
    tool_request = json.loads(tool_source.read_text())
    dump(args.stage / 'native_tool_input.json', tool_request)
    checkpoint = work / 'critic_public_runtime_20261104/checkpoint'
    verified = []
    tick = time.monotonic()
    for path in sorted(checkpoint.glob('*.safetensors')):
        record = json.loads((checkpoint.parent / f'acquisition_files.{path.name}.json').read_text())[0]
        observed = sha(path)
        assert observed == record['sha256'] and path.stat().st_size == record['bytes']
        st = path.stat()
        verified.append({'path': str(path), 'sha256': observed, 'bytes': st.st_size,
                         'mtime_ns': st.st_mtime_ns, 'inode': st.st_ino})
    assert len(verified) == 4
    code = [repo / 'scripts/autodl' / n for n in (
        'shared_native_model_host.py', 'prepare_shared_model_probe.py',
        'run_shared_model_probe.py', 'phase_policy.py', 'phase_deployment.py',
        'phase_objective.py', 'routed_multimodal.py')]
    code.append(work / 'paris_local_run_results_20261104/local_vlm_sdpa.py')
    protocol = {
        'status': 'frozen_before_model_inference',
        'checkpoint': str(checkpoint),
        'checkpoint_revision': '63e235a0c0d6e5bc2f8f3485214df4457fd2e672',
        'max_model_forwards': 4, 'max_total_output_tokens': 1536,
        'phase_total_output_tokens_per_condition': 512, 'decision_output_cap': 192,
        'phase_total_seconds_per_condition': 90, 'maximum_forward_seconds_soft': 90,
        'time_limit_semantics': 'HF max_time is soft; actor480/parent540 second deadlines are hard process limits',
        'max_actor_wall_seconds': 480, 'max_parent_wall_seconds': 540,
        'fixed_order': ['separate_direct_v3', 'native_tool', 'paired_phase_v3'],
        'phase_conditions': ['separate_direct_v3', 'paired_phase_v3'],
        'sampling': 'greedy; existing PhasePolicy seeds; no prompt or sample retries',
        'phase_input': str((args.stage / 'current_observation.json').resolve()),
        'phase_input_sha256': sha(args.stage / 'current_observation.json'),
        'tool_input': str((args.stage / 'native_tool_input.json').resolve()),
        'tool_input_sha256': sha(args.stage / 'native_tool_input.json'),
        'fixed_source_unit': component['unit_id'],
        'source_material': str(source), 'source_material_sha256': sha(source),
        'source_tool_request': str(tool_source), 'source_tool_request_sha256': sha(tool_source),
        'state_file': str(state_path), 'state_file_sha256': sha(state_path),
        'code_sha256': {str(p): sha(p) for p in code},
        'weight_sha256_verified': verified,
        'weight_verification_seconds': time.monotonic() - tick,
        'paid_API_calls': 0, 'training_steps': 0, 'policy_action_dispatch': False,
        'scientific_scope': 'Single shared-model deployment/runtime qualification. '
            'Existing declared source room, no new room, no preference truth, '
            'no learned method comparison, no native asset-manager rerun.',
        'old_protocols_reopened': False, 'automatic_retries': False}
    dump(args.stage / 'protocol.json', protocol)
    attempts = []
    def reject(*a, **k):
        attempts.append('blocked'); raise RuntimeError('Preflight networking prohibited')
    socket.socket.connect = reject
    socket.create_connection = reject
    sys.path.insert(0, str(work / 'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine
    from phase_deployment import deployment_input, decision_stop_ids
    from PIL import Image
    engine = LocalQwenEngine(checkpoint, args.stage / 'preflight', max_pixels=65536,
                             max_input_tokens=16384, max_new_tokens=512)
    decoded = []
    try:
        for row in images:
            with Image.open(row['path']) as im:
                decoded.append(im.convert('RGB'))
        lengths = {}
        for stage in ('action', 'decision'):
            batch = deployment_input(engine.processor, messages, decoded,
                                     arm='separate_phase_v3', stage=stage)
            lengths[stage] = {'input_tokens': int(batch['input_ids'].shape[1]),
                             'input_shapes': {k: list(v.shape) for k, v in batch.items()}}
            assert lengths[stage]['input_tokens'] + 512 < 16384
        native = copy.deepcopy(tool_request['messages'])
        if tool_request.get('response_format'):
            native.append({'role': 'user', 'content':
                'Return only one valid JSON object. Do not add markdown or commentary.'})
        _, tool_encoded = engine.encode(native)
        assert tool_encoded['input_tokens'] + 512 < 16384
        assert engine._model is None and not attempts
        dump(args.stage / 'preflight.json', {'status': 'passed', 'model_loaded': False,
            'phase_inputs': lengths, 'tool_input_tokens': tool_encoded['input_tokens'],
            'decision_stop_ids': decision_stop_ids(engine.processor.tokenizer),
            'paid_API_calls': 0, 'model_forwards': 0, 'network_attempts': 0,
            'future_paired_action_prefix': 'Not fabricated; must be built from actual generated D',
            'protocol_sha256': sha(args.stage / 'protocol.json')})
        print(json.dumps({'preflight': 'passed', 'phase_lengths': lengths,
                          'tool_tokens': tool_encoded['input_tokens'], 'model_loaded': False}))
    finally:
        for im in decoded:
            im.close()


if __name__ == '__main__':
    main()
