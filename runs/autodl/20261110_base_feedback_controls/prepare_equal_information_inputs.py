"""Prepare same-evidence format controls; encode on CPU without model loading.

The author's before-state memory exposes earlier public poses omitted from the
old compact JSON. Add those same past observations to BOTH future inputs. Do
not mislabel the previous run as a matched pure-format comparison.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
WORK = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')


def main():
    start = time.monotonic()
    out = OUT/'input_v2'
    out.mkdir(exist_ok=False)
    source = WORK/'native_execution_inventory_20261104/source/Designer/agent_rollout.py'
    pin = read(OUT/'audit_v1/source_receipt.json')
    assert sha(source) == pin['rollout_sha256']
    fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'build_messages')
    scope = {'Path': Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    run = REPO/'runs/autodl/20261110_bounded_base_execution/recovery_v1'
    past = []
    prepared = []
    for t in sorted(run.glob('turn_*')):
        req = read(t/'request.json')
        brief, raw = req[-1]['content'][-1]['text'].split('\nCurrent evidence:\n', 1)
        obs = read(t/'request.json')
        common = json.loads(raw)
        common['previous_observed_target_poses'] = list(past)
        exact = json.dumps(common, ensure_ascii=False, separators=(',', ':'))
        direct = read(t/'request.json')
        direct[-1]['content'][-1]['text'] = brief+'\nCurrent evidence:\n'+exact
        memory = (OUT/'audit_v1'/t.name/'native_memory_before.txt').read_text().rstrip()
        images = [Path(i['image']) for i in req[-1]['content'] if i['type'] == 'image']
        native = scope['build_messages'](req[0]['content'], brief,
            memory+'\n\nExact common evidence:\n'+exact, images)
        assert direct[0] == native[0]
        assert direct[-1]['content'][:-1] == native[-1]['content'][:-1]
        assert json.loads(direct[-1]['content'][-1]['text'].split('\nCurrent evidence:\n')[1]) == common
        assert json.loads(native[-1]['content'][-1]['text'].split('\nExact common evidence:\n')[1]) == common
        dump(out/(t.name+'_direct.json'), direct)
        dump(out/(t.name+'_native_format.json'), native)
        prepared.append(t.name)
        target = read(t/'calls.json')[0]['arguments']['object_id']
        obj = next(o for o in common['current_objects'] if o['object_id'] == target)
        past.append({'source_turn': t.name, 'source_request_sha256': sha(t/'request.json'),
                     'object_id': target, 'transform': obj['transform']})

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    attempted = []
    def reject(*args, **kwargs):
        attempted.append(True)
        raise RuntimeError('No network allowed in CPU input encoding')
    socket.socket.connect = reject
    socket.create_connection = reject
    sys.path.insert(0, str(WORK/'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine
    import torch
    torch.set_num_threads(1)
    engine = LocalQwenEngine(WORK/'critic_public_runtime_20261104/checkpoint', out/'unused_model_calls',
                            max_pixels=65536, max_input_tokens=8192, max_new_tokens=1024)
    rows = []
    for name in prepared:
        first, a = engine.encode(read(out/(name+'_direct.json')))
        second, b = engine.encode(read(out/(name+'_native_format.json')))
        assert engine._model is None
        assert a['images'] == b['images']
        assert torch.equal(first['pixel_values'], second['pixel_values'])
        assert torch.equal(first['image_grid_thw'], second['image_grid_thw'])
        rows.append({'unit': name, 'direct_input_tokens': a['input_tokens'],
            'native_format_input_tokens': b['input_tokens'], 'same_image_tensors': True,
            'direct_request_sha256': sha(out/(name+'_direct.json')),
            'native_request_sha256': sha(out/(name+'_native_format.json')),
            'image_receipts': a['images']})
        del first, second
    result = {'status': 'same_evidence_inputs_encoded_no_model_loaded', 'rows': rows,
        'cpu_encodings': 6, 'new_model_calls': 0, 'training_updates': 0, 'paid_api_calls': 0,
        'network_attempts': len(attempted), 'cuda_initialized': torch.cuda.is_initialized(),
        'model_loaded': engine._model is not None, 'seconds': time.monotonic()-start,
        'source_sha256': sha(Path(__file__)),
        'scope': 'Same raw common evidence and images, native text derived from public prior observations. More formatting tokens are real cost. Old saved model run is NOT a matched pure-format arm.',
        'not_supported': 'This keeps the restricted custom system and unannotated images; not a full official prompt/observation reproduction. No behavior gain measured.'}
    assert not result['model_loaded'] and not result['cuda_initialized']
    dump(out/'result.json', result)
    print(json.dumps({k: result[k] for k in ['status','cpu_encodings','new_model_calls','seconds','cuda_initialized']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
