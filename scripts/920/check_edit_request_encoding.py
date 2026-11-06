"""Check canonical requests with the fixed processor and official OTA targets."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists(): raise ValueError('Output exists')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import torch
    from transformers import AutoProcessor
    from edit_supervision import encode_request_targets
    torch.set_num_threads(2)
    plan = read(a.study/'ENCODING_PLAN.json')
    source = Path(plan['official_source'])
    sys.path.insert(0, str(source/'src'))
    from scene_observation import configure_image_budget
    spec = importlib.util.spec_from_file_location('canonical_official', source/'src/sft_dataset.py')
    up = importlib.util.module_from_spec(spec); sys.modules[spec.name] = up; spec.loader.exec_module(up)
    processor = AutoProcessor.from_pretrained(plan['model'], local_files_only=True, trust_remote_code=False)
    configure_image_budget(processor, plan['budget']['max_pixels'])
    data = read(a.study/'DATASET.json'); cache = read(a.study/'ENCODING.json'); manifest = read(a.requests)
    assert manifest['status'] == 'passed'
    results, pairs = [], {}
    print(datetime.now(timezone.utc).isoformat(), 'begin44canonicalCPUencodings; no model weights', flush=True)
    for row in manifest['training']:
        print(row['id'], row['arm'], 'encoding', flush=True)
        assert sha(row['request']) == row['request_sha256']
        original = next(x for x in data['rows'] if x['id'] == row['id'])['documents'][row['arm']]
        assert sha(original['path']) == original['sha256']
        samples = up.build_step_samples(Path(original['path']), a.study.resolve().parents[3])
        assert len(samples) == 1
        ex, am, dm, boundary = encode_request_targets(read(row['request']), samples[0], processor, up,
                                                     max_length=plan['budget']['max_sequence_tokens'])
        old = next(x for x in cache['records'] if x['id'] == row['id'] and x['arm'] == row['arm'])
        assert sha(old['tensor_path']) == old['tensor_sha256']
        batch = torch.load(old['tensor_path'], map_location='cpu', weights_only=True)
        assert torch.equal(ex['labels'][am], batch['labels'][0][batch['action_mask'][0]])
        assert torch.equal(ex['labels'][dm], batch['labels'][0][batch['decision_mask'][0]])
        assert torch.equal(ex['pixel_values'], batch['pixel_values'])
        assert torch.equal(ex['image_grid_thw'], batch['image_grid_thw'])
        assert bool((ex['labels'][:boundary] == -100).all())
        assert bool((ex['labels'][boundary:] != -100).all())
        prompt_hash = hashlib.sha256(ex['input_ids'][:boundary].numpy().tobytes()).hexdigest()
        pairs.setdefault(row['id'], []).append(prompt_hash)
        results.append(dict(id=row['id'], arm=row['arm'], input_tokens=len(ex['input_ids']), prompt_tokens=boundary,
                            action_tokens=int(am.sum()), decision_tokens=int(dm.sum()),
                            deployment_prompt_exact=True, original_target_tokens_exact=True, original_pixels_exact=True))
    assert len(results) == 44 and all(len(v) == 2 and len(set(v)) == 1 for v in pairs.values())
    assert not torch.cuda.is_initialized()
    result = dict(status='passed', recorded_at=datetime.now(timezone.utc).isoformat(), examples=len(results),
                  maximum_sequence_tokens=max(x['input_tokens'] for x in results),
                  sequence_limit=plan['budget']['max_sequence_tokens'], records=results,
                  request_manifest_sha256=sha(a.requests),
                  code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['edit_supervision.py', 'check_edit_request_encoding.py']},
                  model_loaded=False, cuda_initialized=False, training_updates=0, model_calls=0,
                  limits=['No weights updated or predictions produced.', 'Only training observations tokenized; deployment token budget not yet verified.',
                          'Common request builder removes serialization differences, not task-distribution or professional-supervision limitations.'])
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','examples','maximum_sequence_tokens','sequence_limit','model_calls','training_updates']}), flush=True)


if __name__ == '__main__': main()
