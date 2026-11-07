"""Verify corrected candidate targets with real offline processor; no model."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    require(not a.out.exists(), 'Use a fresh result path')
    plan = read(a.plan)
    for p, expected in plan['input_hashes'].items():
        require(sha(p) == expected, f'Frozen input changed: {p}')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import torch
    from transformers import AutoProcessor
    from edit_supervision import encode_request_targets
    torch.set_num_threads(2)
    source = Path(plan['official_source'])
    sys.path.insert(0, str(source / 'src'))
    from scene_observation import configure_image_budget
    spec = importlib.util.spec_from_file_location('candidate_official', source / 'src/sft_dataset.py')
    up = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = up
    spec.loader.exec_module(up)
    processor = AutoProcessor.from_pretrained(plan['model'], local_files_only=True, trust_remote_code=False)
    configure_image_budget(processor, plan['budget']['max_pixels'])
    data = read(plan['candidate_dataset'])
    cache = read(plan['historical_encoding'])
    records, pairs = [], {}
    changed = {'paper_table_modes_build_09', 'waterloo_edit_03', 'storage_edit_01', 'fhouse_perimeter_01'}
    print(datetime.now(timezone.utc).isoformat(), 'encode44correctedCandidatesCPU; no weights', flush=True)
    for row in data['rows']:
        for arm, ref in row['documents'].items():
            print(row['id'], arm, flush=True)
            req = row['requests'][arm]
            require(sha(ref['path']) == ref['sha256'] and sha(req['path']) == req['sha256'], 'Candidate changed')
            doc = read(ref['path'])
            samples = up.build_step_samples(Path(ref['path']), Path(plan['repo_root']))
            require(len(samples) == 1, 'Expected one official sample')
            expected = json.dumps(row['decision_target'], ensure_ascii=False) if arm == 'decision_action' and row['decision_target'] else ''
            require(samples[0].assistant_content == expected, 'Official sample lost/changed decision')
            ex, am, dm, boundary = encode_request_targets(read(req['path']), samples[0], processor, up,
                                                        max_length=plan['budget']['max_sequence_tokens'])
            old = next(x for x in cache['records'] if x['id'] == row['id'] and x['arm'] == arm)
            require(sha(old['tensor_path']) == old['tensor_sha256'], 'Historical tensors changed')
            batch = torch.load(old['tensor_path'], map_location='cpu', weights_only=True)
            require(torch.equal(ex['labels'][am], batch['labels'][0][batch['action_mask'][0]]), 'Action target changed')
            require(torch.equal(ex['pixel_values'], batch['pixel_values']), 'Image pixels changed')
            require(torch.equal(ex['image_grid_thw'], batch['image_grid_thw']), 'Image grid changed')
            old_d = batch['labels'][0][batch['decision_mask'][0]]
            same_d = torch.equal(ex['labels'][dm], old_d)
            should_change = arm == 'decision_action' and row['id'] in changed
            require(same_d != should_change, 'Unexpected decision token change')
            require(bool(dm.any()) == bool(expected), 'Decision supervision missing/unexpected')
            decoded = processor.tokenizer.decode(ex['labels'][dm], skip_special_tokens=False)
            require(not expected or expected in decoded, 'Decision text absent from supervised tokens')
            require(bool((ex['labels'][:boundary] == -100).all()), 'Input supervised')
            require(bool((ex['labels'][boundary:] != -100).all()), 'Unsupervised assistant gap')
            ph = hashlib.sha256(ex['input_ids'][:boundary].numpy().tobytes()).hexdigest()
            pairs.setdefault(row['id'], []).append(ph)
            records.append(dict(id=row['id'], arm=arm, sequence_tokens=len(ex['input_ids']),
                prompt_tokens=boundary, action_tokens=int(am.sum()), decision_tokens=int(dm.sum()),
                historical_decision_tokens=int(batch['decision_mask'].sum()),
                decision_target_changed=not same_d, decision_text=expected,
                action_targets_exact=True, pixels_exact=True, prompt_hash=ph))
    require(len(records) == 44 and all(len(v) == 2 and len(set(v)) == 1 for v in pairs.values()), 'Arm prompts differ')
    require(sum(bool(x['decision_tokens']) for x in records) == 8, 'Decision position count differs')
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    result = dict(status='passed', recorded_at=datetime.now(timezone.utc).isoformat(),
        candidate_dataset_sha256=sha(plan['candidate_dataset']), plan_sha256=sha(a.plan),
        records=records, examples=44, decision_positions=8,
        changed_decision_documents=sum(x['decision_target_changed'] for x in records),
        maximum_sequence_tokens=max(x['sequence_tokens'] for x in records),
        budget=plan['budget'], model_loaded=False, cuda_initialized=False, model_calls=0, training_updates=0,
        limits=['Encoding proves target inclusion and input isolation, not learning or professional quality.',
                'No new actions, source projects, human review, no-edit target or training authorization.'])
    with a.out.open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'examples', 'decision_positions', 'changed_decision_documents', 'maximum_sequence_tokens']}), flush=True)


if __name__ == '__main__':
    main()
