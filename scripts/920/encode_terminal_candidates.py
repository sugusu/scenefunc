"""CPU-only terminal encoding and combined-corpus Trainer consumption.

No model weights, optimizer, real backward, generation or training entry point.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return dict(path=str(path), sha256=sha(path))
def require(ok, message):
    if not ok: raise ValueError(message)
def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    a = ap.parse_args(); plan = read(a.plan)
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import torch
    from transformers import AutoProcessor
    from terminal_supervision import encode_terminal
    from terminal_training_dataset import CandidateEditExamples
    from train_edit_learning_study import normalization_check
    torch.set_num_threads(2)
    source = Path(plan['official_source']); sys.path.insert(0, str(source/'src'))
    from scene_observation import configure_image_budget
    spec = importlib.util.spec_from_file_location('terminal_official', source/'src/sft_dataset.py')
    official = importlib.util.module_from_spec(spec); sys.modules[spec.name] = official; spec.loader.exec_module(official)
    processor = AutoProcessor.from_pretrained(plan['model'], local_files_only=True, trust_remote_code=False)
    configure_image_budget(processor, plan['budget']['max_pixels'])
    root = Path(plan['output']); runtime = Path(plan['runtime'])/'encoding'
    require(not runtime.exists(), 'Refuse to overwrite encoding'); runtime.mkdir()
    data = read(root/'terminal_corpus/DATASET.json'); old = read(plan['base_encoding'])
    records = []; examples = []; checks = []
    collator = official.StepwiseCollator(processor.tokenizer.pad_token_id)
    for row in data['rows']:
        for key in ('input', 'request', 'target', 'document', 'measured_evidence', 'state', 'view_binding'):
            require(sha(row[key]['path']) == row[key]['sha256'], 'Changed '+key)
        for image in row['images']: require(sha(image['path']) == image['sha256'], 'Changed image')
        samples = official.build_step_samples(Path(row['document']['path']), Path(plan['repo_root']))
        require(len(samples) == 1, 'Official loader lost observed terminal turn')
        request, target = read(row['request']['path']), read(row['target']['path'])
        ex, control, decision, boundary = encode_terminal(request, samples[0], target, processor,
            max_length=plan['budget']['max_sequence_tokens'])
        require(bool((ex['labels'][:boundary] == -100).all()) and not bool(decision.any()), 'Prompt leakage or spurious decision target')
        examples.append(ex)
        batch = collator([ex]); batch['action_mask'] = control.unsqueeze(0); batch['decision_mask'] = decision.unsqueeze(0)
        path = runtime/(row['id']+'.pt'); torch.save(batch, path)
        for arm in ('action_only', 'decision_action'):
            records.append(dict(id=row['id'], arm=arm, family_id=row['family_id'],
                target_kind='terminal_control', tensor_path=str(path), tensor_sha256=sha(path),
                input_tokens=len(ex['input_ids']), prompt_tokens=boundary, action_tokens=int(control.sum()),
                tool_tokens=0, terminal_tokens=int(control.sum()), decision_tokens=0,
                prompt_sha256=hashlib.sha256(ex['input_ids'][:boundary].numpy().tobytes()).hexdigest(),
                action_target_sha256=hashlib.sha256(ex['labels'][control].numpy().tobytes()).hexdigest(),
                deployment_prompt_exact=True, input_receipts_masked=True, shared_tensor_across_arms=True))
        checks.append(row['id']+':official_sample_and_explicit_done_serialization')
        print(row['id'], 'tokens', len(ex['input_ids']), 'terminal', int(control.sum()), flush=True)
    padded = collator(examples)
    require(bool((padded['labels'][padded['attention_mask']==0] == -100).all()), 'Padding supervised')
    combined = [dict(r, target_kind='tool_action', tool_tokens=r['action_tokens'], terminal_tokens=0) for r in old['records']] + records
    for r in old['records']: require(sha(r['tensor_path']) == r['tensor_sha256'], 'Old tensor changed')
    consumption = {}
    for arm in ('action_only', 'decision_action'):
        selected = [r for r in combined if r['arm'] == arm]
        ds = CandidateEditExamples(selected, expected_examples=24, expected_families=6)
        consumption[arm] = normalization_check(ds, runtime/'trainer_fixture'/arm)
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    totals = {arm: {k:sum(r[k] for r in combined if r['arm']==arm)
                   for k in ('input_tokens','action_tokens','tool_tokens','terminal_tokens','decision_tokens')}
              for arm in consumption}
    write(root/'ENCODING.json', dict(schema='scenefunc-edit-plus-terminal-encoding-v1', records=combined,
        base_encoding=ref(plan['base_encoding']), terminal_dataset=ref(root/'terminal_corpus/DATASET.json'),
        totals=totals, trainer_consumption=consumption, checks=checks,
        new_encoded_observations=2, old_observations_reencoded=0,
        objective='Project-balanced control/action NLL plus decision NLL where present. Terminal control shares the action objective; it is not a professional-decision target.',
        normalization_change='Waterloo control denominator now includes two terminal examples. Both arms change identically; this is not an equal-total-supervised-token mechanism comparison.',
        model_calls=0, training_updates=0, real_backward=False, training_authorized=False))
    print('PASS combined CPU Trainer consumption; no training', flush=True)


if __name__ == '__main__':
    main()
