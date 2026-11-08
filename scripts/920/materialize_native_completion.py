"""Bounded CPU candidate preparation using native OTA completion; no training."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from native_completion_supervision import native_request, completion_content


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def locked(reference):
    require(sha(reference['path']) == reference['sha256'], 'Changed source: ' + reference['path'])
    return read(reference['path'])


def prepare(plan):
    from native_completion_supervision import build_native_request, NATIVE_PROTOCOL, OLD_PROTOCOL
    output = Path(plan['output']) / 'corpus'
    require(not output.exists(), 'Candidate already exists; do not overwrite')
    source = read(plan['tool_dataset'])
    terminals = read(plan['terminal_dataset'])
    require(len(source['rows']) == 22 and len(terminals['rows']) == 2, 'Frozen sample count changed')
    rows = []
    for row, terminal in [(r, False) for r in source['rows']] + [(r, True) for r in terminals['rows']]:
        inputs = locked(row['input'] if terminal else row['execution_inputs'])
        old_request = locked(row['request'] if terminal else row['requests']['action_only'])
        request, accounting = build_native_request(**inputs)
        require(request == native_request(old_request), 'Deployment input changed beyond protocol')
        require(request[1:] == old_request[1:], 'User/state/history/knowledge/images changed')
        require(request[0]['content'].replace(NATIVE_PROTOCOL, OLD_PROTOCOL) == old_request[0]['content'], 'Unexpected system change')
        for im in row['images']:
            require(sha(im['path']) == im['sha256'], 'Observation changed')
        dest = output / row['id']
        write(dest / 'request.json', request)
        documents = {}
        target = locked(row['target']) if terminal else None
        for arm in ('action_only', 'decision_action'):
            if not terminal:
                require(locked(row['requests'][arm]) == old_request, 'Arm deployment inputs differ')
            document = locked(row['document'] if terminal else row['documents'][arm])
            original = copy.deepcopy(document)
            if terminal:
                document['turns'][0]['content'] = completion_content(target)
                require(document['turns'][0].get('tool_calls') == [], 'Terminal has a tool call')
                document['provenance'].update(serialization='native-ota-completion-v1',
                    official_ota_serialization_overridden=False)
            document['provenance']['prospective_deployment_protocol'] = 'native-completion-v1'
            require(document['task_context'] == original['task_context'], 'Task context changed')
            if not terminal:
                require(document['turns'] == original['turns'], 'Action or reason target changed')
            path = dest / arm / 'episode.json'
            write(path, document)
            documents[arm] = ref(path)
        require(not target or completion_content(target) not in json.dumps(request, ensure_ascii=False), 'Completion target leaked')
        rows.append(dict(id=row['id'], family_id=row['family_id'],
            target_kind='terminal_control' if terminal else 'tool_action', documents=documents,
            request=ref(dest / 'request.json'), input=row['input'] if terminal else row['execution_inputs'],
            images=row['images'], target=target, source=row, history_accounting=accounting))
    write(output / 'DATASET.json', dict(schema='native-completion-candidate-v1', rows=rows,
        counts=dict(examples=24, tool_actions=22, terminal_states=2, normalization_groups=6),
        model_calls=0, training_updates=0, new_professional_decisions=0,
        prompt_change='Only the shared completion protocol sentence, identically in all arms.',
        source_datasets=[ref(plan['tool_dataset']), ref(plan['terminal_dataset'])]))
    print('Prepared24native-completion candidates; shared deployment roundtrip passed', flush=True)


def encode(plan):
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import torch
    from transformers import AutoProcessor
    from edit_supervision import encode_request_targets
    from native_completion_supervision import encode_native_completion
    from terminal_training_dataset import CandidateEditExamples
    from train_edit_learning_study import normalization_check
    torch.set_num_threads(2)
    source = Path(plan['official_source'])
    sys.path.insert(0, str(source / 'src'))
    from scene_observation import configure_image_budget
    spec = importlib.util.spec_from_file_location('native_completion_official', source / 'src/sft_dataset.py')
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    processor = AutoProcessor.from_pretrained(plan['model'], local_files_only=True, trust_remote_code=False)
    configure_image_budget(processor, plan['max_pixels'])
    collator = official.StepwiseCollator(processor.tokenizer.pad_token_id)
    runtime = Path(plan['runtime'])
    require(not runtime.exists(), 'Refuse to overwrite tensors')
    runtime.mkdir(parents=True)
    corpus = Path(plan['output']) / 'corpus/DATASET.json'
    data = read(corpus)
    old = {(r['arm'], r['id']): r for r in read(plan['old_encoding'])['records']}
    records = []
    targets = []
    for row in data['rows']:
        request = locked(row['request'])
        examples = []
        pair = []
        for arm in ('action_only', 'decision_action'):
            doc = locked(row['documents'][arm])
            samples = official.build_step_samples(Path(row['documents'][arm]['path']), Path(plan['repo_root']))
            require(len(samples) == 1, 'Official loader lost sample')
            terminal = row['target_kind'] == 'terminal_control'
            if terminal:
                ex, control, decision, boundary = encode_native_completion(request, samples[0], row['target'], processor, official, plan['max_sequence_tokens'])
            else:
                ex, control, decision, boundary = encode_request_targets(request, samples[0], processor, official, plan['max_sequence_tokens'])
            previous = old[arm, row['id']]
            require(sha(previous['tensor_path']) == previous['tensor_sha256'], 'Historical tensor changed')
            old_batch = torch.load(previous['tensor_path'], map_location='cpu', weights_only=True)
            old_labels = old_batch['labels'].reshape(-1)
            old_control = old_batch['action_mask'].reshape(-1)
            old_decision = old_batch['decision_mask'].reshape(-1)
            if not terminal:
                require(torch.equal(ex['labels'][control], old_labels[old_control]), 'Tool target changed')
                require(torch.equal(ex['labels'][decision], old_labels[old_decision]), 'Reason target changed')
            require(torch.equal(ex['pixel_values'], old_batch['pixel_values']), 'Image pixels changed')
            require(torch.equal(ex['image_grid_thw'], old_batch['image_grid_thw']), 'Image grid changed')
            require(bool((ex['labels'][:boundary] == -100).all()), 'Input/receipt supervision leakage')
            require(torch.equal(control | decision, ex['labels'] != -100) and not bool((control & decision).any()), 'Target coverage mismatch')
            decoded_prompt = processor.tokenizer.decode(ex['input_ids'][:boundary], skip_special_tokens=False)
            if doc['turns'][0]['content']:
                require(doc['turns'][0]['content'] not in decoded_prompt, 'Gold answer in input')
            batch = collator([ex])
            batch['action_mask'] = control.unsqueeze(0)
            batch['decision_mask'] = decision.unsqueeze(0)
            path = runtime / arm / (row['id'] + '.pt')
            path.parent.mkdir(exist_ok=True)
            torch.save(batch, path)
            thash = lambda t: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
            record = dict(id=row['id'], arm=arm, family_id=row['family_id'], target_kind=row['target_kind'],
                tensor_path=str(path), tensor_sha256=sha(path), input_tokens=len(ex['input_ids']),
                prompt_tokens=boundary, action_tokens=int(control.sum()), decision_tokens=int(decision.sum()),
                tool_tokens=0 if terminal else int(control.sum()), terminal_tokens=int(control.sum()) if terminal else 0,
                prompt_sha256=thash(ex['input_ids'][:boundary]), action_target_sha256=thash(ex['labels'][control]),
                deployment_prompt_exact=True, input_receipts_masked=True, pixels_unchanged=True,
                historical_control_target_unchanged=not terminal)
            records.append(record)
            pair.append(record)
            examples.append(ex)
            if terminal:
                targets.append(dict(id=row['id'], arm=arm, previous_control_tokens=previous['action_tokens'],
                    new_control_tokens=int(control.sum()), text=processor.tokenizer.decode(ex['labels'][control], skip_special_tokens=False)))
        require(pair[0]['prompt_sha256'] == pair[1]['prompt_sha256'], 'Arm prompt mismatch')
        require(pair[0]['action_target_sha256'] == pair[1]['action_target_sha256'], 'Arm control target mismatch')
        padded = collator(examples)
        require(bool((padded['labels'][padded['attention_mask'] == 0] == -100).all()), 'Padding supervised')
        print(row['id'], 'official CPU encoding passed for both arms', flush=True)
    consumption = {}
    for arm in ('action_only', 'decision_action'):
        selected = [r for r in records if r['arm'] == arm]
        consumption[arm] = normalization_check(CandidateEditExamples(selected, expected_examples=24, expected_families=6), runtime / 'trainer_fixture' / arm)
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    totals = {arm: {key: sum(r[key] for r in records if r['arm'] == arm)
                   for key in ('input_tokens', 'action_tokens', 'tool_tokens', 'terminal_tokens', 'decision_tokens')}
              for arm in consumption}
    write(Path(plan['output']) / 'ENCODING.json', dict(status='official_serialization_and_CPU_Trainer_consumption_passed',
        dataset=ref(corpus), records=records, totals=totals, terminal_targets=targets,
        trainer_consumption=consumption, model_loaded=False, model_calls=0, training_updates=0,
        cuda_initialized=False, real_backward=False, training_authorized=False,
        limits='Candidate preparation only. Official sample builder, target serializer and collator; shared deployment prefix replaces legacy prompt. Native-format completion remains researcher control supervision, not a sourced decision. No model rollout or learning benefit demonstrated.'))
    print('PASS48CPUrecords;0model/optimizer/realbackward', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'encode'], required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    require(plan['model_calls'] == plan['training_updates'] == plan['scene_edits'] == 0, 'CPU-only scope required')
    for path, digest in plan['inputs'].items():
        require(sha(path) == digest, 'Frozen input changed: ' + path)
    print('Start bounded native-completion candidate ' + args.phase, flush=True)
    (prepare if args.phase == 'prepare' else encode)(plan)


if __name__ == '__main__':
    main()
