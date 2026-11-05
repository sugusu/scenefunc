"""Measure a locked multi-root construction corpus with the official CPU loader.

No model, forward, optimizer or training entry point is provided. Input tensors
are inspected in memory; the public result contains hashes and token counts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    repo, source = Path(plan['repo']), Path(plan['official_source'])
    for name, digest in plan['code_sha256'].items():
        require(sha(Path(__file__).parent / name) == digest, 'Code changed: ' + name)
    for name, digest in plan['official_source_hashes'].items():
        require(sha(source / name) == digest, 'Loader changed: ' + name)
    for name, digest in plan['processor_hashes'].items():
        require(sha(Path(plan['model']) / name) == digest, 'Processor changed: ' + name)
    index_path = repo / plan['index']['path']
    require(sha(index_path) == plan['index']['sha256'], 'Index changed')
    index = read(index_path)
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    import torch
    from transformers import AutoProcessor
    from decision_pairing import mask_role, validate_triplet
    torch.set_num_threads(4)
    sys.path.insert(0, str(source / 'src'))
    spec = importlib.util.spec_from_file_location('locked_corpus_dataset', source / 'src/sft_dataset.py')
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    from scene_observation import configure_image_budget
    processor = AutoProcessor.from_pretrained(plan['model'], local_files_only=True)
    configure_image_budget(processor, plan['budget']['max_pixels'])
    tensor_sha = lambda t: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
    cases, records = [], []
    for case in index['cases']:
        manifest_path = repo / case['candidate_manifest']
        require(sha(manifest_path) == case['candidate_manifest_sha256'], 'Manifest changed')
        data, root = read(manifest_path), Path(case['server_input_root'])
        encoded, docs, row_for_path = {}, {}, {}
        for row in data['rows']:
            for ref in row['documents'].values():
                row_for_path[ref['path']] = row
            require(len(row['images']) == 5, 'Expected five observed views')
            for ref in row['images']:
                require(sha(root / ref['path']) == ref['sha256'], 'Observed image changed')
        for refs in data['arms'].values():
            for ref in refs:
                path = root / ref['path']
                require(sha(path) == ref['sha256'], 'Episode changed')
                if ref['path'] in encoded:
                    require(encoded[ref['path']]['role'] == ref['supervised_role'], 'Conflicting role')
                    continue
                doc = read(path)
                row = row_for_path[ref['path']]
                require(doc['turns'][0]['images'] == [x['path'] for x in row['images']], 'View order changed')
                ds = upstream.StepwiseSFTDataset(path.parent, root, processor,
                    max_length=plan['budget']['max_sequence_tokens'])
                require(len(ds) == 1, 'Expected isolated sample')
                raw = ds[0]
                ex = mask_role(raw, processor.tokenizer, ref['supervised_role'])
                positions = (ex['labels'] != -100).nonzero().flatten()
                target = ex['labels'][positions]
                require(len(target) > 0 and len(ex['image_grid_thw']) == 5, 'Missing targets or views')
                require(torch.equal(target, ex['input_ids'][positions]), 'Target IDs changed')
                require(bool((ex['labels'][raw['labels'] == -100] == -100).all()), 'Prompt supervised')
                decoded = processor.tokenizer.decode(ex['input_ids'], skip_special_tokens=False)
                require('EVAL_ONLY_SOURCE_CONSTRUCTION' not in decoded, 'Evaluation sentinel leaked')
                require(doc['evaluation_only']['after_state'] not in decoded, 'Future state path leaked')
                context = json.loads(doc['task_context']['scene_description'])
                require(doc['task_context']['scene_description'] in decoded, 'Task context lost')
                require(all(k in context for k in ['original_request', 'objects', 'asset_catalog',
                    'recent_actual_tool_receipts', 'physical_unknowns']), 'Required context missing')
                if case['case'] == 'paper_table_modes':
                    expected = ('furnish_everyday' if row['step'] <= 6 else
                                'prepare_guest_reserve' if row['step'] <= 8 else 'restore_everyday')
                    require(context['activity_context']['phase'] == expected, 'Wrong activity phase')
                category = ('reason' if ref['supervised_role'] == 'reason' else
                            'source_action' if row['source_principle_target'] else 'ordinary_action')
                record = dict(case=case['case'], step=row['step'], path=ref['path'],
                    role=ref['supervised_role'], category=category, input_tokens=len(ex['input_ids']),
                    supervised_tokens=len(target), input_sha256=tensor_sha(ex['input_ids']),
                    target_sha256=tensor_sha(target), image_sha256=tensor_sha(ex['pixel_values']),
                    image_instances=len(ex['image_grid_thw']), evaluation_metadata_in_input=False,
                    target_text=processor.tokenizer.decode(target, skip_special_tokens=False))
                encoded[ref['path']] = dict(record, example=ex, target_ids=target.tolist(),
                    prompt_ids=raw['input_ids'][:int((raw['labels'] != -100).nonzero()[0])].tolist())
                docs[ref['path']] = doc
                records.append(record)
        for row in data['rows']:
            if row['source_principle_target']:
                paths = {key: ref['path'] for key, ref in row['documents'].items()}
                validate_triplet(*(docs[paths[key]] for key in ['reason', 'paired', 'separate']))
                p, s = encoded[paths['paired']], encoded[paths['separate']]
                require(p['prompt_ids'] == s['prompt_ids'], 'Arm prompts differ')
                require(p['input_sha256'] != s['input_sha256'], 'Pairing has no effect')
                batch = upstream.StepwiseCollator(processor.tokenizer.pad_token_id)(
                    [p['example'], s['example']])
                padding = batch['attention_mask'] == 0
                require(bool(padding.any()) and bool((batch['labels'][padding] == -100).all()),
                    'Actual unequal-length collator padding not masked')
            else:
                require(list(row['documents']) == ['shared'], 'Ordinary action differs by arm')
        arm_stats = {}
        arm_rows = {arm: [encoded[ref['path']] for ref in refs] for arm, refs in data['arms'].items()}
        require(len(arm_rows['paired']) == len(arm_rows['separate']), 'Different sample counts')
        for p, s in zip(arm_rows['paired'], arm_rows['separate']):
            require(p['step'] == s['step'] and p['role'] == s['role'], 'Arm order differs')
            require(p['target_ids'] == s['target_ids'], 'Arm targets differ')
            require(p['image_sha256'] == s['image_sha256'], 'Arm pixels differ')
        for arm, rows in arm_rows.items():
            by_category = {key: sum(r['supervised_tokens'] for r in rows if r['category'] == key)
                           for key in ['reason', 'source_action', 'ordinary_action']}
            total = sum(by_category.values())
            arm_stats[arm] = dict(examples=len(rows), input_tokens=sum(r['input_tokens'] for r in rows),
                supervised_tokens=total, supervised_by_category=by_category,
                image_instances=sum(r['image_instances'] for r in rows),
                max_sequence_tokens=max(r['input_tokens'] for r in rows),
                category_token_fraction={k: v / total for k, v in by_category.items()})
        item = dict(case=case['case'], family_id=case['family_id'],
            manifest_sha256=sha(manifest_path), unique_examples=len(encoded), arms=arm_stats,
            commands=dict(Counter(row['command']['name'] for row in data['rows'])),
            source_steps=[row['step'] for row in data['rows'] if row['source_principle_target']])
        cases.append(item)
        print(json.dumps(dict(case=item['case'], arms=arm_stats), ensure_ascii=False), flush=True)
    require(len(records) == plan['budget']['unique_examples'], 'Unique sample count mismatch')
    totals = {}
    for arm in ['separate', 'paired']:
        totals[arm] = {key: sum(c['arms'][arm][key] for c in cases)
                       for key in ['examples', 'input_tokens', 'supervised_tokens', 'image_instances']}
        totals[arm]['supervised_by_category'] = {key: sum(c['arms'][arm]['supervised_by_category'][key]
            for c in cases) for key in ['reason', 'source_action', 'ordinary_action']}
        totals[arm]['equal_source_loss_mass_by_category'] = {key: sum(
            c['arms'][arm]['category_token_fraction'][key] for c in cases) / len(cases)
            for key in ['reason', 'source_action', 'ordinary_action']}
        require(totals[arm]['examples'] == plan['budget']['examples_per_arm'], 'Arm count mismatch')
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    result = dict(status='mixed_corpus_encoded_without_model', ended_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(args.plan), index_sha256=sha(index_path), cases=cases, totals=totals, records=records,
        checks=dict(arm_target_sequences_equal=True, arm_image_tensors_equal=True,
                    same_prompt_at_source_actions=True, source_action_prefix_differs=True,
                    padding_masked=True, all_task_contexts_preserved=True,
                    paper_activity_phases_preserved=True, evaluation_sentinel_excluded=True),
        model_loaded=False, model_forward_calls=0, optimizer_updates=0, cuda_initialized=False,
        training_admitted=False, training_authorized=False,
        limits=['Equal targets do not imply equal input tokens, FLOPs or information quality.',
                'Source loss masses are arithmetic implications, not a new Trainer execution.',
                'Three source principles in two related reconstruction templates; no independent test.',
                'Source reasons support strategies, not the demonstrated coordinates or physical use.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps(dict(status=result['status'], totals=totals)), flush=True)


if __name__ == '__main__':
    main()
