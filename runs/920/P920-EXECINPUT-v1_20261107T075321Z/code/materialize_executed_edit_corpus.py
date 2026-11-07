"""Materialize deployable inputs and real CPU training batches; no training mode."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from edit_request import tools_and_scope, validate_action, normalize_context
from executed_edit_request import build_executed_request
from shared_action_contract import schemas_from_observation, digest


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: f.write(json.dumps(v, ensure_ascii=False, indent=2) + '\n')
def ref(p): return dict(path=str(p), sha256=sha(p))
def require(ok, message):
    if not ok: raise ValueError(message)


def materialize(plan, output):
    require(not output.exists(), 'Refuse to overwrite a corpus')
    root = Path(plan['repo_root']); parent = root.parent
    data = read(plan['candidate_dataset']); rows = []; tests = []
    activity_events = read(plan['activity_events'])['events'] if plan.get('activity_events') else {}
    native = schemas_from_observation({'system': read(data['rows'][0]['documents']['action_only']['path'])['system']})
    for row in data['rows']:
        docs = {}
        for arm, source in row['documents'].items():
            require(sha(source['path']) == source['sha256'], 'Candidate document changed')
            docs[arm] = read(source['path'])
        doc = docs['action_only']; context = json.loads(doc['task_context']['scene_description'])
        provenance = doc['provenance']; evaluation = doc['evaluation_only']
        step = provenance.get('step', provenance.get('step_index', 1))
        state = evaluation.get('after_state', evaluation.get('actual_after_state'))
        state = state.get('path') if isinstance(state, dict) else state
        path = Path(state); path = path if path.is_absolute() else parent/path
        trace_path = path.parent/'STEPS.json'
        history = []; evidence = {'candidate': row['documents']['action_only']}
        if trace_path.exists():
            trace = read(trace_path)
            require([x['index'] for x in trace] == list(range(1, len(trace)+1)), 'Nonconsecutive native trace')
            current = next(x for x in trace if x['index'] == step)
            require([current['command']] == row['command'], 'Target and executed command differ')
            for old in trace[:step-1]:
                history.append(dict(attempt=old['index'], origin='researcher_reconstruction',
                    calls=[old['command']], outcome=dict(kind='native_reply', receipt=old['reply']),
                    before_sha256=old['before_sha256'], after_sha256=old['after_sha256']))
            if history:
                require(history[-1]['after_sha256'] == current['before_sha256'], 'History not bound to current native step')
            # Current targets and future replies only verify provenance here;
            # they are not passed to build_executed_request.
            evidence['trace'] = ref(trace_path)
            evidence['current_before_sha256'] = current['before_sha256']
            legacy = context.get('recent_actual_tool_receipts')
            if legacy:
                require([x['results'] for x in legacy] == [x['outcome']['receipt']['results'] for x in history[-len(legacy):]], 'Embedded history differs from native receipts')
            if 'previous_actual_tool_receipt' in context:
                require(context['previous_actual_tool_receipt'] == history[-1]['outcome']['receipt'], 'Previous receipt differs')
        else:
            require(row['id'] == 'air_edit_01' and step == 1, 'Missing trajectory for a noninitial step')
            require(not context.get('previous_actual_tool_receipt') and not context.get('recent_actual_tool_receipts'), 'Unverified history')
            evidence['history_scope'] = 'Initial retained AIR editing task; no preceding attempts asserted.'
        # Verify actual pre-action state matches the state represented to the
        # learner, including object transforms; never use a post-action view.
        if trace_path.exists():
            candidates = [path.parent/f'step_{step}_before_state.json', path.parent/f'step_{step:02d}_before.json']
            before = next((p for p in candidates if p.exists()), None)
            require(before is not None, 'Pre-action state artifact missing')
            state_doc = read(before)
            # Native replay canonical_hash uses ordinary JSON separators, not
            # the compact request-contract digest.
            native_hash = hashlib.sha256(json.dumps(state_doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            require(native_hash == evidence['current_before_sha256'], 'Before state hash differs')
            objects = normalize_context({'objects': state_doc['objects']})['objects']
            represented = normalize_context(context)['objects']
            require(set(objects) == set(represented), 'Pre-action identities differ')
            for oid, obj in represented.items():
                require(obj['transform'] == objects[oid]['transform'], 'Pre-action transforms differ')
            evidence['pre_action_state'] = ref(before)
        tools, scope = tools_and_scope(doc['system'], native)
        images = doc['turns'][0]['images']
        for image in row['images']: require(sha(image['path']) == image['sha256'], 'Image changed')
        if row['id'] in activity_events:
            event = activity_events[row['id']]
            require(set(event) == {'request', 'authorship', 'scope'}, 'Unexpected phase annotation fields')
            require(event['authorship'] == 'researcher_specified_pre_action_activity', 'Phase authorship missing')
            # An explicitly declared new task condition, not a reconstruction
            # of a historical user message or a post-action score. No target
            # object, pose, reference layout or preferred strategy is supplied.
            context = copy.deepcopy(context)
            context['current_activity_request'] = copy.deepcopy(event)
        args = dict(context=context, tools=tools, scope=scope, images=images,
                    history=history, next_attempt=step, policy=plan['input_contract'])
        request, accounting = build_executed_request(**args)
        for call in row['command']: validate_action(call, request)
        inputs_path = output/'inputs'/f"{row['id']}.json"; write(inputs_path, args)
        request_path = output/'requests'/f"{row['id']}.json"; write(request_path, request)
        # Same public builder consumes serialized pre-action inputs at training
        # and future deployment. No assistant target parameter is accepted.
        require(build_executed_request(**read(inputs_path))[0] == request, 'Deployment reconstruction differs')
        new = copy.deepcopy(row); new.update(documents={}, requests={}, execution_inputs=ref(inputs_path),
            history_accounting=accounting, execution_evidence=evidence,
            advertised_tools=[t['name'] for t in tools])
        for arm, original in docs.items():
            require(original['task_context'] == doc['task_context'] and original['turns'][0]['images'] == images, 'Arm inputs differ')
            dest = output/'episodes'/arm/row['id']/'episode.json'
            copied = copy.deepcopy(original)
            copied['provenance']['prospective_input_contract'] = plan['input_contract']['name']
            if row['id'] in activity_events:
                copied['task_context']['scene_description'] = json.dumps(context, ensure_ascii=False, sort_keys=True)
                copied['provenance']['activity_request_authorship'] = 'researcher_specified_pre_action_activity_not_historical_user_quote'
            write(dest, copied); new['documents'][arm] = ref(dest); new['requests'][arm] = ref(request_path)
        rows.append(new)
        if history:
            bad = copy.deepcopy(args); bad['history'][-1]['attempt'] = step
            try: build_executed_request(**bad)
            except ValueError: tests.append(row['id'] + ':current_outcome_rejected')
            else: raise ValueError('Current target outcome leaked into history')
    result = dict(schema='scenefunc-executed-edit-corpus-v1', rows=rows,
        counts=dict(actions=len(rows), documents=2*len(rows), source_families=len({r['family_id'] for r in rows}),
                    decision_positions=sum(r['decision_target'] is not None for r in rows),
                    source_principles=len({r['decision_id'] for r in rows if r['decision_target']}),
                    examples_with_history=sum(bool(r['history_accounting']['included_attempts']) for r in rows),
                    explicit_activity_requests=len(activity_events)),
        input_contract=plan['input_contract'], candidate_dataset=ref(plan['candidate_dataset']),
        regression_checks=tests, training_authorized=False, training_updates=0, model_calls=0,
        limits=['No new source decision, action or independent project.',
                'Historical receipts were already present in 17 candidate inputs; this migrates them and adds verified earlier commands, not new experiential knowledge.',
                'Reconstructed execution history is not model text or historical designer action.',
                'Native query tools are not invented or silently added; advertised per-task schemas are preserved.',
                'No no-edit targets, recovery/rejection demonstrations or independent whole-room evaluation added.'])
    write(output/'DATASET.json', result)
    print(json.dumps(result['counts']), flush=True)


def encode(plan, corpus, output):
    require(not output.exists(), 'Refuse to overwrite tensors')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import torch
    from transformers import AutoProcessor
    from edit_supervision import encode_request_targets
    torch.set_num_threads(2)
    source = Path(plan['official_source']); sys.path.insert(0, str(source/'src'))
    from scene_observation import configure_image_budget
    spec = importlib.util.spec_from_file_location('executed_official_dataset', source/'src/sft_dataset.py')
    official = importlib.util.module_from_spec(spec); sys.modules[spec.name] = official; spec.loader.exec_module(official)
    processor = AutoProcessor.from_pretrained(plan['model'], local_files_only=True, trust_remote_code=False)
    configure_image_budget(processor, plan['budget']['max_pixels'])
    data = read(corpus/'DATASET.json'); old = read(plan['historical_encoding']); records = []
    collator = official.StepwiseCollator(processor.tokenizer.pad_token_id)
    def tensor_hash(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
    for row in data['rows']:
        require(sha(row['execution_inputs']['path']) == row['execution_inputs']['sha256'], 'Input changed')
        # The native environment materializes and round-trips the shared
        # builder (including jsonschema). The model environment consumes that
        # exact hashed request; do not mix CP311 dependencies into CP310.
        request = read(row['requests']['action_only']['path'])
        pair = {}; prompt_hashes = []
        for arm, document in row['documents'].items():
            require(sha(document['path']) == document['sha256'], 'Document changed')
            require(sha(row['requests'][arm]['path']) == row['requests'][arm]['sha256'], 'Request changed')
            require(request == read(row['requests'][arm]['path']), 'Training/deployment request differs')
            samples = official.build_step_samples(Path(document['path']), Path(plan['repo_root']))
            require(len(samples) == 1, 'Expected one official step')
            ex, am, dm, boundary = encode_request_targets(request, samples[0], processor, official,
                                                         max_length=plan['budget']['max_sequence_tokens'])
            old_row = next(x for x in old['records'] if x['id'] == row['id'] and x['arm'] == arm)
            require(sha(old_row['tensor_path']) == old_row['tensor_sha256'], 'Historical tensor changed')
            previous = torch.load(old_row['tensor_path'], map_location='cpu', weights_only=True)
            require(torch.equal(ex['labels'][am], previous['labels'][0][previous['action_mask'][0]]), 'Action changed')
            require(torch.equal(ex['pixel_values'], previous['pixel_values']), 'Observation pixels changed')
            require(torch.equal(ex['image_grid_thw'], previous['image_grid_thw']), 'Image grid changed')
            require(bool((ex['labels'][:boundary] == -100).all()) and bool((ex['labels'][boundary:] != -100).all()), 'Supervision gap/leakage')
            expected = json.dumps(row['decision_target'], ensure_ascii=False) if arm == 'decision_action' and row['decision_target'] else ''
            require(samples[0].assistant_content == expected and bool(dm.any()) == bool(expected), 'Decision target differs')
            rendered = processor.tokenizer.decode(ex['input_ids'][:boundary], skip_special_tokens=False)
            require('EVAL_ONLY_' not in rendered, 'Evaluation sentinel in prompt')
            require(not expected or expected not in rendered, 'Gold decision in prompt')
            event = read(row['execution_inputs']['path'])['context'].get('current_activity_request')
            require(not event or event['request'] in rendered, 'Current activity request lost during tokenization')
            batch = collator([ex]); batch['action_mask'] = am.unsqueeze(0); batch['decision_mask'] = dm.unsqueeze(0)
            target = output/'tensors'/arm/(row['id']+'.pt'); target.parent.mkdir(parents=True, exist_ok=True); torch.save(batch, target)
            prompt_hashes.append(tensor_hash(ex['input_ids'][:boundary])); pair[arm] = ex
            records.append(dict(id=row['id'], arm=arm, family_id=row['family_id'],
                tensor_path=str(target), tensor_sha256=sha(target), input_tokens=len(ex['input_ids']),
                prompt_tokens=boundary, action_tokens=int(am.sum()), decision_tokens=int(dm.sum()),
                prompt_sha256=prompt_hashes[-1], action_target_sha256=tensor_hash(ex['labels'][am]),
                history_attempts=row['history_accounting']['included_attempts'],
                deployment_prompt_exact=True, input_receipts_masked=True, original_actions_pixels_exact=True))
        require(len(set(prompt_hashes)) == 1, 'Arm prompt differs')
        padded = collator(list(pair.values()))
        require(bool((padded['labels'][padded['attention_mask']==0] == -100).all()), 'Padding supervised')
        print(row['id'], 'materialized both CPU batches', flush=True)
    # Consume the actual tensors through the existing training Dataset and
    # Trainer normalization path. No real model, optimizer or backward occurs.
    from train_edit_learning_study import EditExamples, normalization_check
    consumption = {}
    for arm in ('action_only', 'decision_action'):
        consumption[arm] = normalization_check(EditExamples([r for r in records if r['arm']==arm]), output/'trainer_fixture'/arm)
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    totals = {arm: {key: sum(r[key] for r in records if r['arm']==arm)
                     for key in ('input_tokens', 'action_tokens', 'decision_tokens')} for arm in consumption}
    result = dict(status='materialized_and_consumed_by_CPU_Trainer', records=records,
        dataset=ref(corpus/'DATASET.json'), totals=totals, trainer_normalization=consumption,
        maximum_sequence_tokens=max(r['input_tokens'] for r in records),
        training_authorized=False, optimizer_updates=0, model_loaded=False, cuda_initialized=False,
        checks=dict(official_target_serialization=True, official_collator=True,
                    shared_deployment_builder=True, every_assistant_token_supervised=True,
                    same_arm_prompt_actions_pixels=True, history_masked=True),
        caveat='Real tensor consumption and Trainer scaling only; no real-model forward, fit, rollout or design improvement.')
    write(output/'ENCODING.json', result)
    print(json.dumps(dict(status=result['status'], maximum_sequence_tokens=result['maximum_sequence_tokens'], totals=totals)), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--mode', choices=['materialize', 'encode'], required=True)
    ap.add_argument('--corpus', type=Path, required=True); ap.add_argument('--out', type=Path)
    a = ap.parse_args(); plan = read(a.plan)
    for p, expected in plan['input_hashes'].items(): require(sha(p) == expected, 'Frozen dependency changed: '+p)
    print(datetime.now(timezone.utc).isoformat(), a.mode, 'CPU only; zero model calls/updates', flush=True)
    if a.mode == 'materialize': materialize(plan, a.corpus)
    else:
        require(a.out is not None, 'Encoding output required'); encode(plan, a.corpus, a.out)


if __name__ == '__main__': main()
