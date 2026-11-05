"""Assemble and inspect a sourced development batch; never load model weights.

One recorded four-edit program is aggregated into a target, explicitly not an
online step trajectory. All other examples are shared byte-for-byte by arms.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from audit_native_review import export
from decision_pairing import DecisionUnitDataset, validate_triplet
from native_checkpoint import canonical_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def locked(ref):
    path = Path(ref['path'])
    if sha(path) != ref['sha256']:
        raise ValueError('Changed input: '+str(path))
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); plan = json.loads(args.plan.read_text())
    out, root = args.out.resolve(), Path(plan['root']).resolve()
    out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    print(now(), 'begin sourced development batch audit; zero model/training calls', flush=True)
    result = {'started_at': now(), 'plan_sha256': sha(args.plan), 'model_calls': 0,
              'training_updates': 0, 'training_eligible': False, 'status': 'running'}
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        sys.dont_write_bytecode = True
        for name, digest in plan['scripts'].items():
            if sha(Path(__file__).parent/name) != digest:
                raise ValueError('Changed script: '+name)
        inputs = {name: locked(ref) for name, ref in plan['inputs'].items()}
        source = Path(plan['source'])
        for row in locked(plan['source_lock'])['files']:
            if sha(source/row['source_path']) != row['sha256']:
                raise ValueError('Official source changed')
        cards = inputs['knowledge']['records']
        assert len(cards) == 6 and len({c['family_id'] for c in cards}) == 6
        assert all(not c['training_eligible'] for c in cards)
        families = {p['family_id']: p for p in inputs['registry_before']['projects']}
        assert all(families[c['family_id']]['allocation'] in ('development', 'development_source_review') for c in cards)
        common, retrieval = [], []
        for card in cards:
            answer = {'condition': card['condition_summary'], 'decision': card['decision_scope']['source_decision'],
                      'reason': card['rationale_summary'], 'source': card['professional_evidence']['primary_crosscheck']['url'],
                      'scope_limit': card['decision_scope'].get('scope_boundary', card['decision_scope'].get('unsupported_part'))}
            assert answer['scope_limit']
            provenance = {'family_id': card['family_id'], 'decision_id': card['decision_id'],
                          'role': 'attributed_knowledge', 'audit_only': True, 'training_eligible': False,
                          'origin': 'researcher paraphrase of cited professional explanation; not a raw brief or expert review'}
            episode = {'stage': 'qa', 'system': 'Recall the attributed project decision and its conditions. Do not treat it as a universal rule or an executed outcome.',
                       'provenance': provenance,
                       'turns': [{'role': 'user', 'content': 'What decision and reason are documented for this project condition? State the source and limits.\n'+canonical({'condition': answer['condition'], 'source': answer['source']})},
                                 {'role': 'assistant', 'content': '', 'answer': canonical(answer)}]}
            path = out/'episodes'/'knowledge'/card['decision_id']/'episode.json'
            save(path, episode); common.append((card['decision_id'], path, episode, 0))
            retrieval.append({'family_id': card['family_id'], 'decision_id': card['decision_id'], 'knowledge': answer})
        save(out/'RETRIEVAL_CORPUS.json', {'audit_only': True, 'records': retrieval,
             'scope': 'Development corpus contract, not final training/test retrieval data. No original source images.'})
        review = export(inputs['bukit_review'], root, [c['candidate_id'] for c in inputs['bukit_review']['candidates']])
        assert json.loads(review['turns'][-1]['answer'])['verdict'] == 'insufficient_evidence'
        path = out/'episodes'/'review'/'bukit'/'episode.json'; save(path, review)
        common.append(('bukit_objective_review', path, review, 10))

        # Validate actual initial-state binding and retain the recorded program.
        binding, state, ctx, views = (inputs[k] for k in ('ono_binding', 'ono_state', 'ono_context', 'ono_views'))
        assert canonical_hash(state) == binding['state_sha256'] == canonical_hash(binding['native_state'])
        assert canonical_hash(ctx) == binding['current_context_sha256']
        assert views['binding']['sha256'] == plan['inputs']['ono_binding']['sha256']
        assert views['state']['sha256'] == plan['inputs']['ono_state']['sha256']
        image_refs = []
        for expected, actual, direction in zip(views['images'], binding['images'], ('topdown', 'south', 'east', 'west', 'north')):
            assert expected == actual
            path = Path(actual['path']).resolve()
            assert path.is_relative_to(root) and path.name.endswith('.'+direction+'.png') and sha(path) == actual['sha256']
            image_refs.append({'path': str(path.relative_to(root)), 'sha256': actual['sha256']})
        assert len(views['images']) == len(binding['images']) == len(image_refs) == 5
        steps = inputs['ono_steps']; assert len(steps) == 4
        for i, step in enumerate(steps):
            assert step['index'] == i+1
            assert all(r['output']['success'] for r in step['reply']['results'])
            assert step['command']['name'] == 'move_furniture_tool'
            assert step['command']['arguments']['object_id'] == 'recon_high_storage_0'
            if i: assert steps[i-1]['after_sha256'] == step['before_sha256']
        objects = [{k: obj[k] for k in ('object_id', 'object_type', 'name', 'transform', 'bbox_min', 'bbox_max', 'immutable')}
                   for obj in ctx['objects'].values()]
        task = {'request': ctx['design_request'], 'request_provenance': ctx['request_provenance'],
                'unknowns': ctx['unknowns'], 'objects': objects,
                'program_scope': 'Propose a four-call move program from this initial observation. Calls are executed sequentially. This aggregated program example does not supply or simulate intermediate observations.'}
        reason = '在兼顾低层高空间的开敞感与收纳时，来源采用主区低矮家具、入口安排高柜的策略。本重建据此尝试迁移高柜，并保留其身份与其他对象；是否减少遮挡、是否损害其他活动须由执行后检查决定。'
        doc = {'stage': 'furniture', 'system': 'Use the supplied reconstructed scene. Only move_furniture_tool(object_id, x, y, yaw) is available; x/y are metres and yaw is degrees. Only recon_high_storage_0 is editable. Retain all objects. Output the ordered program; do not assert unmeasured outcomes.',
               'task_context': {'scene_description': canonical(task)},
               'provenance': {'family_id': 'paris_19_remote_studio', 'decision_id': 'ono_low_furniture_entry_storage',
                              'role': 'source_conditioned_reconstructed_program', 'audit_only': True, 'training_eligible': False,
                              'historical_or_online_designer_trace': False, 'new_execution': False,
                              'scope': 'Four actual sequential native calls aggregated under one initial observation; not four independently observed assistant steps. Conditional geometric benefit, no whole-room preference label.'},
               'evaluation_only': {'sentinel': 'DBATCH_EVALUATION_ONLY_SENTINEL', 'outcome': inputs['ono_packet']['outcome']},
               'turns': [{'role': 'assistant', 'content': reason, 'images': [r['path'] for r in image_refs],
                          'tool_calls': [s['command'] for s in steps]}]}
        docs = {name: deepcopy(doc) for name in ('paired', 'separate', 'reason')}
        docs['separate']['turns'][0]['content'] = ''; docs['reason']['turns'][0]['tool_calls'] = []
        validate_triplet(docs['reason'], docs['paired'], docs['separate'])
        refs = {}
        for name, episode in docs.items():
            path = out/'episodes'/'program'/name/'episode.json'; save(path, episode)
            refs[name] = {'path': str(path.relative_to(root)), 'sha256': sha(path)}
        manifest = {'schema': 'scenefunc-decision-budget-audit-v1', 'audit_only': True,
                    'fixed_length': plan['budget']['common_pair_pad_tokens'],
                    'decisions': [{'id': 'ono_source_conditioned_program', **refs, 'images': image_refs}]}
        save(out/'PROGRAM_MANIFEST.json', manifest)
        save(out/'BATCH_MANIFEST.json', {'audit_only': True, 'training_eligible': False,
             'common': [{'id': name, 'path': str(p.relative_to(root)), 'sha256': sha(p)} for name, p, _, _ in common],
             'program_manifest': {'path': str((out/'PROGRAM_MANIFEST.json').relative_to(root)), 'sha256': sha(out/'PROGRAM_MANIFEST.json')},
             'independent_project_families': 6, 'examples_per_arm': 9})

        sys.path.insert(0, str(source/'src'))
        import sft_dataset as upstream
        import torch
        from transformers import AutoProcessor
        from scene_observation import configure_image_budget
        torch.set_num_threads(4)
        processor = AutoProcessor.from_pretrained(plan['processor'], local_files_only=True)
        configure_image_budget(processor, plan['budget']['image_max_pixels'])
        labels = lambda ex: ex['labels'][ex['labels'] != -100]
        text = lambda ex: processor.tokenizer.decode(ex['input_ids'], skip_special_tokens=False)
        checks, records, batches, examples = {}, {}, {}, {}
        for arm in ('paired', 'separate'):
            rows, names = [], []
            for name, path, episode, count in common:
                ds = upstream.StepwiseSFTDataset(path.parent, root, processor, max_length=plan['budget']['max_sequence_tokens'])
                assert len(ds) == 1
                ex = ds[0]
                assert len(ex.get('image_grid_thw', [])) == count
                target = processor.tokenizer.decode(labels(ex), skip_special_tokens=False)
                answer = json.loads(target.split('<answer>', 1)[1].split('</answer>', 1)[0])
                assert answer == json.loads(episode['turns'][-1]['answer'])
                first = int((ex['labels'] != -100).nonzero()[0])
                assert bool((ex['labels'][:first] == -100).all()) and torch.equal(ex['labels'][first:], ex['input_ids'][first:])
                rows.append(ex); names.append(name)
            unit = DecisionUnitDataset(out/'PROGRAM_MANIFEST.json', root, processor, upstream, arm,
                                       max_length=plan['budget']['max_sequence_tokens'])[0]
            rows.extend(unit['examples']); names.extend(('ono_reason', 'ono_program'))
            assert len(rows) == 9
            batch = upstream.StepwiseCollator(processor.tokenizer.pad_token_id)(rows)
            length = batch['input_ids'].shape[1]; fixed = plan['budget']['common_pair_pad_tokens']
            assert length <= fixed
            for key, fill in [('input_ids', processor.tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100), ('mm_token_type_ids', 0)]:
                if key in batch: batch[key] = torch.nn.functional.pad(batch[key], (0, fixed-length), value=fill)
            records[arm] = {'examples': len(rows), 'image_instances': len(batch['image_grid_thw']),
                'supervised_tokens': sum(len(labels(ex)) for ex in rows),
                'unpadded_tokens': sum(len(ex['input_ids']) for ex in rows),
                'batch_shapes': {k: list(v.shape) for k, v in batch.items()},
                'components': [{'id': name, 'tokens': len(ex['input_ids']), 'supervised_tokens': len(labels(ex)),
                    'target_sha256': hashlib.sha256(labels(ex).numpy().tobytes()).hexdigest(),
                    'input_sha256': hashlib.sha256(ex['input_ids'].numpy().tobytes()).hexdigest()}
                    for name, ex in zip(names, rows)]}
            examples[arm], batches[arm] = rows, batch
            print(arm, records[arm]['examples'], 'examples;', records[arm]['supervised_tokens'], 'targets;', records[arm]['image_instances'], 'images', flush=True)
        p, s = examples['paired'], examples['separate']
        checks['all_nine_targets_exactly_equal'] = all(torch.equal(labels(x), labels(y)) for x, y in zip(p, s))
        checks['eight_common_examples_tensor_identical'] = all(x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x) for x, y in zip(p[:-1], s[:-1]))
        checks['only_program_conditioning_differs'] = reason in text(p[-1]) and reason not in text(s[-1]) and not torch.equal(p[-1]['input_ids'], s[-1]['input_ids'])
        checks['reason_not_supervised_twice_in_program'] = reason not in processor.tokenizer.decode(labels(p[-1]), skip_special_tokens=False)
        checks['twenty_equal_images'] = all(len(b['image_grid_thw']) == 20 for b in batches.values()) and all(torch.equal(batches['paired'][k], batches['separate'][k]) for k in ('pixel_values', 'image_grid_thw'))
        checks['same_padded_shapes'] = all(v.shape == batches['separate'][k].shape for k, v in batches['paired'].items())
        checks['padding_masked'] = all(bool((b['labels'][b['attention_mask'] == 0] == -100).all()) for b in batches.values())
        checks['same_causal_target_count'] = int((batches['paired']['labels'][:, 1:] != -100).sum()) == int((batches['separate']['labels'][:, 1:] != -100).sum())
        checks['evaluation_metadata_not_in_input'] = all('DBATCH_EVALUATION_ONLY_SENTINEL' not in text(ex) for ex in p+s)
        checks['no_cuda_initialized'] = not torch.cuda.is_initialized()
        retrieval_cases = []
        for family in ('paris_19_remote_studio', 'singapore_bukit_batok_interim_spaces', '__unseen_contract_query__'):
            found = [r for r in retrieval if r['family_id'] != family]
            assert all(r['family_id'] != family for r in found)
            assert len(found) == (6 if family.startswith('__') else 5)
            retrieval_cases.append({'query_family': family, 'result_ids': [r['decision_id'] for r in found],
                                    'result_count': len(found), 'knowledge_sha256': hashlib.sha256(canonical(found).encode()).hexdigest()})
        save(out/'RETRIEVAL_AUDIT.json', {'cases': retrieval_cases, 'family_exclusion_passed': True,
             'scope': 'Deterministic include-all other-family reference; no ranking, model calls or final evaluation. Unknown query is a contract fixture, not a real held-out project.'})
        checks['query_family_exclusion'] = True
        save(out/'SUPERVISED_TARGETS.json', {arm: [processor.tokenizer.decode(labels(ex), skip_special_tokens=False) for ex in rows] for arm, rows in examples.items()})
        result.update(checks=checks, arms=records, unique_families=6, source_conditioned_programs=1,
                      program_calls=4, knowledge_records=6, objective_reviews=1, professional_reviews=0)
        assert all(checks.values()), checks
        result.update(status='passed_development_data_audit', exit_code=0,
            limitation='Material parity and loader behavior only. One aggregated program, no online policy, real loss/optimizer/FLOP parity, learned design skill, independent evaluation or authorized training.')
    except Exception as exc:
        result.update(status='failed', exit_code=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['ended_at'] = now(); save(out/'RESULT.json', result)
        print(now(), result['status'], flush=True)


if __name__ == '__main__':
    main()
