"""Audit five actual step contexts through the official CLI; never train.

One sourced decision supplies five dependent steps. Only step one attributes
a professional principle; later reason fields are researcher execution plans.
Both arms supervise exactly the same reason and action tokens once each.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan, out = read(args.plan), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = {'started_at': now(), 'status': 'running', 'plan_sha256': sha(args.plan),
              'model_forward_calls': 0, 'optimizer_updates': 0, 'new_native_actions': 0,
              'training_admitted': False, 'stages': {}}
    save(out/'RESULT.json', result)
    print(now(), 'begin true-step source pairing audit, CPU only', flush=True)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                          PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        sys.dont_write_bytecode = True
        repo, root, source = (Path(plan[k]) for k in ('repo', 'input_root', 'source'))
        scripts = repo/'scripts/920'
        for name, digest in plan['code_sha256'].items():
            assert sha(scripts/name) == digest, name
        for name, digest in plan['source_sha256'].items():
            assert sha(source/name) == digest, name
        for spec in plan['processor_files']:
            assert sha(Path(plan['model'])/spec['path']) == spec['sha256']
        assert sha(plan['patch']) == plan['patch_sha256']
        packet = Path(plan['packet'])
        for name, digest in plan['packet_sha256'].items():
            assert sha(packet/name) == digest, name
        from native_checkpoint import canonical_hash
        from decision_pairing import validate_triplet
        steps, summary = read(packet/'STEPS.json'), read(packet/'SUMMARY.json')
        assert len(steps) == 5 and summary['conditional_rule_passed']
        labels = ['initial', 'step_1', 'step_2', 'step_3', 'step_4']
        manifest = {'schema': 'scenefunc-decision-budget-audit-v1', 'audit_only': True,
                    'fixed_length': plan['budget']['sequence_tokens'], 'decisions': []}
        attribution, context_checks = [], []
        for index, label in enumerate(labels, 1):
            doc = read(packet/'ota_candidates'/f'step_{index:02d}.json')
            state, ctx = read(packet/(label+'_state.json')), read(packet/(label+'_context.json'))
            metric, binding = read(packet/(label+'_measurement.json')), read(packet/(label+'_state_binding.json'))
            step = steps[index-1]
            assert canonical_hash(state) == step['before_sha256'] == binding['state_sha256']
            assert canonical_hash(state) == canonical_hash(binding['native_state'])
            assert canonical_hash(ctx) == binding['current_context_sha256']
            if index > 1:
                assert steps[index-2]['after_sha256'] == step['before_sha256']
            assert doc['turns'][0]['tool_calls'] == [step['command']]
            assert step['reply']['results'][0]['output']['success']
            inputs = json.loads(doc['task_context']['scene_description'])
            assert inputs['available_measurement'] == metric
            assert inputs['design_request'] == ctx['design_request']
            assert {obj['object_id'] for obj in inputs['objects']} == set(state['objects'])
            for obj in inputs['objects']:
                actual = state['objects'][obj['object_id']]
                for key in ('name', 'object_type', 'bbox_min', 'bbox_max', 'transform', 'immutable'):
                    assert obj[key] == actual[key], (index, key)
            if index > 1:
                assert inputs['previous_actual_tool_receipt'] == steps[index-2]['reply']
            else:
                assert 'previous_actual_tool_receipt' not in inputs
            # Preserve the source candidate; the derived arms remove this same
            # prompt-level rationale so it cannot silently condition both actions.
            removed_principle = inputs.pop('source_principle')
            doc['task_context']['scene_description'] = json.dumps(inputs, ensure_ascii=False, sort_keys=True)
            doc['provenance'].update(audit_only=True, training_eligible=False,
                                     source_episode_group=plan['source_run'], dependent_step=index)
            doc['evaluation_only']['sentinel'] = 'EVAL_ONLY_SENTINEL_STORAGE_STEPS'
            image_refs = []
            for image, original, direction in zip(binding['images'], doc['turns'][0]['images'],
                                                  ('topdown', 'south', 'east', 'west', 'north')):
                path = Path(image['path']).resolve()
                assert path == (root/original).resolve() and path.is_relative_to(root)
                assert path.name.endswith('.'+direction+'.png') and sha(path) == image['sha256']
                image_refs.append({'path': str(path.relative_to(root)), 'sha256': image['sha256']})
            assert len(image_refs) == len(binding['images']) == len(doc['turns'][0]['images']) == 5
            docs = {arm: deepcopy(doc) for arm in ('paired', 'separate', 'reason')}
            docs['separate']['turns'][0]['content'] = ''
            docs['reason']['turns'][0]['tool_calls'] = []
            validate_triplet(docs['reason'], docs['paired'], docs['separate'])
            refs = {}
            for arm, candidate in docs.items():
                path = out/'episodes'/f'step_{index:02d}'/arm/'episode.json'
                save(path, candidate)
                refs[arm] = {'path': str(path.relative_to(root)), 'sha256': sha(path)}
            manifest['decisions'].append({'id': f'{plan["source_run"]}_step_{index:02d}',
                                          **refs, 'images': image_refs})
            attribution.append({'step': index, 'source_principle_in_reason_target': index == 1,
                                'professional_origin': 'Source concentration principle only; all coordinates/order are researcher construction.' if index == 1 else None,
                                'researcher_plan': True, 'reason': doc['turns'][0]['content'],
                                'removed_common_input_field': removed_principle})
            context_checks.append({'step': index, 'state_sha256': step['before_sha256'],
                                   'actual_context_sha256': binding['current_context_sha256'],
                                   'measurement_sha256': sha(packet/(label+'_measurement.json')),
                                   'images': image_refs, 'command': step['command'],
                                   'previous_receipt_correct': True})
        save(out/'MANIFEST.json', manifest)
        save(out/'SOURCE_ATTRIBUTION.json', {'independent_source_decisions': 1,
             'dependent_observed_steps': 5, 'source_principle_targets': 1, 'researcher_only_plan_targets': 4,
             'records': attribution, 'common_input_retains_source_scope_caveats': True,
             'scope': 'Only explicit source_principle field removed; shared unknowns still explain the source-to-proxy scope. Loader contrast is reason-before-action conditioning, not removal of all knowledge or proof of professional learning.'})
        save(out/'STEP_BINDINGS.json', context_checks)
        overlay = out/'overlay'
        for name in plan['source_sha256']:
            dest = overlay/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/name, dest)
        (overlay/'src/__init__.py').write_text('')
        for name in ('decision_pairing.py', 'decision_entry_audit.py'):
            shutil.copy2(scripts/name, overlay/'src'/name)
        patch = subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', plan['patch']],
                               cwd=overlay, capture_output=True, text=True)
        (out/'patch.log').write_text(patch.stdout+patch.stderr)
        assert patch.returncode == 0 and sha(overlay/'src/train_sft.py') == plan['patched_entry_sha256']
        env = dict(os.environ, PYTHONPATH=str(overlay))
        reports = {}
        for arm in ('paired', 'separate'):
            target = out/arm/'BATCH.json'
            command = [sys.executable, str(overlay/'src/train_sft.py'), '--model-path', plan['model'],
                       '--repo-root', str(root), '--max-length', str(plan['budget']['sequence_tokens']),
                       '--max-pixels', str(plan['budget']['image_max_pixels']), '--use-cpu', '--no-bf16',
                       '--dataloader-workers', '0', '--per-device-batch-size', '5',
                       '--output-dir', str(out/arm/'output'), '--dry-run',
                       '--decision-manifest', str(out/'MANIFEST.json'), '--decision-arm', arm,
                       '--decision-audit-report', str(target)]
            receipt = out/(arm+'_execution')
            rc = subprocess.call([sys.executable, str(scripts/'run_stage.py'), '--receipt-dir', str(receipt),
                                  '--timeout', str(plan['budget']['entry_seconds_each']), '--', *command],
                                 cwd=overlay, env=env)
            result['stages'][arm] = read(receipt/'receipt.json')
            save(out/'RESULT.json', result)
            assert rc == result['stages'][arm]['exit_code'] == 0
            reports[arm] = read(target)
        p, s = reports['paired'], reports['separate']
        checks = {
            'all_five_actual_step_contexts_bound': len(context_checks) == 5,
            'all_steps_selected_by_actual_cli': all(x['dataset_decisions'] == x['first_batch_decisions'] == 5 and x['first_batch_examples'] == 10 for x in reports.values()),
            'all_ten_targets_identical': all(x['target_ids'] == y['target_ids'] for x, y in zip(p['rows'], s['rows'])),
            'five_reason_inputs_identical': all(p['rows'][i]['active_input_sha256'] == s['rows'][i]['active_input_sha256'] for i in range(0, 10, 2)),
            'five_action_inputs_different': all(p['rows'][i]['active_input_sha256'] != s['rows'][i]['active_input_sha256'] for i in range(1, 10, 2)),
            'reason_conditioning_only_in_paired_actions': all(p['rows'][i]['reason_present_in_input'] and not s['rows'][i]['reason_present_in_input'] for i in range(1, 10, 2)),
            'reason_supervised_once_per_step': all(x['rows'][i]['reason_present_in_targets'] == (i % 2 == 0) for x in reports.values() for i in range(10)),
            'fifty_image_instances_same_tensors': all(x['image_instances'] == 50 for x in reports.values()) and all(p['batch_tensor_sha256'][k] == s['batch_tensor_sha256'][k] for k in ('pixel_values', 'image_grid_thw')),
            'same_padded_shapes': p['batch_shapes'] == s['batch_shapes'],
            'same_supervised_token_count': p['total_supervised_tokens'] == s['total_supervised_tokens'],
            'same_causal_shifted_targets': sum(x['causal_shifted_targets'] for x in p['rows']) == sum(x['causal_shifted_targets'] for x in s['rows']),
            'evaluation_metadata_absent': not any(row['evaluation_only_sentinel_present'] for x in reports.values() for row in x['rows']),
            'no_model_or_training': all(not x['cuda_initialized'] and not x['model_weights_loaded'] and not x['trainer_instantiated'] and x['optimizer_updates'] == x['model_forward_calls'] == 0 for x in reports.values()),
            'upstream_unchanged': all(sha(source/n) == h for n, h in plan['source_sha256'].items()),
        }
        result.update(checks=checks, arms=reports, source_decisions=1, observed_steps=5,
                      unique_images=25, source_principle_targets=1, researcher_only_plan_targets=4,
                      original_candidates_changed=False, explicit_source_principle_field_removed_symmetrically=True,
                      common_source_scope_caveats_retained=True,
                      unpadded_tokens={arm: sum(row['input_tokens'] for row in x['rows']) for arm, x in reports.items()})
        assert all(checks.values()), [k for k, v in checks.items() if not v]
        result.update(status='passed_actual_cli_step_pairing_audit', exit_code=0,
                      limitation='True pre-step state/view/receipt binding and equal supervised material verified. Both arms also learn state-to-reason; difference is reason before action in the same sequence. Not a pure test of all state/knowledge association, actual loss/FLOP parity, training benefit, human preference or independent generalization.')
    except BaseException as exc:
        result.update(status='failed', exit_code=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['ended_at'] = now()
        save(out/'RESULT.json', result)
        print(now(), result['status'], flush=True)


if __name__ == '__main__':
    main()
