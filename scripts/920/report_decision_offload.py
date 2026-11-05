"""Verify saved pilot artifacts on CPU, without loading the base or training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    root = args.run.resolve()
    print(datetime.now(timezone.utc).isoformat(), 'verify saved pilot on CPU, no model or optimization', flush=True)
    torch.set_num_threads(4)
    cfg = read(root/'PLAN.json'); runtime = Path(cfg['runtime_directory'])
    out = Path(cfg['training_output_root'])
    approval = read(root/'USER_APPROVAL.json')
    assert approval['approved'] and approval['plan_sha256'] == sha(root/'PLAN.json')
    for path, digest in cfg['code_files'].items():
        assert sha(Path(path)) == digest, path
    batch = read(out/'BATCH.json'); outer = read(runtime/'authorized_batch_execution/receipt.json')
    assert batch['status'] == 'completed' and batch['successful_arms'] == 2 and outer['exit_code'] == 0
    encoding = read(Path(cfg['encoding']['path']))
    assert sha(Path(cfg['encoding']['path'])) == cfg['encoding']['sha256']
    summary, manifest, finals = {}, [], {}
    for arm in cfg['arm_order']:
        arm_dir = out/arm
        run = read(arm_dir/'RUN.json'); rows = read(arm_dir/'MICRO_LOSS.json')
        receipt = read(out/(arm+'_execution')/'receipt.json')
        records = [x for x in encoding['records'] if x['component'] == 'reason' or x['arm'] == arm]
        assert len(records) == 20 and len(rows) == 40
        assert run['status'] == 'completed' and run['optimizer_updates'] == 2 and run['backward_completed'] == 40
        assert run['same_process_adapter_save_zero_reload_exact'] and receipt['exit_code'] == 0
        assert run['checkpoint_input_stats']['checkpoint_forward_calls'] == 1440
        denominators = {s: sum(x['target_count'] for x in records if x['source_decision'] == s)
                        for s in {x['source_decision'] for x in records}}
        cycles = []
        for cycle in range(2):
            part = rows[cycle*20:(cycle+1)*20]
            for source, loss in zip(records, part):
                assert loss['optimizer_step_before'] == cycle
                assert loss['target_count'] == source['target_count']
                assert abs(loss['weight'] - 20/(3*denominators[source['source_decision']])) < 1e-12
                assert math.isfinite(loss['nll_sum'])
            objective = sum(x['nll_sum'] * x['weight'] / 20 for x in part)
            logged = next(x for x in run['training_log'] if x.get('step') == cycle+1 and 'loss' in x)
            assert abs(objective-logged['loss']) <= 5.1e-5
            assert math.isfinite(logged['grad_norm'])
            cycles.append({'updates_before_cycle': cycle, 'source_balanced_objective': objective,
                           'logged_loss': logged['loss'], 'grad_norm': logged['grad_norm'],
                           'target_tokens': sum(x['target_count'] for x in part)})
        final = load_file(str(arm_dir/'adapter/adapter_model.safetensors'), device='cpu')
        assert len(final) == 144 and sum(v.numel() for v in final.values()) == 3833856
        assert all('lora_' in n and '.language_model.' in n for n in final)
        assert finite_state(final)
        checkpoints = []
        previous = None
        for step in (1, 2):
            cp = arm_dir/'checkpoints'/f'checkpoint-{step}'
            state = read(cp/'trainer_state.json'); assert state['global_step'] == step
            weights = load_file(str(cp/'adapter_model.safetensors'), device='cpu')
            assert set(weights) == set(final) and finite_state(weights)
            optim = torch.load(cp/'optimizer.pt', map_location='cpu', weights_only=True)
            assert len(optim['state']) == 144 and finite_state(optim)
            assert all(int(x['step']) == step for x in optim['state'].values())
            changed = None if previous is None else sum(not torch.equal(weights[n], previous[n]) for n in weights)
            if step == 2:
                assert all(torch.equal(weights[n], final[n]) for n in weights)
                assert changed > 0
            checkpoints.append({'step': step, 'adapter_tensors': len(weights),
                                'finite_optimizer_states': len(optim['state']),
                                'changed_since_previous_checkpoint': changed})
            previous = weights
        finals[arm] = final
        summary[arm] = {'updates': 2, 'micro_backwards': 40, 'cycles': cycles,
                        'process_seconds': receipt['elapsed_seconds'],
                        'trainer_seconds': run['training_log'][-1]['train_runtime'],
                        'peak_allocated_bytes': run['peak_allocated_bytes'],
                        'peak_allocated_GiB': run['peak_allocated_bytes']/1024**3,
                        'changed_adapter_tensors_from_initial': run['changed_adapter_tensors'],
                        'checkpoint_checks': checkpoints,
                        'final_equals_checkpoint_2': True,
                        'same_process_save_zero_reload_exact': True,
                        'cold_optimizer_resume_tested': False}
        for f in sorted(arm_dir.rglob('*')):
            if f.is_file():
                manifest.append({'path': str(f), 'bytes': f.stat().st_size, 'sha256': sha(f)})
    different = sum(not torch.equal(finals['paired'][n], finals['separate'][n]) for n in finals['paired'])
    assert different > 0 and not torch.cuda.is_initialized()
    write(root/'CHECKPOINT_MANIFEST.json', {'files_remain_on_server': True, 'files': manifest})
    result = {'run_id': cfg['run_id'], 'status': 'four_update_mechanism_pilot_completed',
              'verified_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(root/'PLAN.json'),
              'total_optimizer_updates': 4, 'total_micro_backwards': 80, 'arms': summary,
              'batch_started_at': outer['started_at'], 'batch_ended_at': outer['ended_at'],
              'batch_elapsed_seconds': outer['elapsed_seconds'], 'batch_exit_code': outer['exit_code'],
              'different_adapter_tensors_between_arms': different,
              'final_adapter_quality_evaluated': False, 'design_or_generalization_claim': False,
              'interpretation': 'Matched four-update training and same-process save/reload completed. Both second-cycle losses rose slightly; these are training-context objectives before updates, not final-adapter evaluation or proof of pairing benefit.',
              'limitations': ['Three source decisions and ten dependent reconstructed steps, including researcher action explanations.',
                             'Paired/separate contexts already differ before learning; their loss gap is not a learning benefit.',
                             'No cold optimizer resume, independent-room generation or human review performed.',
                             'Two brief unrelated CPU input-build attempts ran during training (1.520266s/1 and 1.921717s/0); runtime is observational, not an isolated throughput benchmark.']}
    write(root/'TRAINING_RESULT.json', result)
    lines = ['arm,updates_before_cycle,source_balanced_objective,grad_norm,target_tokens,peak_allocated_GiB']
    for arm, row in summary.items():
        for c in row['cycles']:
            lines.append(f"{arm},{c['updates_before_cycle']},{c['source_balanced_objective']:.9f},{c['grad_norm']:.9f},{c['target_tokens']},{row['peak_allocated_GiB']:.6f}")
    (root/'TRAINING_METRICS.csv').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
