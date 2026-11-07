"""Read-only acceptance for evaluation-only continuation with carried failure costs.

Does not run models, edit scenes, choose checkpoints, or change frozen metrics.
Uses stdlib only; writes reports into a new output directory.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def review(public):
    execution = read(public / 'execution_receipt/receipt.json')
    if execution['status'] != 'completed' or execution.get('exit_code') != 0:
        raise ValueError('Review requires an actually completed successful batch')
    study = read(public / 'STUDY_PLAN.json')
    train_path = Path(study['training_plan']['path'])
    train = read(train_path)
    auth = read(public / 'AUTHORIZATION.json')
    assert auth['approved'] is True
    assert sha(public / 'STUDY_PLAN.json') == auth['study_plan_sha256']
    assert sha(train_path) == auth['plan_sha256'] == study['training_plan']['sha256']
    assert auth['new_training_updates'] == 0
    original_execution = read(train_path.parent / 'EXECUTION.json')
    assert original_execution['exit_code'] != 0, 'Original failed batch must remain failed'
    for file, digest in {**train['code_files'], **study['frozen_files']}.items():
        assert sha(file) == digest, file
    training = read(study['training_result'])
    evaluation = read(public / 'EVALUATION_BATCH.json')
    assert training['status'] == 'completed' and training['exit_code'] == 0
    assert evaluation['status'] == 'completed_pending_scientific_review'
    assert evaluation['exit_code'] == 0
    arms, seconds = {}, 0.0
    for arm in train['arm_order']:
        path = Path(train['training_output_root']) / arm
        result = read(path / 'RUN.json')
        receipt = read(path.parent / (arm + '_receipt') / 'receipt.json')
        assert result['status'] == 'completed' and result['optimizer_updates'] == 8
        assert result['backward_completed'] == 24 * 8
        assert result['same_process_adapter_save_zero_reload_exact'] is True
        assert receipt['exit_code'] == 0 and receipt['elapsed_seconds'] <= 7200
        seconds += receipt['elapsed_seconds']
        adapter = Path(result['adapter'])
        arms[arm] = dict(updates=8, backwards=192, elapsed_seconds=receipt['elapsed_seconds'],
            peak_allocated_bytes=result['peak_allocated_bytes'],
            adapter={str(adapter / name): sha(adapter / name)
                     for name in ('adapter_config.json', 'adapter_model.safetensors')},
            training_log=result['training_log'],
            cold_optimizer_resume_tested=result['cold_process_optimizer_resume_tested'])
    assert len(evaluation['cells']) == len(study['cells']) == 12
    cells, aggregate, gpu_seconds = [], defaultdict(lambda: defaultdict(int)), 0.0
    carried_calls, carried_seconds = 0, 0.0
    for expected, row in zip(study['cells'], evaluation['cells']):
        assert row['id'] == expected['id'] and row['exit_code'] == 0
        runtime = Path(expected['runtime'])
        result = read(runtime / 'rollout/RESULT.json')
        assert result['status'] == 'policy_completed_pending_comparison'
        assert result['model_calls'] <= 6 and result['tool_attempts'] <= 6
        assert result['gpu_subprocess_seconds'] <= 600
        gpu_seconds += result['gpu_subprocess_seconds']
        initial = {x['requirement']: x['pass'] for x in result['initial_review']['checks']}
        final = result['final_audit']['review']
        checks = {x['requirement']: x['pass'] for x in final['checks']}
        assert checks.keys() == initial.keys(), 'Evaluation check identities changed'
        lost = sorted(k for k, value in initial.items() if value is True and checks[k] is not True)
        assert lost == sorted(row['newly_lost_checks'])
        scoped = final['requested_activity_status'] == 'pass' and not lost
        assert scoped == row['scoped_improvement_without_new_loss']
        assert result['initial_review']['requested_activity_status'] != 'pass', 'Do not count retention as new activity gain'
        rounds = result['rounds']
        # A done output has no execution; it is never a state change.
        changes = sum(x['execution']['before_sha256'] != x['execution']['after_sha256']
                      for x in rounds if x.get('execution'))
        rejected = sum(bool(x.get('rejection')) for x in rounds)
        cell_public = public / 'evaluation' / row['id']
        bound = read(cell_public / 'PLAN.json')
        assert sha(cell_public / 'PLAN.json') == row['plan_sha256']
        assert bound == read(expected['template']), 'Bound cell differs from frozen template'
        assert read(runtime / 'rollout_receipt/receipt.json')['exit_code'] == 0
        assert read(runtime / 'export_receipt/receipt.json')['exit_code'] == 0
        interrupted = bound.get('interrupted_rollout')
        if interrupted:
            old = Path(interrupted['directory'])
            for name, digest in interrupted['sha256'].items():
                assert sha(old / name) == digest
            old_result = read(old / 'RESULT.json')
            assert old_result['status'] == 'failed' and old_result['model_calls'] == 1
            assert old_result['native_calls'] == 0
            for name in ('request.json', 'model/raw_output.txt', 'model/RESULT.json'):
                assert sha(old / 'round_000' / name) == sha(runtime / 'rollout/round_000' / name)
            assert rounds[0].get('rejection'), 'Original malformed output must remain rejected'
            carried_calls += old_result['model_calls']
            carried_seconds += old_result['gpu_subprocess_seconds']
        if expected['arm'] != 'frozen_base_knowledge_feedback':
            adapter = bound['adapter']
            for name, digest in adapter['sha256'].items():
                assert sha(Path(adapter['directory']) / name) == digest
        child_seconds = 0.0
        assert len(rounds) == result['model_calls']
        for turn, round_row in enumerate(rounds):
            rd = runtime / 'rollout' / f'round_{turn:03d}'
            model = read(rd / 'model/RESULT.json')
            assert model['status'] == 'completed' and model['training_updates'] == 0
            assert model['paid_api_calls'] == 0 and model['model_calls'] == 1
            assert model['seed'] == bound['inference']['seed'] + turn
            assert len(model['encoding']['images']) == 5
            assert model['encoding']['input_tokens'] <= bound['budget']['max_input_tokens']
            assert model['output_tokens'] <= bound['budget']['max_new_tokens']
            assert sha(rd / 'request.json') == model['request_sha256']
            assert sha(rd / 'model/raw_output.txt') == model['raw_output_sha256']
            if bound['adapter']:
                assert model['adapter_reload_exact'] is True and model['adapter'] == bound['adapter']
            if interrupted and turn == 0:
                child_seconds += old_result['gpu_subprocess_seconds']
                assert round_row['new_model_call'] is False
            else:
                receipt = read(rd / 'model_receipt/receipt.json')
                assert receipt['exit_code'] == 0
                child_seconds += receipt['elapsed_seconds']
        assert abs(child_seconds - result['gpu_subprocess_seconds']) < 1e-6
        checkpoint = runtime / 'rollout/final_checkpoint'
        manifest = read(checkpoint / 'manifest.json')
        for name, info in manifest['files'].items():
            file = (checkpoint / name).resolve()
            assert file.is_relative_to(checkpoint.resolve())
            assert file.stat().st_size == info['bytes'] and sha(file) == info['sha256']
        export = read(cell_public / 'EXPORT_RESULT.json')
        assert export['status'] == 'geometry_export_roundtrip_checked'
        assert sha(export['glb']['path']) == export['glb']['sha256']
        assert sha(checkpoint / 'manifest.json') == export['input_manifest_sha256']
        images = list((cell_public / 'recorded/views').glob('*'))
        assert len(images) == 5 and all(x.stat().st_size > 0 for x in images)
        observation = read(runtime / 'rollout/final_observation.json')
        assert len(observation['image_paths']) == 5
        for i, source in enumerate(observation['image_paths']):
            assert sha(source) == sha(cell_public / 'recorded/views' / f'{i}_{Path(source).name}')
        cell = dict(row, initial_activity=result['initial_review']['requested_activity_status'],
            rejected_outputs=rejected, state_changes=changes, stop_reason=result['stop_reason'],
            checkpoint_manifest_sha256=sha(checkpoint / 'manifest.json'),
            glb=export['glb'], views=[str(x) for x in sorted(images)])
        cells.append(cell)
        a = aggregate[row['arm']]
        for key, value in dict(cells=1, activity_pass=final['requested_activity_status'] == 'pass',
                scoped_gain=scoped, new_loss=bool(lost), model_calls=result['model_calls'],
                rejections=rejected, native_calls=result['native_calls'], state_changes=changes,
                model_done=result['stop_reason'] == 'model_done').items():
            a[key] += value
    assert abs(gpu_seconds - evaluation['gpu_subprocess_seconds']) < 1e-6
    assert gpu_seconds <= 7200 and seconds + gpu_seconds <= 21600
    assert carried_calls == 1 and abs(carried_seconds - 24.04812056897208) < 1e-8
    new_calls = sum(c['model_calls'] for c in cells) - carried_calls
    new_seconds = gpu_seconds - carried_seconds
    assert new_calls <= auth['max_additional_model_calls'] == 71
    assert new_seconds <= auth['max_additional_gpu_subprocess_seconds']
    return dict(status='artifact_and_accounting_checks_passed',
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        scientific_conclusion='See observed counts; no automatic professional/generalization acceptance.',
        training=arms, training_seconds=seconds, evaluation_gpu_seconds=gpu_seconds,
        total_gpu_subprocess_seconds=seconds + gpu_seconds, aggregate=dict(aggregate), cells=cells,
        new_training_updates=0, carried_model_calls=carried_calls,
        carried_gpu_seconds=carried_seconds, additional_model_calls=new_calls,
        additional_gpu_seconds=new_seconds, original_execution=original_execution,
        limitations=['One exposed development family; two conditions and two seeds are not four independent projects.',
            'Joint training adds 1332 decision tokens; not an equal-material pairing-mechanism test.',
            'Initial 0.6m bathroom access failure remains an absolute defect; whole-room usability unknown.',
            'No independent professional preference evaluation or physical-use validation.',
            'No causal comparison against earlier runs with different tasks and inputs.'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--public', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = review(args.public.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'FINAL_ACCEPTANCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cells', 'training')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
