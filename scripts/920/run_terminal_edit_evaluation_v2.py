"""Evaluation-only revision: accepted adapters, one preserved interrupted round, no resampling."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from native_project import sha256
from run_decision_loss import save


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args()
    study = read(args.study)
    auth = read(args.authorization)
    if (auth.get('approved') is not True or not auth.get('user_messages')
            or auth.get('study_plan_sha256') != sha256(args.study)
            or auth.get('plan_sha256') != study['training_plan']['sha256']):
        raise PermissionError('Exact new study approval required')
    for path, digest in study['frozen_files'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Frozen study file changed: '+path)
    if len(study['cells']) != 12 or study['evaluation_gpu_seconds'] != 7200:
        raise ValueError('Study evaluation budget changed')
    public = Path(study['public_directory'])
    if (public/'EVALUATION_BATCH.json').exists():
        raise FileExistsError('Batch already attempted; inspect saved work, never restart')
    training = read(study['training_plan']['path'])
    training_result = read(Path(study['training_result']))
    if (training_result['status'] != 'completed'
            or training_result['plan_sha256'] != study['training_plan']['sha256']):
        raise ValueError('Both approved adapters must finish before the fixed comparison')
    for arm in training['arm_order']:
        row = read(Path(training['training_output_root'])/arm/'RUN.json')
        if (row['status'] != 'completed' or row['optimizer_updates'] != 8
                or row['plan_sha256'] != study['training_plan']['sha256']
                or not row['same_process_adapter_save_zero_reload_exact']):
            raise ValueError('Unaccepted final adapter: '+arm)
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = dict(status='running', started_at=now(), study_plan_sha256=sha256(args.study),
                  cells=[], gpu_subprocess_seconds=0.)
    scripts = Path(__file__).parent
    def progress():
        save(public/'EVALUATION_BATCH.json', result)
        status = read(public/'STATUS.json')
        status.update(state='evaluation_'+result['status'], updated_at=now(),
            evaluation_cells_completed=sum(c['exit_code']==0 for c in result['cells']), evaluation_cells_attempted=len(result['cells']), evaluation_cells_planned=12,
            latest_verifiable_artifact=str(public/'EVALUATION_BATCH.json'), failure_reason=result.get('error'))
        save(public/'STATUS.json', status)
        save(public.parent/'latest.json', dict(run_id=study['run_id'], stage=status['state'],
            updated_at=status['updated_at'], run_directory=study['runtime_directory']))
    progress()
    try:
        for cell in study['cells']:
            plan = read(cell['template'])
            prepared = read(Path(plan['prepared_directory'])/'RESULT.json')
            if prepared['status'] != 'prepared_without_model':
                raise ValueError('Current input was not prepared')
            if cell['arm'] != 'frozen_base_knowledge_feedback':
                adapter = Path(plan['adapter']['directory'])
                for name, digest in plan['adapter']['sha256'].items():
                    if sha256(adapter/name) != digest:
                        raise ValueError('Accepted adapter changed: '+str(adapter/name))
            cell_public, runtime = Path(plan['public_directory']), Path(cell['runtime'])
            planfile = cell_public/'PLAN.json'
            if planfile.exists():
                raise FileExistsError('Cell already bound; no restart')
            save(planfile, plan)
            command = [sys.executable, str(scripts/'run_stage.py'), '--receipt-dir',
                str(runtime/'rollout_receipt'), '--timeout', '1800', '--', study['native_python'],
                str(scripts/'run_edit_policy_executed_v2.py'), '--plan', str(planfile),
                '--phase', 'rollout', '--out', str(runtime/'rollout')]
            print(now(), 'start', cell['id'], flush=True)
            code = subprocess.call(command)
            report_path = runtime/'rollout/RESULT.json'
            report = read(report_path) if report_path.exists() else {}
            result['gpu_subprocess_seconds'] += report.get('gpu_subprocess_seconds', 0.)
            row = dict(id=cell['id'], condition=cell['condition'], seed=cell['seed'], arm=cell['arm'],
                exit_code=code, result_path=str(report_path),
                receipt=read(runtime/'rollout_receipt/receipt.json'),
                plan_sha256=sha256(planfile), model_calls=report.get('model_calls'),
                tool_attempts=report.get('tool_attempts'), native_calls=report.get('native_calls'),
                gpu_seconds=report.get('gpu_subprocess_seconds'), whole_room_success='unknown')
            review = report.get('final_audit', {}).get('review', {})
            checks = review.get('checks', [])
            initial = {r['requirement']:r['pass'] for r in report.get('initial_review', {}).get('checks', [])}
            lost = [r['requirement'] for r in checks if initial.get(r['requirement']) is True and r['pass'] is not True]
            row.update(activity_status=review.get('requested_activity_status', 'unknown'),
                overall_measured_status=review.get('measured_requirements_status', 'unknown'),
                newly_lost_checks=lost, absolute_failed_checks=[r['requirement'] for r in checks if r['pass'] is False],
                unknown_checks=[r['requirement'] for r in checks if r['pass'] is None])
            row['scoped_improvement_without_new_loss'] = bool(code == 0 and report.get('status') != 'failed' and initial and checks and
                row['activity_status'] == 'pass' and not lost)
            result['cells'].append(row)
            # Preserve raw outcomes before any error terminates the comparison.
            recorded = cell_public/'recorded'
            for name in ('RESULT.json', 'final_state.json', 'final_audit.json', 'restored_audit.json', 'final_observation.json'):
                source = runtime/'rollout'/name
                if source.exists():
                    recorded.mkdir(parents=True, exist_ok=True); shutil.copy2(source, recorded/name)
            for rd in sorted((runtime/'rollout').glob('round_*')):
                for name in ('request.json', 'tool_reply.json', 'validation_error.json', 'model/raw_output.txt', 'model/RESULT.json'):
                    source, target = rd/name, recorded/rd.name/name
                    if source.exists():
                        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
            observation = runtime/'rollout/final_observation.json'
            if observation.exists():
                for i, path in enumerate(read(observation)['image_paths']):
                    target = recorded/'views'/f'{i}_{Path(path).name}'
                    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
            progress()
            if code or report.get('status') == 'failed':
                raise RuntimeError('Infrastructure failure; keep all outcomes and stop without retry')
            if result['gpu_subprocess_seconds'] > study['evaluation_gpu_seconds']:
                raise RuntimeError('Evaluation GPU time ceiling exceeded')
            bundle = runtime/'rollout/final_checkpoint'
            export = dict(bundle=str(bundle), manifest_sha256=sha256(bundle/'manifest.json'),
                code_sha256={n:plan['code_sha256'][n] for n in ('export_catalog_geometry.py', 'inspect_native_mesh_contacts.py', 'native_project.py')},
                budget=dict(output_max_bytes=150000000))
            export_path = cell_public/'EXPORT_PLAN.json';save(export_path, export)
            code = subprocess.call([sys.executable, str(scripts/'run_stage.py'), '--receipt-dir',
                str(runtime/'export_receipt'), '--timeout', '120', '--', study['native_python'],
                str(scripts/'export_catalog_geometry.py'), '--plan', str(export_path), '--out', str(runtime/'export')])
            if code:
                raise RuntimeError('Saved terminal export failed; do not rerun model')
            save(cell_public/'EXPORT_RESULT.json', read(runtime/'export/RESULT.json'))
        fields = ['id','condition','seed','arm','activity_status','scoped_improvement_without_new_loss',
                  'overall_measured_status','whole_room_success','tool_attempts','native_calls','model_calls','gpu_seconds']
        with (public/'LEARNING_RESULTS.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader();writer.writerows(result['cells'])
        result.update(status='completed_pending_scientific_review', ended_at=now(), exit_code=0)
    except BaseException as exc:
        result.update(status='failed', ended_at=now(), exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        progress()


if __name__ == '__main__':
    main()
