"""CPU orchestration fixtures: no model, training, rendering or native calls."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import run_terminal_edit_evaluation as runner
from native_project import sha256
from run_decision_loss import save


def exercise(root, fail_first=False):
    root.mkdir(parents=True, exist_ok=False)
    save(root/'STATUS.json', {})
    train = dict(training_output_root=str(root/'training'), arm_order=['action_only','decision_action'])
    save(root/'TRAIN_PLAN.json', train)
    train_sha = sha256(root/'TRAIN_PLAN.json')
    save(root/'TRAINING_RESULT.json', dict(status='completed', plan_sha256=train_sha))
    for arm in train['arm_order']:
        adapter = root/'training'/arm/'adapter'
        save(adapter.parent/'RUN.json', dict(status='completed', optimizer_updates=8,
            plan_sha256=train_sha, same_process_adapter_save_zero_reload_exact=True))
        for name in ('adapter_config.json','adapter_model.safetensors'):
            adapter.mkdir(exist_ok=True);(adapter/name).write_text('CPU_FIXTURE_NOT_ADAPTER')
    save(root/'prepared/RESULT.json', dict(status='prepared_without_model'))
    cells=[]
    for mode in ('clear_floor','desk_work'):
        for seed in (17,29):
            for arm in ('frozen_base_knowledge_feedback','action_only','decision_action'):
                ident=f'{mode}_{seed}_{arm}'; public=root/'cells'/ident
                p=dict(prepared_directory=str(root/'prepared'), public_directory=str(public), adapter=None,
                    code_sha256={n:'fixture' for n in ('export_catalog_geometry.py','inspect_native_mesh_contacts.py','native_project.py')})
                if arm!='frozen_base_knowledge_feedback':p['adapter_pending_binding']=str(root/'training'/arm/'adapter')
                save(public/'TEMPLATE.json',p)
                cells.append(dict(id=ident,condition=mode,seed=seed,arm=arm,template=str(public/'TEMPLATE.json'),runtime=str(root/'runtime'/ident)))
    study=dict(run_id='CPU_FIXTURE',public_directory=str(root),runtime_directory=str(root/'runtime'),native_python='CPU_FIXTURE',
        evaluation_gpu_seconds=7200,training_plan=dict(path=str(root/'TRAIN_PLAN.json'),sha256=train_sha),cells=cells,
        frozen_files={c['template']:sha256(Path(c['template'])) for c in cells})
    save(root/'STUDY.json',study)
    save(root/'AUTH.json',dict(approved=True,user_messages=['CPU fixture only, no actual authorization'],
        study_plan_sha256=sha256(root/'STUDY.json'),plan_sha256=train_sha))
    calls=[]
    def child(command):
        calls.append(command);out=Path(command[command.index('--out')+1]);receipt=Path(command[command.index('--receipt-dir')+1])
        save(receipt/'receipt.json',dict(exit_code=1 if fail_first else 0,fixture_only=True))
        if 'run_edit_policy_executed.py' in ' '.join(command):
            initial=dict(checks=[dict(requirement='preserve',**{'pass':True}),dict(requirement='bath',**{'pass':False})])
            final=dict(checks=initial['checks'],requested_activity_status='pass',measured_requirements_status='fail')
            save(out/'RESULT.json',dict(status='failed' if fail_first else 'policy_completed_pending_comparison',
                initial_review=initial,final_audit=dict(review=final),model_calls=1,native_calls=1,tool_attempts=1,gpu_subprocess_seconds=1.))
            save(out/'final_checkpoint/manifest.json',{})
        else:save(out/'RESULT.json',dict(fixture_only=True))
        return 1 if fail_first else 0
    argv=['run_terminal_edit_evaluation.py','--study',str(root/'STUDY.json'),'--authorization',str(root/'AUTH.json')]
    with patch.object(runner.subprocess,'call',child),patch.object(sys,'argv',argv):
        try:runner.main()
        except RuntimeError:
            if not fail_first:raise
        else:
            if fail_first:raise AssertionError('Failed cell did not stop batch')
        try:runner.main()
        except FileExistsError:pass
        else:raise AssertionError('Duplicate batch accepted')
    result=json.loads((root/'EVALUATION_BATCH.json').read_text())
    assert len(result['cells'])==(1 if fail_first else 12)
    assert len(calls)==(1 if fail_first else 24)
    assert all('train_' not in ' '.join(c) for c in calls)
    assert all(r['absolute_failed_checks']==['bath'] and r['whole_room_success']=='unknown' for r in result['cells'])
    assert result['status']==('failed' if fail_first else 'completed_pending_scientific_review')
    if not fail_first:assert len((root/'LEARNING_RESULTS.csv').read_text().splitlines())==13
    return dict(cells=len(result['cells']),simulated_subprocesses=len(calls),duplicate_batch_rejected=True,
                inherited_failure_retained=True,model_calls=0,native_calls=0,training_updates=0)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    result=dict(success=exercise(args.out/'success'),failure=exercise(args.out/'failure',True))
    save(args.out/'RESULT.json',dict(status='passed',fixtures=result,scope='CPU service doubles only'))
    print('PASS: 12-cell accounting/export, first-error stop, duplicate rejection, retained absolute failure; zero actual model/native/training calls.')
