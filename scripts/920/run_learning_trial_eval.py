"""Bind final approved adapters, run the frozen gate, then conditional development.

No training, checkpoint selection, retries, or budget increases. Existing rooms
are restored afresh for each arm. Default CLI only verifies the frozen proposal.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from native_project import sha256
from score_activity_transition import checked_review


def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def locked(p,h):
    if sha256(Path(p))!=h:raise ValueError('Frozen input changed: '+str(p))


def validate(path):
    p=read(path)
    if p['schema']!='bounded-room-learning-v1' or p['gate_rule']!='any_trained_arm_all_four_without_new_failure':raise ValueError('Unsupported trial')
    t=p['training_plan'];locked(t['path'],t['sha256']);train=read(t['path'])
    if train['arm_order']!=['action_only','paired'] or train['total_training_seconds']!=p['budget']['training_seconds']:raise ValueError('Training arms/budget changed')
    for name,h in p['inputs'].items():locked(name,h)
    for stage,n in [('gate',12),('development',12)]:
        cells=p['stages'][stage]
        if len(cells)!=n or len({c['cell_id'] for c in cells})!=n:raise ValueError('Wrong cell matrix')
        calls=seconds=0;groups={}
        for c in cells:
            for name,h in c['frozen_inputs'].items():locked(name,h)
            q=read(c['template']);calls+=q['budget']['model_calls'];seconds+=q['budget']['gpu_seconds_per_cell']
            if c['arm'] not in ['frozen_base','action_only','paired'] or q['adapter'] is not None:raise ValueError('Template must not contain old adapter')
            if q['cell_id']!=c['cell_id'] or q['inference']['seed']!=c['seed']:raise ValueError('Cell identity mismatch')
            for name,h in q['code_sha256'].items():locked(Path(__file__).parent/name,h)
            locked(q['upstream_parser']['path'],q['upstream_parser']['sha256'])
            # Exactly the same prompt, measurements, knowledge, tools and budget within a condition/seed.
            comparable=copy.deepcopy(q)
            for field in ['cell_id','public_directory','runtime_directory']:comparable.pop(field)
            key=(c['condition'],c['seed'])
            if key in groups and groups[key]!=comparable:raise ValueError('Unequal evaluation information/budget')
            groups[key]=comparable
        if calls!=p['budget'][stage+'_model_calls'] or seconds!=p['budget'][stage+'_gpu_seconds']:raise ValueError('Evaluation ceiling mismatch')
    if p['budget']['total_gpu_subprocess_seconds']!=sum(p['budget'][k] for k in ('training_seconds','gate_gpu_seconds','development_gpu_seconds')):raise ValueError('Total budget mismatch')
    return p,train


def room_score(report, stage):
    if report.get('status')!='policy_completed_pending_comparison':raise ValueError('Incomplete/failed trajectory')
    initial=report['initial_review'];final=report['final_audit']['review']
    before,after=checked_review(initial),checked_review(final)
    if before.keys()!=after.keys() or initial['requested_mode']!=final['requested_mode']:raise ValueError('Changing requirements')
    newly_failed=[k for k in before if before[k] is True and after[k] is False]
    newly_unknown=[k for k in before if before[k] is True and after[k] is None]
    activity=final['measured_requirements_status'] if stage=='gate' else final['requested_activity_status']
    return dict(accepted=activity=='pass' and not newly_failed and not newly_unknown,
        initial_activity=initial.get('requested_activity_status',initial['measured_requirements_status']),final_activity=activity,
        all_declared_requirements=final['measured_requirements_status'],newly_failed=newly_failed,newly_unknown=newly_unknown,
        remaining_failed=[k for k,v in after.items() if v is False],remaining_unknown=[k for k,v in after.items() if v is None],
        valid_completion=report.get('valid_completion',False),response_terminated=report.get('response_terminated',False),
        model_calls=report['model_calls'],native_calls=report['native_calls'],tool_attempts=report['tool_attempts'],
        gpu_subprocess_seconds=report['gpu_subprocess_seconds'],whole_room_success='unknown')


def gate_decision(rows):
    counts={arm:sum(x['arm']==arm and x['score']['accepted'] for x in rows) for arm in ('frozen_base','action_only','paired')}
    if len(rows)!=12 or any(sum(x['arm']==a for x in rows)!=4 for a in counts):raise ValueError('Incomplete gate matrix')
    return dict(continue_development=any(counts[a]==4 for a in ('action_only','paired')),accepted_by_arm=counts,
        rule='At least one trained arm must pass all four complete training-domain trajectories; all three arms evaluated in development if open.')


def bind_adapters(train, root):
    result=read(root/'TRAINING_RESULT.json')
    if result.get('status')!='completed' or result.get('plan_sha256')!=sha256(root/'TRAIN_PLAN.json'):raise ValueError('Training not completed under exact plan')
    adapters={'frozen_base':None};evidence={}
    for arm in train['arm_order']:
        path=Path(train['training_output_root'])/arm/'RUN.json';run=read(path)
        recorded=result['arms'][arm]
        if recorded.get('exit_code')!=0 or run!=recorded['training'] or run.get('status')!='completed' or run.get('optimizer_updates')!=train['updates_per_arm'] or run.get('same_process_adapter_save_zero_reload_exact') is not True:raise ValueError('Final adapter receipt mismatch')
        directory=Path(train['training_output_root'])/arm/'adapter'
        if Path(run['adapter'])!=directory:raise ValueError('Unexpected adapter path')
        adapters[arm]=dict(directory=str(directory),sha256={f:sha256(directory/f) for f in ('adapter_config.json','adapter_model.safetensors')})
        evidence[arm]=dict(training_receipt=str(path),sha256=sha256(path),adapter=adapters[arm])
    write(root/'ADAPTER_BINDING.json',dict(training_result_sha256=sha256(root/'TRAINING_RESULT.json'),arms=evidence))
    return adapters


def execute_stage(p, stage, adapters, root, call=subprocess.call):
    scripts=Path(__file__).parent;cells=[];folder=root/'evaluation'/stage
    for spec in p['stages'][stage]:
        plan=read(spec['template']);plan['adapter']=adapters[spec['arm']]
        target=Path(plan['public_directory'])/'PLAN.json';write(target,plan)
        frozen={**spec['frozen_inputs'],str(target):sha256(target)}
        if plan['adapter']:
            frozen.update({str(Path(plan['adapter']['directory'])/f):h for f,h in plan['adapter']['sha256'].items()})
        cells.append(dict(cell_id=spec['cell_id'],condition=spec['condition'],arm=spec['arm'],plan=str(target),
            command=[p['native_python'],str(scripts/'run_learning_room_policy.py'),'--plan',str(target),'--phase','rollout','--out',spec['output']],
            output=spec['output'],receipt=spec['receipt'],wall_seconds=spec['wall_seconds'],frozen_inputs=frozen))
    write(folder/'STATUS.json',dict(status='ready',completed=0,planned=12))
    batch=dict(run_id=p['run_id'],training_updates=0,cells=cells,budget=dict(model_calls=p['budget'][stage+'_model_calls'],gpu_subprocess_seconds=p['budget'][stage+'_gpu_seconds']))
    path=folder/'BATCH_PLAN.json';write(path,batch)
    command=[sys.executable,str(scripts/'run_stage.py'),'--receipt-dir',str(folder/'batch_receipt'),'--timeout',str(sum(c['wall_seconds'] for c in cells)+120),'--',sys.executable,str(scripts/'run_matched_trajectory_batch.py'),'--plan',str(path)]
    if call(command):raise RuntimeError('First infrastructure failure in '+stage+'; stop without retry')
    rows=[dict(cell_id=c['cell_id'],condition=c['condition'],arm=c['arm'],score=room_score(read(Path(c['output'])/'RESULT.json'),stage)) for c in cells]
    write(folder/'SCORES.json',rows);return rows


def run_evaluation_stages(p, adapters, root, result, progress, execute=execute_stage):
    rows=execute(p,'gate',adapters,root);result['gate']=gate_decision(rows);progress()
    if result['gate']['continue_development']:
        result['development']=execute(p,'development',adapters,root)
        result['status']='completed_pending_scientific_review'
    else:result['status']='stopped_training_domain_gate_failed'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    p,train=validate(a.plan);root=a.plan.parent
    if not a.execute:
        print(json.dumps(dict(status='proposal_validated',training_updates=0,model_calls=0,cells=24)));return
    result_path=root/'EVALUATION_RESULT.json'
    # Atomic single-start claim; incomplete runs cannot be silently restarted.
    with result_path.open('x') as f:json.dump(dict(status='starting'),f)
    now=lambda:datetime.now(timezone.utc).isoformat()
    result=dict(status='running',started_at=now(),plan_sha256=sha256(a.plan),training_updates=0)
    def progress():
        write(result_path,result);status=read(root/'STATUS.json')
        status.update(status=result['status'],state=result['status'],updated_at=now(),latest_verifiable_artifact=str(result_path),failure_reason=result.get('error'))
        write(root/'STATUS.json',status)
        write(root.parent/'latest.json',dict(run_id=p['run_id'],stage=result['status'],updated_at=now(),run_directory=train['runtime_directory']))
    progress()
    try:
        adapters=bind_adapters(train,root)
        run_evaluation_stages(p,adapters,root,result,progress)
        result['exit_code']=0
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',exit_code=1);raise
    finally:
        result['ended_at']=now();progress()

if __name__=='__main__':main()
