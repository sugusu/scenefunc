"""Exercise the exact proposal, adapter binding and conditional dispatch on CPU."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from run_learning_trial_eval import validate, room_score, gate_decision, bind_adapters, run_evaluation_stages, execute_stage, read, write
from native_project import sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise ValueError('Do not overwrite validation')
    checks=[];p,train=validate(a.plan);checks.append('frozen_24_cell_matrix_and_equal_information')
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,FileNotFoundError):checks.append(name);return
        raise AssertionError('Not rejected: '+name)
    base={'status':'policy_completed_pending_comparison','model_calls':1,'native_calls':0,'tool_attempts':0,'gpu_subprocess_seconds':1.,
          'initial_review':dict(requested_mode='desk_work',measured_requirements_status='fail',requested_activity_status='fail',checks=[{'requirement':'activity','pass':False},{'requirement':'bath','pass':False},{'requirement':'preserve','pass':True}]),
          'final_audit':{'review':dict(requested_mode='desk_work',measured_requirements_status='fail',requested_activity_status='pass',checks=[{'requirement':'activity','pass':True},{'requirement':'bath','pass':False},{'requirement':'preserve','pass':True}])}}
    scored=room_score(base,'development');assert scored['accepted'] and scored['remaining_failed']==['bath'];checks.append('inherited_failure_retained_without_erasing_activity_gain')
    for value in (False,None):
        bad=copy.deepcopy(base);bad['final_audit']['review']['checks'][-1]['pass']=value
        assert not room_score(bad,'development')['accepted']
    checks.append('new_failure_or_unknown_never_accepted')
    assert not room_score(base,'gate')['accepted'];checks.append('gate_requires_all_declared_checks')
    bad=copy.deepcopy(base);bad['final_audit']['review']['measured_requirements_status']='pass'
    reject('forged_aggregate_rejected',lambda:room_score(bad,'development'))
    arms=['frozen_base','action_only','paired'];rows=[dict(arm=arm,score=dict(accepted=arm=='frozen_base')) for arm in arms for _ in range(4)]
    assert not gate_decision(rows)['continue_development'];checks.append('base_success_alone_cannot_open_training_gate')
    reject('incomplete_gate_rejected',lambda:gate_decision(rows[:-1]))
    for ready,expected in [(False,['gate']),(True,['gate','development'])]:
        rr=copy.deepcopy(rows)
        for x in rr:
            if x['arm']=='paired':x['score']['accepted']=ready
        calls=[];result={}
        def fake(p,stage,adapters,root):calls.append(stage);return rr if stage=='gate' else []
        run_evaluation_stages(p,{},a.plan.parent,result,lambda:None,execute=fake)
        assert calls==expected
    checks.append('actual_dispatch_skips_failed_gate_and_keeps_all_arms_after_pass')
    calls=[]
    def fail(*args):calls.append(args[1]);raise RuntimeError('infrastructure failure')
    reject('first_stage_error_stops',lambda:run_evaluation_stages(p,{},a.plan.parent,{},lambda:None,execute=fail));assert calls==['gate']
    with tempfile.TemporaryDirectory(prefix='learning_control_') as td:
        t=Path(td);bad=copy.deepcopy(p);bad['budget']['gate_model_calls']+=1;write(t/'bad.json',bad)
        reject('budget_tamper_rejected',lambda:validate(t/'bad.json'))
        write(t/'TRAIN_PLAN.json',train)
        synthetic=copy.deepcopy(train);synthetic['training_output_root']=str(t/'training')
        training_result=dict(status='completed',plan_sha256=sha256(t/'TRAIN_PLAN.json'),arms={})
        for arm in train['arm_order']:
            folder=t/'training'/arm;adapter=folder/'adapter';adapter.mkdir(parents=True)
            (adapter/'adapter_config.json').write_text('{}');(adapter/'adapter_model.safetensors').write_bytes(b'fixture only; never loaded')
            run=dict(status='completed',optimizer_updates=8,same_process_adapter_save_zero_reload_exact=True,adapter=str(adapter))
            write(folder/'RUN.json',run);training_result['arms'][arm]=dict(exit_code=0,training=run)
        write(t/'TRAINING_RESULT.json',training_result);bound=bind_adapters(synthetic,t)
        assert bound['frozen_base'] is None and set(bound)==set(arms);checks.append('final_adapter_binding_hashes_actual_files')
        run=read(t/'training/paired/RUN.json');run['optimizer_updates']=7;write(t/'training/paired/RUN.json',run)
        reject('partial_or_disagreeing_training_refused',lambda:bind_adapters(synthetic,t))
        # Real stage function must stop on its subprocess failure before scoring.
        local=copy.deepcopy(p)
        for c in local['stages']['gate']:
            q=read(c['template']);q['public_directory']=str(t/'public'/c['cell_id']);z=t/'templates'/c['cell_id']/'PLAN.json';write(z,q);c['template']=str(z)
        reject('batch_failure_never_scored_or_retried',lambda:execute_stage(local,'gate',bound,t,call=lambda cmd:1))
    write(a.out,dict(status='passed',checks=checks,check_groups=len(checks),model_calls=0,training_updates=0,native_calls=0,
        limitation='Synthetic controller boundaries only; native restore/encoding reported separately.'))
    print(json.dumps(dict(status='passed',check_groups=len(checks))))

if __name__=='__main__':main()
