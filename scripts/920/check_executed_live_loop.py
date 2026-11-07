"""Exercise the actual prospective runner with explicit CPU service doubles.

Uses two archived model proposals/replies plus synthetic rejection/stop events.
No model, GPU, rendering, native execution or scientific performance test.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import run_edit_policy_executed as runner
from compact_catalog_context import restore_catalog
from edit_request import tools_and_scope, dumps
from executed_live_policy import POLICY, build_session_request, bind_outcome
from native_project import sha256
from shared_action_contract import context_from_request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--conditional-fixture',action='store_true',help='Exercise new conditional feedback/transition branch with explicitly synthetic review rows')
    p.add_argument('--condition-evaluator',choices=['evaluate_conditional_room.py','evaluate_nesting_room.py'],default='evaluate_conditional_room.py')
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    for path,h in plan['input_hashes'].items():
        if sha256(Path(path))!=h:raise ValueError('Input changed: '+path)
    a.out.mkdir(parents=True,exist_ok=False);sources={}
    def read(path,raw=False):
        path=Path(path);sources[str(path)]=sha256(path);return path.read_text() if raw else json.loads(path.read_text())
    root=Path(plan['recorded_rollout']);old=read(root/'RESULT.json')
    original_plan=read(plan['recorded_plan']);prepared=Path(original_plan['prepared_directory'])
    house=read(prepared/'checkpoint/house_state.json');rid=next(iter(house['rooms']))
    states=[house['rooms'][rid]]+[read(root/f'round_{i:03d}/checkpoint/house_state.json')['rooms'][rid] for i in range(2)]
    messages=[read(root/f'round_{i:03d}/request.json') for i in range(3)]
    observations=[dict(system=m[0]['content'],image_paths=[c['image'] for c in m[1]['content'] if c['type']=='image']) for m in messages]
    contexts=[context_from_request(m) for m in messages]
    raw=[read(root/f'round_{i:03d}/model/raw_output.txt',True) for i in range(2)]
    raw+=['<tool_call>{broken json}</tool_call>', '<done>Declared activity and whole-room quality remain unverified.']
    task=read(prepared/'checkpoint/task.json')
    audits=[read(prepared/'initial_audit.json')]+[read(root/f'round_{i:03d}/audit.json') for i in range(2)]
    if a.conditional_fixture:
        audits=[dict(state_sha256=runner.canonical_hash(s),limits=['Synthetic CPU review, not a room outcome'],
            feedback=dict(synthetic_current_marker='condition_state_'+str(i)),
            review=dict(requested_mode='synthetic_condition',checks=[dict(requirement='synthetic_requirement',**{'pass':i==2})],
                measured_requirements_status='pass' if i==2 else 'fail')) for i,s in enumerate(states)]
    pointer={'state':0,'model':0};events=[]
    def current_context():
        i=pointer['state'];ctx=copy.deepcopy(contexts[i]);ctx['objects']=copy.deepcopy(states[i]['objects'])
        if 'catalog_descriptions' in ctx:
            ctx['asset_catalog']=restore_catalog(ctx);ctx.pop('catalog_descriptions');ctx.pop('catalog_limitations')
        ctx['prior_tool_receipts']=[]
        return ctx
    session=SimpleNamespace(scene=SimpleNamespace(room_id=rid,to_state_dict=lambda:copy.deepcopy(states[pointer['state']])),
        current_context=current_context,project=SimpleNamespace(task=task))
    server=SimpleNamespace(_state=SimpleNamespace(session=session))
    def write(path,data):runner.write(Path(path),data)
    class Client:
        def __init__(self,app):pass
        def close(self):pass
        def post(self,route,json):
            events.append(route);data={}
            if route=='/session/observation':data=observations[pointer['state']]
            elif route=='/session/checkpoint':
                b=Path(json['output_directory']);write(b/'house_state.json',dict(rooms={rid:states[pointer['state']]}))
                data={'fixture_checkpoint':str(b)}
            elif route=='/session/step':
                i=pointer['state']
                if json['tool_calls']!=[old['rounds'][i]['call']]:raise ValueError('Unexpected fixture call')
                data=old['rounds'][i]['reply'];pointer['state']+=1
            return SimpleNamespace(status_code=200,json=lambda:copy.deepcopy(data))
    def child(cmd,env=None):
        # All child processes are intercepted; run_stage itself is tested by
        # the real enclosing receipt, not by these simulated internal receipts.
        directory=Path(cmd[cmd.index('--receipt-dir')+1]);body=cmd[cmd.index('--')+1:]
        write(directory/'receipt.json',dict(elapsed_seconds=0,exit_code=0,fixture_only=True))
        target=Path(body[body.index('--out')+1])
        if body[1].endswith('edit_study_model_step.py'):
            i=pointer['model'];target.mkdir(parents=True)
            (target/'raw_output.txt').write_text(raw[i]);write(target/'RESULT.json',dict(output_tokens=1,fixture_only=True))
            pointer['model']+=1
        else:write(target,audits[pointer['state']])
        return 0
    live=copy.deepcopy(original_plan)
    if a.conditional_fixture:live['evaluation']['script']=a.condition_evaluator
    live.update(input_contract=POLICY,cell_id='cpu_fixture_not_model',public_directory=str(a.out/'fixture_public'),
        gpu_lock_paths=[str(a.out/'fixture.lock')],gpu_uuid='GPU-CPU-FIXTURE',
        budget=dict(model_calls=4,tool_calls=4,gpu_seconds_per_cell=100,seconds_per_model_call=30))
    deps=['run_edit_policy_executed.py','executed_live_policy.py','executed_edit_request.py','edit_request.py',
          'compact_catalog_context.py','shared_action_contract.py','catalog_project_session.py','native_checkpoint.py',
          'run_stage.py','edit_study_model_step.py',live['evaluation'].get('script','evaluate_edit_room.py')]
    if a.conditional_fixture:deps+=['conditional_room_metrics.py','score_activity_transition.py','build_activity_completion_reviews.py']
    if a.conditional_fixture and a.condition_evaluator=='evaluate_nesting_room.py':deps+=['nesting_room_metrics.py']
    live['code_sha256']={n:sha256(Path(__file__).with_name(n)) for n in deps}
    livepath=a.out/'FIXTURE_PLAN_NOT_FOR_NATIVE_LAUNCH.json';write(livepath,live)
    print('Exercise actual runner control flow with CPU service doubles; zero real model/native calls.',flush=True)
    with patch.object(runner,'create_app',return_value=(None,server)), \
         patch('fastapi.testclient.TestClient',Client), \
         patch.object(runner.subprocess,'call',child), \
         patch.object(runner.subprocess,'check_output',return_value='GPU-CPU-FIXTURE,0,24000'), \
         patch.object(runner.os,'sched_setaffinity'), \
         patch.object(sys,'argv',['run_edit_policy_executed.py','--plan',str(livepath),'--phase','rollout','--out',str(a.out/'loop')]):
        runner.main()
    result=read(a.out/'loop/RESULT.json');checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    check('runner_exits_after_done',result['stop_reason']=='model_done' and result['status']=='policy_completed_pending_comparison')
    check('four_simulated_generations_three_attempts_two_native_service_calls',
          (result['model_calls'],result['tool_attempts'],result['native_calls'])==(4,3,2))
    check('restore_observed_and_audited_before_first_request','/session/observation' in events[:4] and (a.out/'loop/restored_audit.json').exists())
    check('actual_recorded_replies_preserved',all(result['rounds'][i]['execution']['outcome']['receipt']==old['rounds'][i]['reply'] for i in range(2)))
    rejection=result['rounds'][2]['execution']
    check('malformed_proposal_not_invented_as_call',rejection['calls']==[] and rejection['outcome']['native_call_made'] is False)
    check('rejection_binds_unchanged_state',rejection['before_sha256']==rejection['after_sha256']==result['rounds'][1]['after_sha256'])
    last=read(a.out/'loop/round_003/request.json');text=last[1]['content'][-1]['text']
    check('next_input_contains_actual_rejection','validation_rejection' in text and rejection['outcome']['message'] in text)
    check('history_keeps_two_complete_attempts',result['rounds'][3]['input_history']['included_attempts']==[2,3])
    check('final_checkpoint_and_audit_retained',(a.out/'loop/final_checkpoint/house_state.json').exists() and 'final_audit' in result)
    check('done_does_not_create_success_label','whole_room_success' not in result)
    if a.conditional_fixture:
        check('current_condition_measurements_in_model_request','condition_state_2' in text)
        check('transition_scored_from_actual_initial_and_final_audits',result['transition_score']['outcome']=='resolved_declared_deficit')
        check('costs_do_not_count_rejection_as_execution',result['transition_score']['costs']['validation_rejections']==1)
        check('condition_pass_never_whole_room_success',result['transition_score']['whole_room_success']=='unknown')
    good=result['rounds'][:3]
    # An extra model reference/gold field is not accepted into shared input.
    clean=build_session_request(session,observations[2],live,good,{},1)
    poisoned=copy.deepcopy(good);poisoned[0]['gold_reason']='DO_NOT_EXPOSE'
    check('gold_fields_not_read',build_session_request(session,observations[2],live,poisoned,{},1)==clean)
    for label,history in [('missing_attempt',good[1:]),('unbound_state',copy.deepcopy(good))]:
        if label=='unbound_state':history[-1]['execution']['after_sha256']='a'*64
        try:build_session_request(session,observations[2],live,history,{},1)
        except ValueError:checks.append(label+':rejected')
        else:raise AssertionError(label)
    try:bind_outcome(1,'a'*64,'b'*64,rejection='invalid')
    except ValueError:checks.append('state_changing_rejection_rejected')
    else:raise AssertionError('state_changing_rejection')
    write(a.out/'RESULT.json',dict(status='passed',checks=checks,input_hashes=sources,
        actual_runner_loop_exercised_with_doubles=True,real_model_calls=0,real_native_calls=0,training_updates=0,
        simulated_model_calls=4,simulated_native_calls=2,
        conditional_fixture=a.conditional_fixture,
        limits='CPU doubles verify wiring and control flow only. Native rendering, model generation, GPU scheduling and live latency are not tested. Archived calls are not rerun.'))
    print(f'PASS {len(checks)} integration checks; no real model/native execution.',flush=True)


if __name__=='__main__':main()
