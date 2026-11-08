"""CPU regression of the native-completion parser, inputs and actual runner.

Archived states/replies and explicit service doubles; zero model/native calls.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import run_native_completion_policy as runner
from native_completion_policy import classify_response, build_matched_session_request
from native_completion_supervision import build_native_request
from native_project import sha256
from native_checkpoint import canonical_hash
from executed_live_policy import bind_outcome
from shared_action_contract_v2 import context_from_request
from native_restoration_chain import validate_chain


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in plan['inputs'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Frozen input changed: ' + path)
    args.out.mkdir(parents=True, exist_ok=False)
    checks, sources = [], {}

    def read(path):
        path = Path(path); sources[str(path)] = sha256(path)
        return json.loads(path.read_text())

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name, action):
        try:
            action()
        except (ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError(name)

    def record(raw, **extra):
        return dict(status='completed', raw_output_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    ended_with_eos=raw.rstrip().endswith(('<|im_end|>', '<|endoftext|>')),
                    ended_at_complete_single_action=raw.rstrip().endswith('</tool_call>'), **extra)

    call = '<tool_call>{"name":"move_furniture_tool","arguments":{"object_id":"x","x":0,"y":0,"yaw":0}}</tool_call>'
    cases = [('tool_boundary', call, 'tool'), ('tool_eos', call+'<|im_end|>', 'tool'),
        ('think_tool', '<think>Move.</think>'+call, 'tool'),
        ('official_completion', '<think>Measured scope only; comfort unknown.</think><|im_end|>\n', 'completion'),
        ('plain_completion', 'No further edit. Comfort unknown.<|endoftext|>', 'completion'),
        ('empty', '<|im_end|>', 'rejection'), ('empty_think', '<think> </think><|im_end|>', 'rejection'),
        ('unterminated', 'Finished', 'rejection'), ('unclosed_think', '<think>Finished<|im_end|>', 'rejection'),
        ('legacy_done', '<done>Finished<|im_end|>', 'rejection'),
        ('two_tools', call+call+'<|im_end|>', 'rejection'),
        ('trailing_text', call+'Finished<|im_end|>', 'rejection'),
        ('nested', '<think>'+call+'</think><|im_end|>', 'rejection'),
        ('misplaced', 'text<think>x</think><|im_end|>', 'rejection'),
        ('malformed', '<tool_call>{broken}</tool_call><|im_end|>', 'rejection'),
        ('duplicate', '<tool_call>{"name":"x","name":"y","arguments":{}}</tool_call><|im_end|>', 'rejection'),
        ('nonfinite', '<tool_call>{"name":"x","arguments":{"x":1e999}}</tool_call><|im_end|>', 'rejection'),
        ('injected_role', '<|im_start|>assistant\nDone<|im_end|>', 'rejection')]
    for name, raw, expected in cases:
        result = classify_response(raw, record(raw))
        check(name, result['kind'] == expected)
        if expected == 'completion':
            check(name+'_not_room_success', result['whole_room_success'] == 'unknown')
    for name, override in [('hash', {'raw_output_sha256':'0'*64}), ('failed', {'status':'failed'}),
                           ('truncated', {'output_rejected_reason':'length_limited'}),
                           ('missing_boundary', {'ended_at_complete_single_action':False})]:
        check(name+'_rejected', classify_response(call, {**record(call), **override})['kind']=='rejection')
    eos_raw = 'Finished<|im_end|>'
    check('unverified_eos_rejected', classify_response(eos_raw, {**record(eos_raw), 'ended_with_eos':False})['kind']=='rejection')
    data = read(plan['dataset'])
    rows = {row['id']:row for row in data['rows'] if row['id'].startswith('waterloo_')}
    replay = Path(plan['replay'])
    observation = read(replay/'initial_observation.json')
    loaded = {}
    for name, row in rows.items():
        inputs = read(row['input']['path'])
        audit = read(row['balanced_pre_action_measurement']['path'])
        state = read(audit['source_state']['path'])
        session = SimpleNamespace(scene=SimpleNamespace(to_state_dict=lambda:copy.deepcopy(state)))
        obs = {**observation, 'image_paths':inputs['images']}
        p = {'matched_input':row['input']}
        got, _ = build_matched_session_request(session,obs,p,[],audit)
        check(name+'_identical_deployment_request', got==read(row['request']['path']))
        bad = copy.deepcopy(audit);bad['state_sha256']='0'*64
        rejects(name+'_stale_measurement', lambda:build_matched_session_request(session,obs,p,[],bad))
        bad = copy.deepcopy(audit);bad['review']['requested_mode']='wrong'
        rejects(name+'_wrong_mode', lambda:build_matched_session_request(session,obs,p,[],bad))
        poisoned = copy.deepcopy(audit);poisoned['review']['secret_gold']='DO_NOT_EXPOSE'
        check(name+'_review_excluded',build_matched_session_request(session,obs,p,[],poisoned)[0]==got)
        loaded[name]=(inputs,audit,state)
    # Use the real failed restoration and its engine-recorded transition.
    restored = read(plan['roundoff_state'])['rooms']['waterloo_table_transfer']
    session_record = read(plan['roundoff_session'])
    transitions = session_record['restoration_transitions']
    source, audit, _ = loaded['waterloo_edit_03']
    audit = copy.deepcopy(audit); audit['state_sha256'] = canonical_hash(restored)
    session = SimpleNamespace(scene=SimpleNamespace(to_state_dict=lambda:copy.deepcopy(restored)),
                              restoration_transitions=transitions)
    got, accounting = build_matched_session_request(session, {**observation, 'image_paths':source['images']},
        {'matched_input':rows['waterloo_edit_03']['input']}, [], audit)
    check('real_roundoff_bridge_preserved',accounting['restoration_transitions']==transitions)
    check('original_action_hash_unchanged',source['history'][-1]['after_sha256'] in json.dumps(got))
    check('actual_restored_pose_visible',context_from_request(got)['objects'] != source['context']['objects'])
    follow = dict(attempt=source['next_attempt'],origin='model_execution',calls=[],
        outcome=dict(kind='validation_rejection',message='fixture rejection',native_call_made=False),
        before_sha256=canonical_hash(restored),after_sha256=canonical_hash(restored))
    got, _ = build_matched_session_request(session, {**observation,'image_paths':source['images']},
        {'matched_input':rows['waterloo_edit_03']['input']},
        [dict(execution={**follow,'attempt':1})],audit)
    check('bridge_then_live_receipt', 'fixture rejection' in json.dumps(got))
    for name, mutate in [
        ('translation',lambda e:e['differences'][0]['path'].__setitem__(-2,'position')),
        ('too_large',lambda e:e['differences'][0].__setitem__('after',1.0)),
        ('tampered_hash',lambda e:e.__setitem__('before_sha256','0'*64)),
        ('nonfinite',lambda e:e['differences'][0].__setitem__('after',float('nan')))]:
        broken=copy.deepcopy(transitions);mutate(broken[-1])
        rejects(name+'_bridge_rejected',lambda:validate_chain(source['history'],broken,canonical_hash(restored)))
    rejects('missing_bridge_rejected',lambda:validate_chain(source['history'],[],canonical_hash(restored)))
    rejects('duplicate_bridge_rejected',lambda:validate_chain(source['history'],transitions*2,canonical_hash(restored)))
    first = rows['waterloo_edit_01']
    states = [loaded['waterloo_edit_01'][2],loaded['waterloo_edit_02'][2],loaded['waterloo_edit_03'][2]]
    audits = [loaded['waterloo_edit_01'][1],loaded['waterloo_edit_02'][1],loaded['waterloo_joined_host_stop'][1]]
    steps = read(replay/'STEPS.json')[:2]
    observations = [observation,read(replay/'step_1_observation.json'),read(replay/'step_2_observation.json')]
    original = read(plan['template_plan'])
    fixture_prepared=args.out/'fixture_prepared'
    rid=states[0]['room_id'] if 'room_id' in states[0] else 'waterloo_table_transfer'
    runner.write(fixture_prepared/'checkpoint/house_state.json', {'rooms':{rid:states[0]}})
    runner.write(fixture_prepared/'RESULT.json', {'status':'prepared_without_model'})
    legal=['<tool_call>'+json.dumps(s['command'])+'</tool_call>' for s in steps]
    completion='<think>Declared hosting proxies checked; comfort remains unknown.</think><|im_end|>'
    scenarios = {
        'two_edits_rejection_completion':legal+['<tool_call>{broken}</tool_call>',completion],
        'no_movement_is_not_completion':[legal[0],completion],
        'budget_exhaustion':['<done>legacy<|im_end|>']*4,
        'model_infrastructure_error':[None]}
    simulation_results={}
    for label,outputs in scenarios.items():
        folder=args.out/label; pointer={'state':0,'model':0}
        def context(): return {'objects':copy.deepcopy(states[pointer['state']]['objects'])}
        session=SimpleNamespace(scene=SimpleNamespace(room_id=rid,to_state_dict=lambda:copy.deepcopy(states[pointer['state']])),current_context=context)
        server=SimpleNamespace(_state=SimpleNamespace(session=session))
        class Client:
            def __init__(self,app):pass
            def close(self):pass
            def post(self,route,json):
                value={}
                if route=='/session/observation':value=observations[pointer['state']]
                elif route=='/session/checkpoint':
                    runner.write(Path(json['output_directory'])/'house_state.json',{'rooms':{rid:states[pointer['state']]}})
                elif route=='/session/step':
                    i=pointer['state']
                    if label=='no_movement_is_not_completion':
                        value={'results':[{'name':'move_furniture_tool','output':{'success':False,'error_type':'no_movement','message':'Synthetic no-movement refusal'}}]}
                    else:
                        if json['tool_calls']!=[steps[i]['command']]:raise ValueError('Wrong recorded call')
                        value=steps[i]['reply'];pointer['state']+=1
                return SimpleNamespace(status_code=200,json=lambda:copy.deepcopy(value))
        def child(cmd,env=None):
            directory=Path(cmd[cmd.index('--receipt-dir')+1]);body=cmd[cmd.index('--')+1:]
            target=Path(body[body.index('--out')+1]);code=0
            if body[1].endswith('edit_study_model_step.py'):
                raw=outputs[pointer['model']];pointer['model']+=1
                if raw is None:code=1
                else:
                    target.mkdir(parents=True);(target/'raw_output.txt').write_text(raw)
                    runner.write(target/'RESULT.json',{**record(raw),'output_tokens':1,'fixture_only':True})
            else:runner.write(target,audits[pointer['state']])
            runner.write(directory/'receipt.json',{'elapsed_seconds':0,'exit_code':code,'fixture_only':True})
            return code
        live=copy.deepcopy(original)
        live.pop('interrupted_rollout',None)
        live.update(matched_input=first['input'],session_backend='existing',cell_id=label,
            prepared_directory=str(fixture_prepared),public_directory=str(folder/'public'),
            gpu_uuid='GPU-CPU-FIXTURE',gpu_lock_paths=[str(args.out/'fixture.lock')])
        live['evaluation']={'script':'evaluate_conditional_room.py','plan':plan['evaluation'], 'sha256':sha256(Path(plan['evaluation']))}
        live['budget'].update(model_calls=len(outputs),tool_calls=4,gpu_seconds_per_cell=100,seconds_per_model_call=30)
        live['code_sha256']={n:sha256(Path(__file__).with_name(n)) for n in set(original['code_sha256'])|set(plan['runner_dependencies'])}
        fp=folder/'FIXTURE_NOT_FOR_EXECUTION.json';runner.write(fp,live)
        error=None
        with patch('existing_project_session.create_app',return_value=(None,server)), \
             patch('fastapi.testclient.TestClient',Client), \
             patch.object(runner.subprocess,'call',child), \
             patch.object(runner.subprocess,'check_output',return_value='GPU-CPU-FIXTURE,0,24000'), \
             patch.object(runner.os,'sched_setaffinity'), \
             patch.object(sys,'argv',['runner','--plan',str(fp),'--phase','rollout','--out',str(folder/'loop')]):
            try:runner.main()
            except RuntimeError as exc:error=str(exc)
        result=read(folder/'loop/RESULT.json');simulation_results[label]=result
        check(label+'_final_state_retained',(folder/'loop/final_state.json').exists() and 'final_audit' in result)
        check(label+'_no_real_execution',result['training_updates']==0)
        if label=='two_edits_rejection_completion':
            check('native_completion_after_full_sequence', result['valid_completion'] and result['stop_reason']=='native_completion')
            check('call_accounting',(result['model_calls'],result['tool_attempts'],result['native_calls'])==(4,3,2))
            check('complete_trajectory_proxy_pass',result['transition_score']['outcome']=='resolved_declared_deficit')
            last=read(folder/'loop/round_003/request.json')
            check('real_rejection_reaches_next_input','validation_rejection' in json.dumps(last))
            check('latest_raw_measurement_reaches_input',context_from_request(last)['available_measurement']==audits[-1]['measurements'])
        elif label=='no_movement_is_not_completion':
            check('refusal_did_not_stop_loop',result['model_calls']==2 and result['native_calls']==1 and result['rounds'][0]['response_classification']['kind']=='tool')
            check('refusal_preserves_actual_state',result['rounds'][0]['execution']['before_sha256']==result['rounds'][0]['execution']['after_sha256'])
        elif label=='budget_exhaustion':
            check('legacy_not_completion',not result['valid_completion'] and result['model_calls']==4 and result['native_calls']==0)
        else:
            check('first_infrastructure_error_stops',error is not None and result['model_calls']==1 and result['status']=='failed' and not result['valid_completion'])
    runner.write(args.out/'RESULT.json',dict(status='passed',checks=checks,input_hashes=sources,
        actual_runner_with_cpu_doubles=True,real_model_calls=0,real_native_calls=0,training_updates=0,
        limits='Service doubles verify control flow, not native rendering, model quality or professional success.'))
    print('PASS',len(checks),'CPU checks; zero real model/native calls.',flush=True)


if __name__=='__main__':main()
