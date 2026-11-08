"""Recorded-output parity and prospective runner control flow with CPU doubles."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import run_upstream_stop_policy as runner
from upstream_termination_policy import load_parser, classify_for_execution
from native_project import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    p = read(args.plan)
    for path, digest in p['inputs'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Frozen source changed: ' + path)
    args.out.mkdir(parents=True, exist_ok=False)
    parser = load_parser(p['upstream_parser'])
    observed = []
    for result_path in p['recorded_results']:
        result = read(result_path)
        for row in result['rounds']:
            raw, record = row['raw_output'], row['model']
            actual = classify_for_execution(raw, record, parser)
            assert actual['upstream_parsed_calls'] == parser(raw)[1]
            if not parser(raw)[1]:
                assert actual['kind'] == 'completion' and actual['stop_reason'] == 'no_tool_call'
            observed.append(dict(file=result_path, turn=row['turn'], kind=actual['kind'],
                                 stop_reason=actual.get('stop_reason')))
    fixture = read(p['fixture_plan'])
    data = read(fixture['dataset'])
    rows = {r['id']: r for r in data['rows']}
    first = rows['waterloo_edit_01']
    audits = [read(rows[name]['balanced_pre_action_measurement']['path'])
              for name in ('waterloo_edit_01', 'waterloo_edit_02')]
    states = [read(a['source_state']['path']) for a in audits]
    replay = Path(fixture['replay'])
    observations = [read(replay/'initial_observation.json'), read(replay/'step_1_observation.json')]
    step = read(replay/'STEPS.json')[0]
    rid = 'waterloo_table_transfer'
    prepared = args.out/'prepared'
    runner.write(prepared/'checkpoint/house_state.json', {'rooms': {rid: states[0]}})
    runner.write(prepared/'RESULT.json', {'status': 'prepared_without_model'})
    legal = '<think>Unclosed, but actual upstream tool extraction applies.\n<tool_call>'+json.dumps(step['command'])+'</tool_call>'
    scenarios = {'malformed_stop':['<think>Finished<|im_end|>'],
                 'valid_stop':['<think>Unknown comfort.</think><|im_end|>'],
                 'extracted_tool_then_stop':[legal, '<think>Finished<|im_end|>'],
                 'infrastructure_error':[None]}
    reports = {}
    for label, outputs in scenarios.items():
        folder = args.out/label
        pointer = dict(state=0, model=0)
        session = SimpleNamespace(scene=SimpleNamespace(room_id=rid,
            to_state_dict=lambda: copy.deepcopy(states[pointer['state']])),
            current_context=lambda: dict(objects=copy.deepcopy(states[pointer['state']]['objects'])))
        server = SimpleNamespace(_state=SimpleNamespace(session=session))
        class Client:
            def __init__(self, app): pass
            def close(self): pass
            def post(self, route, json):
                value = {}
                if route == '/session/observation': value = observations[pointer['state']]
                elif route == '/session/checkpoint':
                    runner.write(Path(json['output_directory'])/'house_state.json', {'rooms':{rid:states[pointer['state']]}})
                elif route == '/session/step':
                    assert json['tool_calls'] == [step['command']]
                    value = step['reply']; pointer['state'] += 1
                return SimpleNamespace(status_code=200, json=lambda:copy.deepcopy(value))
        def child(cmd, env=None):
            directory = Path(cmd[cmd.index('--receipt-dir')+1]); body = cmd[cmd.index('--')+1:]
            target = Path(body[body.index('--out')+1]); code=0
            if body[1].endswith('edit_study_model_step.py'):
                raw=outputs[pointer['model']]; pointer['model']+=1
                if raw is None: code=1
                else:
                    target.mkdir(parents=True); (target/'raw_output.txt').write_text(raw)
                    runner.write(target/'RESULT.json', dict(status='completed', output_tokens=1,
                        raw_output_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                        ended_with_eos=raw.endswith('<|im_end|>'),
                        ended_at_complete_single_action=raw.endswith('</tool_call>'), fixture_only=True))
            else: runner.write(target, audits[pointer['state']])
            runner.write(directory/'receipt.json', dict(elapsed_seconds=0, exit_code=code, fixture_only=True))
            return code
        live = read(fixture['template_plan']); live.pop('interrupted_rollout', None)
        live.update(completion_protocol='upstream-stop-v1', upstream_parser=p['upstream_parser'],
            matched_input=first['input'], session_backend='existing', cell_id=label,
            prepared_directory=str(prepared), public_directory=str(folder/'public'),
            gpu_uuid='GPU-CPU-FIXTURE', gpu_lock_paths=[str(args.out/'fixture.lock')])
        live['evaluation'] = dict(script='evaluate_conditional_room.py', plan=fixture['evaluation'],
                                 sha256=sha256(Path(fixture['evaluation'])))
        # A larger fake budget proves early termination rather than exhaustion.
        live['budget'].update(model_calls=4, tool_calls=4, gpu_seconds_per_cell=100, seconds_per_model_call=30)
        names = set(live['code_sha256']) | set(fixture['runner_dependencies']) | {'run_upstream_stop_policy.py','upstream_termination_policy.py'}
        live['code_sha256'] = {n:sha256(Path(__file__).with_name(n)) for n in names}
        fp=folder/'FIXTURE_NOT_FOR_EXECUTION.json'; runner.write(fp,live)
        error=None
        with patch('existing_project_session.create_app',return_value=(None,server)), \
             patch('fastapi.testclient.TestClient',Client), \
             patch.object(runner.subprocess,'call',child), \
             patch.object(runner.subprocess,'check_output',return_value='GPU-CPU-FIXTURE,0,24000'), \
             patch.object(runner.os,'sched_setaffinity'), \
             patch.object(sys,'argv',['runner','--plan',str(fp),'--phase','rollout','--out',str(folder/'loop')]):
            try: runner.main()
            except RuntimeError as exc: error=str(exc)
        result=read(folder/'loop/RESULT.json')
        assert (folder/'loop/final_state.json').exists() and 'final_audit' in result
        assert result['model_calls'] == len(outputs)
        assert result['native_calls'] == (1 if label=='extracted_tool_then_stop' else 0)
        if label=='infrastructure_error':
            assert error and result['status']=='failed' and not result['response_terminated']
        else:
            assert error is None and result['stop_reason']=='no_tool_call' and result['response_terminated']
            assert result['valid_completion'] == (label=='valid_stop')
            assert result['tool_attempts'] == result['native_calls']
        reports[label] = dict(model_calls_in_fixture=result['model_calls'],
            native_calls_in_fixture=result['native_calls'], stop_reason=result['stop_reason'],
            format_valid=result['valid_completion'], final_state_saved=True)
    runner.write(args.out/'RESULT.json',dict(status='passed',recorded_outputs=observed,
        actual_runner_cpu_doubles=reports,real_model_calls=0,real_native_calls=0,training_updates=0,
        limits='Parser/control-flow acceptance only. Not new trajectories, native rendering or model-quality results.'))
    print(json.dumps(dict(status='passed', recorded_outputs=len(observed), runner_scenarios=len(reports))))


if __name__=='__main__': main()
