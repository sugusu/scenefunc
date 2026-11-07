"""Real saved-bundle measurements plus explicitly synthetic adversarial states."""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

from conditional_room_metrics import measure_endpoint
from native_project import sha256, validate_bundle
from score_activity_transition import score_transition


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); p = json.loads(a.plan.read_text()); a.out.mkdir(parents=True, exist_ok=False)
    for path, h in p['input_hashes'].items():
        if sha256(Path(path)) != h:raise ValueError('Input changed: '+path)
    checks, results = [], {}
    def check(label, ok):
        if not ok:raise AssertionError(label)
        checks.append(label)
    def save(name, obj):
        with (a.out/name).open('x') as f:json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False);f.write('\n')
    print('Read-only actual checkpoint measurement; synthetic adversarial states are not scene executions.', flush=True)
    for case in p['cases']:
        target = a.out/(case['id']+'.json')
        subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_conditional_room.py')),
            '--plan', case['evaluation_plan'], '--bundle', case['bundle'], '--out', str(target)], check=True)
        result = json.loads(target.read_text());results[case['id']] = result
        check(case['id']+':expected_condition', result['review']['measured_requirements_status'] == case['expected_status'])
        check(case['id']+':fixed_denominator',len(result['review']['checks']) == case['expected_requirements'])
        check(case['id']+':feedback_bounded',len(json.dumps(result['feedback'],ensure_ascii=False)) < 12000)
        check(case['id']+':feedback_matches_review',result['feedback']['checks'] == {x['requirement']:x['pass'] for x in result['review']['checks']})
    plan = json.loads(Path(p['cases'][0]['evaluation_plan']).read_text())
    root = Path(plan['initial_bundle']); manifest, house, _, _ = validate_bundle(root)
    rid = manifest['selected_room_id'];initial = house['rooms'][rid];room = root/('room_'+rid)
    program = plan['program'];baseline = results['initial_work']['review']
    poisoned = dict(program, tool_calls=[{'reference_action':'never expose'}])
    try:measure_endpoint(initial,initial,room,root,poisoned,'separate_work')
    except ValueError as exc:check('reference_action_program_rejected','declared requirements only' in str(exc))
    else:raise AssertionError('Reference program admitted into feedback')
    synthetic = {}
    def evaluate(label, state):
        result = measure_endpoint(state,initial,room,root,program,'separate_work')
        save('synthetic_'+label+'.json',dict(provenance='in_memory_test_not_native_execution',**result))
        synthetic[label]=result['review']
        check(label+':denominator_preserved', {x['requirement'] for x in baseline['checks']} ==
              {x['requirement'] for x in result['review']['checks']})
        return {x['requirement']:x['pass'] for x in result['review']['checks']}, result
    bad = copy.deepcopy(initial);del bad['objects'][program['table_ids'][1]]
    values, _ = evaluate('missing_table',bad)
    check('missing_table_is_failure_not_crash',values['all_original_instances_and_assets'] is False and values['separate_table_gap'] is False)
    bad = copy.deepcopy(initial);cid=program['work_chair_ids'][1]
    bad['objects'][cid]['bbox_max'][0] += .1
    values, result = evaluate('changed_chair',bad)
    check('replacement_not_silently_measured_as_original',values['all_original_instances_and_assets'] is False and values['assigned_seat_alignment_'+cid] is None)
    bad = copy.deepcopy(initial);bad['objects'][cid]['scale_factor']=2
    try:measure_endpoint(bad,initial,room,root,program,'separate_work')
    except ValueError as exc:check('malformed_nonunit_scaling_explicitly_rejected','Unexpected object scaling' in str(exc))
    else:raise AssertionError('Nonunit scale outside native measurement contract')
    bad = copy.deepcopy(initial);tid=program['table_ids'][1]
    bad['objects'][tid]['transform']['rotation_wxyz']=[math.cos(.2),math.sin(.2),0,0]
    values, _ = evaluate('tilted_table',bad)
    check('unsupported_tilt_not_success',values['separate_table_gap'] is None)
    bad = copy.deepcopy(initial);bad['objects'][cid]['transform']['rotation_wxyz']=[math.cos(.35),0,0,math.sin(.35)]
    values, result = evaluate('yaw_rotated_chair',bad)
    check('arbitrary_chair_yaw_measured',result['measurements']['measurement_support'][cid]=='supported_known_proxy')
    check('yaw_result_has_finite_edge_gap',math.isfinite(result['measurements']['seat_alignment'][cid]['chair_envelope_to_table_edge_gap_m']))
    # No clamping of off-room query points into a reachable boundary cell.
    bad_program=copy.deepcopy(program);bad_program['access_settings']['approach_points_xy_m']['bed_side']=[100.,100.]
    result=measure_endpoint(initial,initial,room,root,bad_program,'separate_work')
    check('off_room_target_cannot_pass',all(not s['targets']['bed_side']['reachable_from_entry'] for s in result['measurements']['fixed_access']['scenarios']))
    transitions={}
    for name, left, right, expected in [
        ('keep_satisfied','initial_work','initial_work','maintained_declared_requirements'),
        ('repair_hosting','initial_host','joined_host','resolved_declared_deficit'),
        ('damage_work','initial_work','joined_work','introduced_declared_failure')]:
        score=score_transition(results[left]['review'],results[right]['review'],attempts=0 if left==right else 2,
            native_calls=0 if left==right else 2,state_changes=0 if left==right else 2)
        check(name+':outcome',score['outcome']==expected);transitions[name]=score
    save('RESULT.json',dict(status='passed',checks=checks,
        actual_saved_bundle_cases=len(p['cases']),synthetic_state_cases=len(synthetic),
        synthetic_off_room_target_cases=1,transitions=transitions,real_model_calls=0,native_edit_calls=0,training_updates=0,
        input_hashes=p['input_hashes'],
        limits='Measurements of saved evidence and in-memory synthetic tests only. No new scene rollout or independent project outcome. Transition action costs refer to historical evidence; synthetic states were not executed.'))
    print('PASS',len(checks),'checks',flush=True)


if __name__=='__main__':main()
