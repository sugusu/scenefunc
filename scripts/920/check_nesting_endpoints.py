"""Read-only admission checks for new evaluator on saved checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from native_project import sha256, validate_bundle
from nesting_room_metrics import measure_nesting
from score_activity_transition import score_transition


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();p=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    for path,h in p['input_hashes'].items():
        if sha256(Path(path))!=h:raise ValueError('Frozen admission input changed: '+path)
    results={};checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    for c in p['cases']:
        output=a.out/(c['id']+'.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_nesting_room.py')),
            '--plan',c['plan'],'--bundle',c['bundle'],'--out',str(output)],check=True)
        r=json.loads(output.read_text());results[c['id']]=r
        check(c['id']+':activity',r['review']['requested_activity_status']==c['expected_activity'])
        check(c['id']+':bath_failure_preserved',any(x['requirement']=='access_bath_entry_width_0.6' and x['pass'] is False for x in r['review']['checks']))
        check(c['id']+':not_whole_room_pass',r['review']['measured_requirements_status']=='fail' and r['review']['whole_room_success']=='unknown')
    plan=json.loads(Path(p['cases'][1]['plan']).read_text());root=Path(plan['initial_bundle'])
    m,h,_,_=validate_bundle(root);initial=h['rooms'][m['selected_room_id']];room=root/('room_'+m['selected_room_id'])
    program=plan['program'];mode='desk_work'
    baseline=measure_nesting(initial,initial,room,root,program,mode)
    keys={x['requirement'] for x in baseline['review']['checks']}
    synthetic={}
    for name in ('missing_stool','changed_stool','blocked_work_patch'):
        state=copy.deepcopy(initial)
        if name=='missing_stool':del state['objects'][program['stool_id']]
        elif name=='changed_stool':state['objects'][program['stool_id']]['bbox_max'][0]+=.3
        else:state['objects']['recon_mobile_table_0']['transform']['translation']=[-1.64,0,.75]
        r=measure_nesting(state,initial,room,root,program,mode);synthetic[name]=r
        check(name+':denominator',keys=={x['requirement'] for x in r['review']['checks']})
        predicates={x['requirement']:x['pass'] for x in r['review']['checks']}
        if name=='missing_stool':check(name+':fails',predicates['work_stool_in_declared_region'] is False)
        elif name=='changed_stool':check(name+':unknown',predicates['work_stool_in_declared_region'] is None)
        else:check(name+':fails',predicates['desk_work_patch_supported_and_unoccupied'] is False)
    try:measure_nesting(initial,initial,room,root,dict(program,reference_actions=[]),mode)
    except ValueError:checks.append('reference_solution_in_program_rejected')
    else:raise AssertionError('Reference solution in scoring inputs')
    transitions={}
    for mode, count in [('clear_floor',2),('desk_work',4)]:
        score=score_transition(results['initial_'+mode]['review'],results[mode+'_'+mode]['review'],
            attempts=count,native_calls=count,state_changes=count)
        check(mode+':local_gain_not_whole_room_success',bool(score['repaired']) and not score['measured_success'])
        check(mode+':no_added_failed_requirements',not score['newly_failed'])
        transitions[mode]=score
    with (a.out/'RESULT.json').open('x') as f:json.dump(dict(status='passed',checks=checks,
        actual_saved_state_evaluations=len(results),synthetic_only=synthetic,transitions=transitions,
        real_model_calls=0,native_edits=0,training_updates=0),f,ensure_ascii=False,indent=2);f.write('\n')
    print('PASS',len(checks),'new evaluator checks',flush=True)


if __name__=='__main__':main()
