"""Wire existing observed room states into a prospective conditional edit task pack.

This reuses evidence, not model/native execution. Evaluation-only references are
kept outside model requests. Source-exposed calibration is not an unseen test.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from edit_request import normalize_context
from executed_edit_request import build_executed_request
from score_activity_transition import score_transition


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); plan=json.loads(a.plan.read_text())
    for path,h in plan['input_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h: raise ValueError('Input changed: '+path)
    a.out.mkdir(parents=True,exist_ok=False)
    refs={}
    def read(path):
        path=Path(path);b=path.read_bytes();refs[str(path)]=hashlib.sha256(b).hexdigest();return json.loads(b)
    def write(name,obj):
        p=a.out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    reviews=read(plan['reviews']); source=Path(plan['source_directory'])
    summary=read(source/'SUMMARY.json');program=read(source/'FROZEN_PROGRAM.json')
    index={(x['state'],x['requested_mode']):x for x in reviews['records']}
    modes=('separate_work','joined_four_place_hosting');tasks=[];scores=[];checks=[]
    print('Build4conditionalTasksAndScoreRecordedTransitions; CPU, zero new scenes/models/updates',flush=True)
    for state in ('initial','step_2'):
        template=read(Path(plan['corpus_inputs'])/('waterloo_edit_01.json' if state=='initial' else 'waterloo_edit_03.json'))
        native=read(source/(state+'_state.json'))
        row=next(x for x in summary['rows'] if x['state']==state)
        binding=read(Path(plan['parent'])/row['binding_ref']['path'])
        if refs[str(Path(plan['parent'])/row['binding_ref']['path'])]!=row['binding_ref']['sha256']:raise ValueError('Binding changed')
        if binding['native_state']!=native:raise ValueError('Image/native state mismatch')
        for im in binding['images']:
            b=Path(im['path']).read_bytes();h=hashlib.sha256(b).hexdigest()
            if h!=im['sha256']:raise ValueError('View changed')
            refs[im['path']]=h
        if [x['path'] for x in binding['images']]!=template['images']:raise ValueError('Different pre-state views')
        for oid,obj in normalize_context(template['context'])['objects'].items():
            if obj['transform']!=native['objects'][oid]['transform']:raise ValueError('Stale object transform')
        for mode in modes:
            case=state+'__'+mode
            doc=read(index[(state,mode)]['document']['path'])
            if refs[index[(state,mode)]['document']['path']]!=index[(state,mode)]['document']['sha256']:raise ValueError('Review input changed')
            observation=json.loads(doc['turns'][0]['content'].split('\n',1)[1])
            inp=copy.deepcopy(template)
            # Fresh task begins in a furnished room; do not replay the source's
            # historic sequence or current-phase rationale as a desired action.
            inp.update(history=[],next_attempt=1)
            ctx=inp['context'];ctx.pop('recent_actual_tool_receipts',None);ctx.pop('previous_actual_tool_receipt',None)
            ctx['current_requested_mode']=mode
            ctx['current_activity_request']=dict(request=(
                '现在两人同时独立工作，各需可用的独立桌面与坐席。' if mode=='separate_work' else
                '现在四人围坐开展共享桌面活动，需要四个可用餐位及坐席。'),
                authorship='researcher_specified_pre_action_activity',
                scope='仅调整本次活动；保留原任务其他用途。现状充分可用时允许不改；停止时列明未知，不宣称整屋验收。')
            ctx['available_measurement']=observation['measurement']
            ctx['declared_activity_requirements']=observation['requirements']
            ctx['current_geometry_evidence']=observation['recorded_geometry_check']
            ctx['current_identity_evidence']=observation['identity_comparison']
            request,accounting=build_executed_request(**inp)
            text=json.dumps(request,ensure_ascii=False)
            if any(x in text for x in ('mode_checkpoints','further_edit_under_measured_scope','stop_action_training_admitted')):
                raise ValueError('Evaluation label leaked')
            write(Path('tasks')/case/'inputs.json',inp)
            write(Path('tasks')/case/'request.json',request)
            write(Path('evaluation_only')/case/'initial_review.json',index[(state,mode)]['review'])
            tasks.append(dict(id=case,initial_state=state,requested_mode=mode,
                source_family='waterloo',split='source_exposed_calibration_only',
                input_file=f'tasks/{case}/inputs.json',request_file=f'tasks/{case}/request.json',
                initial_review=f'evaluation_only/{case}/initial_review.json',
                state_sha256=index[(state,mode)]['state_sha256'],history_accounting=accounting))
            # Existing reference transitions, not a new replay or model response.
            transitions=[('keep_current',state,0),('recorded_two_edits','step_2' if state=='initial' else 'after',2)]
            if state=='initial' and mode=='separate_work':transitions.append(('recorded_full_return','after',4))
            for label,endpoint,cost in transitions:
                scored=score_transition(index[(state,mode)]['review'],index[(endpoint,mode)]['review'],
                    attempts=cost,native_calls=cost,state_changes=cost)
                scores.append(dict(task=case,reference=label,endpoint=endpoint,
                    evidence_kind='recorded_researcher_transition' if cost else 'identity_baseline_no_model',score=scored))
    for task in tasks:
        outcomes=[s for s in scores if s['task']==task['id']]
        if sum(x['score']['measured_success'] for x in outcomes)<1:raise ValueError('No feasible reference')
        checks.append(task['id']+':feasible_reference_and_noop_scored')
    expected={('initial__separate_work','keep_current'):'maintained_declared_requirements',
        ('initial__separate_work','recorded_two_edits'):'introduced_declared_failure',
        ('initial__joined_four_place_hosting','keep_current'):'remaining_declared_deficit',
        ('initial__joined_four_place_hosting','recorded_two_edits'):'resolved_declared_deficit',
        ('step_2__separate_work','keep_current'):'remaining_declared_deficit',
        ('step_2__separate_work','recorded_two_edits'):'resolved_declared_deficit',
        ('step_2__joined_four_place_hosting','keep_current'):'maintained_declared_requirements',
        ('step_2__joined_four_place_hosting','recorded_two_edits'):'introduced_declared_failure',
        ('initial__separate_work','recorded_full_return'):'maintained_declared_requirements'}
    for row in scores:
        if row['score']['outcome']!=expected[(row['task'],row['reference'])]:raise ValueError('Unexpected transition classification')
        checks.append(row['task']+':'+row['reference'])
    r=index[('initial','separate_work')]['review']
    unknown=copy.deepcopy(r);unknown['checks'][0]['pass']=None;unknown['measured_requirements_status']='unknown'
    if score_transition(r,unknown,attempts=0,native_calls=0,state_changes=0)['measured_success']:raise ValueError('Unknown accepted')
    checks.append('unknown_not_success')
    for label,final,cost in [('missing_coverage',{**r,'checks':r['checks'][:-1]},0),
                              ('invalid_cost',r,-1),('changed_condition',index[('initial',modes[1])]['review'],0)]:
        try:score_transition(r,final,attempts=cost,native_calls=0,state_changes=0)
        except ValueError:checks.append(label+':rejected')
        else:raise ValueError('Bad transition accepted: '+label)
    write(Path('TASKS.json'),dict(tasks=tasks,planned_model_calls=0,training_authorized=False,
        independent_test=False,unknowns=program['unknowns'],
        scope='A four-task calibration pack, not a native launch plan. Existing current measurements are visible equally to every arm; reference actions/next states are evaluation-only.'))
    write(Path('evaluation_only/REFERENCE_SCORES.json'),scores)
    write(Path('RESULT.json'),dict(status='passed',checks=checks,tasks=4,reference_transitions=len(scores),
        source_families=1,new_model_calls=0,new_native_calls=0,training_updates=0,
        input_hashes=refs,whole_room_success='unknown',
        note='Reuses previously accepted single-state reviews; adds transition scoring and current-state edit requests. No new professional or performance labels.'))
    print(f'PASS {len(checks)} checks; four tasks, nine reference transitions, no rerun.',flush=True)


if __name__=='__main__':main()
