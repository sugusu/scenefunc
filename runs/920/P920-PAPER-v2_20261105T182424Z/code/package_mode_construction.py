"""Bind actual construction observations to shared actions and one sourced reason."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from native_checkpoint import canonical_hash
from native_project import sha256
from compact_catalog_context import OBJECT_FIELDS


def read(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out=a.out.resolve();plan=read(a.plan)
    for name,h in plan['code_sha256'].items():assert sha256(Path(__file__).parent/name)==h,name
    public=Path(plan['public']);native=Path(plan['native']);root=Path(plan['input_root'])
    assert read(native.parent/'native_execution/receipt.json')['exit_code']==0
    result=read(native/'RESULT.json');count=plan['action_count'];assert result['tool_calls']==count and result['observations']==count+1
    assert result['status']=='reference_constructed_pending_independent_audit'
    design=read(public/'SUPERVISION_DESIGN.json');reason_step=design['source_reason_step'];assert 1 <= reason_step <= count
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[];arms={'separate':[],'paired':[]};input_changes=[]
    steps=read(native/'STEPS.json');assert len(steps)==count
    for index,step in enumerate(steps,1):
        label='initial' if index==1 else f'step_{index-1:02d}'
        obs=read(native/(label+'_observation.json'));state=read(native/f'step_{index:02d}_before.json')
        binding=read(Path(obs['state_binding']))
        assert canonical_hash(state)==step['before_sha256']==binding['state_sha256']
        assert canonical_hash(binding['native_state'])==canonical_hash(state)
        if index>1:assert steps[index-2]['after_sha256']==step['before_sha256']
        context= json.loads(obs['room_prompt'].split('\nCURRENT NATIVE STATE:\n',1)[1])
        assert canonical_hash(context)==binding['current_context_sha256']
        assert context['objects']==state['objects']
        assert step['reply']['results'][0]['output']['success'] is True
        refs=[]
        for path,item in zip(obs['image_paths'],binding['images']):
            f=Path(path);assert f.resolve()==Path(item['path']).resolve() and sha256(f)==item['sha256']
            refs.append({'path':str(f.relative_to(root)),'sha256':item['sha256']})
        assert len(refs)==5
        # Keep all physical facts, remove source strategies and supervision bookkeeping.
        inputs={'design_request':context['design_request'],'original_request':context['original_request'],
                'request_provenance':context['request_provenance'],
                'objects':{oid:{key:obj[key] for key in OBJECT_FIELDS} for oid,obj in context['objects'].items()},
                'asset_bindings':context['asset_bindings'],'asset_catalog':context['asset_catalog'],
                'recent_actual_tool_receipts':context['prior_tool_receipts'][-2:],
                'physical_unknowns':plan['physical_unknowns']}
        mode=read(public/'MODE_PLAN.json')
        inputs['activity_context']={'phase':'furnish_everyday' if index<=6 else 'prepare_guest_reserve' if index<=8 else 'restore_everyday',
            'guest_floor_reserve_xy_m':mode['guest_reserve_xy_m'],
            'reserve_provenance':'Researcher task requirement, not original surveyed dimensions or a furniture target pose.',
            'sleeping_use_verified':False}
        if index>=9:
            prior=read(native/'step_06_after.json')
            inputs['previous_observed_everyday_table_and_chairs']={oid:prior['objects'][oid]['transform'] for oid in ['mobile_table_0','chair_left_0','chair_right_0']}
        text=json.dumps(inputs,ensure_ascii=False,sort_keys=True)
        assert design['source_reason'] not in text
        doc={'stage':'furniture','system':obs['system'],
             'task_context':{'scene_description':text},
             'provenance':{'family_id':plan['family_id'],'run_id':public.name,'step':index,
                'kind':'researcher_reconstructed_native_construction_candidate',
                'historical_designer_trajectory':False,'training_eligible':False,'training_authorized':False,'source_reason_only':index==reason_step,'researcher_execution_plan_supervised':False},
             'evaluation_only':{'sentinel':'EVAL_ONLY_SOURCE_CONSTRUCTION',
                'after_state':str(native/f'step_{index:02d}_after.json'),'actual_reply':step['reply']},
             'turns':[{'role':'assistant','content':'','images':[x['path'] for x in refs],
                       'tool_calls':[step['command']]}]}
        documents={}
        if index==reason_step:
            for component in ['reason','paired','separate']:
                candidate=deepcopy(doc)
                if component!='separate':candidate['turns'][0]['content']=design['source_reason']
                if component=='reason':candidate['turns'][0]['tool_calls']=[]
                path=a.out/'episodes'/f'step_{index:02d}'/component/'episode.json';write(path,candidate)
                documents[component]={'path':str(path.relative_to(root)),'sha256':sha256(path),
                                      'supervised_role':'reason' if component=='reason' else 'action'}
            for arm in arms:arms[arm]+=[documents['reason'],documents[arm]]
        else:
            path=a.out/'episodes'/f'step_{index:02d}'/'shared/episode.json';write(path,doc)
            documents['shared']={'path':str(path.relative_to(root)),'sha256':sha256(path),'supervised_role':'action'}
            for arm in arms:arms[arm].append(documents['shared'])
        input_changes.append({'step':index,'raw_observation_sha256':sha256(native/(label+'_observation.json')),
            'original_unknowns_retained_in_raw_record':context['unknowns'],
            'replacement_physical_unknowns':plan['physical_unknowns'],
            'reason':'Source strategy and supervision/lineage metadata are not scene facts or resident constraints. They stay in provenance, not the common prompt.',
            'retained_object_fields':list(OBJECT_FIELDS),'prior_receipts_kept':2})
        rows.append({'step':index,'state_sha256':step['before_sha256'],'binding_sha256':sha256(Path(obs['state_binding'])),
                     'command':step['command'],'images':refs,'documents':documents,
                     'source_principle_target':index==reason_step,'ordinary_action_only':index!=reason_step})
    assert len(arms['paired'])==len(arms['separate'])==count+1
    report={'status':'native_construction_candidates_with_one_source_only_reason',
        'created_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha256(a.plan),
        'independent_source_decisions':1,'native_construction_steps':count,'shared_action_only_steps':count-1,
        'reason_samples_per_arm':1,'action_samples_per_arm':count,'arms':arms,'rows':rows,
        'new_model_calls':0,'training_updates':0,'training_admitted':False,
        'scope':'Actual native actions with state-bound observations. Exactly one source-only reason; researcher coordinates/order are not reason targets. Fixed fixtures, substitutes and unknown functional use retained.'}
    write(a.out/'DATASET.json',report);write(a.out/'INPUT_TRANSFORMATIONS.json',input_changes)
    print(json.dumps({k:report[k] for k in ['status','native_construction_steps','shared_action_only_steps']}))


if __name__=='__main__':main()
