"""Export actual table-mode evidence and candidate supervision, not training."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from audit_native_review import observation, export
from native_checkpoint import canonical_hash
from preflight_sourced_trajectory import inspect


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2); f.write('\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args()
    R=Path(__file__).resolve().parents[2];D=R.parent;P=R/'runs/920'/a.run_id
    T=D/'runtime'/a.run_id;replay=T/'replay';prog=read(P/'FROZEN_PROGRAM.json')
    result=read(replay/'RESULT.json');steps=read(replay/'STEPS.json')
    assert result['status']=='native_sequence_completed' and result['tool_calls']==4 and result['observations']==5
    assert [s['command'] for s in steps]==prog['tool_calls']
    ref=lambda p:{'path':str(p.resolve().relative_to(D)),'sha256':sha(p)}
    def candidate(label):
        obs=read(replay/(label+'_observation.json'))
        return {'candidate_id':label,'state':ref(replay/(label+'_state.json')),
            'context':ref(replay/(label+'_context.json')),'observation':ref(replay/(label+'_observation.json')),
            'binding':ref(Path(obs['state_binding']))}
    labels=['initial','step_1','step_2','step_3','after'];rows=[];states={};metrics={}
    for label in labels:
        c=candidate(label);ob=observation(D,c);state=read(replay/(label+'_state.json'))
        m=read(replay/(label+'_measurement.json'));states[label]=state;metrics[label]=m
        geom=inspect(state)
        fixed=[sum(x['reachable_from_entry'] for x in s['targets'].values()) for s in m['fixed_access']['scenarios']]
        seating=[{k:v['reachable_from_entry'] for k,v in s['targets'].items()} for s in m['seat_approaches']]
        rows.append({'state':label,'state_ref':c['state'],'binding_ref':c['binding'],
            'view_paths':ob['images'],'mode_geometry':m['mode_geometry'],'table_gap_m':m['table_geometry']['footprint_gap_m'],
            'fixed_access_counts':fixed,'seat_approaches':seating,'seat_alignment':m['seat_alignment'],
            'actual_geometry':geom,'largest_empty_rectangle_m2':m['floor_rectangle']['largest_grid_empty_rectangle_m2']})
        (P/'views').mkdir(exist_ok=True);shutil.copy2(D/ob['images'][0],P/'views'/(label+'_topdown.png'))
    initial,final=states['initial'],states['after'];moving=['recon_table_b_0','recon_chair_b_0']
    unchanged=[k for k in initial['objects'] if k not in moving]
    assert set(initial['objects'])==set(final['objects']) and len(final['objects'])==27
    assert all(initial['objects'][k]==final['objects'][k] for k in unchanged)
    returns={}
    for oid in moving:
        left=copy.deepcopy(initial['objects'][oid]);right=copy.deepcopy(final['objects'][oid])
        lt=left.pop('transform');rt=right.pop('transform');assert left==right
        translation=float(np.linalg.norm(np.array(lt['translation'])-rt['translation']))
        q1=np.asarray(lt['rotation_wxyz']);q2=np.asarray(rt['rotation_wxyz'])
        angle=math.degrees(2*math.acos(min(1.,abs(float(np.dot(q1,q2)/(np.linalg.norm(q1)*np.linalg.norm(q2)))))))
        returns[oid]={'translation_m':translation,'orientation_degrees':angle}
    checkpoint_files=0
    for i,step in enumerate(steps,1):
        before=read(replay/f'step_{i}_before_state.json');after=read(replay/f'step_{i}_after_state.json')
        assert canonical_hash(before)==step['before_sha256']==canonical_hash(states[labels[i-1]])
        assert canonical_hash(after)==step['after_sha256']==canonical_hash(states[labels[i]])
        assert step['reply']['results'][0]['output']['success'] and step['actual_sweep']['clear_under_declared_sweep_model']
        checkpoint=replay/f'checkpoint_{i}'
        for name,item in read(checkpoint/'manifest.json')['files'].items():
            assert sha(checkpoint/name)==item['sha256'] and (checkpoint/name).stat().st_size==item['bytes'];checkpoint_files+=1
    rowmap={r['state']:r for r in rows}
    required_seats={}
    for label,mode in prog['mode_checkpoints'].items():
        ids=prog['work_chair_ids'] if mode=='separate_work' else list(rowmap[label]['seat_alignment'])
        required_seats[label]=all(all(s[x] for x in ids) for s in rowmap[label]['seat_approaches'])
    checks={'frozen_checkpoint_modes':all(rowmap[k]['mode_geometry'][v] for k,v in prog['mode_checkpoints'].items()),
        'all_fixed_activities_at_every_state':all(r['fixed_access_counts']==[7,7,7] for r in rows),
        'required_seat_approaches':all(required_seats.values()),
        'all_actual_envelopes_clear':all(not r['actual_geometry']['positive_3d_AABB_pairs'] and not r['actual_geometry']['outside_declared_wall_inner_faces'] for r in rows),
        'return_within_frozen_tolerance':all(x['translation_m']<=.01 and x['orientation_degrees']<=.2 for x in returns.values()),
        'all_objects_dimensions_assets_retained':True,'non_target_objects_exact':True,
        'actual_calls_match_frozen_program':True,'states_views_receipts_and_checkpoint_files_verified':True}
    passed=all(checks.values());stamp=datetime.now(timezone.utc).isoformat()
    summary={'run_id':a.run_id,'recorded_at':stamp,'status':'conditional_mode_transition_demonstration' if passed else 'mixed_or_failed_mode_transition',
        'native_tool_calls':4,'observations':5,'images':25,'objects_preserved':27,'non_target_objects_exactly_unchanged':25,
        'rows':rows,'return_errors':returns,'required_seat_approaches':required_seats,'checks':checks,
        'conditional_rule_passed':passed,'checkpoint_files_verified':checkpoint_files,
        'intermediate_cost':'After moving the work stool but before its table, neither complete mode is available; the same holds during the return transition.',
        'mode_tradeoff':'Joined hosting intentionally lacks the separate-table work condition; no universal best layout or simultaneous multi-activity pass is inferred.',
        'professional_preference':None,'whole_room_success':None,'model_generated':False,'training_updates':0,'unknowns':prog['unknowns']}
    save(P/'SUMMARY.json',summary);source=read(P/'SOURCE_RECORD.json')
    packet={'decision_id':source['decision_id'],'family_id':source['family_id'],'allocation':source['allocation'],
        'kind':'researcher_constructed_transfer_of_reported_professional_decision','source':ref(P/'SOURCE_RECORD.json'),
        'source_scope':'Named designer selects two separable/joinable wheeled tables for clients wanting flexible configurations. Four places, room dimensions and movement order are researcher choices.',
        'researcher_assumptions':prog['unknowns'],'initial':candidate('initial'),'hosting':candidate('step_2'),'returned':candidate('after'),
        'actual_steps':ref(replay/'STEPS.json'),'all_step_state_view_bindings':rows,'effects':ref(P/'SUMMARY.json'),
        'outcome':{'conditional_mode_transition':passed,'professional_preference':None,'whole_room_success':None,'core_training_admitted':False},
        'supervision_scope':{'source_decisions':1,'dependent_action_steps':4,'historical_designer_trace':False,
            'professional_principle_in_first_explanation_only':True,'remaining_explanations':'researcher execution plan','actual_loader_checked':False}}
    save(P/'DECISION_PACKET.json',packet)
    rationales=['报道中Dess Chew为需要灵活配置的住户选择两张可分可合的带轮桌。本迁移准备把分桌工作改为合桌待客；先移动第二张桌对应的凳子。这一步次序和坐标是研究者实现，仍须检查实际通路。',
        '把第二张桌移到第一张桌旁，保留桌子实例和尺寸；完成后核实台面关系、四个声明餐位以及其他活动接近。这是研究者操作计划，不是设计师历史命令。',
        '待客阶段结束，把第二张桌移回分桌位置；保留两桌与所有其他对象，接下来再恢复对应工作凳。',
        '把第二张桌的工作凳恢复到声明工作位置；用实际终态核查两个独立工位和整屋接近，不能预先宣称精确回位或物理搬运成功。']
    for i,step in enumerate(steps):
        label=labels[i];ob=observation(D,candidate(label));context=read(replay/(label+'_context.json'))
        objects=[{'object_id':oid,**{k:obj[k] for k in ('name','object_type','bbox_min','bbox_max','transform','immutable')}}
                 for oid,obj in states[label]['objects'].items()]
        inputs={'original_request':None,'design_request':context['design_request'],'request_provenance':'adapted',
            'objects':objects,'available_measurement':metrics[label],'unknowns':prog['unknowns'],
            'current_requested_mode':'joined_four_place_hosting' if i<2 else 'separate_work'}
        if i:inputs['previous_actual_tool_receipt']=steps[i-1]['reply']
        episode={'stage':'furniture','system':'Use move_furniture_tool(object_id,x,y,yaw) in metres/degrees. Preserve objects and dimensions. Only table_b and chair_b are editable. Do not assert future measured success.',
            'task_context':{'scene_description':json.dumps(inputs,ensure_ascii=False,sort_keys=True)},
            'provenance':{'family_id':source['family_id'],'allocation':source['allocation'],'episode_group':a.run_id,
                'step_index':i+1,'kind':'source_guided_reconstructed_action_candidate','reason_origin':'reported source principle plus researcher plan' if i==0 else 'researcher execution plan',
                'historical_designer_trace':False,'training_eligible':False,'training_authorized':False},
            'evaluation_only':{'actual_after_state':ref(replay/f'step_{i+1}_after_state.json'),'summary':ref(P/'SUMMARY.json')},
            'turns':[{'role':'assistant','content':rationales[i],'images':ob['images'],'tool_calls':[step['command']]}]}
        save(P/'ota_candidates'/f'step_{i+1:02d}.json',episode)
    record={'review_id':a.run_id+'_objective_modes','created_at':stamp,
        'provenance':{'kind':'objective_geometric_mode_review','family_id':source['family_id'],'allocation':source['allocation'],
            'unique_review_unit':1,'professional_review':False,'training_eligible':False,'training_authorized':False},
        'brief':read(replay/'initial_context.json')['design_request'],'source_context':{'url':source['professional_evidence']['primary_crosscheck']['url'],'supported':packet['source_scope']},
        'question':'比较分桌、合桌和恢复状态，说明各自支持的声明活动及转换代价；证据是否足以判断真实整屋质量？',
        'assumptions':prog['unknowns'],'candidates':[candidate(x) for x in ('initial','step_2','after')],
        'measurement_input':{x:metrics[x] for x in ('initial','step_2','after')},
        'judgment':{'verdict':'insufficient_evidence','ranking_groups':[],
            'claims':[{'candidate_id':x,'object_ids':prog['table_ids'],'activity':'工作与待客模式',
                'assessment':json.dumps(rowmap[x]['mode_geometry'])+'; fixed access counts '+str(rowmap[x]['fixed_access_counts'])+' under declared proxies. Different requested modes do not create an unconditional ranking.',
                'evidence_ids':['measures_'+x]} for x in ('initial','step_2','after')],
            'evidence':[{'evidence_id':'measures_'+x,'artifact':ref(replay/(x+'_measurement.json'))} for x in ('initial','step_2','after')],
            'unknowns':prog['unknowns']+['No independent professional preference or model learning evidence.']}}
    save(P/'OBJECTIVE_REVIEW.json',record);save(P/'QA_EPISODE_CANDIDATE.json',export(record,D,['initial','step_2','after']))
    save(P/'EXPORT_CHECKS.json',{'state_bound_viewsets':5,'images_checked':25,'checkpoints_checked':4,
        'source_decisions':1,'ota_step_candidates':4,'objective_QA_candidates':1,'new_loader_run':False,'training_admitted':False,
        'future_state_in_evaluation_metadata_only':True,'code_sha256':sha(Path(__file__))})
    shutil.copy2(__file__,P/'code'/Path(__file__).name)
    print(json.dumps({'status':summary['status'],'checks':checks,'return_errors':returns}),flush=True)


if __name__=='__main__':main()
