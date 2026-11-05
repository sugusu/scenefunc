"""Report a terminal independent-room attempt against the retained requirements.

This is evidence accounting, not a new trained judge or an aggregate score.
Role names identify catalog proxies; they do not establish human functionality.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from native_project import sha256, validate_bundle
from native_checkpoint import canonical_hash, rotation_roundtrip_differences


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


ROLE_NAMES={
 'sleep':{'queen_bed'},'cook':{'cooker','kitchen_counter','kitchen_sink','refrigerator'},
 'hygiene':{'toilet','shower','washroom_vanity'},'storage':{'bookshelf','storage_bin'},
 'daily_work_meal':{'dining_table','dining_chair_1'},'guest_meal':{'dining_table','dining_chair_1'},
}
UNKNOWN={
 'sleep':['Sleep surface, approach and actual use are not certified by asset presence.'],
 'cook':['Water/power/heat/exhaust installation and appliance articulation are unverified.'],
 'hygiene':['Door leaves/closed-state privacy, plumbing and sanitary operation are unverified.'],
 'storage':['Internal volume, required capacity and opening/retrieval motions are unverified.'],
 'daily_work_meal':['Desk/dining surface, seated reach and transition effort are unverified.'],
 'guest_meal':['Two simultaneous usable places and retained sleeping/hygiene require evidence; seats alone do not prove them.'],
 'access':['Square-body grid routes are declared conservative proxies, not human or accessibility certification.'],
 'editable_complete':['Native manifest closure and geometry preview are checked; a cold native session restore remains separate.'],
 'terrace_access':['Only an indoor approach point is measured; door operation, threshold level and outdoor route remain unknown.'],
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    plan=read(a.plan);root=Path(plan['runtime_directory']);roll=root/'rollout';receipt=read(root/'rollout_execution/receipt.json');result=read(roll/'RESULT.json')
    if receipt['status']!='completed' or receipt['exit_code']!=0 or result['status']!='policy_attempt_completed_pending_independent_audit':
        raise ValueError('Only a terminal completed attempt may receive this final report; failures need a separately labelled partial report')
    reqspec=plan['evaluation']['requirements'];assert sha256(Path(reqspec['path']))==reqspec['sha256'];reqs=read(reqspec['path'])['requirements'];assert len(reqs)==9
    audit=read(root/'FINAL_AUDIT.json');checkpoint=roll/'final_checkpoint'
    manifest,state,runtime,task=validate_bundle(checkpoint);rid=manifest['selected_room_id'];native=state['rooms'][rid]
    assert audit['input_manifest_sha256']==sha256(checkpoint/'manifest.json')
    rotation_roundtrip_differences(read(roll/'final_state.json'),native)
    obs=read(roll/'final_observation.json');binding=read(obs['state_binding'])
    assert canonical_hash(binding['native_state'])==binding['state_sha256']
    rotation_roundtrip_differences(binding['native_state'],native)
    assert len(obs['image_paths'])==5 and obs['image_paths']==[x['path'] for x in binding['images']]
    for image in binding['images']:assert sha256(Path(image['path']))==image['sha256']
    exported=read(root/'export/RESULT.json');assert exported['input_manifest_sha256']==sha256(checkpoint/'manifest.json')
    assert sha256(Path(exported['glb']['path']))==exported['glb']['sha256']
    actual_ids={oid for oid,o in native['objects'].items() if o['object_type']=='furniture'}
    assert actual_ids=={x['object_id'] for x in audit['inventory']}==set(exported['furniture_object_ids'])
    models=[];actions=[];bindings_verified=0
    for directory in sorted(roll.glob('round_*')):
        model=read(directory/'model/RESULT.json');assert model['status']=='completed'
        assert model['request_sha256']==sha256(directory/'request.json')
        assert model['training_updates']==0 and model['dtype']=='bfloat16'
        assert model.get('adapter')==plan['adapter']
        models.append(model)
        reply_path=directory/'tool_reply.json'
        if not reply_path.exists():continue
        cp=directory/'checkpoint';mm,ss,rr,tt=validate_bundle(cp)
        observation=read(directory/'observation.json');bb=read(observation['state_binding'])
        assert canonical_hash(bb['native_state'])==bb['state_sha256']
        rotation_roundtrip_differences(bb['native_state'],ss['rooms'][rid])
        assert len(observation['image_paths'])==5 and observation['image_paths']==[x['path'] for x in bb['images']]
        for image in bb['images']:assert sha256(Path(image['path']))==image['sha256']
        bindings_verified+=1
        calls=read(directory/'parsed_calls.json');replies=read(reply_path)['results'];assert len(calls)==len(replies)==1
        actions.append({'turn':directory.name,'requested':calls[0],'actual_reply':replies[0],'native_success':isinstance(replies[0].get('output'),dict) and replies[0]['output'].get('success') is True,'checkpoint_manifest_sha256':sha256(cp/'manifest.json'),'after_state_sha256':bb['state_sha256']})
    assert len(models)==result['model_calls']<=plan['budget']['model_calls'];assert len(actions)==result['tool_calls']<=plan['budget']['tool_calls']
    assert models[0]['request_sha256']==read(Path(plan['prepared_directory'])/'RESULT.json')['request_sha256']
    counts=dict(Counter(x['name'] for x in audit['inventory']));contacts=[x['objects'] for x in audit['positive_aabb_pairs'] if x['triangle_surface_intersection_detected']]
    requirements=[]
    for req in reqs:
        key=req['id'];roles=ROLE_NAMES.get(key,set());objects=[x for x in audit['inventory'] if x['name'] in roles];ids={x['object_id'] for x in objects}
        access=[]
        for scenario in audit['access']['scenarios']:
            if key=='access':targets=scenario['targets']
            elif key=='terrace_access':targets={n:v for n,v in scenario['targets'].items() if n=='terrace_inside_approach'}
            else:targets={n:v for n,v in scenario['targets'].items() if n.split(':')[0] in ids or (key=='hygiene' and n=='bath_inside_approach')}
            if targets:access.append({'square_body_width_m':scenario['square_body_width_m'],'targets':targets})
        row={'id':key,'requirement':req['requirement'],'activity_states':req['states'],'overall_status':'unknown_not_accepted','catalog_proxy_objects':objects,'conditional_access':access,'observed_triangle_contacts':[c for c in contacts if ids.intersection(c)],'unknowns':UNKNOWN[key],'evidence_needed':req['evidence_needed']}
        if roles:row['available_catalog_role_counts']={name:counts.get(name,0) for name in sorted(roles)}
        if key=='guest_meal':row['observed_dining_chairs']=counts.get('dining_chair_1',0)
        if key=='editable_complete':row['verified_subclaims']={'native_manifest_closed':True,'actual_furniture_ids_preserved_in_export':True,'five_views_match_final_state':True,'per_action_checkpoints_and_images_verified':bindings_verified,'cold_native_restore_tested':False,'glb':exported['glb']}
        if key in ['access','terrace_access']:
            row['proxy_results']=[{'square_body_width_m':s['square_body_width_m'],'all_declared_targets_reachable':bool(s['targets']) and all(v['reachable_from_entry'] for v in s['targets'].values()),'unreachable_targets':[n for n,v in s['targets'].items() if not v['reachable_from_entry']]} for s in access]
        requirements.append(row)
    report={'run_id':plan['run_id'],'arm':plan['comparison_arm'],'reported_at':datetime.now(timezone.utc).isoformat(),'status':'terminal_attempt_evidence_reported','plan_sha256':sha256(a.plan),'actual_exit_code':receipt['exit_code'],'actual_started_at':receipt['started_at'],'actual_ended_at':receipt['ended_at'],'actual_seconds':receipt['elapsed_seconds'],'stop_reason':result['stop_reason'],'model_calls':len(models),'tool_calls':len(actions),'native_accepted_actions':sum(x['native_success'] for x in actions),'failed_actions':[x for x in actions if not x['native_success']],'output_tokens':sum(x['output_tokens'] for x in models),'generation_seconds':sum(x['generation_seconds'] for x in models),'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in models),'input_token_range':[min(x['encoding']['input_tokens'] for x in models),max(x['encoding']['input_tokens'] for x in models)],'training_updates':0,'inventory_counts':counts,'furniture_count':len(actual_ids),'furniture_ids':sorted(actual_ids),'triangle_contacts':contacts,'outside_nominal_interior':[x['object_id'] for x in audit['bounds'] if x['outside_nominal_room_xy']],'immutable_records_unchanged':audit['immutable_records_unchanged'],'requirements':requirements,'actions':actions,'checkpoint':str(checkpoint),'final_state_sha256':canonical_hash(native),'final_view_binding':str(obs['state_binding']),'glb':exported['glb'],'professional_review':False,'whole_room_accepted':False,'generalization_claim':False,'scope':'Postprocessing the previously declared nine requirements and fixed geometric diagnostics. Missing human/functional evidence remains unknown; object presence and low loss are not success. No aggregate score or new threshold.'}
    a.out.mkdir(exist_ok=False);write(a.out/'REPORT.json',report)
    lines=['# '+plan['comparison_arm']+'：整屋输出证据','','实际运行 '+str(round(receipt['elapsed_seconds'],3))+' 秒，退出0；'+str(len(models))+' 次模型调用、'+str(len(actions))+' 次工具动作。','','| 用途 | 目录对象证据 | 结论范围 |','|---|---|---|']
    for x in requirements:lines.append('| '+x['id']+' | '+(', '.join(y['object_id'] for y in x['catalog_proxy_objects']) or '见路径/状态证据')+' | 未知项保留，未验收完整用途 |')
    lines+=['','实际三角面接触：`'+json.dumps(contacts,ensure_ascii=False)+'`。','越出室内边界：`'+json.dumps(report['outside_nominal_interior'],ensure_ascii=False)+'`。','','完整分项、实际动作、路径场景和成本见 REPORT.json。这是单个未封存研究房壳的描述性输出，不是专业、训练收益或泛化结论。']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['arm','model_calls','tool_calls','furniture_count','whole_room_accepted']}),flush=True)


if __name__=='__main__':main()
