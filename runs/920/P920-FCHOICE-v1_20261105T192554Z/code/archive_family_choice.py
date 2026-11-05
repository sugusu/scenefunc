"""Archive executed alternatives and condition-specific objective review records."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import jsonschema
from PIL import Image
from native_project import sha256
from native_checkpoint import canonical_hash


def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--allowed-failed-stage',action='append',default=[]);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];p=repo/'runs/920'/a.run;t=repo.parent/'runtime'/a.run
    choice=read(p/'CHOICE_PLAN.json');measurement=read(p/'MEASUREMENT.json');visual=read(p/'VISUAL_REVIEW.json')
    now=datetime.now(timezone.utc).isoformat();copies=[];summaries={};candidate_inputs={}
    def cp(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert sha256(src)==sha256(dst)
        copies.append(dict(source=str(src),path=str(dst.relative_to(p)),sha256=sha256(src)))
    for label in choice['candidates']:
        n=t/label/'native';receipt=read(t/label/'native_stage/receipt.json');assert receipt['exit_code']==0
        steps=read(n/'STEPS.json');assert len(steps)==1 and steps[0]['reply']['results'][0]['output']['success'] is True
        initial_observation=read(n/'initial_observation.json')
        tools,_=json.JSONDecoder().raw_decode(initial_observation['system'].split('Available executable tools:\n',1)[1])
        schemas={x['name']:x['parameters'] for x in tools}
        call=steps[0]['command'];jsonschema.Draft202012Validator(schemas[call['name']]).validate(call['arguments'])
        for index,key in enumerate(['initial','step_01']):
            observation=read(n/(key+'_observation.json'));binding=read(Path(observation['state_binding']))
            state=read(n/(key+'_state.json'))
            assert binding['state_sha256']==canonical_hash(state)==canonical_hash(binding['native_state'])
            assert len(observation['image_paths'])==len(binding['images'])==5
            cp(Path(observation['state_binding']),p/label/'bindings'/Path(observation['state_binding']).name)
            for path,b in zip(observation['image_paths'],binding['images']):
                src=Path(path);assert src.resolve()==Path(b['path']).resolve() and sha256(src)==b['sha256']
                with Image.open(src) as im:im.verify()
                if index==1 or (label=='perimeter' and src.name.endswith('.topdown.png')):
                    cp(src,p/label/'views'/src.name)
        for src in n.glob('*.json'):cp(src,p/label/'native'/src.name)
        for folder in ['preflight','export','cold_restore']:
            cp(t/label/folder/'RESULT.json',p/label/folder/'RESULT.json')
        cp(t/label/'FINAL_AUDIT.json',p/label/'FINAL_AUDIT.json')
        for name in ['manifest.json','house_state.json','runtime.json','task.json','catalog.json','session_checkpoint.json']:
            src=n/'checkpoint'/name
            if src.exists():cp(src,p/label/'checkpoint_metadata'/name)
        audit=read(t/label/'FINAL_AUDIT.json');restore=read(t/label/'cold_restore/RESULT.json')
        assert restore['receipt_count']==1 and restore['asset_bindings_exact'] and restore['reserved_ids_preserved']
        m=measurement['candidates'][label]
        summaries[label]=dict(native_receipt=receipt,measured=m,triangle_contact_pairs=audit['surface_intersection_pairs'],
            outside_room=sum(x['outside_nominal_room_xy'] for x in audit['bounds']),
            generic_failed_approaches={str(s['square_body_width_m']):[name for name,x in s['targets'].items()
                if not x['reachable_from_entry']] for s in audit['access']['scenarios']},
            export=read(t/label/'export/RESULT.json')['glb'],checkpoint=str(n/'checkpoint'),
            checkpoint_manifest_sha256=sha256(n/'checkpoint/manifest.json'),
            restore_receipt=read(t/label/'restore_stage/receipt.json'))
        candidate_inputs[label]=dict(state_sha256=m['final_state_sha256'],
            final_views=[dict(path=x,sha256=sha256(Path(x))) for x in read(n/'final_observation.json')['image_paths']],
            actual_command=call,action_record_sha256=sha256(n/'STEPS.json'))
    stages=[];failed=[];excluded_running=[]
    for src in t.rglob('receipt.json'):
        if not src.parent.name.endswith('_stage'):continue
        receipt=read(src)
        relative=src.parent.relative_to(t)
        if receipt['status']=='running':
            assert any(Path(x).name==Path(__file__).name for x in receipt['command']), 'Other stage still running'
            excluded_running.append(str(relative));continue
        if receipt['exit_code']!=0:
            assert str(relative) in a.allowed_failed_stage, 'Unacknowledged failed stage'
            failed.append(str(relative))
        for name in ['receipt.json','stage.log']:cp(src.parent/name,p/'execution'/relative/name)
        stages.append(dict(stage=str(relative),receipt=receipt))
    assert set(failed)==set(a.allowed_failed_stage)
    for name in ['prepare_family_choice.py','revise_family_choice_orientation.py','configure_family_choice.py',
                 'run_family_choice_batch.py','construct_and_switch_room.py','measure_family_choice.py',
                 'audit_family_choice_geometry.py','postprocess_family_choice.py','archive_family_choice.py']:
        cp(repo/'scripts/920'/name,p/'code'/name)
    initial=read(t/'perimeter/native/initial_state.json')
    packet=dict(created_at=now,source=read(p/'SOURCE_RECORD.json'),task_variants=read(p/'TASK_VARIANTS.json'),
        input_manifest=read(p/'INPUT_LOCK.json'),researcher_initial_state=initial,
        candidates=candidate_inputs,measured_consequences=measurement,
        source_evidence_supports='Keeping existing furniture within a relatively open common-living organisation.',
        source_does_not_support=['Our dimensions, poses or current source-object IDs.',
            'The contrasting visual-distraction request or a professional ranking of these layouts.'],
        historical_designer_trajectory=False,independent_professional_review=False,
        visual_review=visual,training_eligible=False,training_authorized=False)
    write(p/'DECISION_PACKET.json',packet)
    review=[]
    for variant in packet['task_variants']['variants']:
        review.append(dict(id=a.run+'_'+variant['id'],
            input=dict(original_request=None,common_request=packet['task_variants']['shared_task']['design_request'],
                condition_request=variant['request'],candidate_states_and_views=candidate_inputs,
                declared_indicator=variant['indicator']),
            evaluation_only=dict(kind='programmatic_conditional_geometry_label_not_expert_preference',
                ranking=measurement['conditional_proxy_rankings'][variant['id']],
                exact_measurements={label:dict(clear_rays=x['measured']['clear_rays'],
                    all_declared_access=x['measured']['all_declared_access']) for label,x in summaries.items()},
                limitations=measurement['limits'],visual_review_status=visual['status']),
            provenance=dict(family_id=choice['family_id'],task_origin=variant['provenance']),
            training_eligible=False))
    write(p/'REVIEW_CANDIDATES.json',dict(status='two_condition_objective_review_candidates_not_professional_labels',
        records=review,source_family_count=1,independent_project_count=1,
        condition_variants_are_not_independent_projects=True,model_calls=0,training_updates=0))
    write(p/'ARCHIVE_CHECKS.json',dict(recorded_at=now,native_actions=2,actual_observations=4,
        image_hash_and_decode_checks=20,actual_schema_valid_calls=2,public_views=11,
        visual_review_file='VISUAL_REVIEW.json',stages=stages,copies=copies,
        failed_postprocessing=failed,currently_excluded_own_receipt=excluded_running,
        summaries=summaries,model_calls=0,training_updates=0))
    recovery=['# 恢复记录','', '完整资产、GLB和可编辑检查点留服务器；以下命令记录不重新生成结果。','']
    for label,x in summaries.items():
        recovery += [f"- {label}: `{x['checkpoint']}`，manifest SHA256 `{x['checkpoint_manifest_sha256']}`。",
                     f"  GLB `{x['export']['path']}`，SHA256 `{x['export']['sha256']}`。"]
    recovery += ['', '各候选 RESTORE_PLAN.json 和 execution 下的实际命令给出固定原生环境及冷加载流程。复查选择新输出目录，不能覆盖既有receipt或结果。原项目版本、上游运行库与资产指纹在checkpoint元数据中；本机丢失大型资产时须按绑定来源重新取得对应授权资产，Git中的哈希不能重建资产本身。']
    (p/'RECOVERY.md').write_text('\n'.join(recovery)+'\n')
    print(json.dumps(dict(run=a.run,copied_files=len(copies),actions=2,views_checked=20,review_candidates=2)))


if __name__=='__main__':main()
