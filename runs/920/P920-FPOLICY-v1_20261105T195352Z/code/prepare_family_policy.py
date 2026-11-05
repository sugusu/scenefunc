"""Freeze two no-training policy attempts; exclude candidate layouts and outcomes."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from native_project import sha256,validate_bundle


def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-run',required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];now=datetime.now(timezone.utc)
    run='P920-FPOLICY-v1_'+now.strftime('%Y%m%dT%H%M%SZ');p=repo/'runs/920'/run;t=repo.parent/'runtime'/run
    p.mkdir();t.mkdir();source=repo/'runs/920'/a.source_run
    source_plan=read(source/'perimeter_PLAN.json');base=read(repo/'runs/920/P920-KBASE-v1_20261105T151743Z/strong/PLAN.json')
    variants=read(source/'TASK_VARIANTS.json');choice=read(source/'CHOICE_PLAN.json')
    obs=repo.parent/'runtime'/a.source_run/'perimeter/native/initial_observation.json'
    observation=read(obs)
    support=('This is an already furnished shared living and dining room, not an empty dwelling. '
        'Preserve all eight existing furniture instances, their dimensions and the immutable room. '
        'The original task plus current activity condition determines what needs changing. '
        'Retain four usable dining places, sofa use and storage access while keeping entrance/window circulation. '
        'The designated observer uses the northeast dining stool recon_seat_ne_0; the other activity is on recon_sofa_0. '
        'Objects may be repositioned; no particular object or target position is prescribed. '
        'The full current object table and five views are already provided, so repeated state queries are not progress. '
        'If needed, get_current_scene_state takes {}. Follow exact advertised tool schemas. '
        'One action per turn, count rejected actions, and use actual feedback. If the room already meets the condition, '
        'you may finish with <done> and describe remaining unknowns. Do not certify human comfort, storage capacity or acoustics.\n')
    write(p/'EVALUATION_PLAN.json',dict(frozen_at=now.isoformat(),source_choice_run=a.source_run,
        initial_bundle=source_plan['bundle'],initial_manifest_sha256=source_plan['manifest_sha256'],
        observer_object_id='recon_seat_ne_0',target_object_id='recon_sofa_0',
        sight=choice['sight'],sight_coordinates='Source eye/target positions converted to local object frames, then followed through actual final poses.',
        access_settings={k:choice['measurement'][k] for k in ['grid_m','assumed_body_height_m','square_body_widths_m','entry_center_xy_m','entry_inward_normal_xy']},
        fixed_access={'entry_inside':[-2.95,-1.6],'window_front':[3.05,1.3]},
        seat_ids=['recon_seat_nw_0','recon_seat_ne_0','recon_seat_sw_0','recon_seat_se_0'],
        table_id='recon_dining_table_0',place_width_depth_m=[.55,.30],seat_table_gap_m=[.05,.4],
        functional_front_ids=['recon_sofa_0','recon_tall_storage_0','recon_sideboard_0'],front_offset_m=.35,
        primary_outcomes=['identity/dimension retention','actual sight-ray count','four nonoverlapping tabletop place rectangles',
            'all role-relative access points at three widths','mesh/contact and schema failures','cost'],
        visual_rule={'shared_visual_contact':'all15 clear','reduced_visual_distraction':'all15 blocked'},
        label_scope='Declared strict geometric indicators, not human preference or complete professional success.',
        code_sha256={n:sha256(repo/'scripts/920'/n) for n in ['evaluate_family_policy.py','measure_family_choice.py','measure_room_access.py','inspect_native_mesh_contacts.py','preflight_sourced_trajectory.py']}))
    plans={}
    for variant in variants['variants']:
        arm=variant['id'];apub=p/arm;art=t/arm;apub.mkdir();art.mkdir()
        bundle=art/'input_bundle';shutil.copytree(source_plan['bundle'],bundle)
        manifest,house,_,task=validate_bundle(bundle)
        task['design_request']=variants['shared_task']['design_request']+'\n当前活动条件：'+variant['request']+'\n本次餐桌活动使用东北侧餐凳（recon_seat_ne_0）。可以自行安排家具位置，也可以在已满足时结束。'
        write(bundle/manifest['inputs']['task'],task)
        manifest['project']={**manifest['project'],'project_id':run+'_'+arm,'provenance':'declared_development_condition_inference_not_sealed_test'}
        manifest['files']={str(f.relative_to(bundle)):dict(bytes=f.stat().st_size,sha256=sha256(f))
            for f in sorted(bundle.rglob('*')) if f.is_file() and f.name!='manifest.json'}
        write(bundle/'manifest.json',manifest);validate_bundle(bundle)
        (apub/'PROMPT_SUPPLEMENT.txt').write_text(support)
        q=copy.deepcopy(base)
        q.update(run_id=run,comparison_arm=arm,condition=arm,frozen_at=now.isoformat(),
            question='Does the frozen base change actual room edits with the visual-contact requirement on an identical furnished initial state?',
            public=str(apub),runtime=str(art),public_directory=str(apub),runtime_directory=str(art),
            prepared_directory=str(art/'prepare'),bundle=str(bundle),manifest_sha256=sha256(bundle/'manifest.json'),
            latest_file=str(art/'latest.json'),adapter=None,
            prompt_supplement={'path':str(apub/'PROMPT_SUPPLEMENT.txt'),'sha256':sha256(apub/'PROMPT_SUPPLEMENT.txt')},
            source_observation=dict(path=str(obs),sha256=sha256(obs),state_binding_sha256=sha256(Path(observation['state_binding']))))
        q['budget'].update(model_calls=4,tool_calls=4,rollout_seconds=900,prepare_seconds=180,output_max_bytes=2*1024**3)
        q['evaluation']={'plan':str(p/'EVALUATION_PLAN.json'),'sha256':sha256(p/'EVALUATION_PLAN.json'),
            'condition_is_researcher_constructed':True,'sealed_test':False}
        q['inference']['context']='Same objects, images, tool schemas, general guidance and budget; only current activity request differs. No source rationale, candidate coordinates, measurement labels or reference actions in input.'
        q['prompt_representation']='furnished_room_two_activity_conditions_same_general_guidance'
        q['code_sha256']={name:sha256(repo/'scripts/920'/name) for name in base['code_sha256']}
        write(apub/'PLAN.json',q);plans[arm]=dict(path=str(apub/'PLAN.json'),sha256=sha256(apub/'PLAN.json'))
    plan=dict(run_id=run,frozen_at=now.isoformat(),public=str(p),runtime=str(t),arms=plans,
        execution_order=list(plans),question='Frozen base under opposite use conditions; no training and no supplied candidate action.',
        budget=dict(arms=2,total_model_calls_max=8,total_tool_calls_max=8,training_updates=0,retries=0),
        code_sha256={name:sha256(repo/'scripts/920'/name) for name in ['run_family_policy_batch.py','run_knowledge_policy.py','catalog_adapter_step.py','run_stage.py']})
    write(p/'PLAN.json',plan)
    write(p/'STATUS.json',dict(run_id=run,status='frozen_prepared_inputs',completed=0,planned=2,
        updated_at=now.isoformat(),configuration_sha256=sha256(p/'PLAN.json'),runtime_directory=str(t),
        failure_reason=None,next_step='Check GPU0, prepare both requests, execute once in fixed order.'))
    write(repo/'runs/920/latest.json',dict(run_id=run,stage='two_condition_policy_frozen',updated_at=now.isoformat(),run_directory=str(t)))
    Path('/tmp/fpolicy_current').write_text(run);print(run)


if __name__=='__main__':main()
