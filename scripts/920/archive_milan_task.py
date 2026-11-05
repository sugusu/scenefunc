"""Archive independent-family task preparation, accepted/rejected references and limits."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from jsonschema import validate
from native_checkpoint import canonical_hash


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args();p=a.public.resolve();t=a.runtime.resolve()
    print(datetime.now(timezone.utc).isoformat(),'archive native task and rejected alternative; no new execution',flush=True)
    gate=read(p/'EXECUTION_GATE.json');assert gate['candidates']['east']['admitted'] and not gate['candidates']['central']['admitted']
    assert not (t/'central/native').exists()
    receipt_paths=[t/'build_stage',t/'initial_audit_stage',t/'final_audit_stage',t/'measurement_stage',t/'restore_stage',t/'export_stage',t/'east/preflight_stage',t/'east/native_stage',t/'central/preflight_stage']
    receipts={}
    for d in receipt_paths:
        rel=d.relative_to(t);v=read(d/'receipt.json');assert v['exit_code']==0
        receipts[str(rel)]=v;shutil.copytree(d,p/rel)
    for name in ['INITIAL_AUDIT.json','FINAL_AUDIT.json','MEASUREMENT.json']:
        shutil.copy2(t/name,p/name)
    for folder in ['build','east/preflight','east/native','central/preflight','cold_restore','export']:
        out=p/folder;out.mkdir(parents=True,exist_ok=True)
        for f in (t/folder).glob('*.json'):shutil.copy2(f,out/f.name)
    for label,bundle in [('input_metadata',t/'build/bundle'),('checkpoint_metadata',t/'east/native/checkpoint')]:
        out=p/label;out.mkdir()
        for n in ['manifest.json','house_state.json','runtime.json','task.json','catalog.json','provenance.json']:
            if (bundle/n).exists():shutil.copy2(bundle/n,out/n)
    native=t/'east/native';steps=read(native/'STEPS.json');assert len(steps)==2
    bindings=[];views=[]
    for i in range(3):
        name='initial' if i==0 else f'step_{i:02d}'
        obs=read(native/(name+'_observation.json'));bpath=Path(obs['state_binding']);binding=read(bpath);state=read(native/(name+'_state.json'))
        assert binding['state_sha256']==canonical_hash(state)==canonical_hash(binding['native_state'])
        context=json.loads(obs['room_prompt'].split('\nCURRENT NATIVE STATE:\n',1)[1]);assert canonical_hash(context)==binding['current_context_sha256']
        assert len(obs['image_paths'])==len(binding['images'])==5
        for item,path in zip(binding['images'],obs['image_paths']):
            f=Path(path);assert f==Path(item['path']) and sha(f)==item['sha256']
            with Image.open(f) as im:im.verify()
            if i in [0,2] or '.topdown.' in f.name:
                (p/'views').mkdir(exist_ok=True);shutil.copy2(f,p/'views'/f.name);views.append(f.name)
        (p/'bindings').mkdir(exist_ok=True);shutil.copy2(bpath,p/'bindings'/bpath.name)
        bindings.append(dict(step=i,state_sha256=binding['state_sha256'],images=binding['images']))
        if i<2:
            tools,_=json.JSONDecoder().raw_decode(obs['system'].split('Available executable tools:\n',1)[1]);schemas={x['name']:x['parameters'] for x in tools};command=steps[i]['command']
            validate(command['arguments'],schemas[command['name']]);assert steps[i]['before_sha256']==binding['state_sha256']
            assert steps[i]['reply']['results'][0]['output']['success']
        if i>0:assert steps[i-1]['after_sha256']==binding['state_sha256']
    generic={}
    for name in ['INITIAL','FINAL']:
        v=read(t/(name+'_AUDIT.json'));assert v['surface_intersection_pairs']==0 and not any(x['outside_nominal_room_xy'] for x in v['bounds'])
        generic[name]=[dict(width_m=s['square_body_width_m'],total=len(s['targets']),reachable=sum(x['reachable_from_entry'] for x in s['targets'].values()),failed=[k for k,x in s['targets'].items() if not x['reachable_from_entry']]) for s in v['access']['scenarios']]
    v=read(t/'MEASUREMENT.json');nr=receipts['east/native_stage'];now=datetime.now(timezone.utc).isoformat()
    summary=dict(run_id=p.name,status='independent_task_reference_completed_with_rejected_alternative',family_id='milan_home_office_set_architects',
        prospective_allocation='test',sealed_test=False,training_or_retrieval_admitted=False,source_layout_reconstructed=False,
        candidate_proposals=2,preflight_admitted=1,actual_native_candidates=1,actual_tool_calls=2,model_calls=0,training_updates=0,
        initial_furniture=13,final_furniture=15,retained_initial_furniture=13,observations=3,images=15,public_visually_reviewed_images=11,
        conditional_access=v['conditional_access'],workspace=v['workspace'],sight_clear=v['sight']['clear'],sight_total=v['sight']['total'],generic_access=generic,
        whole_dwelling_accepted=False,professional_review=False,formal_evaluation_ready=False,
        geometric_diagnostic_input_available=True,multiple_fully_admissible_responses_demonstrated=False,
        native_started_at=nr['started_at'],native_ended_at=nr['ended_at'],native_seconds=nr['elapsed_seconds'],actual_exit_code=0,
        cold_restore_seconds=receipts['restore_stage']['elapsed_seconds'],export_seconds=receipts['export_stage']['elapsed_seconds'],
        export=read(t/'export/RESULT.json')['glb'],interpretation='One actual conditioned geometry reference, not model output or independent design-learning success. Central candidate rejected at0.6m; no retry.')
    write(p/'SUMMARY.json',summary)
    write(p/'VISUAL_REVIEW.json',dict(reviewed_images=views,reviewer='AI inspection, not independent professional',findings=['All original13 furniture remain and desk/chair added at east location.', 'Initial bookshelf overlaps the living-window projection; unchanged by candidate. Daylight/window operation not accepted.', 'Bed left-side approach has no connected route under declared proxies; right-side point remains reachable.', 'Office chair faces its table, approximately away from sofa: unblocked sight segments do not establish a seated worker naturally sees sofa without turning.', 'No door leaves, installed utility verification or body articulation.'],geometry_after_review_changed=False))
    write(p/'ARCHIVE_CHECKS.json',dict(state_bound_observations=3,decoded_hashed_images=15,schema_valid_successful_actions=2,all_prior_stages_exit_zero=True,rejected_candidate_native_actions=0,bindings=bindings))
    write(p/'EVALUATION_BOUNDARY.json',dict(input_allowed=['TASK_INPUT.json','input_metadata/*','original native input bundle and current observations'],
        input_forbidden=['SOURCE_RECORD.json final-design details','PROTOCOL.json candidate_plans','east_PLAN.json','central_PLAN.json','any reference actions/states/final images','MEASUREMENT.json or answer labels'],
        same_other_project_knowledge_pool_only=True,current_family_excluded_from_training_and_retrieval=True,
        inference_protocol_frozen=False,model_evaluation_run=False,source_pretraining_exposure='unknown',analyst_source_exposure=True,
        next_step='Freeze bounded identical model/tool/token budgets and requirements for this new task, then run a diagnostic baseline; preserve window, seating and operational unknowns. Do not tune these two references or claim a sealed whole-home benchmark.'))
    s=read(p/'STATUS.json');s.update(status=summary['status'],updated_at=now,completed=2,planned=2,completed_unit='candidate dispositions: one executed, one rejected',ended_at=nr['ended_at'],actual_exit_code=0,
        failure_reason='Central candidate failed frozen0.6m access gate; not a process error.',latest_verifiable_artifact=str(p/'SUMMARY.json'),next_step=read(p/'EVALUATION_BOUNDARY.json')['next_step'])
    write(p/'STATUS.json',s);write(p.parent/'latest.json',dict(run_id=p.name,stage=s['status'],updated_at=now,run_directory=str(t)))
    print(json.dumps({k:summary[k] for k in ['status','actual_native_candidates','actual_tool_calls','sight_clear','model_calls']}),flush=True)


if __name__=='__main__':main()
