"""Archive an actually completed two-condition policy batch and preserve failures."""
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
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];p=repo/'runs/920'/a.run;t=repo.parent/'runtime'/a.run
    batch=read(t/'BATCH.json');assert batch['status']=='two_condition_attempts_complete_pending_review'
    assert read(t/'batch_execution/receipt.json')['exit_code']==0
    plan=read(p/'PLAN.json');now=datetime.now(timezone.utc).isoformat();rows={};copies=[]
    def cp(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha256(src)==sha256(dst)
        copies.append(dict(path=str(dst.relative_to(p)),source=str(src),sha256=sha256(src)))
    all_bindings=[]
    for arm in plan['execution_order']:
        ar=t/arm;apub=p/arm;roll=ar/'rollout';result=read(roll/'RESULT.json')
        initial=read(ar/'INITIAL_EVALUATION.json');evaluation=read(ar/'EVALUATION.json');audit=read(ar/'FINAL_AUDIT.json')
        observation=read(ar/'prepare/observation.json')
        tools,_=json.JSONDecoder().raw_decode(observation['system'].split('Available executable tools:\n',1)[1])
        schemas={tool['name']:tool['parameters'] for tool in tools};schema_errors=[];parse_errors=[]
        observations=[('initial',observation)];commands=[]
        for rounddir in sorted(roll.glob('round_*')):
            calls=read(rounddir/'parsed_calls.json') if (rounddir/'parsed_calls.json').exists() else []
            for call in calls:
                commands.append(call)
                if call.get('_parse_error') or 'name' not in call or 'arguments' not in call:
                    parse_errors.append(dict(round=rounddir.name,call=call));continue
                errors=(list(jsonschema.Draft202012Validator(schemas[call['name']]).iter_errors(call['arguments']))
                        if call['name'] in schemas else ['unadvertised tool'])
                if errors:schema_errors.append(dict(round=rounddir.name,call=call,errors=[str(x) for x in errors]))
            if (rounddir/'observation.json').exists():observations.append((rounddir.name,read(rounddir/'observation.json')))
            for name in ['request.json','parsed_calls.json','tool_reply.json','observation.json','next_request.json']:
                if (rounddir/name).exists():cp(rounddir/name,apub/'rollout'/rounddir.name/name)
            for name in ['RESULT.json','ENCODING.json','output_ids.json','raw_output.txt']:
                if (rounddir/'model'/name).exists():cp(rounddir/'model'/name,apub/'rollout'/rounddir.name/'model'/name)
        for label,obs in observations:
            binding=read(Path(obs['state_binding']))
            assert len(obs['image_paths'])==len(binding['images'])==5
            assert canonical_hash(binding['native_state'])==binding['state_sha256']
            cp(Path(obs['state_binding']),apub/'bindings'/Path(obs['state_binding']).name)
            for path,b in zip(obs['image_paths'],binding['images']):
                assert Path(path).resolve()==Path(b['path']).resolve() and sha256(Path(path))==b['sha256']
                with Image.open(path) as im:im.verify()
            all_bindings.append(dict(arm=arm,label=label,state_sha256=binding['state_sha256'],images=binding['images']))
        final_obs=read(roll/'final_observation.json');final_binding=read(Path(final_obs['state_binding']))
        final_state=read(roll/'final_state.json')
        assert canonical_hash(final_state)==final_binding['state_sha256']
        for path in final_obs['image_paths']:cp(Path(path),apub/'views'/Path(path).name)
        for name in ['RESULT.json','final_state.json','final_observation.json']:
            cp(roll/name,apub/'rollout'/name)
        for name in ['RESULT.json','request.json','observation.json']:
            cp(ar/'prepare'/name,apub/'prepare'/name)
        for name in ['INITIAL_EVALUATION.json','EVALUATION.json','FINAL_AUDIT.json']:cp(ar/name,apub/name)
        for folder in ['export','cold_restore']:cp(ar/folder/'RESULT.json',apub/folder/'RESULT.json')
        for name in ['manifest.json','house_state.json','runtime.json','task.json','catalog.json','session_checkpoint.json']:
            src=roll/'final_checkpoint'/name
            if src.exists():cp(src,apub/'checkpoint_metadata'/name)
        write(apub/'ACTION_SCHEMA_REVIEW.json',dict(budgeted_tool_attempts=len(commands),schema_errors=schema_errors,parse_errors=parse_errors,
            note='Native acceptance and schema validity are separate; invalid arguments are retained even if ignored by the engine.'))
        rows[arm]=dict(model_calls=result['model_calls'],tool_calls=result['tool_calls'],stop_reason=result['stop_reason'],
            schema_mismatch_calls=len(schema_errors),parse_error_attempts=len(parse_errors),commands=commands,
            initial_clear_rays=initial.get('clear_rays'),final_clear_rays=evaluation.get('clear_rays'),
            initial_proxy_pass=initial['conditional_proxy_pass'],final_proxy_pass=evaluation['conditional_proxy_pass'],
            identity_dimensions_preserved=evaluation['identity_and_dimensions_preserved'],
            four_declared_places=evaluation.get('four_declared_places'),role_access=evaluation.get('all_role_relative_access'),
            missing_ids=evaluation['missing_ids'],triangle_contacts=audit['surface_intersection_pairs'],
            outside_furniture=[x['object_id'] for x in audit['bounds'] if x['outside_nominal_room_xy']],
            generic_failed_targets={str(s['square_body_width_m']):[k for k,v in s['targets'].items() if not v['reachable_from_entry']] for s in audit['access']['scenarios']},
            rollout_receipt=read(ar/'rollout_execution/receipt.json'),export=read(ar/'export/RESULT.json')['glb'],
            checkpoint=str(roll/'final_checkpoint'),checkpoint_manifest_sha256=sha256(roll/'final_checkpoint/manifest.json'),
            restore_receipt=read(ar/'restore_execution/receipt.json'))
    # Only terminal receipts are copied. The current archive's own receipt is finalized afterwards.
    excluded=[];receipts=[]
    for src in sorted(t.rglob('receipt.json')):
        if not src.parent.name.endswith('_execution'):continue
        receipt=read(src);relative=src.relative_to(t)
        if receipt['status']=='running':
            assert any(Path(x).name==Path(__file__).name for x in receipt['command'])
            excluded.append(str(relative));continue
        assert receipt['exit_code']==0, 'Do not silently accept failed inference or postprocessing'
        cp(src,p/'execution'/relative)
        cp(src.parent/'stage.log',p/'execution'/relative.parent/'stage.log');receipts.append(receipt)
    cp(t/'BATCH.json',p/'BATCH.json')
    write(p/'SUMMARY.json',dict(status='completed_development_behavior_comparison',recorded_at=now,arms=rows,
        same_commands_both_conditions=rows[plan['execution_order'][0]]['commands']==rows[plan['execution_order'][1]]['commands'],
        model_calls=sum(x['model_calls'] for x in rows.values()),tool_calls=sum(x['tool_calls'] for x in rows.values()),
        training_updates=0,sealed_test=False,professional_evaluation=False,
        interpretation='Already-satisfied condition retention and improvement of the other condition must be distinguished. No training benefit claim.'))
    write(p/'ARCHIVE_CHECKS.json',dict(recorded_at=now,bindings=all_bindings,
        observation_bindings=len(all_bindings),image_hash_decode_checks=len(all_bindings)*5,
        copies=copies,receipts=receipts,excluded_running_self=excluded,
        visual_review='Separate VISUAL_REVIEW.json; hash checks alone are not a visual inspection.'))
    for name in ['archive_family_policy.py','prepare_family_policy.py','run_family_policy_batch.py','evaluate_family_policy.py',
                 'run_knowledge_policy.py','catalog_adapter_step.py','audit_family_choice_geometry.py']:
        cp(repo/'scripts/920'/name,p/'code'/name)
    print(json.dumps({arm:{k:v[k] for k in ['model_calls','tool_calls','initial_clear_rays','final_clear_rays','final_proxy_pass','schema_mismatch_calls']} for arm,v in rows.items()}))


if __name__=='__main__':main()
