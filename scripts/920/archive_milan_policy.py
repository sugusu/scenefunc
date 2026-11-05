"""Archive one terminal Milan policy attempt, including rejected/invalid outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import jsonschema
from PIL import Image
from native_checkpoint import canonical_hash
from native_project import sha256


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run', required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];p=repo/'runs/920'/a.run;t=repo.parent/'runtime'/a.run
    plan=read(p/'PLAN.json');roll=t/'rollout';result=read(roll/'RESULT.json')
    receipt=read(t/'rollout_stage/receipt.json');assert receipt['status']!='running'
    selection=read(p/'TERMINAL_SELECTION.json');bundle=Path(selection['bundle'])
    assert sha256(bundle/'manifest.json')==selection['manifest_sha256']
    copies=[]
    print(datetime.now(timezone.utc).isoformat(), 'archive terminal frozen-base attempt; no model or scene edits', flush=True)
    def cp(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert sha256(src)==sha256(dst)
        copies.append(dict(path=str(dst.relative_to(p)),source=str(src),sha256=sha256(src)))
    observation=read(t/'prepare/observation.json')
    tools,_=json.JSONDecoder().raw_decode(observation['system'].split('Available executable tools:\n',1)[1])
    schemas={x['name']:x['parameters'] for x in tools}
    observations=[('initial',observation)];actions=[];schema_errors=[];parse_errors=[];models=[]
    for rd in sorted(roll.glob('round_*')):
        calls=read(rd/'parsed_calls.json') if (rd/'parsed_calls.json').exists() else []
        reply=read(rd/'tool_reply.json') if (rd/'tool_reply.json').exists() else None
        for call in calls:
            actions.append(dict(round=rd.name,call=call,dispatched=reply is not None,
                                reply=reply))
            if call.get('_parse_error') or 'name' not in call or 'arguments' not in call:
                parse_errors.append(dict(round=rd.name,call=call));continue
            errors=(list(jsonschema.Draft202012Validator(schemas[call['name']]).iter_errors(call['arguments']))
                    if call['name'] in schemas else ['unadvertised tool'])
            if errors:schema_errors.append(dict(round=rd.name,call=call,errors=[str(x) for x in errors]))
        if (rd/'observation.json').exists():observations.append((rd.name,read(rd/'observation.json')))
        for name in ['request.json','parsed_calls.json','tool_reply.json','observation.json','next_request.json']:
            if (rd/name).exists():cp(rd/name,p/'rollout'/rd.name/name)
        for name in ['RESULT.json','ENCODING.json','output_ids.json','raw_output.txt']:
            if (rd/'model'/name).exists():cp(rd/'model'/name,p/'rollout'/rd.name/'model'/name)
        if (rd/'model/RESULT.json').exists():models.append(read(rd/'model/RESULT.json'))
    bindings=[];unique_images=set();unique_states=set()
    for label,obs in observations:
        binding=read(Path(obs['state_binding']))
        assert len(obs['image_paths'])==len(binding['images'])==5
        assert canonical_hash(binding['native_state'])==binding['state_sha256']
        context=json.loads(obs['room_prompt'].split('\nCURRENT NATIVE STATE:\n',1)[1])
        assert canonical_hash(context)==binding['current_context_sha256']
        cp(Path(obs['state_binding']),p/'bindings'/(label+'.json'))
        for image_path,item in zip(obs['image_paths'],binding['images']):
            f=Path(image_path);assert f.resolve()==Path(item['path']).resolve() and sha256(f)==item['sha256']
            with Image.open(f) as im:im.verify()
            unique_images.add(item['sha256'])
        if binding['state_sha256'] not in unique_states:
            cp(Path(obs['image_paths'][0]),p/'views'/(label+'.topdown.png'))
        unique_states.add(binding['state_sha256'])
        bindings.append(dict(label=label,state_sha256=binding['state_sha256'],images=binding['images']))
    final=read(Path(selection['observation']));fb=read(Path(final['state_binding']))
    state=read(bundle/'house_state.json')['rooms'][read(bundle/'manifest.json')['selected_room_id']]
    assert canonical_hash(state)==fb['state_sha256']
    for path in final['image_paths']:cp(Path(path),p/'views'/('final.'+Path(path).name.split('.')[-2]+'.png'))
    for folder in ['prepare','rollout']:
        for f in (t/folder).glob('*.json'):cp(f,p/folder/f.name)
    for f in (t/'prepare/encode').glob('*.json'):cp(f,p/'prepare/encode'/f.name)
    for name in ['EVALUATOR_CALIBRATION.json','INITIAL_EVALUATION.json','EVALUATION.json','FINAL_AUDIT.json']:
        cp(t/name,p/name)
    for folder in ['export','cold_restore']:
        for f in (t/folder).glob('*.json'):cp(f,p/folder/f.name)
    for name in ['manifest.json','house_state.json','runtime.json','task.json','catalog.json','session_checkpoint.json']:
        if (bundle/name).exists():cp(bundle/name,p/'checkpoint_metadata'/name)
    receipts=[];excluded=[]
    for f in sorted(t.rglob('receipt.json')):
        rec=read(f)
        if rec['status']=='running':
            assert any(Path(x).name==Path(__file__).name for x in rec['command'])
            excluded.append(str(f));continue
        cp(f,p/'execution'/f.relative_to(t));cp(f.parent/'stage.log',p/'execution'/f.parent.relative_to(t)/'stage.log')
        receipts.append(rec)
    evaluation=read(t/'EVALUATION.json');audit=read(t/'FINAL_AUDIT.json')
    mutations=[row for row in result['rounds'] if row.get('after_sha256') is not None and row['before_sha256']!=row['after_sha256']]
    costs=dict(model_calls=result['model_calls'],tool_calls=result['tool_calls'],state_changing_calls=len(mutations),
               encoded_input_tokens=sum(x.get('encoding',{}).get('input_tokens',0) for x in models),
               generated_tokens=sum(x.get('output_tokens',0) for x in models),
               model_load_seconds=sum(x.get('load_seconds',0) for x in models),
               generation_seconds=sum(x.get('generation_seconds',0) for x in models),
               peak_allocated_bytes=max([x.get('peak_allocated_bytes',0) for x in models] or [0]),
               peak_reserved_bytes=max([x.get('peak_reserved_bytes',0) for x in models] or [0]),
               rollout_receipt=receipt,
               model_attempts=[dict(status=x['status'],error=x.get('error'),input_tokens=x.get('encoding',{}).get('input_tokens'),output_tokens=x.get('output_tokens')) for x in models])
    write(p/'ACTION_SCHEMA_REVIEW.json',dict(actions=actions,schema_errors=schema_errors,parse_errors=parse_errors,
        note='Schema validity is audited separately from engine acceptance. Invalid native-accepted queries are not counted as design edits.'))
    summary=dict(run_id=a.run,status='terminal_frozen_base_diagnostic_audited',recorded_at=datetime.now(timezone.utc).isoformat(),
        costs=costs,stop_reason=result['stop_reason'],execution_status=result['status'],actual_exit_code=receipt['exit_code'],
        failure_reason=result.get('error'),schema_mismatch_calls=len(schema_errors),parse_error_calls=len(parse_errors),
        furniture_count=evaluation['furniture_count'],original_retained=sum(evaluation['retained_original_ids'].values()),
        original_count=len(evaluation['retained_original_ids']),identity_preserved=all(evaluation['identity_assets_dimensions_preserved'].values()),
        immutable_preserved=all(evaluation['immutable_unchanged'].values()),role_access=evaluation['role_relative_access'],
        dedicated_work_geometry_found=evaluation['dedicated_work_geometry_found'],
        conditional_work_pairs=[{k:x[k] for k in ['table_id','chair_id','clear_rays','total_rays','chair_toward_sofa_cos']} for x in evaluation['workspace_pairs'] if x['conditional_dedicated_work_geometry']],
        mesh_intersections=audit['surface_intersection_pairs'],outside=[x['object_id'] for x in audit['bounds'] if x['outside_nominal_room_xy']],
        generic_access=[dict(width_m=s['square_body_width_m'],planned=len(s['targets']),reachable=sum(x['reachable_from_entry'] for x in s['targets'].values()),failed=[k for k,x in s['targets'].items() if not x['reachable_from_entry']]) for s in audit['access']['scenarios']],
        training_updates=0,professional_evaluation=False,sealed_test=False,whole_dwelling_success='unknown_not_accepted',
        interpretation='One source-inspired independent-family furnished-home revision diagnostic. Neither reference similarity nor object presence establishes learning or whole-home quality.',
        checkpoint=selection,export=read(t/'export/RESULT.json')['glb'])
    write(p/'SUMMARY.json',summary)
    write(p/'ARCHIVE_CHECKS.json',dict(observations=len(bindings),image_hash_decode_checks=len(bindings)*5,
        unique_image_hashes=len(unique_images),unique_native_state_hashes=len(unique_states),bindings=bindings,copies=copies,
        terminal_receipts=receipts,excluded_running_self=excluded,visual_review='See separately recorded VISUAL_REVIEW.json.'))
    print(json.dumps({k:summary[k] for k in ['status','furniture_count','dedicated_work_geometry_found','schema_mismatch_calls','actual_exit_code']}),flush=True)


if __name__=='__main__':main()
