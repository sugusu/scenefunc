"""Archive the bounded Pine source-to-action record and verify observation bindings."""
import argparse,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
from jsonschema import validate
from native_checkpoint import canonical_hash

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args();p=a.public.resolve();t=a.runtime.resolve()
    print(datetime.now(timezone.utc).isoformat(),'archive Pine reconstruction, source attribution and limitations',flush=True)
    steps=read(t/'native/STEPS.json');assert len(steps)==4
    nr=read(t/'native_receipt/receipt.json');assert nr['exit_code']==0
    for f in t.glob('*_receipt'):
        if f.name=='archive_receipt':continue
        receipt=read(f/'receipt.json');assert receipt['exit_code']==0
        shutil.copytree(f,p/f.name,dirs_exist_ok=True)
    for name in ['INITIAL_AUDIT.json','FINAL_AUDIT.json','CONSEQUENCES.json']:
        shutil.copy2(t/name,p/name)
    for directory in ['native','preflight','build','cold_restore','export','download','previews']:
        (p/directory).mkdir(exist_ok=True)
        for f in (t/directory).glob('*.json'):shutil.copy2(f,p/directory/f.name)
    for f in (t/'previews').glob('*.png'):shutil.copy2(f,p/'previews'/f.name)
    observations=[];records=[]
    for i in range(5):
        label='initial' if i==0 else f'step_{i:02d}'
        obs=read(t/'native'/f'{label}_observation.json');state=read(t/'native'/f'{label}_state.json');b=read(obs['state_binding'])
        assert canonical_hash(state)==b['state_sha256']==canonical_hash(b['native_state'])
        context=json.loads(obs['room_prompt'].split('\nCURRENT NATIVE STATE:\n',1)[1]);assert canonical_hash(context)==b['current_context_sha256']
        assert len(obs['image_paths'])==len(b['images'])==5
        for item,path in zip(b['images'],obs['image_paths']):
            f=Path(path);assert f==Path(item['path']) and sha(f)==item['sha256']
            with Image.open(f) as im:im.verify()
            if i in [0,4] or '.topdown.' in f.name:
                (p/'views').mkdir(exist_ok=True);shutil.copy2(f,p/'views'/f.name)
        (p/'bindings').mkdir(exist_ok=True);shutil.copy2(obs['state_binding'],p/'bindings'/Path(obs['state_binding']).name)
        observations.append(dict(step=i,state_sha256=b['state_sha256'],images=b['images']))
        if i<4:
            tools,_=json.JSONDecoder().raw_decode(obs['system'].split('Available executable tools:\n',1)[1]);schemas={x['name']:x['parameters'] for x in tools};step=steps[i]
            validate(step['command']['arguments'],schemas[step['command']['name']]);assert step['before_sha256']==b['state_sha256']
            assert step['reply']['results'][0]['output']['success'] is True
            records.append(dict(step=i+1,observation=str((p/'native'/f'{label}_observation.json').relative_to(p)),before_state_sha256=b['state_sha256'],command=step['command'],receipt=step['reply'],after_state_sha256=step['after_sha256'],action_authorship='Research reconstruction, not historical designer operation',reason_attribution='Source supports combined designer window-workstation/resident old-desk reuse; chair moves/additions and coordinates/order are research implementation.'))
        if i>0:assert steps[i-1]['after_sha256']==b['state_sha256']
    before=read(t/'INITIAL_AUDIT.json');after=read(t/'FINAL_AUDIT.json');c=read(t/'CONSEQUENCES.json')
    generic={label:[dict(width_m=s['square_body_width_m'],reachable=sum(x['reachable_from_entry'] for x in s['targets'].values()),total=len(s['targets']),failed=[k for k,v in s['targets'].items() if not v['reachable_from_entry']]) for s in audit['access']['scenarios']] for label,audit in [('before',before),('after',after)]}
    summary=dict(run_id=p.name,status='source_conditioned_transfer_executed_with_limits',new_source_families=1,new_knowledge_records=1,native_actions=4,observations=5,image_instances=25,initial_furniture=4,final_furniture=6,old_desk_and_chair_instances_retained=True,
        mesh_intersections_before_after=[before['surface_intersection_pairs'],after['surface_intersection_pairs']],outside_before_after=[sum(x['outside_nominal_room_xy'] for x in audit['bounds']) for audit in [before,after]],
        common_access=c['common_access'],generic_access=generic,work=c['work'],seat_proxies=c['simultaneous_seat_square_proxies'],
        native_started_at=nr['started_at'],native_ended_at=nr['ended_at'],native_elapsed_seconds=nr['elapsed_seconds'],native_exit_code=nr['exit_code'],
        cold_restore=read(t/'cold_restore/RESULT.json')['status'],geometry_export=read(t/'export/RESULT.json')['glb'],model_calls=0,training_updates=0,professional_reviews=0,training_admitted=False,official_loader_encoding_this_run=False,whole_room_accepted=False,
        limits=['One assumed transfer room, not surveyed Pine or complete apartment.','Source assigns window furniture to designer and reused old desk to residents.','No guest-bed unfolding, comfort, knee clearance, storage capacity, moving path or professional preference validation.','Source identity reuse is supported; native IDs identify substitute initial instances.','Existing 39 canonical action candidates are not expanded by these four unencoded evidence records.'])
    write(p/'SUMMARY.json',summary)
    write(p/'DECISION_PACKET.json',dict(source=read(p/'SOURCE_RECORD.json'),conditions=read(p/'TASK_INPUT.json'),reconstruction_protocol='RECONSTRUCTION_PROTOCOL.json',records=records,consequences='CONSEQUENCES.json',whole_room_diagnostics=['INITIAL_AUDIT.json','FINAL_AUDIT.json'],editable_checkpoint=str(t/'native/checkpoint'),geometry_preview=summary['geometry_export'],historical_operations=False,training_admitted=False,positive_professional_demonstration=False,canonical_training_manifest_member=False))
    write(p/'VALIDATION.json',dict(bound_observations=5,hashed_decoded_images=25,schema_valid_successful_actions=4,old_instance_and_asset_checks=True,all_prearchive_stage_exit_codes_zero=True,observations=observations))
    status=read(p/'STATUS.json');now=datetime.now(timezone.utc).isoformat();status.update(status=summary['status'],updated_at=now,ended_at=nr['ended_at'],actual_exit_code=nr['exit_code'],latest_verifiable_artifact=str(p/'DECISION_PACKET.json'),failure_reason=None,next_step='Use the attributed source/action/consequence record to broaden decision data; no tuned rerun, automatic training admission or extra optimizer update.')
    status['completed_planned']['native_edits']=[4,4];status['completed_planned']['observations']=[5,5];write(p/'STATUS.json',status)
    write(p.parent/'latest.json',dict(run_id=p.name,stage=status['status'],updated_at=now,run_directory=str(t)))
    print(json.dumps(dict(status=summary['status'],native_actions=4,images=25,training_updates=0)),flush=True)

if __name__=='__main__':main()
