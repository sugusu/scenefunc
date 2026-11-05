"""Archive a finite native construction and verify its actual data bindings."""
import argparse,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import jsonschema
from PIL import Image
from native_project import sha256
from native_checkpoint import canonical_hash

def read(p):return json.loads(p.read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--allowed-failed-stage',action='append',default=[]);a=ap.parse_args()
    p=a.public.resolve();t=a.runtime.resolve();n=t/'native';steps=read(n/'STEPS.json');count=read(p/'EXECUTION_PLAN.json')['budget']['tool_calls'];assert len(steps)==count
    stages={f.parent.name:read(f) for f in sorted(t.glob('*execution/receipt.json')) if read(f)['status']!='running'}
    failed={k:x for k,x in stages.items() if x['exit_code']!=0}
    assert set(failed)==set(a.allowed_failed_stage), 'Every failed postprocessing stage must be explicitly acknowledged'
    assert not any(k in failed for k in ['native_execution','build_execution','preflight_execution','audit_execution','restore_execution','export_execution','package_execution']), 'Core execution failure is not an accepted result'
    copies=[]
    def cp(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha256(src)==sha256(dst)
        copies.append({'source':str(src),'destination':str(dst.relative_to(p)),'sha256':sha256(src),'bytes':src.stat().st_size})
    tools,_=json.JSONDecoder().raw_decode(read(n/'initial_observation.json')['system'].split('Available executable tools:\n',1)[1]);schemas={x['name']:x['parameters'] for x in tools}
    for x in steps:
        call=x['command'];jsonschema.Draft202012Validator(schemas[call['name']]).validate(call['arguments']);assert x['reply']['results'][0]['output']['success'] is True
    images=[]
    for i in range(count+1):
        label='initial' if i==0 else f'step_{i:02d}';obs=read(n/(label+'_observation.json'));binding=read(Path(obs['state_binding']));state=read(n/(label+'_state.json'))
        assert canonical_hash(state)==binding['state_sha256']==canonical_hash(binding['native_state']);assert len(obs['image_paths'])==len(binding['images'])==5
        cp(Path(obs['state_binding']),p/'bindings'/Path(obs['state_binding']).name)
        for path,b in zip(obs['image_paths'],binding['images']):
            src=Path(path);assert src.resolve()==Path(b['path']).resolve() and sha256(src)==b['sha256']
            with Image.open(src) as im:im.verify()
            cp(src,p/'views'/src.name);images.append(path)
    assert len(images)==len(set(images))==5*(count+1)
    data=read(t/'dataset/DATASET.json');assert data['native_construction_steps']==count
    targets={}
    for arm,entries in data['arms'].items():
        targets[arm]=[]
        for e in entries:
            f=t/e['path'];assert sha256(f)==e['sha256'];doc=read(f);turn=doc['turns'][0]
            targets[arm].append((e['supervised_role'],turn['content'] if e['supervised_role']=='reason' else turn['tool_calls']))
    assert targets['paired']==targets['separate']
    for f in n.glob('*.json'):cp(f,p/'native'/f.name)
    for f in (t/'dataset').rglob('*.json'):cp(f,p/'dataset'/f.relative_to(t/'dataset'))
    for stage in stages:
        for name in ['receipt.json','stage.log']:cp(t/stage/name,p/'execution'/stage/name)
    for name in ['FINAL_AUDIT.json','SOURCE_ENDPOINT.json']:cp(t/name,p/name)
    for folder in ['export','cold_restore']:cp(t/folder/'RESULT.json',p/folder/'RESULT.json')
    for folder,target in [('build/bundle','input_metadata'),('native/checkpoint','checkpoint_metadata')]:
        for name in ['manifest.json','house_state.json','runtime.json','task.json','catalog.json','construction_provenance.json','session_checkpoint.json']:
            if (t/folder/name).exists():cp(t/folder/name,p/target/name)
    write(p/'ARCHIVE_CHECKS.json',{'checked_at':datetime.now(timezone.utc).isoformat(),'actual_steps':count,'observation_state_bindings':count+1,'images_hash_and_decode_checked':len(images),'schema_mismatch_calls':0,'same_action_and_reason_targets_both_arms':True,'reason_samples_per_arm':1,'action_samples_per_arm':count,'new_source_decisions':0,'reused_source_decisions':1,'failed_postprocessing':failed,'professional_evaluations':0,'training_updates':0,'tokenization_or_training_run':False,'visual_review':'See separate VISUAL_REVIEW.json; hash/decode checks alone are not visual review.','copies':copies})
    print(json.dumps({'steps':count,'images':len(images),'same_targets':True,'copies':len(copies)}))
if __name__=='__main__':main()
