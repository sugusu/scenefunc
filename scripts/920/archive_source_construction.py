"""Validate and publish small construction evidence without replay or training."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args()
    public=a.public.resolve();root=a.runtime.resolve();native=root/'native';copies=[]
    def cp(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert sha256(src)==sha256(dst)
        copies.append({'source':str(src),'destination':str(dst.relative_to(public)),'sha256':sha256(src),'bytes':src.stat().st_size})
    stages={f.parent.name:read(f) for f in sorted(root.glob('*execution/receipt.json')) if read(f)['status']!='running'}
    assert all(v['exit_code']==(1 if k in ['package_execution','archive_execution'] else 0) for k,v in stages.items())
    steps=read(native/'STEPS.json');assert len(steps)==9
    obs0=read(native/'initial_observation.json')
    tools,_=json.JSONDecoder().raw_decode(obs0['system'].split('Available executable tools:\n',1)[1])
    schemas={x['name']:x['parameters'] for x in tools};calls=[]
    for step in steps:
        call=step['command'];errors=[e.message for e in jsonschema.Draft202012Validator(schemas[call['name']]).iter_errors(call['arguments'])]
        calls.append({'step':step['index'],'command':call,'schema_errors':errors,'native_success':step['reply']['results'][0]['output']['success']})
    assert not any(x['schema_errors'] or not x['native_success'] for x in calls)
    images=[]
    for i in range(10):
        label='initial' if i==0 else f'step_{i:02d}'
        obs=read(native/(label+'_observation.json'));binding=read(Path(obs['state_binding']));state=read(native/(label+'_state.json'))
        assert canonical_hash(state)==binding['state_sha256']==canonical_hash(binding['native_state'])
        assert len(obs['image_paths'])==len(binding['images'])==5
        cp(Path(obs['state_binding']),public/'bindings'/Path(obs['state_binding']).name)
        for path,b in zip(obs['image_paths'],binding['images']):
            src=Path(path);assert src.resolve()==Path(b['path']).resolve() and sha256(src)==b['sha256']
            with Image.open(src) as im:im.verify()
            cp(src,public/'views'/src.name);images.append(str(src))
    assert len(images)==len(set(images))==50
    for f in sorted(native.glob('*.json')):cp(f,public/'native'/f.name)
    for folder in ['dataset_v2','dataset']:
        for f in sorted((root/folder).rglob('*.json')):cp(f,public/folder/f.relative_to(root/folder))
    for stage in stages:
        for name in ['receipt.json','stage.log']:cp(root/stage/name,public/'execution'/stage/name)
    for name in ['FINAL_AUDIT.json','ENCODING.json','SOURCE_ENDPOINT.json']:cp(root/name,public/name)
    for name in ['export','cold_restore']:cp(root/name/'RESULT.json',public/name/'RESULT.json')
    for folder,target in [('build/bundle','input_metadata'),('native/checkpoint','checkpoint_metadata')]:
        for name in ['manifest.json','house_state.json','runtime_state.json','task.json','catalog.json','construction_provenance.json','session_checkpoint.json']:
            if (root/folder/name).exists():cp(root/folder/name,public/target/name)
    checks={'recorded_at':datetime.now(timezone.utc).isoformat(),'actual_steps':9,'observation_state_bindings':10,'images_hash_and_decode_checked':50,
        'schema_mismatch_calls':0,'calls':calls,'new_source_families':0,'new_source_decisions':0,'reused_source_decisions':1,
        'visual_review_scope':'Human-facing agent inspected ten images: step_006 and step_009 five views each. Other forty images checked by hashes and decoding only.',
        'visual_review':'Final three cabinets along west side; room populated as recorded. Procedural chairs/bath/bed/cabinets remain envelopes. No visual basis for professional quality, operating doors or seated table usability.',
        'failed_attempt':'package_execution exit 1 from relative output path; v1 code, plan, partial dataset and log preserved. Only packaging corrected; native construction not repeated. Archive v1 read its own running receipt before exit_code existed; fixed by collecting terminal stages only, original failure retained.',
        'copies':copies,'model_calls':0,'training_updates':0,'training_admitted':False}
    write(public/'ARCHIVE_CHECKS.json',checks)
    print(json.dumps({k:checks[k] for k in ['actual_steps','observation_state_bindings','images_hash_and_decode_checked','schema_mismatch_calls']}))

if __name__=='__main__':main()
