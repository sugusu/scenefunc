"""Validate committed evidence against local immutable results, without inference."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    repo=a.repository.resolve();pub=repo/'runs/920/P920-DD-v1_20261104T083214Z';run=a.runtime.resolve()
    manifest=read(pub/'MANIFEST.json')
    for row in manifest['files']:
        file=pub/row['path'];assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],file
    # Check every new published artifact, including files excluded from manifest
    # because the final lifecycle status is updated after this check.
    checked=0
    for base in [pub,repo/'scripts/920']:
        for file in base.rglob('*'):
            if not file.is_file() or '__pycache__' in file.parts:continue
            assert file.suffix not in ['.pt','.glb','.blend','.safetensors','.pem','.key']
            raw=file.read_bytes();assert len(raw)<3*1024**2
            if file.suffix in ['.json','.py','.md','.txt','.log']:
                assert not re.search(rb'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)',raw),file
            if file.suffix=='.json':json.loads(raw)
            if file.suffix=='.py':ast.parse(raw)
            checked+=1
    result=read(pub/'RESULTS.json');assert len(result['rows'])==6 and result['formal_attempts']==result['formal_completed']==4
    assert sum(row['tools']['tool_calls_attempted'] for row in result['rows'] if row['tools'])==9
    assert sum(row['tools']['tool_calls_successful'] for row in result['rows'] if row['tools'])==5
    for row in read(pub/'ARTIFACTS.json')['files']:
        assert sha(row['scene_path'])==row['scene_sha256']
        assert sha(row['checkpoint_path'])==row['checkpoint_sha256']
    for arm in ['A','B']:
        g=read(run/'comparison_inputs'/f'{arm}_G/request.json');u=read(run/'comparison_inputs'/f'{arm}_U/request.json')
        assert g[0]==u[0] and g[1]['content']==u[1]['content'][:-1]
    for label in ['A_G','A_U','B_G','B_U']:
        call=pub/'model_runs'/('formal_'+label)
        model=read(call/'result.json');launch=read(call/'launch.json');input_record=read(pub/'comparison_inputs'/label/'receipt.json')
        assert launch['exit_code']==0 and model['status']=='completed' and model['ended_with_eos']
        assert model['input_tokens']==input_record['input_tokens']<=8192
        assert sha(run/'comparison_inputs'/label/'request.json')==model['request_sha256']==input_record['request_sha256']
        assert model['max_new_tokens']==2048 and model['seed']==0 and model['precision']=='float16'
        assert len(read(call/'parsed_calls.json'))<=12 and sha(call/'raw_output.txt')==model['raw_output_sha256']
        final=Path(read(pub/'revisions'/label/'application.json')['final_directory'])
        views=read(final/'views/receipt.json');assert views['status']=='completed' and len(views['views'])==5
        for image in views['views']:assert sha(image['path'])==image['sha256']
    output={'status':'pass','checked_at':datetime.now(timezone.utc).isoformat(),'manifest_files_verified':len(manifest['files']),
            'new_files_checked':checked,'formal_rounds':4,'scenes':6,'same_start_and_single_U_addition':True,
            'scene_checkpoint_view_hashes_verified':True,'model_receipts_exit_codes_budgets_verified':True,
            'no_weights_meshes_credentials_in_new_public_files':True,'new_model_calls':0,
            'scientific_result':'Negative/limited: no meaningful use repair demonstrated; B-G loses required furniture. Whole-room usability remains unknown.'}
    a.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))


if __name__=='__main__':main()
