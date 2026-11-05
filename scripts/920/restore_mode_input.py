"""Restore the declared baseline input using unchanged source-bundle resources."""
import argparse
import json
from pathlib import Path
import shutil
from native_project import sha256, validate_bundle


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--base',type=Path)
    a=p.parse_args();record=json.loads((a.run/'INPUT_RECONSTRUCTION.json').read_text())
    source=a.base or Path(record['base_bundle'])
    if sha256(source/'manifest.json')!=record['base_manifest_sha256']:
        raise ValueError('Base resources differ; do not silently reconstruct another scene')
    validate_bundle(source)
    shutil.copytree(source,a.out)
    for name in ['manifest.json','house_state.json','runtime.json','task.json']:
        shutil.copy2(a.run/'input'/name,a.out/name)
    if sha256(a.out/'manifest.json')!=record['restored_manifest_sha256']:
        raise ValueError('Restored manifest changed')
    manifest,_,_,_=validate_bundle(a.out)
    print(json.dumps({'status':'restored_and_verified','files':len(manifest['files']),
                      'manifest_sha256':sha256(a.out/'manifest.json'),'output':str(a.out)}))


if __name__=='__main__':main()
