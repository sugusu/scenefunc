"""Copy a native bundle and replace unregistered package URIs by local paths."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from native_project import sha256, validate_bundle


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    source=a.source.resolve();out=a.out.resolve()
    print(datetime.now(timezone.utc).isoformat(),'begin native resource URI relocation',flush=True)
    manifest,_,_,_=validate_bundle(source)
    if out.exists() or out.is_relative_to(source):raise ValueError('Destination must be new and outside source')
    out.mkdir(parents=True)
    for relative in ['manifest.json',*manifest['files']]:
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/relative,target)
    changes=[]
    for relative in manifest['files']:
        if not relative.endswith('.sdf'):continue
        path=out/relative;tree=ET.parse(path);replacements=[]
        for uri in tree.iter('uri'):
            old=(uri.text or '').strip()
            if not old.startswith('package://scene/'):continue
            target=(out/old.removeprefix('package://scene/')).resolve()
            if not target.is_relative_to(out) or not target.is_file():raise ValueError('Invalid source URI')
            uri.text=os.path.relpath(target,path.parent)
            replacements.append({'before':old,'after':uri.text,'resource_sha256':sha256(target)})
        if replacements:
            before=sha256(path);ET.indent(tree);tree.write(path,encoding='utf-8',xml_declaration=True)
            manifest['files'][relative]={'bytes':path.stat().st_size,'sha256':sha256(path)}
            changes.append({'path':relative,'before_sha256':before,'after_sha256':sha256(path),'uris':replacements})
    manifest['resource_path_migration_source_manifest_sha256']=sha256(source/'manifest.json')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_bundle(out)
    changed={c['path'] for c in changes}
    for relative in manifest['files']:
        if relative not in changed and sha256(source/relative)!=sha256(out/relative):
            raise ValueError('Non-URI resource changed')
    report={'status':'relocated','source_manifest_sha256':sha256(source/'manifest.json'),
            'new_manifest_sha256':sha256(out/'manifest.json'),'changed_sdf_files':changes,
            'native_state_unchanged':sha256(source/'house_state.json')==sha256(out/'house_state.json'),
            'geometry_bytes_unchanged':True,'actions_replayed':0}
    (out.parent/(out.name+'_migration.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'sdf_files_changed':len(changes),'native_state_unchanged':True}),flush=True)


if __name__=='__main__':main()
