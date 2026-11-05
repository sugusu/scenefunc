"""Explicitly derive a checkpoint with an appended catalog; preserve old lineage."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from native_project import validate_bundle,sha256
from native_checkpoint import canonical_hash


def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();p=read(a.plan)
    for n,h in p['code_sha256'].items():assert sha256(Path(__file__).parent/n)==h
    old=Path(p['original']);extended=Path(p['extended'])
    assert sha256(old/'manifest.json')==p['original_manifest_sha256'] and sha256(extended/'manifest.json')==p['extended_manifest_sha256']
    om,oh,orr,ot=validate_bundle(old);em,eh,er,et=validate_bundle(extended)
    assert oh==eh and orr==er and ot==et,'Only catalog extensions may be migrated'
    oc=read(old/'catalog.json');ec=read(extended/'catalog.json');n=len(oc['entries'])
    assert ec['schema']==oc['schema'] and ec['entries'][:n]==oc['entries'] and len(ec['entries'])==n+p['added_entries']
    ids=[x['template']['object_id'] for x in ec['entries']];assert len(ids)==len(set(ids))
    os=read(old/'session_checkpoint.json');es=read(extended/'session_checkpoint.json');assert os==es
    assert os['catalog_sha256']==canonical_hash(oc)
    for name,record in om['files'].items():
        if name not in ['catalog.json','provenance.json']:assert sha256(extended/name)==record['sha256'],name
    a.out.mkdir(parents=True,exist_ok=False);b=a.out/'bundle';shutil.copytree(extended,b)
    es['catalog_sha256']=canonical_hash(ec);write(b/'session_checkpoint.json',es)
    assert {k:v for k,v in es.items() if k!='catalog_sha256'}=={k:v for k,v in os.items() if k!='catalog_sha256'}
    proof=dict(derived_at=datetime.now(timezone.utc).isoformat(),kind='explicit_appended_catalog_derivation_not_identical_configuration_resume',
        original=str(old),original_manifest_sha256=p['original_manifest_sha256'],extended=str(extended),extended_manifest_sha256=p['extended_manifest_sha256'],
        old_catalog_sha256=os['catalog_sha256'],new_catalog_sha256=es['catalog_sha256'],old_entries=n,new_entries=len(ec['entries']),
        catalog_hash_kind='native_checkpoint.canonical_hash',old_catalog_file_sha256=sha256(old/'catalog.json'),new_catalog_file_sha256=sha256(b/'catalog.json'),
        unchanged_scene_task_runtime=True,unchanged_old_catalog_templates=True,unchanged_history_reserved_ids_and_runtime_fingerprint=True,
        old_session_sha256=sha256(old/'session_checkpoint.json'),new_session_sha256=sha256(b/'session_checkpoint.json'),
        native_edits=0,model_calls=0,training_updates=0,plan_sha256=sha256(a.plan))
    write(b/'CATALOG_EXTENSION.json',proof)
    prov=read(b/'provenance.json');prov['explicit_catalog_extension']=proof;write(b/'provenance.json',prov)
    em['files']={str(f.relative_to(b)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in sorted(b.rglob('*')) if f.is_file() and f!=b/'manifest.json'}
    write(b/'manifest.json',em);validate_bundle(b)
    write(a.out/'RESULT.json',dict(status='catalog_extension_derived_pending_native_restore',proof=proof,manifest_sha256=sha256(b/'manifest.json')))
    print(json.dumps(dict(status='derived',old_entries=n,new_entries=len(ec['entries']),original_session_untouched=sha256(old/'session_checkpoint.json')==proof['old_session_sha256'])),flush=True)


if __name__=='__main__':main()
