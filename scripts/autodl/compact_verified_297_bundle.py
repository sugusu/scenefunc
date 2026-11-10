"""Retain the sole unique manifest before removing an authorized redundant ZIP.

Every other member must still have its exact audited raw copy. No raw/source
deletion, hardlinks, file renaming or compressed-byte regeneration claim.
"""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    repo=Path(__file__).resolve().parents[2]
    out=repo/'runs/autodl/20261110_complete_native_runtime/storage_audit'
    audit=json.loads((out/'broader_lookup_readonly.json').read_text())
    name='/root/autodl-tmp/scene_generation_next_20261016/work/297_actual_audit/evidence.zip'
    row=next(r for r in audit['archives'] if r['path']==name)
    path=Path(name);assert sha(path)==row['sha256']
    bad=[m for m in row['members'] if m['status']=='unique_or_not_located_keep_archive']
    assert len(bad)==1 and bad[0]['name']=='MANIFEST.json' and bad[0]['bytes']==134439
    retained=out/'retained_unique_297_MANIFEST.json'
    assert not retained.exists()
    with zipfile.ZipFile(path) as z:
        data=z.read('MANIFEST.json');assert hashlib.sha256(data).hexdigest()==bad[0]['sha256']
        retained.write_bytes(data)
    assert sha(retained)==bad[0]['sha256']
    for member in row['members']:
        if member['status']=='directory':continue
        if member['name']=='MANIFEST.json':member['retained_matching_files']=[str(retained)]
        assert any(Path(p).is_file() and sha(p)==member['sha256'] for p in member['retained_matching_files'])
    before=shutil.disk_usage(out).free
    receipt=dict(status='all_member_content_retained_before_archive_deletion',original_archive=row,
        source_raw_files_changed=0,source_raw_files_deleted=0,
        authority='User authorized removing genuinely unnecessary old files; sole unique content explicitly preserved',
        compressed_byte_identical_regeneration_claimed=False,free_before=before)
    target=out/'297_bundle_compaction.json'
    assert not target.exists();target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    assert sha(path)==row['sha256'];path.unlink()
    receipt.update(status='redundant_bundle_removed_unique_manifest_preserved',
        freed_archive_bytes=row['bytes'],free_after=shutil.disk_usage(out).free,
        unique_manifest_path=str(retained),unique_manifest_sha256=sha(retained))
    target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='original_archive'}))


if __name__=='__main__':main()
