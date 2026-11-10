"""Lossless path-preserving archive of NEW geometry with repeated buffers.

No old result/model/source, hardlink or metric change. Build and verify archive
in RAM first. Only then remove the new, byte-redundant buffers/cache and persist
the archive. --restore recreates exact paths/content into an explicit new root;
JSON absolute path strings remain original, so relocation is not claimed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--restore',type=Path)
    a=p.parse_args()
    if a.restore:
        if a.restore.exists():raise FileExistsError('Restore root must be new; no overwrite of evidence')
        with zipfile.ZipFile(a.archive) as z:
            manifest=json.loads(z.read('paths.json'))
            for row in manifest['files']:
                rel=Path(row['path'])
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive path')
                data=z.read('blobs/'+row['sha256'])
                assert digest(data)==row['sha256'] and len(data)==row['bytes']
                dest=a.restore/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        print(json.dumps(dict(status='restored_exact_bytes',files=len(manifest['files']),
            original_absolute_JSON_strings_unchanged=True)))
        return
    if a.source is None:raise ValueError('Explicit NEW source root required')
    root=a.source.resolve()
    if root.name!='actual_floor_geometry_v1' or root.parent.name!='20261110_remote_native_controller':
        raise ValueError('This packaging run is confined to the new admitted geometry output')
    assert not a.archive.exists()
    rows=[];blobs={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('No symlinks in evidence package')
        if path.is_file():
            data=path.read_bytes();h=digest(data);blobs.setdefault(h,data)
            rows.append(dict(path=str(path.relative_to(root)),sha256=h,bytes=len(data)))
    manifest=dict(source_root=str(root),files=rows,original_bytes=sum(r['bytes'] for r in rows),
        unique_bytes=sum(map(len,blobs.values())),scope='Only this NEW run, exact byte reconstruction; no old file deduplication')
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('paths.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        for h,data in blobs.items():z.writestr('blobs/'+h,data)
    data=memory.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        for r in rows:assert digest(z.read('blobs/'+r['sha256']))==r['sha256']
    # Nothing is deleted before the complete independent archive readback.
    deleted=[]
    for r in rows:
        rel=Path(r['path'])
        if '.geometry_cache' in rel.parts or rel.suffix=='.bin':
            target=root/rel;assert digest(target.read_bytes())==r['sha256']
            target.unlink();deleted.append(r)
    free=shutil.disk_usage(root).free
    if free-len(data)<5*1024**3:
        # Keep a recoverable RAM archive if an unrelated writer used the space.
        rescue=Path('/dev/shm/native_floor_geometry_verified_recovery.zip')
        rescue.write_bytes(data)
        raise RuntimeError('Disk reserve prevents persistence; verified archive retained at '+str(rescue))
    a.archive.write_bytes(data)
    with zipfile.ZipFile(a.archive) as z:
        for r in rows:assert digest(z.read('blobs/'+r['sha256']))==r['sha256']
    receipt=dict(status='new_geometry_losslessly_archived',archive=str(a.archive.resolve()),
        archive_sha256=digest(data),archive_bytes=len(data),original_bytes=manifest['original_bytes'],
        unique_bytes=manifest['unique_bytes'],archived_files=len(rows),deleted_new_buffers_and_cache=deleted,
        original_scene_json_sdf_gltf_retained=True,raw_buffers_require_restore_before_loading=True,
        old_files_modified=0,filesystem_hardlinks_used=False,
        disk_free_after_GiB=shutil.disk_usage(root).free/2**30)
    (root/'packaging_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='deleted_new_buffers_and_cache'}))


if __name__=='__main__':main()
