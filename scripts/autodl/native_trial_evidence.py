"""Verified cold restore of the persisted content-addressed native trial bundle."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def restore_solid_evidence(receipt_path, destination):
    receipt_path=Path(receipt_path).resolve()
    receipt=json.loads(receipt_path.read_text());archive=Path(receipt['path'])
    if not archive.is_absolute():
        # Receipt paths produced from repo-relative CLI arguments are relative
        # to the repository, not to the receipt directory.
        repo=next(p for p in receipt_path.parents if (p/'scripts/autodl').is_dir())
        archive=repo/archive
    assert sha(archive)==receipt['sha256']
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    by_hash=defaultdict(list);metadata=None;manifest=None;seen=set()
    with tarfile.open(archive,'r|xz') as tar:
        for item in tar:
            stream=tar.extractfile(item);data=stream.read()
            if item.name=='original_zip_metadata.json':metadata=json.loads(data);continue
            if item.name=='paths.json':
                manifest=json.loads(data)
                for row in manifest['files']:
                    rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
                    by_hash[row['sha256']].append(row)
                continue
            assert item.name.startswith('blobs/') and manifest is not None
            h=item.name.split('/',1)[1];assert hashlib.sha256(data).hexdigest()==h
            assert h not in seen;seen.add(h)
            for row in by_hash[h]:
                assert len(data)==row['bytes']
                target=destination/row['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    assert manifest is not None and metadata is not None and seen==set(by_hash)
    return dict(archive=str(archive),archive_sha256=receipt['sha256'],
                original_root=manifest['original_root'],original_files_restored=len(manifest['files']),
                distinct_blobs_verified=len(seen),original_zip_sha256=metadata['source_zip']['sha256'])


def relocate_restored_paths(root,old_root):
    """Only exact string prefixes in known text artifacts, in a new copy."""
    old=str(old_root).encode();new=str(root).encode();rows=[]
    for p in Path(root).rglob('*'):
        if not p.is_file() or p.suffix not in {'.json','.gltf','.sdf','.yaml','.yml','.xml','.mtl'}:continue
        data=p.read_bytes()
        if old in data:
            p.write_bytes(data.replace(old,new))
            rows.append(dict(path=str(p.relative_to(root)),before_sha256=hashlib.sha256(data).hexdigest(),after_sha256=sha(p)))
    return rows
