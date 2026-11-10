"""Losslessly solid-compress a completed trial's content-addressed evidence.

No source, scene, weights or original ZIP is deleted. Original ZIP metadata
and each uncompressed member hash are retained. This does not promise an
identical reconstruction of the ZIP compression bytes.
"""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path
import shutil
import tarfile
import time
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    original = json.loads(a.receipt.read_text())
    source = Path(original['path'])
    assert sha(source) == original['sha256']
    target = Path('/dev/shm') / (a.out.name + '_solid_evidence.tar.xz')
    if target.exists():
        raise FileExistsError(target)
    tick = time.monotonic()
    members = []
    with zipfile.ZipFile(source) as z:
        for info in z.infolist():
            members.append(dict(name=info.filename, bytes=info.file_size,
                                sha256=hashlib.sha256(z.read(info)).hexdigest(),
                                date_time=list(info.date_time), compress_type=info.compress_type,
                                external_attr=info.external_attr, extra_hex=info.extra.hex(),
                                comment_hex=info.comment.hex()))
        metadata = json.dumps(dict(source_zip=original, zip_comment_hex=z.comment.hex(),
                                   members=members, compressed_byte_identical_regeneration_claimed=False),
                              indent=2).encode()
        with lzma.open(target, 'wb', preset=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as tar:
                item = tarfile.TarInfo('original_zip_metadata.json')
                item.size = len(metadata)
                tar.addfile(item, io.BytesIO(metadata))
                # Adjacent similar native outputs benefit from solid compression.
                for info in z.infolist():
                    data = z.read(info)
                    item = tarfile.TarInfo(info.filename)
                    item.size = len(data)
                    tar.addfile(item, io.BytesIO(data))
    expected = {r['name']: r for r in members}
    verified = []
    with tarfile.open(target, 'r:xz') as tar:
        for item in tar:
            stream = tar.extractfile(item)
            if item.name == 'original_zip_metadata.json':
                assert stream.read() == metadata
                continue
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                h.update(block)
            assert h.hexdigest() == expected[item.name]['sha256']
            assert item.size == expected[item.name]['bytes']
            verified.append(item.name)
    assert set(verified) == set(expected)
    disk_target = a.out / 'actual_scene_evidence.tar.xz'
    required_reserve = 5 * 1024**3 + 4 * 1024**2
    persisted = shutil.disk_usage(a.out).free - target.stat().st_size >= required_reserve
    if persisted:
        shutil.copyfile(target, disk_target)
        assert sha(disk_target) == sha(target)
    result = dict(status='all_original_members_verified', original_receipt=str(a.receipt),
                  original_zip_sha256=original['sha256'], original_zip_bytes=source.stat().st_size,
                  path=str(disk_target if persisted else target), sha256=sha(target),
                  bytes=target.stat().st_size, members_verified=len(verified),
                  original_scene_files=original['files'], persisted_on_data_disk=persisted,
                  original_zip_retained=True, original_absolute_paths_not_relocated=True,
                  seconds=time.monotonic()-tick, compressed_byte_identical_regeneration_claimed=False)
    (a.out/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
