"""Acquire the fixed public layout archive only; never open validation/test data."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

REVISION = '72a2a80c3b96788635b9192b64853e1428d59cb4'
BASE = 'https://huggingface.co/datasets/Metaverse-AI-Lab/M3DLayout/resolve/' + REVISION


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.raw_dir.mkdir(parents=True, exist_ok=False)
    a.out.mkdir(parents=True, exist_ok=False)
    print('Starting fixed-revision layout-only acquisition; no held-out payload inspection.', flush=True)
    files = []
    for remote, local, limit in [('README.md', 'README.md', 200000),
                                  ('layout_dataset/M3DLayout_json.zip', 'M3DLayout_json.zip', 64000000)]:
        path = a.raw_dir / local
        h = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(BASE + '/' + remote, timeout=60) as response, path.open('xb') as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError('Download exceeds frozen byte limit')
                f.write(chunk)
                h.update(chunk)
        files.append(dict(url=BASE + '/' + remote, local_path=str(path), bytes=size, sha256=h.hexdigest()))
        print(json.dumps(files[-1]), flush=True)
    with zipfile.ZipFile(a.raw_dir / 'M3DLayout_json.zip') as z:
        members = [dict(name=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size,
                        crc32=i.CRC, payload_opened=False) for i in z.infolist()]
    result = dict(revision=REVISION, files=files, archive_members=members,
                  held_out_payloads_opened=0, model_calls=0, training_updates=0)
    (a.out / 'ACQUISITION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'members': members, 'status': 'acquired_not_imported'}), flush=True)


if __name__ == '__main__':
    main()
