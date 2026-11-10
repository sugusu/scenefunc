"""Materialize the native SDK wheel closure in an isolated directory.

Uses the inspected uv.lock, its exact artifact hashes, platform wheel tags and
dependency markers. Does not install into an existing environment, run package
setup code, fetch models, or contact model services. Run with native Python 3.11.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tomllib
import urllib.error
import urllib.request
import zipfile

from packaging.markers import Marker
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lock', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.resume)
    (args.out / '.gitignore').write_text('runtime/\n')
    runtime = args.out / 'runtime'
    runtime.mkdir(exist_ok=args.resume)
    packages = tomllib.loads(args.lock.read_text())['package']
    by = {}
    for p in packages:
        by.setdefault(p['name'], []).append(p)
    selected = {}
    pending = ['lxml']
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if len(by[name]) != 1:
            raise ValueError('Ambiguous locked package: ' + name)
        p = selected[name] = by[name][0]
        pending.extend(d['name'] for d in p.get('dependencies', [])
                       if 'marker' not in d or Marker(d['marker']).evaluate())
    tags = {t: i for i, t in enumerate(sys_tags())}
    wheels = []
    for name, p in sorted(selected.items()):
        candidates = []
        for w in p.get('wheels', []):
            filename = w['url'].rsplit('/', 1)[-1]
            compatible = parse_wheel_filename(filename)[3] & tags.keys()
            if compatible:
                candidates.append((min(tags[t] for t in compatible), w))
        if not candidates:
            raise ValueError('No compatible wheel: ' + name)
        w = min(candidates, key=lambda x: x[0])[1]
        wheels.append(dict(name=name, version=p['version'], **w))
    receipt = dict(python=sys.version, lock_sha256=hashlib.sha256(args.lock.read_bytes()).hexdigest(),
                   packages=wheels, installed=[], model_calls=0,
                   scope='Isolated wheel extraction, not full native scene runtime')
    if args.resume:
        previous = json.loads((args.out / 'receipt.json').read_text())
        if previous['lock_sha256'] != receipt['lock_sha256'] or previous['packages'] != wheels:
            raise ValueError('Resume lock/platform differs')
        for installed in previous['installed']:
            for file in installed['files']:
                if hashlib.sha256((runtime / file['path']).read_bytes()).hexdigest() != file['sha256']:
                    raise ValueError('Installed file changed: ' + file['path'])
        receipt = previous
    def save():
        (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    for w in wheels:
        if any(p['name'] == w['name'] for p in receipt['installed']):
            continue
        for attempt in range(3):
            try:
                with urllib.request.urlopen(w['url'], timeout=30) as response:
                    payload = response.read(w['size'] + 1)
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                receipt.setdefault('download_failures', []).append(dict(package=w['name'],
                    attempt=attempt, error=type(exc).__name__ + ': ' + str(exc)))
                save()
                if attempt == 2:
                    raise
        if len(payload) != w['size'] or 'sha256:' + hashlib.sha256(payload).hexdigest() != w['hash']:
            raise ValueError('Wheel integrity failure: ' + w['name'])
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            required = sum(((i.file_size + 4095) // 4096) * 4096 for i in z.infolist()) + 131072
            stat = os.statvfs(args.out)
            if stat.f_bavail * stat.f_frsize - required < 5 * 1024**3 + 1048576:
                raise RuntimeError('Disk reserve would be violated: ' + w['name'])
            files = []
            for entry in z.infolist():
                path = Path(entry.filename)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe archive path')
                if '.data' in path.parts[0]:
                    raise ValueError('Wheel requires a relocation installer: ' + entry.filename)
                if entry.is_dir():
                    continue
                dst = runtime / path
                if dst.exists():
                    raise ValueError('Overlapping package file: ' + str(path))
                dst.parent.mkdir(parents=True, exist_ok=True)
                data = z.read(entry)
                dst.write_bytes(data)
                files.append(dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        receipt['installed'].append(dict(name=w['name'], version=w['version'], files=files))
        save()
        print(w['name'], w['version'], len(files), flush=True)
    receipt['status'] = 'extracted_verified_wheels'
    save()


if __name__ == '__main__':
    main()
