"""Fetch the small, pinned source files reviewed by the CPU evidence audit.

No model weights or inference clients are loaded. Existing differing files are
never overwritten. Downloaded upstream code stays in an ignored local cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

PREFIX = ('https://raw.githubusercontent.com/ZJU-OmniAI/Scene-Designer/'
          'a0565e50ed7d32c0e09132b9311c0b9b91165405/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.index.read_text())
    result = []
    for row in rows:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Nonrelative source path')
        if row['url'] != PREFIX + relative.as_posix():
            raise ValueError('Source URL is not pinned to the reviewed revision')
        target = args.output / relative
        cached = target.exists()
        if cached:
            data = target.read_bytes()
        else:
            with urlopen(row['url'], timeout=30) as response:
                data = response.read(1_000_001)
        if len(data) > 1_000_000 or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Source size/hash mismatch: ' + row['path'])
        if not cached:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
        result.append({'path': row['path'], 'sha256': row['sha256'], 'cached': cached})
    print(json.dumps({'verified_sources': len(result), 'files': result}))


if __name__ == '__main__':
    main()
