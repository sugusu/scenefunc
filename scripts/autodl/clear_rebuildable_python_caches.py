"""Delete only source-backed Python bytecode caches under research directories.

The user authorized removing unused caches to retain actual scene evidence.
Never removes Python source, model files, experiment outputs, app state, shared
libraries or source-less bytecode. All deletions have a before-action receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOTS = [
    '/root/autodl-tmp/layout_experience_pilot',
    '/root/autodl-tmp/scene_generation_next_20261016',
    '/root/autodl-tmp/scene_generation_handoff_20261014_v2',
    '/root/autodl-tmp/agent_method_preparation_20261027',
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    before = shutil.disk_usage(a.out).free
    seen = set()
    rows = []
    for name in ROOTS:
        for entry in Path(name).rglob('*.pyc'):
            if entry.is_symlink() or entry.parent.name != '__pycache__':
                continue
            path = entry.resolve()
            if path in seen:
                continue
            seen.add(path)
            source = path.parent.parent / (path.name.split('.')[0]+'.py')
            if not source.is_file() or source.is_symlink():
                continue
            stat = path.stat()
            rows.append(dict(path=str(path), sha256=sha(path), bytes=stat.st_size,
                             inode=stat.st_ino, mtime_ns=stat.st_mtime_ns,
                             source=str(source), source_sha256=sha(source)))
    receipt = dict(authority='User authorized unused/cache deletion; no raw results deleted',
                   rebuild='Python regenerates each cache from the retained same source on import',
                   rows=rows, deleted=[], free_before=before)
    target = a.out/'receipt.json'
    target.write_text(json.dumps(receipt, indent=2)+'\n')
    for row in rows:
        path, source = Path(row['path']), Path(row['source'])
        assert sha(path) == row['sha256'] and sha(source) == row['source_sha256']
        path.unlink()
        receipt['deleted'].append(row['path'])
    receipt.update(free_after=shutil.disk_usage(a.out).free,
                   cache_bytes=sum(r['bytes'] for r in rows), deleted_count=len(rows),
                   sources_unchanged=all(sha(Path(r['source'])) == r['source_sha256'] for r in rows),
                   deleted_models=0, deleted_experiment_results=0)
    target.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('rows','deleted')}))


if __name__ == '__main__':
    main()
