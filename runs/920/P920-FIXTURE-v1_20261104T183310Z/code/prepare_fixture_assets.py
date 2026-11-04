"""Resolve two prospectively selected fixtures to a bounded pinned asset plan.

This stage queries file metadata only. A separate stage downloads the plan.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    request = json.loads(args.request.read_text())
    source = Path(request['shortlist']['path'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == request['shortlist']['sha256']
    candidates = json.loads(source.read_text())['candidates']
    assert len(candidates) == 2
    plan = {**request, 'resolved_at': datetime.now(timezone.utc).isoformat(), 'files': []}
    api = HfApi(token=True)
    for repo, revision, suffix, role in [
        ('hssd/hssd-models', '208ea6bc22d863bbc1cbaef1829b3e646600dfb6', '.glb', 'mesh'),
        ('hssd/hssd-hab', '4369cb9876214c7fbebcf552eb532380e4d287e4', '.object_config.json', 'coordinate_frame'),
    ]:
        paths = {f"objects/{c['id'][0]}/{c['id']}{suffix}": c for c in candidates}
        entries = api.get_paths_info(repo, list(paths), repo_type='dataset', revision=revision)
        assert {e.path for e in entries} == set(paths)
        for entry in entries:
            c = paths[entry.path]
            lfs = getattr(entry, 'lfs', None)
            expected = {'sha256': lfs.sha256} if lfs else {'git_blob': entry.blob_id}
            plan['files'].append({'repository': repo, 'revision': revision, 'path': entry.path,
                'bytes': entry.size, **expected, 'asset_id': c['id'], 'asset_name': c['name'], 'role': role})
    assert len(plan['files']) == request['budget']['files'] == 4
    assert sum(f['bytes'] for f in plan['files']) <= request['budget']['max_bytes'] <= 32*1024*1024
    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'files': len(plan['files']), 'bytes': sum(f['bytes'] for f in plan['files']), 'downloaded': 0}))


if __name__ == '__main__':
    main()
