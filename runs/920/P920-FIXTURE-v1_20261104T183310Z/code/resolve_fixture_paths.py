"""Resolve up to 128 bathroom fixture IDs against fixed mesh/frame metadata.

Only repository metadata is read. Both observed shard paths and flat paths are
checked; missing files and alternative selections are explicit, never invented.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    rows = json.loads(a.input.read_text())['bathroom_term_candidates']
    rows = [r for r in rows if set(r['wnsynsetkey'].split(',')) & {'toilet.n.02', 'shower_stall.n.01'}]
    assert 0 < len(rows) <= 128
    result = {'started_at': datetime.now(timezone.utc).isoformat(), 'candidate_count': len(rows),
        'scope': 'metadata only, <=128 IDs, two path patterns, two fixed repositories', 'repositories': []}
    api = HfApi(token=True)
    for repo, rev, suffix in [
        ('hssd/hssd-models', '208ea6bc22d863bbc1cbaef1829b3e646600dfb6', '.glb'),
        ('hssd/hssd-hab', '4369cb9876214c7fbebcf552eb532380e4d287e4', '.object_config.json'),
    ]:
        paths = {}
        for r in rows:
            for path in [f"objects/{r['id'][0]}/{r['id']}{suffix}", f"objects/{r['id']}{suffix}"]:
                paths[path] = r['id']
        entries = api.get_paths_info(repo, list(paths), repo_type='dataset', revision=rev)
        found = []
        for e in entries:
            lfs = getattr(e, 'lfs', None)
            found.append({'path': e.path, 'asset_id': paths[e.path], 'bytes': e.size,
                **({'sha256': lfs.sha256} if lfs else {'git_blob': e.blob_id})})
        result['repositories'].append({'repository': repo, 'revision': rev, 'files': found,
            'missing_ids': sorted({r['id'] for r in rows} - {f['asset_id'] for f in found})})
        print(repo, 'matched files', len(found), 'candidate IDs', len(rows), flush=True)
    result['ended_at'] = datetime.now(timezone.utc).isoformat()
    a.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
