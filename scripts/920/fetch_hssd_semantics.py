"""Acquire one pinned metadata CSV and identify bathroom category candidates.

Budget: one CSV <= 8 MiB, no meshes, models, engine or training calls.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = {'started_at': datetime.now(timezone.utc).isoformat(),
        'repository': 'hssd/hssd-hab', 'revision': '4369cb9876214c7fbebcf552eb532380e4d287e4',
        'path': 'semantics/objects.csv', 'budget_bytes': 8*1024*1024,
        'meshes_downloaded': 0, 'credential_values_logged': False}
    code = 0
    try:
        api = HfApi(token=True)
        items = api.get_paths_info(result['repository'], [result['path']],
            repo_type='dataset', revision=result['revision'])
        if len(items) != 1 or not 0 < items[0].size <= result['budget_bytes']:
            raise ValueError('Metadata absent or exceeds budget')
        item = items[0]
        p = Path(hf_hub_download(result['repository'], result['path'],
            repo_type='dataset', revision=result['revision'], cache_dir=args.cache, token=True))
        raw = p.read_bytes()
        if len(raw) != item.size:
            raise ValueError('Metadata byte count mismatch')
        # HSSD objects.csv is a regular Git blob, checked against repository metadata.
        git_blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if git_blob != item.blob_id:
            raise ValueError('Metadata Git blob mismatch')
        rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
        keys = list(rows[0]) if rows else []
        matches = []
        for row in rows:
            # Broad discovery only. False matches remain visible for later review.
            text = json.dumps(row).lower()
            if any(term in text for term in ['toilet', 'shower', 'bathtub']):
                matches.append(row)
        result.update(status='verified_metadata_only', local_path=str(p), bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(), git_blob=git_blob,
            columns=keys, total_records=len(rows), bathroom_term_candidates=matches,
            limitation='Term matches are discovery leads, not selected or usable fixtures; dimensions, mesh identity and native action mapping remain unverified.')
    except Exception as exc:
        code = 1
        result.update(status='failed', error_type=type(exc).__name__,
            http_status=getattr(getattr(exc, 'response', None), 'status_code', None))
    result.update(ended_at=datetime.now(timezone.utc).isoformat(), exit_code=code)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['bathroom_term_candidates', 'columns']}))
    print('bathroom term candidates', len(result.get('bathroom_term_candidates', [])))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
