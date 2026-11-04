"""Inspect only the fixed HSSD repository root, without downloading meshes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = {'started_at': datetime.now(timezone.utc).isoformat(),
        'repository': 'hssd/hssd-models',
        'revision': '208ea6bc22d863bbc1cbaef1829b3e646600dfb6',
        'scope': 'nonrecursive root metadata, at most 100 entries, zero mesh downloads',
        'credential_values_logged': False, 'entries': []}
    code = 0
    try:
        api = HfApi(token=True)
        for item in api.list_repo_tree(report['repository'], repo_type='dataset',
                                      revision=report['revision'], recursive=False):
            if len(report['entries']) >= 100:
                raise ValueError('Root entry limit exceeded')
            report['entries'].append({'path': item.path, 'kind': type(item).__name__,
                'size': getattr(item, 'size', None), 'blob_id': getattr(item, 'blob_id', None)})
        report['status'] = 'completed_metadata_only'
    except Exception as exc:
        code = 1
        report.update(status='failed', error_type=type(exc).__name__,
            http_status=getattr(getattr(exc, 'response', None), 'status_code', None))
    report.update(ended_at=datetime.now(timezone.utc).isoformat(), exit_code=code)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
