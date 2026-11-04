"""Fetch a bounded pinned asset manifest using the standard HF credential store.

No credential values, signed URLs or exception bodies are logged. Raw assets
remain outside Git; this preparation does not establish semantic suitability.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    result = {'started_at': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              'status': 'running', 'files': [], 'credential_values_logged': False}
    print(result['started_at'], 'begin pinned decision asset preparation', flush=True)
    code = 0
    try:
        if len(plan['files']) != 4 or sum(f['bytes'] for f in plan['files']) > 1024 * 1024:
            raise ValueError('Manifest budget exceeded')
        for item in plan['files']:
            downloaded = Path(hf_hub_download(
                repo_id=item['repository'], repo_type='dataset', revision=item['revision'],
                filename=item['path'], cache_dir=plan['cache_directory'], token=True))
            raw = downloaded.read_bytes()
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Asset integrity mismatch')
            result['files'].append({**item, 'local_path': str(downloaded), 'verified': True})
            print('verified', item['asset_id'], item['role'], len(raw), flush=True)
        result['status'] = 'verified'
    except Exception as exc:
        code = 1
        response = getattr(exc, 'response', None)
        result.update(status='failed', error_type=type(exc).__name__,
                      http_status=getattr(response, 'status_code', None))
        print('failed', result['error_type'], result['http_status'], flush=True)
    finally:
        result.update(ended_at=datetime.now(timezone.utc).isoformat(), exit_code=code)
        (args.out / 'DOWNLOAD.json').write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
