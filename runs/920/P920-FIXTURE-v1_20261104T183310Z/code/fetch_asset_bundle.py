"""Fetch a bounded immutable asset plan, checking LFS SHA256 or Git blob IDs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    budget = plan['budget']
    result = {'started_at': datetime.now(timezone.utc).isoformat(), 'files': [],
        'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(), 'status': 'running'}
    code = 0
    try:
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['download_script_sha256']
        assert 0 < len(plan['files']) == budget['files'] <= 64
        assert sum(f['bytes'] for f in plan['files']) <= budget['max_bytes'] <= 32*1024*1024
        for entry in plan['files']:
            assert len(entry['revision']) == 40
            assert bool(entry.get('sha256')) != bool(entry.get('git_blob'))
            p = Path(hf_hub_download(entry['repository'], entry['path'], repo_type='dataset',
                revision=entry['revision'], cache_dir=plan['cache_directory'], token=True))
            raw = p.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
            assert len(raw) == entry['bytes']
            if 'sha256' in entry:
                assert digest == entry['sha256']
            else:
                assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == entry['git_blob']
            result['files'].append({**entry, 'sha256': digest, 'local_path': str(p), 'verified': True})
            print('verified', entry['asset_id'], entry['role'], len(raw), flush=True)
        result['status'] = 'verified'
    except Exception as exc:
        code = 1
        result.update(status='failed', error_type=type(exc).__name__,
            http_status=getattr(getattr(exc, 'response', None), 'status_code', None))
        print('failed', result['error_type'], result['http_status'], flush=True)
    finally:
        result.update(ended_at=datetime.now(timezone.utc).isoformat(), exit_code=code)
        (args.out/'DOWNLOAD.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
