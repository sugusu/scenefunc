"""Fetch the explicit 920 upstream lock; never weaken size/hash verification."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = {'started_at': now(), 'lock_sha256': hashlib.sha256(a.lock.read_bytes()).hexdigest(), 'files': [], 'status': 'running'}
    print(result['started_at'], 'fixed official-source download started', flush=True)
    try:
        for item in json.loads(a.lock.read_text())['files']:
            dest = a.data / 'upstream' / item['path']
            assert dest.resolve().is_relative_to(a.data.resolve())
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raw = dest.read_bytes()
            else:
                # Small pinned source files only. No credentials or retries.
                with urllib.request.urlopen(item['url'], timeout=45) as response:
                    raw = response.read(item['bytes'] + 1)
                assert len(raw) == item['bytes'], item['path']
                assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
                with dest.open('xb') as stream:
                    stream.write(raw)
            assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
            result['files'].append({'path': item['path'], 'bytes': len(raw), 'sha256': item['sha256']})
        result.update(status='verified', exit_code=0)
    except Exception as error:
        result.update(status='failed', exit_code=1, error_type=type(error).__name__)
        raise
    finally:
        result['ended_at'] = now()
        a.receipt.write_text(json.dumps(result, indent=2) + '\n')
        print(result['ended_at'], result['status'], len(result['files']), flush=True)


if __name__ == '__main__':
    main()
