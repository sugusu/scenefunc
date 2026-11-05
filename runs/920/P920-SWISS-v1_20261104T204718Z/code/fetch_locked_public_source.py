"""One bounded public-source request with durable failure and integrity records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    destination = Path(plan['destination'])
    partial = destination.with_name(destination.name + '.partial')
    if destination.exists() or partial.exists() or a.out.exists():
        raise FileExistsError('Preserve existing downloads and receipts; do not restart over them')
    destination.parent.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = {'started_at': now(), 'status': 'running', 'url': plan['url'],
              'plan_sha256': hashlib.sha256(a.plan.read_bytes()).hexdigest(),
              'bytes_received': 0, 'model_calls': 0, 'training_updates': 0}
    def save():
        a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    save()
    print(now(), 'begin bounded public download', plan['url'], flush=True)
    start = time.monotonic()
    last_log = start
    md5, sha = hashlib.md5(), hashlib.sha256()
    try:
        proxies = {'http': plan['proxy'], 'https': plan['proxy']}
        with requests.get(plan['url'], stream=True, timeout=(20, 45), proxies=proxies) as response:
            result.update(http_status=response.status_code, resolved_url=response.url,
                          content_type=response.headers.get('Content-Type'), etag=response.headers.get('ETag'))
            response.raise_for_status()
            declared = response.headers.get('Content-Length')
            if declared and int(declared) > plan['budget']['max_bytes']:
                raise ValueError('Content-Length exceeds frozen budget')
            with partial.open('xb') as stream:
                for block in response.iter_content(2**20):
                    if time.monotonic() - start > plan['budget']['seconds']:
                        raise TimeoutError('Frozen download deadline exceeded')
                    result['bytes_received'] += len(block)
                    if result['bytes_received'] > plan['budget']['max_bytes']:
                        raise ValueError('Frozen download byte budget exceeded')
                    stream.write(block)
                    md5.update(block)
                    sha.update(block)
                    if time.monotonic() - last_log >= 20:
                        save()
                        print(now(), 'received_bytes', result['bytes_received'], flush=True)
                        last_log = time.monotonic()
        result.update(md5=md5.hexdigest(), sha256=sha.hexdigest())
        if 'expected_bytes' in plan and result['bytes_received'] != plan['expected_bytes']:
            raise ValueError('Downloaded size does not match source metadata')
        if 'expected_md5' in plan and result['md5'] != plan['expected_md5']:
            raise ValueError('Downloaded MD5 does not match source metadata')
        if plan.get('kind') == 'json':
            json.loads(partial.read_text())
        partial.rename(destination)
        result.update(status='completed', destination=str(destination), exit_code=0)
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      partial_path=str(partial) if partial.exists() else None, exit_code=1)
        raise
    finally:
        result.update(ended_at=now(), elapsed_seconds=time.monotonic() - start)
        save()
        print(now(), result['status'], 'bytes', result['bytes_received'], flush=True)


if __name__ == '__main__':
    main()
