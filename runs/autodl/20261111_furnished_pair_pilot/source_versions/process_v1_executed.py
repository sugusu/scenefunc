"""Bounded, no-retry subprocess runner with pre-launch accounting."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--folder', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--kind', choices=['model', 'tool'], required=True)
    ap.add_argument('--seconds', type=float, required=True)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    a.folder.mkdir(parents=True, exist_ok=True)
    path = a.folder / (a.label + '_receipt.json')
    if path.exists():
        raise FileExistsError('Existing launch receipt: inspect, never rerun')
    prior = [json.loads(p.read_text()) for p in a.root.glob('B*/turn_*/model_process_receipt.json')]
    used = sum(p.get('elapsed_seconds', p.get('timeout_seconds', 200)) for p in prior)
    if a.kind == 'model' and (len(prior) >= 12 or used >= 1200):
        raise RuntimeError('Global model attempt/time budget exhausted')
    limit = min(a.seconds, 200, 1200-used) if a.kind == 'model' else a.seconds
    receipt = dict(command=cmd, kind=a.kind, status='launching', timeout_seconds=limit,
                   runner_pid=os.getpid(), start_unix=time.time(), paid_api_calls=0)
    write(path, receipt)
    start = time.monotonic()
    with (a.folder / (a.label+'.stdout')).open('w') as out, (a.folder / (a.label+'.stderr')).open('w') as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, start_new_session=True)
        receipt.update(pid=proc.pid, status='running')
        write(path, receipt)
        timed_out = False
        try:
            code = proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
    receipt.update(status='terminal', exit_code=code, timeout=timed_out,
                   elapsed_seconds=time.monotonic()-start)
    write(path, receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    raise SystemExit(0 if code == 0 else 1)


if __name__ == '__main__':
    main()
