"""Run one immutable bounded CPU stage, preserving real start/log/exit evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receipt-dir', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    assert command
    a.receipt_dir.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    record = {'started_at': now(), 'status': 'running', 'command': command, 'timeout_seconds': a.timeout}
    def save():
        temp = a.receipt_dir / 'receipt.tmp'
        temp.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        temp.replace(a.receipt_dir / 'receipt.json')
    start = time.monotonic()
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    env.setdefault('MPLCONFIGDIR', '/tmp/scenefunc920-matplotlib')
    log = a.receipt_dir / 'stage.log'
    with log.open('x', encoding='utf-8') as output:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(pid=proc.pid, start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split(') ', 1)[1].split()[19], log=str(log))
        save()
        print(json.dumps(record), flush=True)
        try:
            code = proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124
    lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
    record.update(status='completed' if code == 0 else 'failed', ended_at=now(), elapsed_seconds=time.monotonic()-start, exit_code=code, first_log=lines[0] if lines else None)
    save()
    print(json.dumps(record), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
