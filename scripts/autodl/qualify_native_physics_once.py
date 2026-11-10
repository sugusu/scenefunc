"""One bounded real local physics-analysis call; no repairs, retry or API.

This is a separate asset-service qualification, not a continuation/reset of
the closed four-attempt furniture diagnostic and not a design-method score.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


WORK = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')


def dump(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def child(out, protocol):
    attempted = []

    def reject(*args, **kwargs):
        attempted.append(True)
        raise RuntimeError('External and loopback networking disabled in inference child')

    socket.socket.connect = reject
    socket.create_connection = reject
    sys.path.insert(0, str(WORK/'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine, LocalVLMService
    messages = read(protocol['input_request'])
    assert sha(protocol['input_request']) == protocol['input_request_sha256']
    engine = LocalQwenEngine(Path(protocol['checkpoint']), out/'model_call',
        max_pixels=65536, max_input_tokens=8192, max_new_tokens=512)
    service = LocalVLMService(engine)
    started = time.monotonic()
    result = dict(status='started', model_subprocess_attempts=1,
                  paid_api_calls=0, training_updates=0, new_complete_scenes=0)
    try:
        returned = service.create_completion(model=protocol['requested_model_label'],
            messages=messages, reasoning_effort=protocol['requested_reasoning_effort'],
            verbosity=protocol['requested_verbosity'], vision_detail=protocol['requested_vision_detail'],
            response_format={'type':'json_object'})
        # Adapter already applies strict json.loads; preserve exact text as well.
        (out/'response_text.txt').write_text(returned)
        dump(out/'response.json', json.loads(returned))
        result['status'] = 'local_vlm_returned_json'
    except BaseException as exc:
        result.update(status='local_vlm_failed_no_retry', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        calls = list((out/'model_call').glob('*/receipt.json'))
        result.update(seconds=time.monotonic()-started, completed_model_outputs=sum(
            read(p).get('status') == 'completed' for p in calls),
            model_receipts=[str(p) for p in calls], network_attempts=len(attempted),
            no_physical_ground_truth_claim=True, full_runtime_ready=False)
        dump(out/'child_result.json', result)
        print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--child', action='store_true')
    a = ap.parse_args()
    stage = a.stage.resolve()
    protocol = read(stage/'protocol.json')
    if a.child:
        child(stage/'actual_v1', protocol)
        return
    out = stage/'actual_v1'
    out.mkdir(exist_ok=False)
    assert shutil.disk_usage(out).free > 5*1024**3 + 16*1024**2
    assert int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text()) >= 24*1024**3
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used',
                                  '--format=csv,noheader,nounits'], text=True)
    name, total, used = gpu.strip().split(', ')
    assert int(total)-int(used) >= 26*1024
    assert sha(protocol['input_request']) == protocol['input_request_sha256']
    assert sha(WORK/'paris_local_run_results_20261104/local_vlm_sdpa.py') == protocol['adapter_sha256']
    lock = read(Path(__file__).resolve().parents[2]/'paris/config/weights.lock.json')
    stats = []
    for row in lock['files']:
        p = Path(protocol['checkpoint'])/row['path']
        assert p.stat().st_size == row['bytes']
        stats.append({'path':str(p), 'bytes':p.stat().st_size, 'mtime_ns':p.stat().st_mtime_ns})
    dump(out/'resources_before.json', {'nvidia_smi':gpu, 'weight_stats':stats,
        'hash_scope':'Sizes/mtime sampled now; full11-file SHA verification completed in bounded readiness, not repeated here.'})
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    command = [str(WORK/'v100_runtime_20261104/runtime/bin/python'), '-B',
               str(Path(__file__).resolve()), '--stage', str(stage), '--child']
    tick = time.monotonic()
    result = {'status':'started', 'timeout':False, 'maximum_attempts':1, 'max_seconds':240,
              'command':command, 'paid_api_calls':0, 'training_updates':0}
    with (out/'stdout.txt').open('w') as stdout, (out/'stderr.txt').open('w') as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        dump(out/'process.json', {'pid':proc.pid, 'process_group':proc.pid})
        try:
            code = proc.wait(timeout=240)
        except subprocess.TimeoutExpired:
            result['timeout'] = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            code = proc.returncode
        finally:
            result.update(status='terminal', returncode=proc.poll(),
                terminal=proc.poll() is not None, seconds=time.monotonic()-tick,
                request_unchanged=sha(protocol['input_request'])==protocol['input_request_sha256'],
                weights_stat_unchanged=all(Path(r['path']).stat().st_size==r['bytes'] and
                    Path(r['path']).stat().st_mtime_ns==r['mtime_ns'] for r in stats))
            dump(out/'process_receipt.json', result)
    print(json.dumps(result), flush=True)
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
