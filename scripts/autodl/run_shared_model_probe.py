"""Bounded new local runtime probe; native Python client, one model actor.

Uses a real archived current observation and a real archived native tool INPUT.
No old model response, teacher D, action dispatch, paid API, optimizer or retry.
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

from shared_native_model_host import SharedNativeClient


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, d):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    args = p.parse_args()
    stage = args.stage.resolve()
    protocol_path = stage / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    preflight = json.loads((stage / 'preflight.json').read_text())
    assert preflight['status'] == 'passed' and preflight['protocol_sha256'] == sha(protocol_path)
    assert protocol['fixed_order'] == ['separate_direct_v3', 'native_tool', 'paired_phase_v3']
    assert protocol['max_model_forwards'] == 4
    out = stage / 'actual_v1'
    out.mkdir(exist_ok=False)
    assert shutil.disk_usage(out).free > 5 * 1024**3 + 4 * 1024**2
    assert int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text()) >= 24 * 1024**3
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    assert int(gpu.strip()) >= 26 * 1024
    inputs = dict(protocol['code_sha256'])
    inputs[protocol['phase_input']] = protocol['phase_input_sha256']
    inputs[protocol['tool_input']] = protocol['tool_input_sha256']
    inputs[protocol['source_material']] = protocol['source_material_sha256']
    inputs[protocol['source_tool_request']] = protocol['source_tool_request_sha256']
    inputs[protocol['state_file']] = protocol['state_file_sha256']
    for path, expected in inputs.items():
        assert sha(path) == expected, path
    for row in protocol['weight_sha256_verified']:
        st = Path(row['path']).stat()
        assert (st.st_size, st.st_mtime_ns, st.st_ino) == (row['bytes'], row['mtime_ns'], row['inode'])
    obs = json.loads(Path(protocol['phase_input']).read_text())
    tool_request = json.loads(Path(protocol['tool_input']).read_text())
    for row in obs['images']:
        assert sha(row['path']) == row['sha256']
        inputs[row['path']] = row['sha256']
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    blocked = []
    def audit(event, args):
        if event == 'socket.connect':
            addr = args[1]
            if not isinstance(addr, tuple) or addr[0] != '127.0.0.1' or addr[1] != port:
                blocked.append('connect'); raise RuntimeError('Only owned model-host loopback allowed')
        if event == 'socket.getaddrinfo' and args[0] not in ('127.0.0.1', 'localhost', '::1'):
            blocked.append('dns'); raise RuntimeError('External DNS prohibited')
    sys.addaudithook(audit)
    actor = None; client = None; log = None
    result = {'status': 'started', 'new_complete_scenes': 0, 'training_steps': 0,
              'paid_API_calls': 0, 'policy_action_dispatch': False, 'whole_scene_success': None}
    start = time.monotonic()
    def deadline(signum, frame):
        raise TimeoutError('New bounded runtime probe exceeded 540 seconds')
    prior_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(540)
    try:
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
        env.pop('PYTHONPATH', None)
        repo = Path(__file__).resolve().parents[2]
        model_python = '/root/autodl-tmp/agent_method_preparation_20261027/work/v100_runtime_20261104/runtime/bin/python'
        log = (out / 'actor.log').open('w')
        actor = subprocess.Popen([model_python, '-B', str(repo / 'scripts/autodl/shared_native_model_host.py'),
            '--stage', str(stage), '--out', str(out / 'actor'), '--port', str(port)],
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        dump(out / 'actor_process.json', {'pid': actor.pid, 'port': port, 'python': model_python})
        client = SharedNativeClient(port, timeout=200)
        wait_start = time.monotonic()
        while True:
            if actor.poll() is not None:
                raise RuntimeError('Owned actor exited before readiness')
            try:
                if client.health()['status'] == 'ready':
                    break
            except Exception:
                pass
            if time.monotonic() - wait_start > 40:
                raise TimeoutError('Actor startup, no restart')
            time.sleep(.1)
        rows = []
        for condition in protocol['fixed_order']:
            if actor.poll() is not None:
                raise RuntimeError('Actor exited; no restart')
            health = client.health()
            if health['incomplete_forward_accounting']:
                raise RuntimeError('Unknown previous forward accounting; stop remaining requests')
            row = {'condition': condition, 'status': 'started'}
            rows.append(row)
            t = time.monotonic()
            try:
                if condition == 'native_tool':
                    text = client.create_completion(**tool_request)
                    row.update(status='returned', text=text,
                               truth_scope='New model estimate on archived real multi-view input; not measured physics truth')
                else:
                    reply = client.phase(arm=condition, **obs)
                    row.update(status='transport_returned', policy_status=reply['status'], result=reply,
                               method_scope='Untrained same base, deployment-format qualification only')
            except Exception as exc:
                row.update(status='failed_no_retry', error=type(exc).__name__ + ': ' + str(exc))
            finally:
                row['seconds'] = time.monotonic() - t
                dump(out / 'client_results.json', rows)
        ledger = json.loads((out / 'actor/shared_model_ledger.json').read_text())
        forwards = ledger['forwards']
        assert len(forwards) <= 4 and ledger['total_output_tokens'] <= 1536
        assert len(ledger['model_loads']) == 1
        assert len({r['model_object_id'] for r in forwards}) == 1
        assert len({r['first_parameter_data_ptr'] for r in forwards}) == 1
        assert not ledger['blocked_network'] and not blocked
        result.update(status='bounded_runtime_probe_completed', client_conditions=rows,
            actual_model_forwards=len(forwards), output_tokens=ledger['total_output_tokens'],
            model_loads=len(ledger['model_loads']), same_model_object_and_parameter_pointer=True,
            original_inputs_unchanged=all(sha(p) == h for p, h in inputs.items()),
            native_client_python=sys.version.split()[0], native_client_executable=sys.executable,
            actor_peak_cuda_allocated_bytes=max((r.get('peak_cuda_allocated_bytes', 0) for r in forwards), default=0),
            actor_peak_cuda_reserved_bytes=max((r.get('peak_cuda_reserved_bytes', 0) for r in forwards), default=0),
            whole_runtime_ready=False, learned_advantage_verified=False,
            previous_closed_protocols_reopened=False)
        assert result['original_inputs_unchanged']
        dump(out / 'input_integrity.json', inputs)
    except BaseException as exc:
        result.update(status='failed_no_retry', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, prior_alarm)
        if client is not None:
            client.close()
        if actor is not None and actor.poll() is None:
            os.killpg(actor.pid, signal.SIGTERM)
            try:
                actor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(actor.pid, signal.SIGKILL); actor.wait(timeout=10)
        if log is not None:
            log.close()
        result.update(total_parent_seconds=time.monotonic() - start,
                      actor_exit_code=actor.returncode if actor is not None else None,
                      actor_process_terminal=actor is not None and actor.poll() is not None,
                      blocked_parent_network=blocked)
        dump(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'client_conditions'}))


if __name__ == '__main__':
    main()
