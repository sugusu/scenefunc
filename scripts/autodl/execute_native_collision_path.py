"""One real native HTTP collision request, using composed furniture settings.

Full AssetManager construction/VLM/SDF services are not claimed. Native app and
client modules load normally; the manager delegator method is AST-isolated to
avoid unrelated unavailable asset-generation imports. No method body changes.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(p, d):
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False)+'\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def server(source, out, port):
    native = module('native_collision_server', source/'server_app.py')
    import flask
    app = native.ConvexDecompositionServerApp()
    @app.after_request
    def receipt(response):
        raw = response.get_data()
        record = {'path': flask.request.path, 'status_code': response.status_code,
                  'response_sha256': hashlib.sha256(raw).hexdigest(), 'response_bytes': len(raw)}
        if flask.request.path == '/generate_collision':
            record['request'] = flask.request.get_json()
            (out/'raw_response.json').write_bytes(raw)
        with (out/'http_events.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        return response
    app.run(host='127.0.0.1', port=port, threaded=False, debug=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--serve', type=int)
    a = ap.parse_args()
    stage, out = a.stage.resolve(), a.out.resolve()
    source = stage/'source/scenesmith/agent_utils/convex_decomposition_server'
    if a.serve is not None:
        server(source, out, a.serve)
        return
    out.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(out).free > 5*1024**3 + 32*1024**2
    tick = time.monotonic()
    protocol = read(stage/'protocol.json')
    geometry = Path(protocol['asset_source'])
    assert sha(geometry) == protocol['asset_source_sha256']
    sources = read(stage/'source_receipt.json')
    for row in sources['files']:
        assert sha(stage/'source'/row['path']) == row['sha256']
    root = Path(__file__).resolve().parents[2]
    native_manager = Path('/root/autodl-tmp/agent_method_preparation_20261027/work/native_room_bridge_20261104/source/scenesmith/agent_utils/asset_manager.py')
    lock = read(root/'paris/config/upstream.lock.json')
    manager_pin = next(r for r in lock['files'] if r['path'].endswith('/asset_manager.py'))
    assert sha(native_manager) == manager_pin['sha256']
    import numpy as np
    import trimesh
    from omegaconf import OmegaConf
    assert trimesh.__version__ == '4.7.4'
    cfg = OmegaConf.create(read(stage/'collision_configuration.json'))
    assert cfg.method == 'vhacd'
    port = None
    for candidate in range(*cfg.server_port_range):
        with socket.socket() as s:
            try:
                s.bind(('127.0.0.1', candidate)); port = candidate; break
            except OSError:
                continue
    assert port is not None
    log = (out/'server.log').open('w')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
        '--stage', str(stage), '--out', str(out), '--serve', str(port)],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    result = {'status': 'started', 'server_pid': proc.pid, 'port': port,
        'new_model_calls': 0, 'paid_api_calls': 0, 'training_updates': 0,
        'new_scene_generations': 0, 'full_runtime_ready': False}
    dump(out/'service_process.json', {'pid': proc.pid, 'process_group': proc.pid,
        'parent_pid': os.getpid(), 'port': port, 'standalone_server': True})
    def deadline(signum, frame):
        raise TimeoutError('260-second one-request process cap')
    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(260)
    try:
        client_module = module('native_collision_client', source/'client.py')
        client = client_module.ConvexDecompositionClient(host='127.0.0.1', port=port, timeout=240)
        started = time.monotonic(); probes = 0
        while True:
            assert proc.poll() is None, 'Native server exited during startup'
            probes += 1
            if client.health_check(): break
            if time.monotonic()-started > 20:
                raise TimeoutError('Native server not healthy within20seconds')
            time.sleep(.1)
        tree = ast.parse(native_manager.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AssetManager')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_generate_collision_geometry')
        scope = {'Path': Path, 'trimesh': trimesh}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(native_manager), 'exec'), scope)
        owner = SimpleNamespace(collision_client=client, collision_method=cfg.method,
            collision_coacd_cfg=cfg.coacd, collision_vhacd_cfg=cfg.vhacd)
        request_start = time.monotonic()
        pieces = scope['_generate_collision_geometry'](owner, geometry)
        request_seconds = time.monotonic()-request_start
        assert pieces
        destination = out/'pieces';destination.mkdir()
        rows = []
        for i, piece in enumerate(pieces):
            assert np.isfinite(piece.vertices).all() and len(piece.faces)
            assert piece.faces.min() >= 0 and piece.faces.max() < len(piece.vertices)
            path = destination/f'piece_{i:03d}.obj';piece.export(path)
            rows.append({'index': i, 'vertices': len(piece.vertices), 'faces': len(piece.faces),
                'convex': bool(piece.is_convex), 'watertight': bool(piece.is_watertight),
                'volume': float(piece.volume), 'bounds': piece.bounds.tolist(),
                'local_path': str(path), 'sha256': sha(path)})
        raw = read(out/'raw_response.json')
        assert raw['success'] and len(raw['collision_pieces']) == len(pieces)
        assert sha(geometry) == protocol['asset_source_sha256']
        result.update(status='native_http_collision_pieces_returned', pieces=rows,
            piece_count=len(pieces), request_seconds=request_seconds, health_probes=probes,
            server_processing_seconds=raw['processing_time_s'], input_path=str(geometry),
            input_sha256=sha(geometry), method='vhacd', parameters=OmegaConf.to_container(cfg.vhacd),
            all_returned_convex=all(r['convex'] for r in rows),
            all_returned_watertight=all(r['watertight'] for r in rows),
            input_unchanged=True, raw_http_response_sha256=sha(out/'raw_response.json'),
            manager_source_sha256=sha(native_manager), native_method_lines=[fn.lineno,fn.end_lineno],
            scope='Real native method delegator, client, Flask app and V-HACD on one existing retrieved mesh. No manager constructor/VLM/canonicalization/Blender/SDF or scene qualification.')
    except BaseException as exc:
        result.update(status='failed_no_retry', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL);proc.wait()
        log.close()
        result.update(server_exit_code=proc.returncode, server_terminal=proc.poll() is not None,
            elapsed_seconds=time.monotonic()-tick,
            parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        dump(out/'result.json', result)
        print(json.dumps({k:result.get(k) for k in ['status','error','piece_count','request_seconds','elapsed_seconds','server_terminal']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
