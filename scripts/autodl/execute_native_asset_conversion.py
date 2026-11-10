"""Execute real native Blender RPC conversion and prepare physics-analysis views.

Uses the already archived author integration patches, a single frozen asset,
and real composed settings. No VLM, guessed mass, SDF or scene is substituted.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
from urllib.parse import urlparse


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    stage, out = a.stage.resolve(), a.out.resolve()
    protocol = json.loads((stage/'protocol.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(out).free > 5*1024**3 + 24*1024**2
    source = Path(protocol['native_source'])
    source_manifest = json.loads((source/'manifest.json').read_text())
    patched = {r['path']: r['after_sha256'] for r in source_manifest['patches']}
    before = {}
    for row in source_manifest['sources']:
        path = source/row['path']
        assert sha(path) == patched.get(row['path'], row['sha256']), str(path)
        before[str(path)] = sha(path)
    asset = Path(protocol['asset_source'])
    assert sha(asset) == protocol['asset_sha256']
    before[str(asset)] = sha(asset)
    sys.path.insert(0, str(source))
    from audit_native_services import compose_native
    cfg = compose_native(Path(protocol['config_source']), Path(protocol['config_integration']))
    settings = cfg.furniture_agent.asset_manager
    render_settings = dict(
        elevation_degrees=float(settings.side_view_elevation_degrees),
        num_side_views=int(settings.num_side_views_for_physics_analysis),
        taa_samples=int(settings.validation_taa_samples),
        include_vertical_views=False,
    )
    dump(out/'render_configuration.json', render_settings)
    import requests
    original_request = requests.sessions.Session.request
    events = []

    def record_request(session, method, url, *args, **kwargs):
        parsed = urlparse(url)
        if parsed.hostname not in ('127.0.0.1', 'localhost'):
            raise RuntimeError('Only loopback RPC is allowed in this run')
        row = dict(method=method, path=parsed.path, host=parsed.hostname,
                   json=kwargs.get('json'), data=kwargs.get('data'))
        events.append(row)
        tick = time.monotonic()
        try:
            response = original_request(session, method, url, *args, **kwargs)
            row.update(status=response.status_code,
                       response_sha256=hashlib.sha256(response.content).hexdigest())
            return response
        except BaseException as exc:
            row['error'] = type(exc).__name__+': '+str(exc)
            raise
        finally:
            row['seconds'] = time.monotonic()-tick
            dump(out/'http_events.json', events)

    requests.sessions.Session.request = record_request
    from scenesmith.agent_utils.blender import BlenderServer
    import numpy as np
    import trimesh
    from PIL import Image
    server = BlenderServer(port_range=(18200, 18210), gpu_id=None,
                           log_file=out/'server.log')
    result = dict(status='started', new_model_calls=0, paid_api_calls=0,
                  training_updates=0, new_scene_generations=0, full_runtime_ready=False)
    tick = time.monotonic()
    child = None

    def deadline(signum, frame):
        raise TimeoutError('300-second bounded conversion/view preparation')

    old_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    try:
        server.start()
        child = server._server_process
        server.wait_until_ready(timeout=60)
        dump(out/'server_process.json', {'pid': child.pid, 'args': child.args})
        target = out/'asset'/'nightstand.gltf'
        t = time.monotonic()
        converted = server.convert_glb_to_gltf(asset, target, export_yup=True, timeout=60)
        result['conversion_seconds'] = time.monotonic()-t
        assert converted == target and target.is_file()

        def merged(path):
            obj = trimesh.load(path, force='scene')
            return obj.to_geometry() if hasattr(obj, 'to_geometry') else obj.dump(concatenate=True)

        initial, final = merged(asset), merged(target)
        # Compare world-transformed vertices, not mesh buffer ordering or labels.
        from scipy.spatial import cKDTree
        a_to_b = cKDTree(final.vertices).query(initial.vertices)[0]
        b_to_a = cKDTree(initial.vertices).query(final.vertices)[0]
        result['geometry_comparison'] = dict(
            original_vertices=len(initial.vertices), original_faces=len(initial.faces),
            converted_vertices=len(final.vertices), converted_faces=len(final.faces),
            original_bounds=initial.bounds.tolist(), converted_bounds=final.bounds.tolist(),
            max_bidirectional_vertex_distance_m=float(max(a_to_b.max(), b_to_a.max())),
            bounds_max_abs_difference_m=float(np.max(np.abs(initial.bounds-final.bounds))),
            interpretation='Conversion integrity in loader world coordinates; not canonical front, semantic or physics validation')
        doc = json.loads(target.read_text())
        refs = [r['uri'] for r in doc.get('buffers', [])+doc.get('images', []) if 'uri' in r]
        for ref in refs:
            assert not ref.startswith(('data:', 'http:', 'https:')), ref
            assert (target.parent/ref).is_file(), ref
        result['external_gltf_resources'] = refs
        t = time.monotonic()
        views = server.render_multiview_for_analysis(mesh_path=target,
            output_dir=out/'physics_views', timeout=180, **render_settings)
        result['view_render_seconds'] = time.monotonic()-t
        assert len(views) == render_settings['num_side_views']
        image_rows = []
        for view in views:
            with Image.open(view) as im:
                im.load()
                image_rows.append(dict(path=str(view), width=im.width, height=im.height,
                                       sha256=sha(view), bytes=view.stat().st_size))
        result.update(status='native_asset_converted_and_physics_views_rendered',
                      views=image_rows, generated_gltf=str(target),
                      vlm_analysis_executed=False, canonicalization_executed=False,
                      physical_metadata_known=False, sdf_generated=False)
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        server.stop()
        requests.sessions.Session.request = original_request
        result.update(seconds=time.monotonic()-tick,
            server_pid=child.pid if child else None,
            server_returncode=child.poll() if child else None,
            server_terminal=child.poll() is not None if child else True,
            source_and_asset_unchanged=all(sha(p) == h for p, h in before.items()),
            parent_peak_rss_mib_only=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            child_peak_rss_not_measured=True,
            versions={n: importlib.metadata.version(n) for n in ('bpy','trimesh','numpy','requests')},
            free_data_directory_bytes=shutil.disk_usage(out).free)
        dump(out/'input_hashes.json', before)
        dump(out/'result.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
