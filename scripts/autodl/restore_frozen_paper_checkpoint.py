"""Restore the archived Paper checkpoint from its exact manifest and resources.

Only already-recovered frozen resources and pinned Git metadata are read. This
does not invent an intermediate session, change a pose or execute a policy.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
META = 'runs/920/P920-PAPER-v2_20261105T182424Z/checkpoint_metadata/'
BASE_MANIFEST = '0cf11058d036d1ec0cd1b589b6dbb39b2465e9ee1b74eee6faed8eddc0d416ac'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); sources = {}; result = {'status': 'started'}
    try:
        assert sha(a.base/'manifest.json') == BASE_MANIFEST
        path = Path('runs/autodl/20261110_corpus_state_recovery/source/native_project.py')
        assert sha(path) == '1c418410b1d5db8606305f200154418dd59844ce1c22fee3afb903286417ad19'
        spec = importlib.util.spec_from_file_location('frozen_native_checkpoint_validator', path)
        native = importlib.util.module_from_spec(spec); sys.modules[spec.name] = native; spec.loader.exec_module(native)
        base, _, _, _ = native.validate_bundle(a.base)
        by_hash = defaultdict(list)
        for relative in ['manifest.json', *base['files']]:
            p = a.base/relative; by_hash[sha(p)].append(p)
        def get(relative):
            raw = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show', PIN+':'+META+relative])
            sources[META+relative] = hashlib.sha256(raw).hexdigest()
            return raw
        manifest_raw = get('manifest.json'); manifest = json.loads(manifest_raw)
        required = sum(r['bytes'] for r in manifest['files'].values())+len(manifest_raw)
        if shutil.disk_usage(a.out).free-required < 5*1024**3:
            raise RuntimeError('Exact checkpoint copy would cross the disk reserve')
        bundle = a.out/'bundle'; bundle.mkdir(); (bundle/'manifest.json').write_bytes(manifest_raw)
        rows = []
        for relative, expected in manifest['files'].items():
            target = native.contained_path(bundle, relative); target.parent.mkdir(parents=True, exist_ok=True)
            if expected['sha256'] in by_hash:
                src = by_hash[expected['sha256']][0]
                shutil.copyfile(src, target); source = str(src)
            else:
                # Published metadata only. Never search an asset by its label
                # or substitute unrelated geometry to fill a missing resource.
                if relative not in {'house_state.json', 'runtime.json', 'session_checkpoint.json'}:
                    raise ValueError('Frozen resource unavailable: '+relative)
                target.write_bytes(get(relative)); source = META+relative
            actual = sha(target)
            assert actual == expected['sha256'] and target.stat().st_size == expected['bytes'], relative
            rows.append({'path': relative, 'sha256': actual, 'bytes': target.stat().st_size, 'source': source})
        copied, state, runtime, task = native.validate_bundle(bundle)
        assert (bundle/'manifest.json').read_bytes() == manifest_raw
        rid = copied['selected_room_id']; room = state['rooms'][rid]
        canonical = hashlib.sha256(json.dumps(room, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        write(a.out/'file_checks.json', rows)
        result.update(status='exact_checkpoint_resources_restored', expected_files=len(rows), exact_files=len(rows),
                      original_validator_passed=True, manifest_rewritten=False,
                      manifest_sha256=sha(bundle/'manifest.json'), room_state_canonical_sha256=canonical,
                      room_id=rid, object_records=len(room['objects']),
                      furniture_objects=sum(o['object_type']=='furniture' for o in room['objects'].values()),
                      frozen_source_session_present=True, original_base_verified=True)
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc)); raise
    finally:
        result.update(pin=PIN, sources=sources, wall_seconds=time.monotonic()-start,
                      native_runtime_instantiated=False, session_resumed=False, renderer_run=False,
                      new_model_calls=0, training_updates=0, new_scene_generations=0,
                      full_scene_success=None)
        write(a.out/'summary.json', result); print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
