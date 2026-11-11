"""Recover frozen Milan asset bytes from already licensed raw local HSSD.

The only acceptance oracle is the OLD canonical GLB SHA256, plus all manifest
resource hashes. Axis search is recovery of the recorded asset frame, not
layout search, semantic reasoning, or new training data.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import trimesh


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, v):
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--raw-root', type=Path, default=Path('/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/official_assets/hssd-models/objects'))
    a = ap.parse_args(); start = time.monotonic()
    if trimesh.__version__ != '4.7.4':
        raise RuntimeError('Use the already installed pinned trimesh4.7.4 recovery dependency')
    metadata = a.root/'source/runs/920/P920-MILAN-TASK-v1_20261105T224015Z/input_metadata'
    manifest = json.loads((metadata/'manifest.json').read_text())
    catalog = json.loads((metadata/'catalog.json').read_text())
    rid = manifest['selected_room_id']; bundle = a.scratch/'bundle'; room = bundle/('room_'+rid)
    helper = Path('runs/autodl/20261111_waterloo_task_search/source/code/build_reference_room.py').resolve()
    assert sha(helper) == '18ae517c85b1c3dffce41e269c83d9062142f2b56da0b96f005f0273f3278146'
    sys.path.insert(0, str(helper.parent))
    spec = importlib.util.spec_from_file_location('frozen_asset_helper', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            r = np.eye(3)[list(perm)]*np.array(signs)[:, None]
            if np.linalg.det(r) > .9:
                rotations.append(r)
    rows = []
    prior_path = a.root/'asset_recovery.json'
    prior = {r['asset_id']: r for r in json.loads(prior_path.read_text())['rows']} if prior_path.exists() else {}
    journal = dict(prior)
    def save(record):
        # Retain already recovered entries not revisited yet. A killed resume
        # must not lose their only transformation/check receipts.
        journal[record['asset_id']] = record
        ordered = [journal[e['binding']['asset_id']] for e in catalog['entries']
                   if e['binding']['asset_id'] in journal]
        dump(prior_path, dict(rows=ordered, wall_seconds=time.monotonic()-start,
            trimesh_version=trimesh.__version__, model_calls=0, training_updates=0))
    for entry in catalog['entries']:
        aid = entry['binding']['asset_id']; template = entry['template']
        out = room/'assets'/aid
        raw = a.raw_root/aid[0]/(aid+'.glb')
        if not raw.is_file():
            raw = a.scratch/'raw_assets'/(aid+'.glb')
        expected = manifest['files']['room_'+rid+'/assets/'+aid+'/canonical.glb']['sha256']
        if out.is_dir():
            old = prior.get(aid)
            assert old and old['exact_resource_recovery']
            assert all(sha(bundle/c['path']) == c['sha256'] for c in old['checks'])
            record = dict(old, reused_verified_asset=True)
            rows.append(record)
            save(record)
            continue
        record = dict(asset_id=aid, name=template['name'], raw_path=str(raw),
                      raw_available=raw.is_file(), attempts=[], exact_resource_recovery=False)
        if raw.is_file():
            record['raw_sha256'] = sha(raw)
            source = trimesh.load(raw, force='scene', process=False)
            recorded_rotation = template['metadata'].get('source_to_native_rotation')
            # Exact recorded fixture rotations include sin(pi) roundoff.
            # Never replace them with mathematically equal integer axes when
            # the goal is exact historical GLB/XML bytes.
            options = ([np.array(recorded_rotation)] if recorded_rotation is not None else [])+rotations
            for index, rotation in enumerate(options):
                check = source.copy(); t = np.eye(4); t[:3, :3] = rotation
                check.apply_transform(t)
                dims = check.extents
                target_dims = np.array(template['bbox_max'])-np.array(template['bbox_min'])
                if not np.allclose(dims, target_dims, rtol=0, atol=1e-6):
                    continue
                candidate = a.scratch/('asset_attempt_'+aid+'_'+str(index))
                if candidate.exists():
                    raise FileExistsError('Do not overwrite an asset recovery attempt')
                values = module.prepare_mesh(dict(source_path=str(raw), sha256=record['raw_sha256'],
                    asset_id=aid, source_to_native_rotation=rotation.tolist()), candidate)
                result = dict(rotation=rotation.tolist(), canonical_sha256=sha(values[0]),
                              canonical_matches=sha(values[0]) == expected)
                record['attempts'].append(result)
                if result['canonical_matches']:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(candidate), str(out))
                    checks = []
                    for p in sorted(out.iterdir()):
                        relative = p.relative_to(bundle).as_posix()
                        expected_row = manifest['files'].get(relative)
                        checks.append(dict(path=relative, sha256=sha(p),
                            exact=expected_row is not None and sha(p) == expected_row['sha256'] and p.stat().st_size == expected_row['bytes']))
                    record.update(checks=checks, exact_resource_recovery=all(x['exact'] for x in checks),
                                  source_to_native_rotation=rotation.tolist(), sdf_visual_rotation=values[3].tolist(),
                                  sdf_visual_shift=values[4].tolist())
                    break
                shutil.rmtree(candidate)
        rows.append(record); save(record)
        print(aid, template['name'], record['raw_available'], record['exact_resource_recovery'], flush=True)


if __name__ == '__main__':
    main()
