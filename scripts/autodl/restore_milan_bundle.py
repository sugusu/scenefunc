"""Reconstruct remaining frozen Milan XML resources; reject byte mismatches.

Source is build_milan_input.py at the recorded Git pin. Metadata are copied
unchanged. Exact GLB/glTF asset recovery must precede this stage. It is a cold
development-input restore, not newly generated architecture or a policy trial.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--native-source', type=Path, required=True)
    a = ap.parse_args(); start = time.monotonic()
    source = a.root/'source/runs/920/P920-MILAN-TASK-v1_20261105T224015Z'
    bundle = a.scratch/'bundle'; manifest = json.loads((bundle/'manifest.json').read_text())
    plan = json.loads((source/'BUILD_PLAN.json').read_text())
    state = json.loads((bundle/'house_state.json').read_text()); rid = manifest['selected_room_id']
    task = json.loads((bundle/'task.json').read_text()); geom = task['geometry_assumptions']
    catalog = json.loads((bundle/'catalog.json').read_text())['entries']
    recovery = json.loads((a.root/'asset_recovery.json').read_text())
    by_asset = {r['asset_id']: r for r in recovery['rows']}
    assert len(by_asset) == 16 and all(r['exact_resource_recovery'] for r in by_asset.values())
    helpers = Path('runs/autodl/20261111_waterloo_task_search/source/code').resolve()
    sys.path.insert(0, str(helpers)); sys.path.insert(0, str(a.native_source.resolve()))
    from build_reference_room import box, save_sdf, sdf_root, add, numbers
    from native_project import import_project, validate_bundle
    from recover_paper_native_resources import rpy_for_resource
    roomdir = bundle/('room_'+rid)
    # The following shell construction directly transcribes the pinned source
    # before native object instantiation; hashes, not a new tolerance, admit it.
    length, width = geom['interior_rectangle'][2:]; height = geom['height']
    root, link = sdf_root('declared_transfer_shell', 'room_geometry_body_link')
    box(link, 'floor', [0, 0, -.025], [length, width, .05], [.72, .64, .54, 1])
    box(link, 'ceiling', [0, 0, height+.025], [length, width, .05], [1, 1, 1, 1], visual=False)
    for name, axis, fixed, low, high, h, normal, cuts in plan['wall_specs']:
        uc = sorted({low, high, *(v for c in cuts for v in c[2:4])})
        zc = sorted({0., h, *(v for c in cuts for v in c[4:6])})
        for i, (left, right) in enumerate(zip(uc, uc[1:])):
            for j, (bottom, top) in enumerate(zip(zc, zc[1:])):
                u, z = (left+right)/2, (bottom+top)/2
                if any(c[2] < u < c[3] and c[4] < z < c[5] for c in cuts):
                    continue
                center = [0., 0., z]; center[axis] = u; center[1-axis] = fixed
                size = [.05, .05, top-bottom]; size[axis] = right-left
                box(link, f'{name}_{i}_{j}', center, size, [.85, .85, .85, 1])
        for label, kind, left, right, bottom, top in cuts:
            if kind == 'window':
                center = [0., 0., (bottom+top)/2]; center[axis] = (left+right)/2; center[1-axis] = fixed
                size = np.array([.015, .015, top-bottom]); size[axis] = right-left
                box(link, label, center, size, [.55, .75, .9, .22], collision=False)
    save_sdf(roomdir/'room.sdf', root)
    for e in catalog:
        template = e['template']; aid = e['binding']['asset_id']; record = by_asset[aid]
        sdf = roomdir/template['sdf_path']; sdf.parent.mkdir(parents=True, exist_ok=True)
        model_name = 'catalog_'+aid if 'front_correction_degrees' in template['metadata'] else template['name']
        root, body = sdf_root(model_name)
        visual = add(body, 'visual', name='source_asset')
        rpy = rpy_for_resource(np.array(record['sdf_visual_rotation']))
        add(visual, 'pose', numbers([*record['sdf_visual_shift'], *rpy]))
        add(add(add(visual, 'geometry'), 'mesh'), 'uri', '../../assets/'+aid+'/model.gltf')
        bounds = np.array([template['bbox_min'], template['bbox_max']])
        box(body, 'collision_envelope', bounds.mean(0), bounds[1]-bounds[0], [1, 1, 1, 1], visual=False)
        save_sdf(sdf, root)
    checks = []
    for name, item in manifest['files'].items():
        p = bundle/name
        actual = sha(p) if p.is_file() else None
        checks.append(dict(path=name, expected=item['sha256'], actual=actual,
                           bytes=p.stat().st_size if p.is_file() else None,
                           exact=actual == item['sha256'] and p.stat().st_size == item['bytes']))
    result = dict(expected_files=len(checks), exact_files=sum(x['exact'] for x in checks), checks=checks,
        manifest_rewritten=False, native_restored=False, original_scene_generated=False,
        model_calls=0, training_updates=0)
    (a.root/'exact_resource_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    assert all(x['exact'] for x in checks), 'Exact resource recovery failed; no substitute manifest'
    validate_bundle(bundle)
    project = import_project(bundle, a.scratch/'native_import')
    from load_paper_native_state import difference
    delta = difference(state['rooms'][rid], project.room.to_state_dict())
    result.update(native_restored=True, raw_state_exact=not delta, state_differences=delta,
                  objects=len(project.room.objects), wall_seconds=time.monotonic()-start)
    (a.root/'exact_resource_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    assert not delta
    print(json.dumps({k: v for k, v in result.items() if k not in ['checks', 'state_differences']}), flush=True)


if __name__ == '__main__':
    main()
