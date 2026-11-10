"""Reconstruct one frozen native input bundle and require original file hashes.

This recovers a researcher-authored development room, not a new generated room
or a historical apartment. Pure XML helpers are executed from pinned source;
the enclosing resource construction is transcribed without SceneSmith/Drake
objects. Every resulting file must match the OLD manifest before acceptance.
No manifest rewriting, pose changes, renderer, model or training is allowed.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from trimesh.transformations import euler_matrix


PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
RUN = 'runs/920/P920-PAPER-v2_20261105T182424Z/'
ASSET_RUN = Path('runs/autodl/20261110_corpus_state_recovery/asset_recovery_v4')
BUILD_HASH = '18ae517c85b1c3dffce41e269c83d9062142f2b56da0b96f005f0273f3278146'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def rpy_for_resource(R):
    """Source formula, Drake v1.49.0 roll_pitch_yaw.cc lines 118-145.

    SciPy supplies the quaternion; this is NOT a substitute Drake runtime.
    Resulting XML bytes are accepted only by the frozen manifest hash.
    https://github.com/RobotLocomotion/drake/blob/v1.49.0/math/roll_pitch_yaw.cc
    """
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    pitch = math.atan2(-R[2, 0], math.sqrt((R[2, 2]**2+R[2, 1]**2+R[1, 0]**2+R[0, 0]**2)/2))
    eps = np.finfo(float).eps
    a = 0.0 if abs(x+z) <= eps and abs(w-y) <= eps else math.atan2(x+z, w-y)
    b = 0.0 if abs(z-x) <= eps and abs(w+y) <= eps else math.atan2(z-x, w+y)
    roll, yaw = a-b, a+b
    def wrap(v):
        return v-2*math.pi if v > math.pi else v+2*math.pi if v < -math.pi else v
    return [wrap(roll), pitch, wrap(yaw)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    if trimesh.__version__ != '4.5.3':
        raise ValueError('Proxy GLBs were originally exported by the geometry runtime 4.5.3')
    if shutil.disk_usage(a.out).free < 5*1024**3+12*1024**2:
        raise RuntimeError('Need 12 MiB above the 5 GiB reserve')
    start = time.monotonic(); sources = {}; result = {'status': 'started'}
    def get(path):
        raw = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show', PIN+':'+path])
        sources[path] = hashlib.sha256(raw).hexdigest()
        return raw
    try:
        source = get('scripts/920/build_reference_room.py')
        assert hashlib.sha256(source).hexdigest() == BUILD_HASH
        ns = {'ET': ET, 'np': np}
        names = {'add', 'numbers', 'box', 'rgba', 'sdf_root', 'save_sdf', 'source_position'}
        funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(funcs) == len(names)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), 'pinned_XML_helpers.py', 'exec'), ns)
        add, numbers, box, rgba, sdf_root, save_sdf, source_position = [ns[n] for n in
            ['add', 'numbers', 'box', 'rgba', 'sdf_root', 'save_sdf', 'source_position']]
        plan_raw = get(RUN+'REFERENCE_BUILD_PLAN.json'); plan = json.loads(plan_raw)
        original_plan = get('runs/920/P920-PAPER-v1_20261105T180747Z/REFERENCE_BUILD_PLAN.json')
        assert original_plan == plan_raw
        initial_raw = get(RUN+'INITIAL_REFERENCE.json'); initial = json.loads(initial_raw)
        assert hashlib.sha256(initial_raw).hexdigest() == plan['initial_reference']['sha256']
        manifest_raw = get(RUN+'input_metadata/manifest.json'); manifest = json.loads(manifest_raw)
        bundle = a.out/'bundle'; bundle.mkdir()
        (bundle/'manifest.json').write_bytes(manifest_raw)
        for name in ['catalog.json', 'construction_provenance.json', 'house_state.json', 'runtime.json', 'task.json']:
            (bundle/name).write_bytes(get(RUN+'input_metadata/'+name))
        rid = manifest['selected_room_id']; directory = bundle/('room_'+rid); directory.mkdir()
        assert rid == plan['room_id'] == 'paper_shared_living_reconstruction'
        asset_rows = {r['asset_id']: r for r in json.loads((ASSET_RUN/'asset_checks.json').read_text())}
        for aid, row in asset_rows.items():
            dest = directory/'assets'/aid; dest.mkdir(parents=True)
            for check in row['checks']:
                src = ASSET_RUN/'canonical'/aid/Path(check['relative']).name
                assert sha(src) == check['expected_sha256']
                shutil.copyfile(src, dest/src.name)

        # Original structural XML computation. Native SceneObject construction
        # is unnecessary for this resource step: frozen state JSON is copied.
        structure = {r['name']: r for r in initial['room_structure']}
        length, _, width = structure['floor']['dimensions']
        height = structure['ceiling']['location'][1]
        shell_xml, shell_link = sdf_root('reconstructed_shell', 'room_geometry_body_link')
        box(shell_link, 'floor', [0, 0, -0.02], [length, width, 0.04], rgba(structure['floor']['color']))
        box(shell_link, 'ceiling', [0, 0, height+0.02], [length, width, 0.04], [1, 1, 1, 1], visual=False)
        openings = [r for r in structure.values() if 'door' in r['name'] or 'window' in r['name']]
        walls = [r for r in structure.values() if 'wall' in r['name']]
        for record in walls:
            center = source_position(record)
            wall_width, wall_height, thickness = record['dimensions']
            along = 0 if record['facing'] in (0, 180) else 1; normal = 1-along
            cuts = []
            for item in openings:
                pos = source_position(item); w, h, _ = item['dimensions']
                if abs(pos[normal]-center[normal]) <= 0.04 and abs(pos[along]-center[along])+w/2 <= wall_width/2+1e-6:
                    cuts.append((pos[along]-w/2, pos[along]+w/2, pos[2]-h/2, pos[2]+h/2))
            low, high = center[along]-wall_width/2, center[along]+wall_width/2
            xcuts = sorted({low, high, *(v for c in cuts for v in c[:2])})
            zcuts = sorted({0.0, wall_height, *(v for c in cuts for v in c[2:])})
            index = 0
            for left, right in zip(xcuts, xcuts[1:]):
                for bottom, top in zip(zcuts, zcuts[1:]):
                    mid_u, mid_z = (left+right)/2, (bottom+top)/2
                    if any(c[0] < mid_u < c[1] and c[2] < mid_z < c[3] for c in cuts):
                        continue
                    xyz = center.copy(); xyz[along] = mid_u; xyz[2] = mid_z
                    size = [thickness, thickness, top-bottom]; size[along] = right-left
                    box(shell_link, record['name']+'_'+str(index), xyz, size, rgba(record['color']))
                    index += 1
        for record in openings:
            position = source_position(record); w, h, depth = record['dimensions']
            size = np.array([w, depth, h] if record['facing'] in (0, 180) else [depth, w, h])
            color = rgba(record['color'])
            if 'window' in record['name']: color[3] = 0.22
            box(shell_link, record['name'], position, size, color, collision='door' in record['name'])
        save_sdf(directory/'room.sdf', shell_xml)

        provenance = []; built = {}; rpy_records = []
        records = sorted(initial['furniture'], key=lambda r: bool(plan['objects'][r['name']].get('parent')))
        for record in records:
            name = record['name']; spec = plan['objects'][name]
            root, link = sdf_root(name); object_dir = directory/'objects'/name; object_dir.mkdir(parents=True)
            if spec['kind'] == 'mesh':
                aid = spec['asset_id']; row = asset_rows[aid]
                bounds = np.array(row['native_bounds']); shift = np.array(row['shift'])
                # Original builder performs this addition even for zero base
                # height. Omitting it preserves -0 in z and changes XML bytes.
                base_height = float(spec.get('reconstructed_base_height', 0))
                if base_height != 0:
                    raise ValueError('This fixed recovery does not add reconstructed bases')
                shift[2] += base_height
                rotation = np.asarray(plan['assets'][aid]['source_to_native_rotation'], dtype=float)
                yup_to_zup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                remaining = rotation @ yup_to_zup.T
                rpy = rpy_for_resource(remaining)
                rpy_records.append({'asset_id': aid, 'rotation': remaining.tolist(), 'rpy': rpy})
                node = add(link, 'visual', name='source_asset'); add(node, 'pose', numbers([*shift, *rpy]))
                add(add(add(node, 'geometry'), 'mesh'), 'uri', os.path.relpath(directory/'assets'/aid/'model.gltf', object_dir))
                box(link, 'collision_envelope', bounds.mean(axis=0), bounds[1]-bounds[0], [1, 1, 1, 1], visual=False)
            elif spec['kind'] == 'procedural_proxy':
                w, h, d = record['dimensions']; bounds = np.array([[-w/2, -d/2, 0], [w/2, d/2, h]])
                box(link, 'declared_proxy', [0, 0, h/2], [w, d, h], rgba(record['color']),
                    collision=spec['object_type'] != 'thin_covering')
                aid = 'procedural_reference:'+name
            else:
                raise ValueError('Unexpected resource kind; do not silently approximate '+spec['kind'])
            save_sdf(object_dir/'object.sdf', root)
            center = source_position(record); position = np.array([center[0], center[1], float(spec.get('base_z', 0))])
            if spec.get('mount'):
                raise ValueError('This fixed recovery has no mounted objects')
            if spec.get('parent'):
                parent = built[spec['parent']]; position[2] = parent['position'][2]+parent['bounds'][1, 2]
            built[name] = {'position': position, 'bounds': bounds}
            provenance.append({'object_id': 'recon_'+name+'_0', 'reference_name': name, 'asset_id': aid, 'kind': spec['kind'],
                'source_dimensions_width_height_depth_m': record['dimensions'], 'actual_bounds_min_max_m': bounds.tolist(),
                'actual_position_m': position.tolist(), 'source_position_x_up_south_m': record['location'],
                'rescaled': False, 'support_parent': spec.get('parent') or spec.get('mount') or 'floor',
                'semantic_front_status': spec.get('front_status', 'not_applicable_or_unknown'), 'reconstruction_note': spec['note']})
        write(bundle/'provenance.json', {'plan_sha256': hashlib.sha256(plan_raw).hexdigest(),
            'reference_sha256': hashlib.sha256(initial_raw).hexdigest(), 'objects': provenance,
            'ceiling': 'collision-only structural ceiling; roof cutaway for views', 'opening_hinges_and_operability': 'unknown; proxy leaves retained as supplied',
            'training_eligible': False})

        # Same pure SDF-to-mesh helper and geometry-runtime version used by the
        # original construction preparation; proxies remain labelled proxies.
        contact_source = get('scripts/920/inspect_native_mesh_contacts.py')
        fn = [n for n in ast.parse(contact_source).body if isinstance(n, ast.FunctionDef) and n.name == 'sdf_visuals']
        assert len(fn) == 1
        mesh_ns = {'ET': ET, 'np': np, 'trimesh': trimesh, 'euler_matrix': euler_matrix}
        exec(compile(ast.Module(body=fn, type_ignores=[]), 'pinned_sdf_visuals.py', 'exec'), mesh_ns)
        for name in ['low_fridge', 'pantry']:
            meshes = [m for _, m in mesh_ns['sdf_visuals'](directory/'objects'/name/'object.sdf', bundle, {})]
            dest = directory/'assets'/('catalog_construct_'+name)/'proxy_geometry.glb'; dest.parent.mkdir(parents=True)
            trimesh.util.concatenate(meshes).export(dest)
        package = get('runs/920/P920-IMPORT-v1_20261104T134317Z/audit/source/package.xml')
        (bundle/'package.xml').write_bytes(package)

        checks = []
        for relative, expected in manifest['files'].items():
            p = bundle/relative
            checks.append({'path': relative, 'expected': expected, 'actual_bytes': p.stat().st_size if p.exists() else None,
                           'actual_sha256': sha(p) if p.exists() else None,
                           'exact': p.exists() and sha(p) == expected['sha256'] and p.stat().st_size == expected['bytes']})
        write(a.out/'file_checks.json', checks); write(a.out/'resource_rotations.json', rpy_records)
        exact = all(r['exact'] for r in checks)
        result.update(status='all_files_exact_pending_native_validation' if exact else 'resource_mismatch',
                      expected_files=len(checks), exact_files=sum(r['exact'] for r in checks),
                      input_manifest_sha256=sha(bundle/'manifest.json'), manifest_rewritten=False)
        if exact:
            module_path = Path('runs/autodl/20261110_corpus_state_recovery/source/native_project.py')
            assert sha(module_path) == '1c418410b1d5db8606305f200154418dd59844ce1c22fee3afb903286417ad19'
            loader = importlib.util.spec_from_file_location('frozen_native_project', module_path)
            native = importlib.util.module_from_spec(loader); sys.modules[loader.name] = native; loader.loader.exec_module(native)
            native.validate_bundle(bundle)
            result.update(status='exact_bundle_and_original_resource_validator_passed', original_validator_passed=True)
        else:
            result['original_validator_passed'] = False
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc)); raise
    finally:
        result.update(pin=PIN, sources=sources, wall_seconds=time.monotonic()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      trimesh_version=trimesh.__version__, numpy_version=np.__version__,
                      native_runtime_instantiated=False, renderer_run=False, new_model_calls=0, training_updates=0,
                      new_scene_generations=0, restored_input_is_development_reconstruction=True,
                      whole_scene_quality_verified=False)
        write(a.out/'summary.json', result); print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
