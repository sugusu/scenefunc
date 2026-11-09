"""Measure all frozen geometry-eligible previews from the actual Paris mesh.

No model or new action generator. The old discretization, scope and geometric
eligibility are preserved. Unary domains and joint compatibility are measured
for EVERY eligible preview; the latter is cross-checked with the existing
ordinary exact solver. This is a finite-pool geometry diagnostic, not human
preference, training, complete scene success, or a new optimization method.
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union


def read(p):
    return json.loads(p.read_text())


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def full_compatibility(left, right):
    """Independent vectorized Cartesian enumeration of rectangle interiors.

    Floating endpoint comparisons deliberately use the old solver's exact
    <=/>= semantics, not the earlier bounded search's area tolerance.
    Chunking bounds memory; all pairs are counted, even after a witness is found.
    """
    count = 0
    first = None
    for offset in range(0, len(left), 128):
        a = left[offset:offset + 128, None, :]
        b = right[None, :, :]
        ok = ((a[:, :, 2] <= b[:, :, 0]) | (b[:, :, 2] <= a[:, :, 0]) |
              (a[:, :, 3] <= b[:, :, 1]) | (b[:, :, 3] <= a[:, :, 1]))
        count += int(ok.sum())
        if first is None and ok.any():
            i, j = np.argwhere(ok)[0]
            first = [offset + int(i), int(j)]
    return {'pairs_checked': len(left) * len(right), 'compatible_pairs': count,
            'first_witness_indices': first}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    w, out = args.work_base.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    old = w / 'joint_use_feedback_20261104/ordinary_control/B'
    legacy = w / 'joint_use_witness_20261104'
    sys.path.insert(0, str(legacy))
    from measure_joint_use import candidates, section_region, top_surface, TASKS
    from strong_pair_control import strong_solve

    result = read(old / 'result.json')
    checkpoint = Path(result['original_checkpoint'])
    glb = checkpoint.parent / 'partial_scene.glb'
    cp = read(checkpoint)
    payload = cp['payload']
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != cp['payload_sha256']:
        raise ValueError('Checkpoint payload is inconsistent')
    if digest(checkpoint) != result['original_checkpoint_sha256']:
        raise ValueError('Original checkpoint changed')
    measured = next(r for r in read(legacy / 'measurements.json')['rows']
                    if r['arm'] == 'B' and r['stage'] == 'mesh_feedback_sdpa_kv')
    if digest(glb) != measured['scene_sha256']:
        raise ValueError('Original actual mesh changed')
    sources = [checkpoint, glb, old / 'candidates.json', old / 'result.json',
               legacy / 'measurements.json', legacy / 'measure_joint_use.py',
               legacy / 'strong_pair_control.py']
    before = {str(p): digest(p) for p in sources}
    raw = read(old / 'candidates.json')
    queries = [{'id': 'keep', 'object_id': None, 'offset_xy_m': [0., 0.],
                'expected_joint': result['before_joint']['status'] == 'witness'}]
    for i, r in enumerate(raw):
        if 'joint' in r:
            queries.append({'id': f'preview_{i:02d}', 'object_id': r['object_id'],
                            'offset_xy_m': r['offset_xy_m'],
                            'expected_joint': r['joint']['status'] == 'witness'})
    dump(out / 'frozen_queries.json', queries)
    scene = trimesh.load(glb, force='scene')
    meshes = {}
    for node in scene.graph.nodes_geometry:
        transform, gid = scene.graph[node]
        m = scene.geometry[gid].copy()
        m.apply_transform(transform)
        meshes[node] = m
    furniture = {o['object_id'] for o in payload['placed_instances']}
    if not furniture <= meshes.keys():
        raise ValueError('Missing furniture mesh')
    # Verify surfaces against the actual meshes and cache the same plane cuts.
    sections = {}
    for surface in measured['surfaces']:
        sid = surface['object_id']
        z, actual = top_surface(meshes[sid])
        archived = shape(surface['surface'])
        if abs(z - surface['z_m']) > 1e-10 or actual.symmetric_difference(archived).area > 1e-10:
            raise ValueError('Surface/mesh mismatch')
        sections[sid] = {oid: section_region(mesh, z + 1e-4)
                         for oid, mesh in meshes.items()}
    setup_seconds = time.perf_counter() - start
    rows = []
    for query in queries:
        t0 = time.perf_counter()
        oid = query['object_id']
        dx, dy = query['offset_xy_m']
        pools = {task: [] for task in TASKS}
        surfaces = []
        for surface in measured['surfaces']:
            sid = surface['object_id']
            poly = shape(surface['surface'])
            if sid == oid:
                poly = translate(poly, dx, dy)
            blocked, uncertain = [], []
            for other, mesh in meshes.items():
                if other == sid:
                    continue
                bounds = mesh.bounds.copy()
                if other == oid:
                    bounds[:, :2] += [dx, dy]
                if not box(*bounds[:, :2].reshape(-1)).intersects(poly):
                    continue
                region, unknown = sections[sid][other]
                if other == oid:
                    region = translate(region, dx, dy)
                if unknown:
                    uncertain.append(other)
                blocked.append(region.intersection(poly))
            free = poly.difference(unary_union(blocked) if blocked else Polygon())
            domains = {task: [] if uncertain else candidates(free, dims, sid, surface['z_m'])
                       for task, dims in TASKS.items()}
            for task, domain in domains.items():
                pools[task].extend(domain)
            surfaces.append({'id': sid, 'free_section': mapping(free),
                             'uncertain_section_objects': uncertain,
                             'domain_sizes': {task: len(d) for task, d in domains.items()}})
        domain_seconds = time.perf_counter() - t0
        arrays = {task: np.asarray([r['polygon'].bounds for r in domain], dtype=np.float64).reshape(-1, 4)
                  for task, domain in pools.items()}
        t1 = time.perf_counter()
        exact = full_compatibility(arrays['work'], arrays['dining'])
        exhaustive_seconds = time.perf_counter() - t1
        t2 = time.perf_counter()
        strong = strong_solve(pools['work'], pools['dining'])
        strong_seconds = time.perf_counter() - t2
        joint = exact['compatible_pairs'] > 0
        if joint != (strong['status'] == 'witness') or joint != query['expected_joint']:
            raise ValueError('Archived, Cartesian and extreme-solvers disagree at ' + query['id'])
        unary = all(len(v) > 0 for v in pools.values())
        if joint and not unary:
            raise ValueError('Joint/unary implication violated')
        artifact = out / (query['id'] + '_domains.npz')
        np.savez_compressed(artifact, **arrays)
        row = {**query, 'finite_separate_witness': unary, 'finite_joint_witness': joint,
               'domain_sizes': {task: len(v) for task, v in pools.items()},
               'exhaustive': exact, 'ordinary_extreme_solver': strong,
               'surface_evidence': surfaces,
               'domain_file': artifact.name, 'domain_file_sha256': digest(artifact),
               'timings_seconds': {'domain_construction': domain_seconds,
                                   'cartesian_count': exhaustive_seconds,
                                   'ordinary_extreme_solver': strong_seconds},
               'whole_scene_success': None}
        dump(out / (query['id'] + '.json'), row)
        rows.append(row)
        print(json.dumps({'candidate': query['id'], 'separate': unary, 'joint': joint,
                          'pool_sizes': row['domain_sizes'],
                          'compatible_pairs': exact['compatible_pairs']}), flush=True)
    after = {str(p): digest(p) for p in sources}
    if before != after:
        raise ValueError('An old source changed')
    conflicts = sum(r['finite_separate_witness'] != r['finite_joint_witness'] for r in rows)
    summary = {'states': len(rows), 'old_parent_scenes': 1, 'new_generated_scenes': 0,
               'model_calls': 0, 'training_updates': 0, 'states_modified': False,
               'furniture_meshes_loaded': len(furniture), 'all_mesh_nodes_loaded': len(meshes),
               'new_independent_room_families': 0, 'measured_scoped_labels': 2 * len(rows),
               'separate_pass': sum(r['finite_separate_witness'] for r in rows),
               'joint_pass': sum(r['finite_joint_witness'] for r in rows),
               'same_candidate_condition_conflicts': conflicts,
               'all_archived_joint_results_match': True,
               'all_cartesian_and_ordinary_results_match': True,
               'geometry_checks_reused_from_archive': True,
               'sources_unchanged': True, 'source_hashes': after,
               'setup_seconds': setup_seconds, 'total_seconds': time.perf_counter() - start,
               'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'scope': 'Fixed 1cm/0-or-90 degree zero-thickness witness domains on mesh top surfaces; not human use, continuous impossibility, activity transitions or preferences.'}
    dump(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
