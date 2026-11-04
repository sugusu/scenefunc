"""Ordinary finite geometric witness search on unchanged, real exported scenes.

Positive means the reported zero-thickness rectangles fit in measured planar
regions. It does not imply a person can use them. Missing witnesses are unknown.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import resource
import time

import numpy as np
import trimesh
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'paris_local_run_results_20261104'
TASKS = {'work': (0.8, 0.5), 'dining': (0.6, 0.4)}


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def section_region(mesh, z):
    if z < mesh.bounds[0, 2] or z > mesh.bounds[1, 2]:
        return Polygon(), False
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return Polygon(), True  # Within the bounds but no closed section evidence.
    region = Polygon()
    unknown = False
    for ring in section.discrete:
        if len(ring) < 4 or np.linalg.norm(ring[0] - ring[-1]) > 1e-6:
            unknown = True
            continue
        polygon = Polygon(ring[:, :2])
        if not polygon.is_valid:
            unknown = True
            continue
        # Conservatively fill every closed ring. This can fill a true hole, but
        # cannot cancel overlapping solid parts through an even-odd operation.
        # Consequently failure to find a footprint never proves infeasibility.
        region = region.union(polygon)
    return region, unknown


def top_surface(mesh):
    z = float(mesh.bounds[1, 2])
    tri = mesh.triangles
    mask = (mesh.face_normals[:, 2] >= 1 - 1e-7) & (np.max(np.abs(tri[:, :, 2] - z), axis=1) <= 1e-5)
    polygons = [Polygon(t[:, :2]) for t in tri[mask]]
    polygons = [p for p in polygons if p.is_valid and p.area > 0]
    return z, unary_union(polygons) if polygons else Polygon()


def candidates(poly, dimensions, surface_id, z):
    if poly.is_empty:
        return []
    xmin, ymin, xmax, ymax = poly.bounds
    safe = poly.buffer(-1e-6)
    prepared = prep(safe)
    found = []
    for rotation, (w, h) in enumerate([dimensions, dimensions[::-1]]):
        if xmax - xmin < w or ymax - ymin < h:
            continue
        xs = list(np.arange(xmin + w / 2, xmax - w / 2 + 1e-9, 0.01)) + [(xmin+xmax)/2]
        ys = list(np.arange(ymin + h / 2, ymax - h / 2 + 1e-9, 0.01)) + [(ymin+ymax)/2]
        for x, y in itertools.product(xs, ys):
            rect = box(x-w/2, y-h/2, x+w/2, y+h/2)
            if prepared.covers(rect):
                found.append({'surface_id': surface_id, 'z': z, 'center_xy': [float(x), float(y)],
                              'size_xy': [w, h], 'orientation_degrees': rotation*90, 'polygon': rect})
    return sorted(found, key=lambda w: (sum((w['center_xy'][i] - (poly.bounds[i]+poly.bounds[i+2])/2)**2 for i in (0,1)), w['orientation_degrees'], w['center_xy']))


def serial(witness):
    if witness is None:
        return None
    return {k: mapping(v) if k == 'polygon' else v for k, v in witness.items()}


def solve(pools, simultaneous, cap):
    if not all(pools.values()):
        return {'status': 'not_found', 'comparisons': 0, 'reason': 'one or more activity pools empty'}
    if not simultaneous:
        return {'status': 'witness', 'comparisons': 0,
                'witnesses': {task: serial(pool[0]) for task, pool in pools.items()}}
    count = 0
    for w in pools['work']:
        for d in pools['dining']:
            count += 1
            # Different heights would need an explicit non-overlap volume test;
            # require disjoint horizontal footprints as a sufficient restriction.
            if w['polygon'].intersection(d['polygon']).area <= 1e-12:
                return {'status': 'witness', 'comparisons': count,
                        'witnesses': {'work': serial(w), 'dining': serial(d)}}
            if count >= cap:
                return {'status': 'not_found', 'comparisons': count, 'reason': 'finite joint budget exhausted'}
    return {'status': 'not_found', 'comparisons': count, 'reason': 'finite candidate pool exhausted'}


def measure(checkpoint):
    start = time.perf_counter()
    state = json.loads(checkpoint.read_text())['payload']
    glb = checkpoint.parent / 'partial_scene.glb'
    scene = trimesh.load(glb, force='scene')
    meshes = {}
    for node in scene.graph.nodes_geometry:
        tf, gid = scene.graph[node]
        m = scene.geometry[gid].copy()
        m.apply_transform(tf)
        meshes[node] = m
    objects = {o['object_id']: o for o in state['placed_instances']}
    assert set(objects).issubset(meshes)
    surfaces = []
    pools = {mode: {task: [] for task in TASKS} for mode in ['asset_surface', 'scene_section_free']}
    for oid, obj in objects.items():
        if obj['name'] not in ['desk', 'office_desk', 'dining_table']:
            continue
        z, poly = top_surface(meshes[oid])
        blockers = []
        unknown = []
        blocked = Polygon()
        for other, mesh in meshes.items():
            if other == oid:
                continue
            # A disjoint XY AABB cannot obstruct this surface even if its section is open.
            if not box(*mesh.bounds[:, :2].reshape(-1)).intersects(poly):
                continue
            region, uncertain = section_region(mesh, z+1e-4)
            overlap = region.intersection(poly)
            if uncertain:
                unknown.append(other)
            if overlap.area > 1e-10:
                blockers.append({'object_id': other, 'overlap_m2': float(overlap.area)})
                blocked = blocked.union(overlap)
        free = poly.difference(blocked)
        row = {'object_id': oid, 'name': obj['name'], 'z_m': z, 'surface_m2': float(poly.area),
               'free_section_m2': float(free.area), 'surface': mapping(poly), 'free_section': mapping(free),
               'blockers': blockers, 'uncertain_section_objects': unknown, 'candidate_counts': {},
               'aabb_is_not_surface': meshes[oid].bounds.tolist()}
        for mode, region in [('asset_surface', poly), ('scene_section_free', free)]:
            row['candidate_counts'][mode] = {}
            for task, dim in TASKS.items():
                c = candidates(region, dim, oid, z)
                row['candidate_counts'][mode][task] = len(c)
                if mode == 'asset_surface' or not unknown:
                    pools[mode][task].extend(c)
        surfaces.append(row)
    output = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
              'scene': str(glb), 'scene_sha256': digest(glb), 'surfaces': surfaces,
              'scope': 'zero-thickness planar footprint; not entire activity or preference',
              'full_scene_success': None, 'states_modified': False, 'results': {}}
    cap = json.loads((ROOT/'PROTOCOL.json').read_text())['joint_search_cap']
    for mode, pool in pools.items():
        output['results'][mode] = {'pool_sizes': {k: len(v) for k,v in pool.items()},
                                  'separate_use': solve(pool, False, cap),
                                  'simultaneous_use': solve(pool, True, cap)}
    output['seconds'] = time.perf_counter() - start
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    t = time.perf_counter()
    rows = []
    for arm, stage in itertools.product(['A', 'B'], ['feedback_turn', 'mesh_feedback_sdpa_kv']):
        result = measure(OLD / f'{arm}_{stage}' / 'tools' / 'checkpoint.json')
        result.update({'arm':arm, 'stage':stage})
        rows.append(result)
        print(json.dumps({'arm':arm,'stage':stage,'surfaces':[{k:s[k] for k in ['object_id','surface_m2','free_section_m2','blockers','uncertain_section_objects','candidate_counts']} for s in result['surfaces']], 'results':result['results']},ensure_ascii=False),flush=True)
    args.out.write_text(json.dumps({'protocol_sha256':digest(ROOT/'PROTOCOL.json'), 'rows':rows,
        'cpu_wall_seconds': time.perf_counter()-t, 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'paid_api_calls':0,'model_inferences':0,'new_scenes':0},ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
