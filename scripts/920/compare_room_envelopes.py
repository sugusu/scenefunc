"""Attribute envelope warnings to a generated input or its asset reconstruction.

This reads two frozen states, never edits either, and does not score usability.
Furniture AABBs exclude thin coverings; walls and opening sweeps are not tested.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(bounds, floor_bounds, excluded):
    boundary, overlaps = {}, {}
    for name, box in sorted(bounds.items()):
        excursion = np.maximum(0, np.maximum(floor_bounds[0]-box[0], box[1]-floor_bounds[1]))
        if excursion.max() > 1e-6:
            boundary[name] = excursion.tolist()
    for left, right in itertools.combinations(sorted(bounds), 2):
        if left in excluded or right in excluded:
            continue
        size = np.minimum(bounds[left][1], bounds[right][1])-np.maximum(bounds[left][0], bounds[right][0])
        if np.all(size > 1e-4):
            overlaps[left+' | '+right] = {'extents_m': size.tolist(), 'volume_m3': float(np.prod(size))}
    return {'boundary': boundary, 'overlaps': overlaps}


def comparison(before, after):
    return {'present_in_both': sorted(before.keys() & after.keys()),
            'introduced_by_reconstruction': sorted(after.keys()-before.keys()),
            'absent_after_reconstruction': sorted(before.keys()-after.keys())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'begin source-to-native envelope attribution', flush=True)
    plan = json.loads(args.plan.read_text())
    inputs = {}
    for key, entry in plan['inputs'].items():
        path = Path(entry['path'])
        if sha(path) != entry['sha256']:
            raise ValueError('Frozen input changed: '+key)
        inputs[key] = json.loads(path.read_text())
    source, native, measured = (inputs[k] for k in ('source', 'native', 'measured'))
    source_bounds, native_bounds, differences = {}, {}, []
    records = source['furniture']
    names = {r['name'] for r in records}
    if len(records) != 21 or len(names) != 21:
        raise ValueError('Expected the complete 21-object initial reference')
    for record in records:
        name = record['name']
        width, height, depth = record['dimensions']
        x, up, south = record['location']
        yaw = math.radians(-record.get('facing', 0))
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        center = np.array([x, -south, up])
        extent = np.abs(rot) @ (np.array([width, depth, height])/2)
        source_bounds[name] = np.array([center-extent, center+extent])
        obj = native['objects']['recon_'+name+'_0']
        q = np.asarray(obj['transform']['rotation_wxyz'])
        if not np.isclose(np.linalg.norm(q), 1, atol=1e-12, rtol=0):
            raise ValueError('Nonunit native quaternion')
        wq, xq, yq, zq = q
        rotation = np.array([
            [1-2*(yq*yq+zq*zq), 2*(xq*yq-zq*wq), 2*(xq*zq+yq*wq)],
            [2*(xq*yq+zq*wq), 1-2*(xq*xq+zq*zq), 2*(yq*zq-xq*wq)],
            [2*(xq*zq-yq*wq), 2*(yq*zq+xq*wq), 1-2*(xq*xq+yq*yq)]])
        lo, hi = np.array(obj['bbox_min']), np.array(obj['bbox_max'])
        pos = np.array(obj['transform']['translation'])
        native_center = pos + rotation @ ((lo+hi)/2)
        native_extent = np.abs(rotation) @ ((hi-lo)/2)
        native_bounds[name] = np.array([native_center-native_extent, native_center+native_extent])
        differences.append({'name': name, 'source_local_width_depth_height_m': [width, depth, height],
            'native_local_width_depth_height_m': (hi-lo).tolist(),
            'center_change_native_xyz_m': (native_center-center).tolist(),
            'source_world_aabb_m': source_bounds[name].tolist(),
            'native_world_aabb_m': native_bounds[name].tolist()})
    geometry = native['room_geometry']
    floor = next(r for r in source['room_structure'] if r['name']=='floor')
    ceiling = next(r for r in source['room_structure'] if r['name']=='ceiling')
    if [geometry['length'], geometry['width'], geometry['wall_height']] != [floor['dimensions'][0], floor['dimensions'][2], ceiling['location'][1]]:
        raise ValueError('Room envelopes differ')
    floor_bounds = np.array([[-geometry['length']/2, -geometry['width']/2, 0],
                            [geometry['length']/2, geometry['width']/2, geometry['wall_height']]])
    excluded = {name for name in names if native['objects']['recon_'+name+'_0']['object_type']=='thin_covering'}
    before = inspect(source_bounds, floor_bounds, excluded)
    after = inspect(native_bounds, floor_bounds, excluded)
    def name_for(oid):
        return native['objects'][oid]['name']
    existing_boundary = {name_for(r['object_id']): r['outside_room_envelope_m'] for r in measured['boundary_excursions']}
    existing_overlap = {' | '.join(sorted(name_for(oid) for oid in r['object_ids'])): r for r in measured['envelope_overlaps']}
    if after['boundary'].keys() != existing_boundary.keys() or after['overlaps'].keys() != existing_overlap.keys():
        raise ValueError('Recomputed native warning sets differ from actual engine observation')
    for name, extent in after['boundary'].items():
        if not np.allclose(extent, existing_boundary[name], atol=1e-12, rtol=0):
            raise ValueError('Native boundary measurement mismatch')
    for key, value in after['overlaps'].items():
        if not np.allclose(value['extents_m'], existing_overlap[key]['aabb_overlap_extents_m'], atol=1e-12, rtol=0):
            raise ValueError('Native pair measurement mismatch')
    result = {'status': 'source_and_reconstruction_compared', 'plan_sha256': sha(args.plan),
        'input_hashes': {k: v['sha256'] for k, v in plan['inputs'].items()},
        'furniture_count': len(names), 'thin_coverings_excluded': sorted(excluded),
        'source_generated_reference': before, 'native_reconstruction': after,
        'boundary_attribution': comparison(before['boundary'], after['boundary']),
        'pair_attribution': comparison(before['overlaps'], after['overlaps']),
        'per_object_changes': differences, 'native_metrics_match_engine_observation': True,
        'model_calls': 0, 'scene_edits': 0, 'training_updates': 0,
        'limits': ['AABB overlap is a warning, not mesh collision or functional failure.',
                  'Input is generated reference geometry, not a surveyed historical room.',
                  'No walls, opening sweeps, reach, seating or activity clearances tested.',
                  'Absent warnings after substitution do not establish a design improvement.',
                  'Do not attribute reconstruction conflicts to model design reasoning.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'boundary_attribution', 'pair_attribution')}), flush=True)


if __name__ == '__main__':
    main()
