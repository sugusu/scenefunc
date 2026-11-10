"""Derive camera matrix candidates from existing native renderer receipts.

No Blender/model/rendering. Matrices are source reconstructions under recorded
camera defaults, not newly measured calibration. Cutaways and annotation
pixels remain a separate contract; photometric training is not admitted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[2]


def load(p):
    return json.loads(p.read_text())


def path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def unit(v):
    norm = math.sqrt(dot(v, v))
    if norm < 1e-10:
        raise ValueError('Degenerate camera frame')
    return [x/norm for x in v]


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists()
    material = REPO / 'runs/autodl/20261110_storage_observation_recovery/material_v4/observation_bindings.json'
    bindings = {r['unit_id']: r for r in load(material)}
    source = REPO / 'runs/autodl/20261110_native_observation_export/checkpoint_renderer'
    sources = [source / 'tools/render_cardinal_views.py', source / 'tools/render_five_views.py',
               source / 'src_critic/blender_render.py']
    hashes = {str(p): sha(p) for p in [material, a.audit / 'view_provenance.json', *sources]}
    rows = []; position_errors = []
    for view in load(a.audit / 'view_provenance.json'):
        binding = bindings.get(view['unit_id'])
        if not binding:
            continue
        ex_path = path(binding['export_receipt']); ex = load(ex_path)
        descriptor_path = Path(ex['blend_path']).with_suffix('.native_structure.json')
        descriptor = load(descriptor_path)
        assert descriptor['blend_sha256'] == ex['blend_sha256']
        if 'native_structure_sha256' in ex:
            expected_descriptor_hash = ex['native_structure_sha256']
            descriptor_proof = 'export_receipt'
        else:
            # The earliest saved export predates the explicit descriptor-hash
            # field. Require its already published Git blob, not an unchecked
            # current-file fallback, and retain this provenance difference.
            relative = descriptor_path.resolve().relative_to(REPO)
            published = subprocess.check_output(['git', 'show', 'HEAD:' + str(relative)], cwd=REPO)
            expected_descriptor_hash = hashlib.sha256(published).hexdigest()
            descriptor_proof = 'already_published_HEAD_blob_missing_old_export_field'
        assert sha(descriptor_path) == expected_descriptor_hash
        hashes[str(ex_path)] = sha(ex_path); hashes[str(descriptor_path)] = sha(descriptor_path)
        image = Path(view['path']); eye = None; defaults = []; expected_error = None
        if view['visibility_receipt']:
            vis_path = Path(view['visibility_receipt']); vis = load(vis_path)
            assert vis['blend_sha256'] == descriptor['blend_sha256']
            hashes[str(vis_path)] = sha(vis_path)
            eye = vis['camera_location']
            center = [(lo+hi)/2 for lo, hi in zip(descriptor['room_bounds']['min'], descriptor['room_bounds']['max'])]
            forward = unit([b-a for a, b in zip(eye, center)])
            right = unit(cross(forward, [0, 0, 1]))
            down = [-v for v in unit(cross(right, forward))]
            r = [right, down, forward]
            lens = 50.0; sensor_width = 36.0
            azimuth = {'east': 0, 'north': 90, 'west': 180, 'south': 270}[vis['view']]
            elevation = math.radians(30.0)
            max_dim = max(b-c for b, c in zip(descriptor['room_bounds']['max'], descriptor['room_bounds']['min']))
            distance = (max_dim/2)/(sensor_width/2/lens)/0.8
            direction = [math.cos(elevation)*math.cos(math.radians(azimuth)),
                         math.cos(elevation)*math.sin(math.radians(azimuth)), math.sin(elevation)]
            expected = [x+distance*y for x, y in zip(center, direction)]
            expected_error = math.sqrt(sum((x-y)**2 for x, y in zip(expected, eye)))
            assert expected_error < 1e-4, (view['unit_id'], vis['view'], expected_error)
            position_errors.append(expected_error)
            defaults.append('30degree elevation consistent with recorded eye and original default; not a new measured viewpoint')
        else:
            meta_path = Path(str(image) + '.meta.json'); meta = load(meta_path)
            hashes[str(meta_path)] = sha(meta_path)
            camera = meta['camera']; assert camera['mode'] == 'persp_top'
            eye = [*camera['center'], camera['distance']]
            lens, sensor_width = camera['lens'], camera['sensor_width']
            r = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            center = [*camera['center'], 0.0]
            defaults.append('zero camera Euler rotation from exact renderer source')
        width, height = view['width'], view['height']
        fx = width*lens/sensor_width; fy = fx
        k = [[fx, 0.0, width/2], [0.0, fy, height/2], [0.0, 0.0, 1.0]]
        translation = [-dot(row, eye) for row in r]
        rt = [row+[translation[i]] for i, row in enumerate(r)] + [[0.0, 0.0, 0.0, 1.0]]
        point = [dot(row, center)+t for row, t in zip(r, translation)]
        assert point[2] > 0
        center_px = [fx*point[0]/point[2]+width/2, fy*point[1]/point[2]+height/2]
        assert max(abs(center_px[0]-width/2), abs(center_px[1]-height/2)) < 1e-8
        defaults.extend(['square pixels, zero camera shifts, horizontal sensor fit at these resolutions; not explicitly serialized in image sidecars',
                         'pixel coordinate convention uses image edges [0,width] x [0,height]; no claim of pixel-center sampling qualification'])
        rows.append({'unit_id': view['unit_id'], 'image_sha256': view['sha256'],
                     'view_path': str(image), 'K_candidate': k, 'world_to_camera_candidate': rt,
                     'camera_convention': '+X right, +Y down, +Z forward; world +Z up',
                     'camera_location': eye, 'source_expected_location_error_m': expected_error,
                     'cutaway_object_ids': view['cutaway_object_ids'],
                     'descriptor_hash_proof': descriptor_proof,
                     'annotation_hash_changed': view['overlay_change_proved_by_hash'],
                     'calibration_status': 'source-derived candidate, not independently measured or Blender-projection validated',
                     'assumptions': defaults, 'photometric_training_admitted': False})
    assert len(rows) == 50
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    a.out.write_text(json.dumps({'status': 'source_camera_candidates_derived_on_cpu',
        'image_references': len(rows), 'distinct_image_hashes': len({r['image_sha256'] for r in rows}),
        'cardinal_source_location_checks': len(position_errors),
        'maximum_source_location_error_m': max(position_errors),
        'source_sha256': hashes, 'rows': rows,
        'new_renders': 0, 'new_model_calls': 0, 'training_updates': 0,
        'limits': 'Source-derived matrices remove a serialization gap; raw overlays/per-view visibility and independent projection/default validation remain. Original images never replaced.'},
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'source_camera_candidates_derived_on_cpu', 'image_references': len(rows),
                      'cardinal_source_location_checks': len(position_errors),
                      'maximum_source_location_error_m': max(position_errors), 'new_renders': 0}))


if __name__ == '__main__':
    main()
