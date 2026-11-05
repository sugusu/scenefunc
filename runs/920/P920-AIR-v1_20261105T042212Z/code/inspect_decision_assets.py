"""Measure pinned candidate meshes in their recorded coordinate frames."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--download', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'begin candidate mesh inspection', flush=True)
    download = json.loads(a.download.read_text())
    if download['status'] != 'verified':
        raise ValueError('Unverified download')
    groups = {}
    for item in download['files']:
        raw = Path(item['local_path']).read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Asset changed')
        groups.setdefault(item['asset_id'], {})[item['role']] = item
    rows = []
    for asset_id, items in groups.items():
        frame = json.loads(Path(items['coordinate_frame']['local_path']).read_text())
        mesh = trimesh.load(items['mesh']['local_path'], force='mesh', process=False)
        up = np.asarray(frame['up'], dtype=float)
        up /= np.linalg.norm(up)
        front = np.asarray(frame['front'], dtype=float)
        front -= up * np.dot(up, front)
        front /= np.linalg.norm(front)
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack([np.cross(front, up), front, up]).T
        if not np.isfinite(transform).all():
            raise ValueError('Invalid recorded frame')
        mesh.apply_transform(transform)
        bounds = mesh.bounds
        if not np.isfinite(bounds).all() or np.any(mesh.extents <= 0):
            raise ValueError('Invalid geometry bounds')
        rows.append({'asset_id': asset_id, 'name': items['mesh']['asset_name'],
                     'source_mesh_sha256': items['mesh']['sha256'],
                     'frame_sha256': items['coordinate_frame']['sha256'],
                     'frame_convention': '+X right, +Y front, +Z up; metadata alignment only',
                     'source_frame': {'up': frame['up'], 'front': frame['front']},
                     'width_depth_height_m': mesh.extents.tolist(),
                     'bounds_m': bounds.tolist(), 'vertex_count': len(mesh.vertices),
                     'face_count': len(mesh.faces), 'rescaled': False,
                     'collision_asset_downloaded': False, 'support_surface_downloaded': False,
                     'semantic_front_reviewed': False, 'mass_or_movability_known': False,
                     'accepted_for_project': False})
    result = {'status': 'measured_not_semantically_accepted', 'assets': rows,
              'download_sha256': hashlib.sha256(a.download.read_bytes()).hexdigest(),
              'trimesh_version': trimesh.__version__, 'model_calls': 0, 'engine_edits': 0,
              'limits': ['No collision, support, texture or functional-suitability acceptance.',
                         'These are candidate substitutes, not historical project objects.']}
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'measured_assets': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
