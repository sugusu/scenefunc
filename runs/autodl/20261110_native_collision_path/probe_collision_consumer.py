"""Consume all returned collision pieces using native SDF geometry operations.

No mass/material is invented; the complete SDF writer is not executed. This
only applies its fixed Y-up→Z-up transform, OBJ export and declare-convex path.
"""
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import trimesh
from pydrake.geometry import Convex


def main():
    root = Path(__file__).resolve().parent
    out = root/'actual_v1'
    assert trimesh.__version__ == '4.7.4'
    stage = json.loads((out/'result.json').read_text())
    source = root/'source/scenesmith/agent_utils/sdf_generator.py'
    ledger = json.loads((root/'source_receipt.json').read_text())
    pin = next(r for r in ledger['files'] if r['path'].endswith('/sdf_generator.py'))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pin['sha256']
    tree = ast.parse(source.read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == 'YUP_TO_ZUP_TRANSFORM' for t in n.targets))
    scope = {'np': np}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), 'exec'), scope)
    transform = scope['YUP_TO_ZUP_TRANSFORM']
    raw = json.loads((out/'raw_response.json').read_text())
    assert raw['success'] and len(raw['collision_pieces']) == stage['piece_count']
    meshes = out/'consumer_pieces';meshes.mkdir(exist_ok=False)
    tick = time.monotonic()
    rows = []
    for i, data in enumerate(raw['collision_pieces']):
        # Same Trimesh construction as the exact native HTTP client.
        piece = trimesh.Trimesh(vertices=data['vertices'], faces=data['faces'])
        piece.apply_transform(transform)
        path = meshes/f'piece_{i:03d}.obj'
        piece.export(path)
        row = {'index': i, 'transformed_obj_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            hull = Convex(str(path)).GetConvexHull()
        except Exception as exc:
            row.update(status='geometry_rejected', error=f'{type(exc).__name__}: {exc}')
        else:
            row.update(status='geometry_returned', type=type(hull).__name__)
            try:
                row.update(vertices=hull.num_vertices(), faces=hull.num_faces())
            except Exception as exc:
                row['diagnostic_metadata_error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
    result = {'status': 'completed', 'pieces': len(rows),
        'returned': sum(r['status']=='geometry_returned' for r in rows),
        'rejected': sum(r['status']=='geometry_rejected' for r in rows),
        'metadata_errors': sum('diagnostic_metadata_error' in r for r in rows),
        'rows': rows, 'source_sdf_writer_sha256': pin['sha256'],
        'native_sdf_collision_lines': [190,225], 'coordinate_transform': transform.tolist(),
        'new_decomposition_requests': 0, 'new_model_calls': 0, 'seconds': time.monotonic()-tick,
        'mass_or_material_supplied': False, 'simulation_or_full_sdf_import': False,
        'scope': 'Actual Convex.GetConvexHull on every output after native coordinate/export steps. Geometry acceptance is not physical fidelity, mass/inertia validity, or scene success.'}
    (out/'drake_consumption_probe_v2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['status','pieces','returned','rejected','metadata_errors','seconds']}))


if __name__ == '__main__':
    main()
