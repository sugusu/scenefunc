"""Persist actual Waterloo start/hosting geometry without new generation.

Reads actual native SDF visuals and recorded transforms. GLB is a geometry
preview, not a physics asset; textures may be simplified by concatenation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import trimesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(exist_ok=False)
    sys.path.insert(0, str(Path('runs/autodl/20261110_public_function_search/original_evaluator').resolve()))
    from inspect_native_mesh_contacts import sdf_visuals, world_transform
    from native_project import validate_bundle
    manifest, state, runtime, _ = validate_bundle(a.bundle)
    rid = manifest['selected_room_id']; room = a.bundle/('room_'+rid)
    states = {'initial': state['rooms'][rid],
              'hosting': json.loads((a.root/'B0_task_aware_ordinary_CSP/hosting_state.json').read_text())}
    receipt = dict(started=time.time(), outputs=[], material_fidelity_verified=False,
                   native_state_authoritative=True, physics_ready=False)
    up = np.eye(4); up[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    for label, s in states.items():
        scene = trimesh.Scene(); cache = {}; ids = []
        for oid, obj in s['objects'].items():
            if not obj['sdf_path']:
                continue
            ids.append(oid)
            for index, (name, mesh) in enumerate(sdf_visuals(room/obj['sdf_path'], a.bundle, cache)):
                mesh.metadata.update(object_id=oid, source_visual=name,
                    representation=obj['metadata']['asset_source'], up='+Y', units='m')
                scene.add_geometry(mesh, node_name=oid+'::'+str(index),
                                   transform=up@world_transform(obj))
        structure = []
        for index, (name, mesh) in enumerate(sdf_visuals(room/s['room_geometry']['sdf_path'], a.bundle, cache)):
            # All shell nodes retained; renderer may hide walls/ceiling only in
            # the view. Geometry exports never omit difficult furniture.
            scene.add_geometry(mesh, node_name='structure::'+name+'::'+str(index), transform=up)
            structure.append(name)
        scene.metadata.update(reconstruction='Waterloo research transfer, not original residence',
                              units='metres', up='+Y', physics_ready=False)
        raw = scene.export(file_type='glb')
        restored = trimesh.load(__import__('io').BytesIO(raw), file_type='glb', force='scene', process=False)
        assert set(restored.graph.nodes_geometry) == set(scene.graph.nodes_geometry)
        errors = []
        for node in scene.graph.nodes_geometry:
            t, geom = scene.graph[node]; m = scene.geometry[geom].copy(); m.apply_transform(t)
            rt, rg = restored.graph[node]; rm = restored.geometry[rg].copy(); rm.apply_transform(rt)
            assert len(m.faces) == len(rm.faces)
            errors.append(float(np.max(np.abs(m.bounds-rm.bounds))))
        assert max(errors) < 1e-5
        out = a.out/(label+'_scene.glb.gz')
        with out.open('wb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as z:
                z.write(raw)
        (a.out/(label+'_room_state.json')).write_text(json.dumps(s, indent=2)+'\n')
        receipt['outputs'].append(dict(label=label, path=str(out.resolve()),
            compressed_bytes=out.stat().st_size, compressed_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            glb_bytes=len(raw), glb_sha256=hashlib.sha256(raw).hexdigest(),
            visual_object_ids=ids, shell_visuals=structure,
            objects_total=len(s['objects']), exported_visual_objects=len(ids),
            geometry_nodes=len(scene.graph.nodes_geometry), maximum_bound_error_m=max(errors)))
        print(label, len(raw), out.stat().st_size, flush=True)
    receipt['wall_seconds'] = time.time()-receipt['started']
    (a.root/'geometry_artifact_manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
