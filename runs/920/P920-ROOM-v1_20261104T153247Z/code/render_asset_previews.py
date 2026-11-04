"""Render actual pinned glTF candidates for visual review, without resizing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct

import bpy
from mathutils import Vector


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    print(datetime.now(timezone.utc).isoformat(), 'begin fixed candidate visual inspection', flush=True)
    records = []
    for item in plan['meshes']:
        source = Path(item['path'])
        if digest(source) != item['sha256']:
            raise ValueError('Candidate source changed')
        raw = source.read_bytes()
        if raw[:4] != b'glTF' or raw[16:20] != b'JSON':
            raise ValueError('Expected glTF 2 binary with leading JSON chunk')
        gltf = json.loads(raw[20:20 + struct.unpack('<I', raw[12:16])[0]])
        if any('uri' in row and not row['uri'].startswith('data:')
               for key in ('buffers', 'images') for row in gltf.get(key, [])):
            raise ValueError('External glTF resources require a separate source lock')
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(source))
        scene = bpy.context.scene
        meshes = [o for o in scene.objects if o.type == 'MESH']
        bpy.context.view_layer.update()
        points = [o.matrix_world @ Vector(v) for o in meshes for v in o.bound_box]
        low = Vector([min(v[k] for v in points) for k in range(3)])
        high = Vector([max(v[k] for v in points) for k in range(3)])
        center, extent = (low + high) / 2, high - low
        # Blender's glTF importer already maps source +Y up to +Z up. The
        # current candidates all record +Y up / -Z front; reject other frames.
        if item['up'] != [0, 1, 0] or item['front'] != [0, 0, -1]:
            raise ValueError('This preview supports only the locked glTF frame')
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 8
        scene.cycles.use_denoising = False
        scene.render.threads_mode = 'FIXED'
        scene.render.threads = 4
        scene.render.resolution_x = scene.render.resolution_y = 512
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.world = bpy.data.worlds.new('InspectionWorld')
        scene.world.use_nodes = True
        scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
        scene.world.node_tree.nodes['Background'].inputs[1].default_value = 0.7
        radius = max(extent)
        bpy.ops.object.light_add(type='AREA', location=center + Vector((radius, -radius, 2 * radius)))
        bpy.context.object.data.energy = 300 * radius * radius
        bpy.context.object.data.shape = 'DISK'
        bpy.context.object.data.size = 2 * radius
        bpy.context.object.rotation_euler = (center - bpy.context.object.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.ops.object.camera_add(location=center + Vector((1.4, -1.8, 1.1)).normalized() * radius * 3)
        camera = bpy.context.object
        camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera.data.type = 'ORTHO'
        camera.data.ortho_scale = radius * 1.7
        scene.camera = camera
        output = a.out / (item['asset_id'] + '.png')
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        records.append({'asset_id': item['asset_id'], 'source_sha256': item['sha256'],
                        'image': str(output), 'image_sha256': digest(output),
                        'imported_bounds_z_up_m': [list(low), list(high)],
                        'imported_extents_z_up_m': list(extent), 'mesh_nodes': len(meshes),
                        'rescaled': False, 'visual_review': 'pending'})
        (a.out / 'PREVIEWS.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'rendered_candidates': len(records)}), flush=True)


if __name__ == '__main__':
    main()
