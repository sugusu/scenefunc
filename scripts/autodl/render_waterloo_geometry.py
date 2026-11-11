"""CPU orthographic geometry preview of an exported Waterloo state.

New diagnostic image; not the author observation pipeline or visual evaluation.
View hides roof and outer wall meshes; GLB/state retain every source object.
"""
import argparse
import gzip
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    t = time.monotonic()
    import bpy
    from mathutils import Vector
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    tmp = Path('/dev/shm/waterloo_task_search_20261111')/(a.label+'_preview.glb')
    tmp.write_bytes(gzip.decompress(a.input.read_bytes()))
    bpy.ops.import_scene.gltf(filepath=str(tmp))
    hidden = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and ((obj.name.startswith('structure::') and 'floor' not in obj.name) or 'light' in obj.name):
            # Cutaway observation only. A hidden proxy light is not deleted
            # from evidence or geometric measurements.
            obj.hide_render = True; hidden.append(obj.name)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = 8
    scene.render.threads_mode = 'FIXED'; scene.render.threads = 4
    scene.render.resolution_x = 800; scene.render.resolution_y = 700
    scene.render.resolution_percentage = 100
    scene.world.color = (.8, .8, .8)
    # Blender's glTF importer has already converted glTF Y-up back to Z-up.
    # Keep the first side previews in evidence; this is the corrected view.
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 8))
    light = bpy.context.object; light.data.energy = 1600; light.data.shape = 'DISK'; light.data.size = 8
    light.rotation_euler = (Vector((0, 0, 0))-light.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.ops.object.camera_add(location=(0, 0, 12))
    cam = bpy.context.object; cam.rotation_euler = (Vector((0, 0, 0))-cam.location).to_track_quat('-Z', 'Y').to_euler()
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = 8.8; scene.camera = cam
    scene.render.filepath = str(a.output.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    a.output.with_suffix('.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-t,
        backend='Blender Cycles CPU', samples=8, hidden_for_cutaway=hidden,
        image_is_author_observation=False, new_scene_generation=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
