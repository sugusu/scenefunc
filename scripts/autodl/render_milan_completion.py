"""CPU cutaway preview; highlights are diagnostic, not native materials."""
import argparse
import gzip
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view',choices=['topdown','oblique'],required=True)
    a=p.parse_args();start=time.monotonic()
    if a.output.exists():raise FileExistsError('Retain previous view, no overwrite')
    import bpy
    from mathutils import Vector
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    glb=a.scratch/('milan_'+a.view+'_preview.glb')
    glb.write_bytes(gzip.decompress(a.input.read_bytes()))
    bpy.ops.import_scene.gltf(filepath=str(glb))
    colors={'ordinary_work':(.08,.45,.85,1),'meal_table':(.85,.45,.10,1),'meal_chair':(.85,.45,.10,1)}
    hidden=[];highlighted=[]
    for obj in bpy.data.objects:
        if obj.type!='MESH':continue
        if obj.name.startswith('structure::') and 'floor' not in obj.name:
            obj.hide_render=True;hidden.append(obj.name)
        for prefix,color in colors.items():
            if obj.name.startswith(prefix):
                mat=bpy.data.materials.new('diagnostic_'+prefix);mat.diffuse_color=color
                obj.data.materials.clear();obj.data.materials.append(mat);highlighted.append(obj.name)
                break
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
    scene.render.threads_mode='FIXED';scene.render.threads=4
    scene.render.resolution_x=1000;scene.render.resolution_y=700;scene.render.resolution_percentage=100
    scene.world.color=(.8,.8,.8)
    bpy.ops.object.light_add(type='AREA',location=(0,0,9));light=bpy.context.object
    light.data.energy=2200;light.data.shape='DISK';light.data.size=10
    light.rotation_euler=(Vector((0,0,0))-light.location).to_track_quat('-Z','Y').to_euler()
    location=(0,0,14) if a.view=='topdown' else (9,-11,12)
    bpy.ops.object.camera_add(location=location);cam=bpy.context.object
    cam.rotation_euler=(Vector((0,0,0))-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.type='ORTHO';cam.data.ortho_scale=11.5 if a.view=='topdown' else 14
    scene.camera=cam;scene.render.filepath=str(a.output.resolve());bpy.ops.render.render(write_still=True)
    a.output.with_suffix('.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,
        renderer='Cycles CPU,8samples,4threads',view=a.view,hidden_cutaway_only=hidden,
        highlighted_objects=highlighted,blue='new workspace',orange='retained dining',
        material_fidelity_claimed=False,author_native_observation=False),indent=2)+'\n')


if __name__=='__main__':
    main()
