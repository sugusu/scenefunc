"""CPU topdown cutaway of actual GLBs; walls omitted only in this view.

Internal partitions stay visible. Blue highlights denote added work furniture,
not original material fidelity or professional preference.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('No render overwrite')
    t=time.monotonic()
    import bpy
    from mathutils import Vector
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(a.input.resolve()))
    hidden=[];highlighted=[];internal=[]
    for obj in bpy.data.objects:
        if obj.type!='MESH':continue
        if obj.name.startswith(('structure::ceiling','structure::south_','structure::north_',
                                'structure::east_','structure::west_')):
            obj.hide_render=True;hidden.append(obj.name)
        elif obj.name.startswith(('structure::office_','structure::bath_')):
            internal.append(obj.name)
        if obj.name.startswith(('new_work_table_0::','new_work_chair_0::')):
            mat=bpy.data.materials.new('diagnostic_work_blue');mat.diffuse_color=(.04,.35,.9,1)
            obj.data.materials.clear();obj.data.materials.append(mat);highlighted.append(obj.name)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
    scene.render.threads_mode='FIXED';scene.render.threads=4
    scene.render.resolution_x=1100;scene.render.resolution_y=650;scene.render.resolution_percentage=100
    scene.world.color=(.8,.8,.8)
    bpy.ops.object.light_add(type='AREA',location=(0,0,8));light=bpy.context.object
    light.data.energy=2200;light.data.shape='DISK';light.data.size=10
    light.rotation_euler=(Vector((0,0,0))-light.location).to_track_quat('-Z','Y').to_euler()
    bpy.ops.object.camera_add(location=(0,0,14));cam=bpy.context.object
    cam.rotation_euler=(Vector((0,0,0))-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.type='ORTHO';cam.data.ortho_scale=11.5;scene.camera=cam
    scene.render.filepath=str(a.output.resolve());bpy.ops.render.render(write_still=True)
    a.output.with_suffix('.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-t,
        renderer='Blender Cycles CPU,8samples,4threads',hidden_cutaway_only=hidden,
        retained_internal_partitions=internal,blue_added_work_furniture=highlighted,
        author_observation=False,material_fidelity_claimed=False),indent=2)+'\n')


if __name__=='__main__':
    main()
