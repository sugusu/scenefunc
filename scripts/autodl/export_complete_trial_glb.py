"""Export the actual final native .blend to an editable GLB, no model."""
import json
from pathlib import Path
import sys
import bpy

source,target=map(Path,sys.argv[1:])
bpy.ops.wm.open_mainfile(filepath=str(source.resolve()))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
if not meshes:raise RuntimeError('No actual scene meshes')
bpy.ops.export_scene.gltf(filepath=str(target.resolve()),export_format='GLB',
    export_cameras=False,export_lights=False,export_extras=True)
print(json.dumps(dict(source=str(source),output=str(target),mesh_objects=len(meshes),bytes=target.stat().st_size)))
