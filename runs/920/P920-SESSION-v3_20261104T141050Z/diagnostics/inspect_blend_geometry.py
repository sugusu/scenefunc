import bpy,json,sys
from pathlib import Path
bpy.ops.wm.open_mainfile(filepath=sys.argv[1])
rows=[{"name":o.name,"type":o.type,"dimensions":list(o.dimensions),"location":list(o.location),"mesh_vertices":len(o.data.vertices) if o.type=="MESH" else None,"mesh_polygons":len(o.data.polygons) if o.type=="MESH" else None} for o in bpy.context.scene.objects]
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2)+"\n")
print(json.dumps(rows),flush=True)
