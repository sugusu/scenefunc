import bpy,json,sys
from pathlib import Path
from mathutils import Vector
bpy.ops.wm.open_mainfile(filepath=sys.argv[1])
rows=[]
for o in bpy.context.scene.objects:
    if o.type!="MESH": continue
    corners=[o.matrix_world @ Vector(c) for c in o.bound_box]
    lo=[min(v[i] for v in corners) for i in range(3)]
    hi=[max(v[i] for v in corners) for i in range(3)]
    dims=[hi[i]-lo[i] for i in range(3)]
    rows.append({"name":o.name,"local_dimensions":list(o.dimensions),"world_bbox_min":lo,"world_bbox_max":hi,"world_dimensions":dims,"upstream_wall_predicate":o.dimensions.z>2 and min(o.dimensions.x,o.dimensions.y)<0.1,"world_wall_predicate":dims[2]>2 and min(dims[:2])<0.1,"matrix_world":[list(row) for row in o.matrix_world]})
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2)+"\n")
print(json.dumps(rows),flush=True)
