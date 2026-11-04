"""Read actual exported state; no repair or threshold tuning.

FCL mesh-surface intersection is a limited geometry observation, not a complete
physics, containment, human-use, semantic or preference score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import trimesh


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert not args.out.exists()
    start=time.monotonic()
    state=json.loads(args.checkpoint.read_text())['payload']
    shell=json.loads(Path(state['frozen_shell']['path']).read_text())
    assert sha(state['frozen_shell']['path'])==state['frozen_shell']['sha256']
    floor=next(o for o in shell['room_structure'] if o['name']=='floor')
    half=np.array([floor['dimensions'][0],floor['dimensions'][2]])/2
    glb=args.checkpoint.parent/'partial_scene.glb'
    scene=trimesh.load(glb,force='scene')
    instances={o['object_id']:o for o in state['placed_instances']}
    assert set(instances)<=set(scene.graph.nodes_geometry)
    furniture=trimesh.collision.CollisionManager()
    architecture=trimesh.collision.CollisionManager()
    rows=[];shell_nodes=[]
    for node in scene.graph.nodes_geometry:
        tf,gid=scene.graph[node];mesh=scene.geometry[gid].copy();mesh.apply_transform(tf)
        if node in instances:
            obj=instances[node]
            assert sha(obj['geometry_path'])==state['geometry_hashes'][obj['geometry_path']]
            furniture.add_object(node,mesh)
            bounds=mesh.bounds
            exceed=np.maximum(-half-bounds[0,:2],bounds[1,:2]-half)
            rows.append({'object_id':node,'name':obj['name'],'description':obj['description'],
                         'world_bounds':bounds.tolist(),
                         'outside_rectangular_floor_m':np.maximum(exceed,0).tolist(),
                         'outside_rectangular_floor':bool(np.any(exceed>1e-5)),
                         'bottom_z_m':float(bounds[0,2]),
                         'top_z_m':float(bounds[1,2])})
        else:
            architecture.add_object(node,mesh);shell_nodes.append(node)
    collision,pairs=furniture.in_collision_internal(return_names=True)
    shell_collision,shell_pairs=furniture.in_collision_other(architecture,return_names=True)
    classified={'floor':[],'ceiling':[],'walls_doors_windows':[]}
    for a,b in sorted(shell_pairs):
        if b in instances:a,b=b,a
        kind='floor' if b.startswith('floor_') else 'ceiling' if b.startswith('ceiling_') else 'walls_doors_windows'
        classified[kind].append([a,b])
    output={'checkpoint':str(args.checkpoint.resolve()),'scene':str(glb.resolve()),'scene_sha256':sha(glb),
        'placed_furniture':len(instances),'shell_nodes':len(shell_nodes),
        'furniture_surface_intersection_pairs':[list(p) for p in sorted(pairs)],
        'architecture_surface_intersection_pairs':classified,
        'out_of_rectangular_floor_objects':[r['object_id'] for r in rows if r['outside_rectangular_floor']],
        'numeric_boundary_tolerance_m':1e-5,
        'floor_reference':{'native_kernel_bottom_z':0,'source_shell_floor_top_z':floor['location'][1]+floor['dimensions'][1]/2,
                           'interpretation':'Report raw contact/intersection evidence. The bridge floor volume and native placement reference differ; do not call this physical support validated.'},
        'objects':rows,'full_scene_success':None,
        'not_checked':['solid containment without triangle intersections','dynamic stability','door swing volume','walking access','functional fronts','storage capacity','work and dining usable surface','aesthetics','material correctness','human preference'],
        'geometry_mutated':False,'elapsed_seconds':time.monotonic()-start}
    args.out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['placed_furniture','furniture_surface_intersection_pairs','architecture_surface_intersection_pairs','out_of_rectangular_floor_objects','elapsed_seconds']},ensure_ascii=False))


if __name__=='__main__':main()
