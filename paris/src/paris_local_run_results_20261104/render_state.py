"""Render actual GLB state with the pinned five-view renderer.

GLB import replaces the unavailable Drake exporter. A declared observation-only
ceiling cutaway allows interior inspection. No geometry is deleted or moved in
the exported scene. This is not a full native observation-parity claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

P=Path(__file__).resolve().parent
W=P.parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--transient-blend',action='store_true',help='Keep the reproducible renderer intermediate in RAM; retain original GLB and all views on data disk')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();source_hash=sha(args.scene)
    receipt={'source_scene':str(args.scene.resolve()),'source_sha256':source_hash,
             'native_renderer':True,'native_drake_exporter':False,
             'native_ID_annotations':False,'physics_validated':False,
             'observation_only_change':'ceiling hidden for all five cutaway views',
             'status':'started','model_calls':0,'paid_api_calls':0}
    temporary=None
    try:
        import bpy
        import numpy as np
        import trimesh
        from mathutils import Vector, Matrix
        source=trimesh.load(args.scene,force='scene')
        source_names=set(source.graph.nodes_geometry)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(args.scene.resolve()))
        # These bridge GLBs store native Z-up coordinates directly. Blender's
        # glTF importer applies its conventional Y-up -> Z-up conversion. Undo
        # that documented convention once at each imported root; do not alter
        # furniture poses or relax the equality check.
        inverse_import=Matrix(((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1)))
        for root in [o for o in bpy.context.scene.objects if o.parent is None]:
            root.matrix_world=inverse_import@root.matrix_world
        bpy.context.view_layer.update()
        receipt['import_axis_compensation']=[list(row) for row in inverse_import]
        imported=[o for o in bpy.context.scene.objects if o.type=='MESH']
        assert {o.name for o in imported}==source_names
        checks=[]
        for obj in imported:
            tf,gid=source.graph[obj.name];mesh=source.geometry[gid].copy();mesh.apply_transform(tf)
            # Exact world vertices rather than object AABBs (which can differ
            # after rotation) establish the import did not reposition assets.
            verts=np.array([list(obj.matrix_world@v.co) for v in obj.data.vertices])
            assert np.allclose([verts.min(0),verts.max(0)],mesh.bounds,atol=1e-6,rtol=0)
            checks.append({'node':obj.name,'bounds_error_m':float(np.abs(np.array([verts.min(0),verts.max(0)])-mesh.bounds).max())})
            original_name=obj.name
            if original_name.startswith(('wall_','bathroom_','floor_','ceiling_','door_','window_')):
                obj.name='geometry_'+original_name
            else:obj.name='asset_'+original_name
            obj['source_node']=original_name
            if original_name.startswith('ceiling_'):obj.hide_render=True
        world=bpy.data.worlds.new('ObservationWorld');world.use_nodes=True
        world.node_tree.nodes['Background'].inputs[0].default_value=(.8,.8,.8,1)
        world.node_tree.nodes['Background'].inputs[1].default_value=.8
        bpy.context.scene.world=world
        sun=bpy.data.lights.new('ObservationSun','SUN');sun.energy=3
        light=bpy.data.objects.new('ObservationSun',sun);bpy.context.scene.collection.objects.link(light)
        light.rotation_euler=(.3,.2,0.)
        if args.transient_blend:
            temporary=tempfile.TemporaryDirectory(prefix='paris_render_',dir='/dev/shm')
        blend=(Path(temporary.name) if temporary else out)/'current_state.blend'
        bpy.ops.wm.save_as_mainfile(filepath=str(blend),compress=True)
        receipt.update(imported_nodes=len(checks),mesh_checks=checks,blend_sha256=sha(blend),
                       transient_reproducible_blend=args.transient_blend)
        renderer=W/'native_observation_bridge_20261104/source/tools/render_five_views.py'
        env=dict(os.environ,SCENE_RENDER_THREADS='6',SCENE_RENDER_SAMPLES='24',PYTHONDONTWRITEBYTECODE='1')
        command=[sys.executable,str(renderer),str(blend),'--out',str(out)]
        with (out/'render.log').open('w') as log:
            run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        receipt['command']=command;receipt['render_exit']=run.returncode
        assert run.returncode==0
        views=[out/f'current_state.{view}.png' for view in ('topdown','south','east','west','north')]
        assert all(p.is_file() for p in views)
        receipt.update(status='completed',views=[{'path':str(p),'sha256':sha(p)} for p in views])
        assert sha(args.scene)==source_hash
    except BaseException as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if temporary is not None:
            temporary.cleanup()
            receipt['transient_blend_deleted_after_use']=True
        receipt.update(elapsed_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in receipt.items() if k!='mesh_checks'}),flush=True)


if __name__=='__main__':main()
