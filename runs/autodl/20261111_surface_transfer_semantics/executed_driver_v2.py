"""Bounded native geometry regression: surface identity is part of a move.

Runs pinned native methods and RoomScene/SupportSurface/PlacementInfo with real
Drake and a synthetic GLB. No model/SDK bootstrap, mesh collision simulation or
whole-scene generation. Original sources remain untouched; one conditional is
changed only in an in-memory ordinary-patch control.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

R = Path(__file__).resolve().parents[2]
W = R.parent.parent
BRIDGE = W / 'native_room_bridge_20261104'
LIVE = R / 'runs/autodl/20261110_live_manager_request/source'
REL = 'scenesmith/manipuland_agents/tools/manipuland_tools.py'


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest_ast(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def load_kernels():
    p = BRIDGE / 'native_kernels.py'
    spec = importlib.util.spec_from_file_location('surface_transfer_native_kernels', p)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_methods(kernels):
    env = kernels.G
    env.update(ConvexHull=ConvexHull, QhullError=QhullError)
    geometry_path = LIVE / 'scenesmith/utils/geometry_utils.py'
    helper = next(n for n in ast.parse(geometry_path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'safe_convex_hull_2d')
    helper = copy.deepcopy(helper)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,helper], type_ignores=[])),
                 str(geometry_path),'exec'),env)
    response_path = LIVE / 'scenesmith/manipuland_agents/tools/response_dataclasses.py'
    wanted = {'ManipulandErrorType', 'Position2D', 'ManipulandOperationResult', 'ManipulandPlacementResult'}
    definitions = [copy.deepcopy(n) for n in ast.parse(response_path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name in wanted]
    assert {n.name for n in definitions} == wanted
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+definitions, type_ignores=[])),
                 str(response_path), 'exec'), env)
    tree = ast.parse((LIVE / REL).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ManipulandTools')
    wanted_methods = {'_get_object_convex_hull_2d', '_is_top_surface', '_validate_convex_hull_footprint',
                      '_move_manipuland_impl', '_create_placement_failure_result'}
    selected = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted_methods]
    assert {n.name for n in selected} == wanted_methods
    method_sources = [dict(symbol=n.name, start=n.lineno, end=n.end_lineno,
                           original_body_ast_sha256=digest_ast(n),
                           omitted_decorators=[ast.unparse(d) for d in n.decorator_list]) for n in selected]
    for n in selected:
        n.decorator_list = []  # Action logging wrapper only; no body changes.
    original = ast.ClassDef(name='OriginalSurfaceTools', bases=[], keywords=[], body=selected, decorator_list=[])
    patched = copy.deepcopy(original); patched.name = 'OrdinaryPatchedSurfaceTools'
    fn = next(n for n in patched.body if n.name == '_move_manipuland_impl')
    guards = [n for n in ast.walk(fn) if isinstance(n, ast.If) and
              ast.unparse(n.test) == 'position_unchanged and rotation_unchanged']
    assert len(guards) == 1
    before = copy.deepcopy(guards[0].test)
    extra = ast.parse('str(scene_obj.placement_info.parent_surface_id) == surface_id', mode='eval').body
    guards[0].test = ast.BoolOp(op=ast.And(), values=[copy.deepcopy(before), extra])
    patch_condition = ast.unparse(guards[0].test)
    # Restore the guard in a copy and prove every method body then matches.
    restored = copy.deepcopy(patched); restored.name = original.name
    rf = next(n for n in restored.body if n.name == '_move_manipuland_impl')
    rg = next(n for n in ast.walk(rf) if isinstance(n, ast.If) and ast.unparse(n.test) == patch_condition)
    rg.test = before
    assert digest_ast(restored) == digest_ast(original)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, original, patched], type_ignores=[])),
                 str(LIVE / REL), 'exec'), env)
    original_fn = next(n for n in original.body if n.name == '_move_manipuland_impl')
    diff = ''.join(difflib.unified_diff(ast.unparse(original_fn).splitlines(True),
                     ast.unparse(fn).splitlines(True), fromfile='original_method', tofile='ordinary_patch_method'))
    return env, method_sources, patch_condition, diff


def fixture(env, cls, path, project_mesh, same_surface=False):
    path.mkdir(parents=True)
    UID = env['UniqueID']; Transform = env['RigidTransform']
    geo = env['RoomGeometry'](sdf_tree=ET.ElementTree(ET.Element('sdf')),
          sdf_path=path/'not_simulated.sdf', width=3., length=3., wall_height=2.5)
    scene = env['RoomScene'](room_geometry=geo, scene_dir=path, room_id='regression')
    plane = trimesh.Trimesh(vertices=[[-.4,-.4,0],[.4,-.4,0],[.4,.4,0],[-.4,.4,0]],
                            faces=[[0,1,2],[0,2,3]], process=False)
    surfaces = [env['SupportSurface'](surface_id=UID(name), bounding_box_min=np.array([-.4,-.4,0]),
                 bounding_box_max=np.array([.4,.4,.3]), transform=Transform([0,0,height]), mesh=plane.copy())
                for name,height in [('lower_shelf',.4),('upper_shelf',.9)]]
    parent = env['SceneObject'](object_id=UID('shelf_fixture'), object_type=env['ObjectType'].FURNITURE,
              name='shelf_fixture', description='Synthetic two-surface branch fixture; not a room benchmark',
              transform=Transform(), support_surfaces=surfaces,
              bbox_min=np.array([-.4,-.4,0]), bbox_max=np.array([.4,.4,1.2]))
    project = env['SceneObject'](object_id=UID('project_piece_0'), object_type=env['ObjectType'].MANIPULAND,
              name='project_piece', description='Synthetic GLB used only for native footprint validation',
              transform=surfaces[0].to_world_pose(np.array([0.,0.]), 0.), geometry_path=project_mesh,
              bbox_min=np.array([-.05,-.05,0]), bbox_max=np.array([.05,.05,.1]),
              placement_info=env['PlacementInfo'](parent_surface_id=surfaces[0].surface_id,
                                 position_2d=np.array([0.,0.]), rotation_2d=0.))
    scene.add_object(parent); scene.add_object(project)
    owner = cls(); owner.scene = scene; owner.support_surfaces = {str(s.surface_id):s for s in surfaces}
    owner.top_surface_overlap_tolerance = 0.
    owner.active_noise_profile = SimpleNamespace(position_xy_std_meters=0., rotation_yaw_std_degrees=0.)
    return owner, project, surfaces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--scratch', type=Path, required=True)
    a = ap.parse_args(); root=a.root.resolve(); scratch=a.scratch.resolve()
    if root.exists() or scratch.exists():
        raise FileExistsError('Fresh regression paths required; do not overwrite a failed attempt')
    root.mkdir(parents=True); scratch.mkdir(parents=True); start=time.monotonic()
    before = {str(p):sha(p) for p in [LIVE/REL, LIVE/'scenesmith/manipuland_agents/tools/response_dataclasses.py',
                                     LIVE/'scenesmith/utils/geometry_utils.py',BRIDGE/'native_kernels.py']}
    source_manifest = read(LIVE.parent/'source_manifest.json')
    expected = next(r['sha256'] for r in source_manifest['files'] if r['path']==REL)
    assert sha(LIVE/REL)==expected
    network=[]
    def deny(*_,**__):
        network.append(True); raise RuntimeError('No network in CPU surface branch replay')
    socket.socket.connect=deny; socket.create_connection=deny
    kernels=load_kernels(); env, methods, guard, diff=load_methods(kernels)
    for item in kernels.READ:
        before[item['path']]=sha(item['path'])
    mesh=trimesh.creation.box(extents=[.1,.1,.1]); mesh.apply_translation([0,0,.05])
    mesh_path=scratch/'project.glb'; mesh.export(mesh_path)
    dump(root/'protocol.json',dict(source_pin=source_manifest['base_pin'], source_sha256=expected,
         cases=['same_frame_same_local','other_frame_same_local','same_frame_changed_local'],
         decision_rule='Predefined native regression cases, not neural scene generation outcomes',
         original_method_body_changed=False, patch_ast_guard_edits=1, patched_guard=guard,
         source_and_target_surface_transforms='Fixed throughout each call; no claim about stale/moving surface frames',
         geometry='Synthetic0.1mGLB and two native0.8m support planes atZ0.4/Z0.9; no HSSD files changed',
         native_types=['RoomScene','RoomGeometry','SceneObject','UniqueID','PlacementInfo','SupportSurface'],
         body_scope=methods, helpers_mocked=False, wrappers_omitted='Action logging wrapper and unused service constructor/SDK bootstrap',
         mesh_collision_physics_or_path_replay=False, model_calls=0, paid_API_calls=0, training_updates=0,
         whole_scenes_generated=0, regression_is_not_research_benchmark=True))
    rows=[]
    for arm,cls in [('original',env['OriginalSurfaceTools']),('ordinary_patch',env['OrdinaryPatchedSurfaceTools'])]:
        for name in ['same_frame_same_local','other_frame_same_local','same_frame_changed_local']:
            owner, obj, surfaces = fixture(env,cls,scratch/f'{arm}_{name}',mesh_path)
            target=surfaces[1] if name=='other_frame_same_local' else surfaces[0]
            pos=np.array([.1,0.]) if name=='same_frame_changed_local' else np.array([0.,0.])
            expected_pose=target.to_world_pose(pos,0.)
            initial_ids=sorted(map(str,owner.scene.objects)); initial_pose=obj.transform.GetAsMatrix4().copy()
            initial_parent=str(obj.placement_info.parent_surface_id)
            request=dict(object_id=str(obj.object_id),surface_id=str(target.surface_id),
                         position_x=float(pos[0]),position_z=float(pos[1]),rotation_degrees=0.)
            response_text=owner._move_manipuland_impl(**request); response=json.loads(response_text)
            final_pose=obj.transform.GetAsMatrix4()
            expected_move=name!='same_frame_same_local'
            postcondition=(str(obj.placement_info.parent_surface_id)==str(target.surface_id) and
                           np.allclose(final_pose,expected_pose.GetAsMatrix4(),atol=1e-12,rtol=0))
            row=dict(arm=arm,case=name,request=request,raw_response=response_text,
                     initial_parent=initial_parent,final_parent=str(obj.placement_info.parent_surface_id),
                     initial_pose=initial_pose.tolist(),expected_pose=expected_pose.GetAsMatrix4().tolist(),
                     final_pose=final_pose.tolist(),expected_move=expected_move,
                     binding_and_world_pose_postcondition=bool(postcondition),
                     tool_reported_success=response['success'], error_type=response.get('error_type'),
                     world_displacement_m=float(np.linalg.norm(final_pose[:3,3]-initial_pose[:3,3])),
                     object_inventory_preserved=sorted(map(str,owner.scene.objects))==initial_ids,
                     object_and_geometry_identity_preserved=obj.geometry_path==mesh_path and str(obj.object_id)=='project_piece_0')
            rows.append(row); dump(root/'results.json',dict(status='running',rows=rows))
            assert row['object_inventory_preserved'] and row['object_and_geometry_identity_preserved']
            if arm=='original' and name=='other_frame_same_local':
                assert not response['success'] and response['error_type']=='no_movement' and not postcondition
                assert np.array_equal(initial_pose,final_pose)
            else:
                assert postcondition and response['success']==expected_move
                if not expected_move:
                    assert response['error_type']=='no_movement'
    assert len(rows)==6 and not network
    source_rows=[dict(path=p,before=h,after=sha(p),same=sha(p)==h) for p,h in before.items()]
    assert all(r['same'] for r in source_rows)
    dump(root/'source_before_after.json',dict(files=source_rows,all_unchanged=True))
    (root/'ordinary_patch.diff').write_text(diff+'\n')
    dump(root/'fixture_geometry.json',dict(glb_sha256=sha(mesh_path), glb_bytes=mesh_path.stat().st_size,
         dimensions_m=[.1,.1,.1],mesh_is_synthetic=True,no_licensed_assets_distributed=True))
    dump(root/'results.json',dict(status='completed',rows=rows,wall_seconds=time.monotonic()-start,
         native_move_invocations=6,model_calls=0,paid_API_calls=0,training_updates=0,network_attempts=0,
         outcome='Original incorrectly rejects different support frame at equal local pose; ordinary one-guard patch repairs this bounded case',
         occurrence_in_previous_scene_or_model_runs='Not established',full_scene_generation_success=False))
    print(json.dumps(dict(status='completed',native_moves=6,model_calls=0,
                          reproduced='cross_surface_local_pose_alias',wall_seconds=time.monotonic()-start)))


if __name__=='__main__':
    main()
