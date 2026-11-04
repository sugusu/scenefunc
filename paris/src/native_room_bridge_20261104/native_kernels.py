"""Pinned native placement kernels with real Drake, without SDK/model boot.

Native class/function bodies are retained; only selected definitions are loaded
so unused generative services are never imported. SDK @function_tool wrapping is
omitted for direct Python invocation of its unchanged public closure. This is a
native placement-kernel replay, not full Scene-Designer inference or rendering.
"""
from __future__ import annotations
import ast,copy,dataclasses,hashlib,importlib.util,json,logging,math,os,sys,uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from enum import Enum
from types import ModuleType
from typing import Any
import numpy as np
import trimesh
from pydrake.all import RigidTransform,RollPitchYaw,RotationMatrix,Quaternion
from pydrake.geometry.optimization import VPolytope

P=Path(__file__).resolve().parent; S=P/'source/scenesmith'; READ=[]
# A real module name is necessary for dataclass field resolution.
m=ModuleType('replayed_native_definitions');sys.modules[m.__name__]=m
G=m.__dict__;G.update(dict(np=np,trimesh=trimesh,ET=ET,Path=Path,Enum=Enum,Any=Any,
 math=math,json=json,logging=logging,os=os,uuid=uuid,hashlib=hashlib,
 dataclass=dataclasses.dataclass,field=dataclasses.field,asdict=dataclasses.asdict,
 deepcopy=copy.deepcopy,RigidTransform=RigidTransform,RollPitchYaw=RollPitchYaw,
 RotationMatrix=RotationMatrix,Quaternion=Quaternion,VPolytope=VPolytope,
 console_logger=logging.getLogger('native-replay')))

def definitions(rel,names=None,methods=None):
 p=S/rel;t=ast.parse(p.read_text());nodes=[]
 for n in t.body:
  if not isinstance(n,(ast.ClassDef,ast.FunctionDef)):continue
  if names is not None and n.name not in names:continue
  n=copy.deepcopy(n)
  if methods is not None and isinstance(n,ast.ClassDef):
   n.body=[x for x in n.body if not isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) or x.name in methods]
  nodes.append(n);READ.append(dict(path=str(p),name=n.name,start=n.lineno,end=n.end_lineno,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),methods=methods))
 if names is not None:assert set(names)=={n.name for n in nodes},(rel,names)
 future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[])),str(p),'exec'),G)

def whole_simple(rel):
 p=S/rel;spec=importlib.util.spec_from_file_location('native_'+p.stem,p);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 for k,v in vars(mod).items():
  if not k.startswith('__'):G[k]=v
 READ.append(dict(path=str(p),whole_module=True,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))

whole_simple('agent_utils/response_datatypes.py')
whole_simple('agent_utils/action_logger.py')
whole_simple('agent_utils/placement_noise.py')
definitions('utils/geometry_utils.py',['compute_aabb_corners'])
definitions('utils/sdf_utils.py',['serialize_rigid_transform','deserialize_rigid_transform'])
definitions('agent_utils/room.py',['_int_to_base36','UniqueID','ObjectType','PlacementInfo','SupportSurface','SceneObject','RoomScene','copy_scene_object_with_new_pose'])
definitions('agent_utils/house.py',['WallDirection','RoomGeometry'])
definitions('agent_utils/asset_registry.py',['AssetRegistry'])
definitions('agent_utils/asset_manager.py',['AssetManager'],['get_asset_by_id','list_available_assets'])
definitions('wall_agents/tools/wall_surface.py',['WallSurface','_compute_wall_rotation','_rigid_transform_to_list','_list_to_rigid_transform'])
for rel in ['furniture_agents/tools/response_dataclasses.py','wall_agents/tools/response_dataclasses.py','ceiling_agents/tools/response_dataclasses.py']:definitions(rel)
definitions('furniture_agents/tools/furniture_tools.py',['FurnitureTools'],['_check_floor_bounds','_add_furniture_to_scene_impl','_create_success_result','_create_failure_result'])
definitions('wall_agents/tools/wall_tools.py',['WallTools'],['_place_wall_object_impl','_create_placement_failure_result'])
definitions('ceiling_agents/tools/ceiling_tools.py',['compute_ceiling_transform','CeilingTools'],['_place_ceiling_object_impl','_create_placement_failure_result'])

def public_callable(owner,rel,name):
 p=S/rel;t=ast.parse(p.read_text());c=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name==type(owner).__name__)
 method=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='_create_tool_closures')
 node=copy.deepcopy(next(x for x in method.body if isinstance(x,ast.FunctionDef) and x.name==name))
 assert len(node.decorator_list)==1 and isinstance(node.decorator_list[0],ast.Name) and node.decorator_list[0].id=='function_tool'
 node.decorator_list=[]
 fac=ast.FunctionDef(name='_bind_closure',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[node,ast.Return(value=ast.Name(id=name,ctx=ast.Load()))],decorator_list=[])
 future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[future,fac],type_ignores=[])),str(p),'exec'),G)
 READ.append(dict(path=str(p),closure=name,start=node.lineno,end=node.end_lineno,body_modified=False,omitted='SDK function_tool wrapper only'))
 return G['_bind_closure'](owner)

def make_tools(scene,registry,surfaces):
 manager=G['AssetManager']();manager.registry=registry
 noise=type('ZeroNoise',(),dict(position_xy_std_meters=0.,rotation_yaw_std_degrees=0.,position_along_wall_std_meters=0.,position_height_std_meters=0.,rotation_std_degrees=0.))()
 result={}
 for cls,rel,name in [('FurnitureTools','furniture_agents/tools/furniture_tools.py','add_furniture_to_scene_tool'),('WallTools','wall_agents/tools/wall_tools.py','place_wall_object'),('CeilingTools','ceiling_agents/tools/ceiling_tools.py','place_ceiling_object')]:
  obj=G[cls]();obj.scene=scene;obj.asset_manager=manager;obj.active_noise_profile=noise
  obj.surfaces_by_id=surfaces;obj.room_bounds=(-scene.room_geometry.length/2,-scene.room_geometry.width/2,scene.room_geometry.length/2,scene.room_geometry.width/2);obj.ceiling_height=scene.room_geometry.wall_height
  result[name]=public_callable(obj,rel,name)
 return result
