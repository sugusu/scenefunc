"""Export a restored development room and parse its actual native SDF resources.

No hand-written placement, mesh replacement, collision oracle or model action.
Parsing a stationary plant establishes loadability, not scene quality/stability.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--imported',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={'status':'started'}
 before={str(p):sha(p) for p in a.imported.rglob('*') if p.is_file()}
 try:
  sys.path.insert(0,str(a.source.resolve()))
  from scenesmith.agent_utils.house import HouseScene
  from pydrake.all import AddMultibodyPlantSceneGraph,DiagramBuilder,Parser
  from pydrake.multibody.parsing import LoadModelDirectives,ProcessModelDirectives
  from pydrake.geometry import Role
  p=Path('runs/autodl/20261110_corpus_state_recovery/source/native_project.py')
  assert sha(p)=='1c418410b1d5db8606305f200154418dd59844ce1c22fee3afb903286417ad19'
  spec=importlib.util.spec_from_file_location('native_validator_geometry',p)
  native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
  manifest,state,runtime,task=native.validate_bundle(a.imported)
  house=HouseScene.from_state_dict(state,house_dir=a.imported.resolve())
  for rid,room in house.rooms.items():
   room.room_type=runtime['rooms'][rid]['room_type'];room._surface_id_counter=runtime['rooms'][rid]['next_surface_id']
   house.layout.room_geometries[rid]=room.room_geometry
  target=a.imported/'native_geometry_check'
  if target.exists():raise FileExistsError('Do not overwrite a native export')
  export=house.assemble(cfg=None,output_name='native_geometry_check')
  result['native_house_export_completed']=True
  builder=DiagramBuilder();plant,scene_graph=AddMultibodyPlantSceneGraph(builder,time_step=0.001)
  parser=Parser(plant,scene_graph);parser.package_map().AddPackageXml(str((a.imported/'package.xml').resolve()))
  models=ProcessModelDirectives(LoadModelDirectives(str(export/'house_furniture_welded.dmd.yaml')),parser)
  plant.Finalize();diagram=builder.Build();context=diagram.CreateDefaultContext()
  inspector=scene_graph.model_inspector()
  model_rows=[{'model_name':m.model_name,'model_instance':int(m.model_instance)} for m in models]
  pose_errors=[]
  rid=manifest['selected_room_id'];room=house.rooms[rid]
  import numpy as np
  from pydrake.all import Quaternion,RotationMatrix
  for oid,obj in room.objects.items():
   raw=state['rooms'][rid]['objects'][oid]['transform']
   # Compare the loaded rotation matrix with one constructed from the frozen
   # quaternion; do not zero or overwrite serialized differences.
   expected=RotationMatrix(Quaternion(np.array(raw['rotation_wxyz']))).matrix()
   pose_errors.append({'object_id':oid,'rotation_matrix_max_abs_error':float(np.max(np.abs(expected-obj.transform.rotation().matrix()))),
     'translation_max_abs_error':float(np.max(np.abs(np.array(raw['translation'])-obj.transform.translation())))})
  write(a.output/'native_models.json',model_rows);write(a.output/'pose_error_measurements.json',pose_errors)
  # Preserve small native outputs in Git; referenced meshes remain in the
  # previously validated imported resource directory.
  outputs=[]
  for file in sorted(export.iterdir()):
   if file.is_file():
    copy=a.output/file.name;copy.write_bytes(file.read_bytes())
    outputs.append({'path':str(file),'sha256':sha(file),'bytes':file.stat().st_size})
  write(a.output/'exported_files.json',outputs)
  result.update(status='native_room_exported_and_drake_parsed',native_models=len(models),
   plant_bodies=plant.num_bodies(),geometry_count=inspector.num_geometries(),
   illustration_geometries=inspector.NumGeometriesWithRole(Role.kIllustration),
   proximity_geometries=inspector.NumGeometriesWithRole(Role.kProximity),
   context_created=context is not None,all_object_ids_retained=set(room.objects)==set(state['rooms'][rid]['objects']),
   rotation_matrix_max_abs_error=max(r['rotation_matrix_max_abs_error'] for r in pose_errors),
   translation_max_abs_error=max(r['translation_max_abs_error'] for r in pose_errors))
 except BaseException as e:
  result.update(status='failed',error=type(e).__name__+': '+str(e));raise
 finally:
  result.update(source_input_files_unchanged=all(Path(p).is_file() and sha(Path(p))==v for p,v in before.items()),
   wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   physics_simulation_run=False,collision_quality_checked=False,renderer_run=False,
   new_model_calls=0,training_updates=0,new_scene_generations=0,full_scene_success=None)
  write(a.output/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
