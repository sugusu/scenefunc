"""Actual native RoomScene -> Drake glTF -> Blender export, without an Agent.

Uses the original save_scene_as_blend and author's integration-patched server.
This wrapper replaces only the model/runtime bootstrap of export_scene_worker.
Original resources and state remain read-only. It is not new scene generation.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import sys
import time

from native_state_override import apply_recorded_state

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source',type=Path,required=True)
 ap.add_argument('--imported',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--state-ledger',type=Path)
 ap.add_argument('--unit')
 ap.add_argument('--checkpoint-labels',action='store_true')
 ap.add_argument('--disk-headroom-mib',type=int,default=32)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 result={'status':'started','model_calls':0,'training_updates':0,'new_scene_generations':0}
 before={str(p):sha(p) for p in a.imported.rglob('*') if p.is_file()}
 start=time.monotonic()
 try:
  if bool(a.state_ledger)!=bool(a.unit):raise ValueError('Unit and state ledger are required together')
  if a.disk_headroom_mib<24:raise ValueError('At least 24 MiB export headroom required')
  if shutil.disk_usage(a.out).free<5*1024**3+a.disk_headroom_mib*1024**2:raise RuntimeError('Export reserve')
  manifest=json.loads((a.source/'manifest.json').read_text())
  patched={r['path']:r['after_sha256'] for r in manifest['patches']}
  for row in manifest['sources']:
   if sha(a.source/row['path'])!=patched.get(row['path'],row['sha256']):raise ValueError('Source changed: '+row['path'])
  sys.path.insert(0,str(a.source.resolve()))
  from scenesmith.agent_utils.house import HouseScene
  from scenesmith.agent_utils.rendering import save_scene_as_blend
  state=json.loads((a.imported/'house_state.json').read_text())
  runtime=json.loads((a.imported/'runtime.json').read_text())
  meta=json.loads((a.imported/'manifest.json').read_text())
  if a.state_ledger:
   state,proof=apply_recorded_state(state,meta['selected_room_id'],ledger=a.state_ledger,unit=a.unit)
   result['frozen_override']=proof
  house=HouseScene.from_state_dict(state,house_dir=a.imported.resolve())
  rid=meta['selected_room_id'];scene=house.rooms[rid]
  scene.room_type=runtime['rooms'][rid]['room_type']
  scene._surface_id_counter=runtime['rooms'][rid]['next_surface_id']
  if a.checkpoint_labels:scene._scenefunc_structure_labels=True
  house.layout.room_geometries[rid]=scene.room_geometry
  current=scene.to_state_dict()
  (a.out/'actual_room_state.json').write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
  result.update(source_room_sha256=canonical(state['rooms'][rid]),runtime_room_sha256=canonical(current),
                objects=len(scene.objects),furniture=sum(o.object_type.value=='furniture' for o in scene.objects.values()))
  target=(a.out/'step_006.blend').resolve()
  save_scene_as_blend(scene=scene,output_path=target,blender_server_host='127.0.0.1',
                      blender_server_port_range=(18100,18110))
  result.update(status='native_blend_exported',blend_path=str(target),blend_bytes=target.stat().st_size,
                blend_sha256=sha(target),same_native_export_function=True)
  if a.checkpoint_labels:
   source=a.source/'src/render_observation.py'
   fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='render_scene_views')
   statement=next(n for n in fn.body if isinstance(n,ast.If) and '_scenefunc_structure_labels' in ast.unparse(n.test))
   exec(compile(ast.Module(body=[statement],type_ignores=[]),str(source),'exec'),{'scene':scene,'blend':target,'Path':Path})
   result['native_structure_sha256']=sha(target.with_suffix('.native_structure.json'))
 except BaseException as e:
  result.update(status='failed',error=type(e).__name__+': '+str(e));raise
 finally:
  loaded={name:{'path':str(Path(mod.__file__).resolve()),'sha256':sha(Path(mod.__file__))}
    for name,mod in sys.modules.items() if name.startswith('scenesmith') and getattr(mod,'__file__',None)}
  (a.out/'loaded_modules.json').write_text(json.dumps(loaded,indent=2)+'\n')
  versions={}
  for name in ['drake','bpy','trimesh','numpy','Pillow']:
   try:versions[name]=importlib.metadata.version(name)
   except importlib.metadata.PackageNotFoundError:versions[name]=None
  result.update(seconds=time.monotonic()-start,parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   versions=versions,input_files_unchanged=all(Path(p).is_file() and sha(Path(p))==h for p,h in before.items()),
   renderer_views_generated=False,full_scene_success=None,
   original_pixels_equivalent=False,free_disk_bytes=shutil.disk_usage(a.out).free)
  (a.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
