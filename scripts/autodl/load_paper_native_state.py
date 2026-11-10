"""Load an exact Paper bundle with unmodified native HouseScene classes.

No policy, renderer, service, physics step or reconstructed target is invoked.
Inputs are copied by the original importer and remain immutable. This is an
actual native object restore, beyond a manifest check; not scene generation.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def difference(a,b,path='$'):
 if isinstance(a,dict) and isinstance(b,dict):
  return sum([difference(a[k],b[k],path+'.'+k) if k in a and k in b else
   [{'path':path+'.'+k,'kind':'missing_key','source_has':k in a,'restored_has':k in b}]
   for k in sorted(a.keys()|b.keys())],[])
 if isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):return [{'path':path,'kind':'length','source':len(a),'restored':len(b)}]
  return sum([difference(x,y,f'{path}[{i}]') for i,(x,y) in enumerate(zip(a,b))],[])
 return [] if a==b else [{'path':path,'kind':'value','source':a,'restored':b}]


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={'status':'started'}
 original=Path('runs/autodl/20261110_corpus_state_recovery/source/native_project.py')
 before={str(p):sha(p) for p in a.bundle.rglob('*') if p.is_file()}
 write(a.output/'input_hashes_before.json',before)
 try:
  m=json.loads((a.source/'manifest.json').read_text())
  assert m['pin']=='e888297a8b0a2f3acbb8634edf9def59cceab836'
  for row in m['files']:
   if row['status']=='downloaded':assert sha(a.source/row['path'])==row['sha256']
  assert sha(original)=='1c418410b1d5db8606305f200154418dd59844ce1c22fee3afb903286417ad19'
  required=sum(p.stat().st_size for p in a.bundle.rglob('*') if p.is_file())
  if shutil.disk_usage(a.output).free-required<5*1024**3+1024**2:raise RuntimeError('Disk reserve')
  sys.path.insert(0,str(a.source.resolve()))
  spec=importlib.util.spec_from_file_location('original_native_import',original)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  project=module.import_project(a.bundle,a.output/'imported')
  result['native_HouseScene_instantiated']=True
  restored=project.house.to_state_dict();expected=json.loads((a.bundle/'house_state.json').read_text())
  delta=difference(expected,restored);write(a.output/'roundtrip_state.json',restored)
  write(a.output/'state_differences.json',delta)
  rid=project.manifest['selected_room_id'];objects=project.room.objects
  loaded={name:{'path':str(Path(mod.__file__).resolve()),'sha256':sha(Path(mod.__file__))}
   for name,mod in sys.modules.items() if name.startswith('scenesmith') and getattr(mod,'__file__',None)}
  write(a.output/'loaded_sources.json',loaded)
  result.update(status='native_state_restored' if not delta else 'native_state_roundtrip_difference',
   native_class=type(project.house).__module__+'.'+type(project.house).__qualname__,
   object_records=len(objects),furniture_objects=sum(o.object_type.value=='furniture' for o in objects.values()),
   native_surface_counter=project.room._surface_id_counter,room_id=rid,
   state_value_differences=len(delta),serialized_state_hash_exact=canonical(restored)==canonical(expected),
   source_room_sha256=canonical(expected['rooms'][rid]),restored_room_sha256=canonical(restored['rooms'][rid]),
   loaded_source_modules=len(loaded),source_importer_sha256=sha(original),
   prior_history_imported=False)
 except BaseException as e:
  result.update(status='failed',error=type(e).__name__+': '+str(e));raise
 finally:
  after={str(p):sha(p) for p in a.bundle.rglob('*') if p.is_file()}
  write(a.output/'input_hashes_after.json',after)
  versions={}
  for name in ['drake','trimesh','numpy','scipy','omegaconf']:
   try:versions[name]=importlib.metadata.version(name)
   except importlib.metadata.PackageNotFoundError:versions[name]=None
  result.update(wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   input_files_unchanged=before==after,versions=versions,new_model_calls=0,training_updates=0,
   new_scene_generations=0,renderer_run=False,physics_run=False,session_resumed=False,
   full_scene_success=None)
  write(a.output/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
