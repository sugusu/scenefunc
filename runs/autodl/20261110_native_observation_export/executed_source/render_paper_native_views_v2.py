"""Execute the pinned five-view renderer and its original live-ID annotation.

The original export uses a different Blender/Drake version: these are new,
versioned observations, not automatically admitted as recovered training PNGs.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def tree_rss(pid):
 """Sample Linux RSS without installing a monitoring dependency."""
 try:
  status=Path(f'/proc/{pid}/status').read_text().splitlines()
  rss=next((int(x.split()[1])*1024 for x in status if x.startswith('VmRSS:')),0)
  children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
  return rss+sum(tree_rss(int(child)) for child in children)
 except (FileNotFoundError,ProcessLookupError,PermissionError):return 0

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source',type=Path,required=True)
 ap.add_argument('--export',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--seconds',type=float,default=1800)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 source=a.source.resolve();out=a.out.resolve();export=a.export.resolve()
 result={'status':'started','model_calls':0,'training_updates':0,'new_scene_generations':0}
 record=export/'result.json';state=export/'actual_room_state.json';blend=export/'step_006.blend'
 before={str(p):sha(p) for p in [record,state,blend]}
 spec=json.loads(record.read_text());assert spec['status']=='native_blend_exported'
 assert spec['blend_sha256']==sha(blend)
 command=[sys.executable,'-B',str(source/'tools/render_five_views.py'),str(blend),'--out',str(out)]
 env=os.environ.copy();env.update(SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8',
   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 result.update(command=command,render_threads=4,render_samples=8,
   source_room_sha256=spec['source_room_sha256'],runtime_room_sha256=spec['runtime_room_sha256'],
   versions=spec['versions'],original_view_set_replaced=False)
 start=time.monotonic();child=None
 try:
  peak=0
  with (out/'renderer.log').open('w') as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll() is None:
    peak=max(peak,tree_rss(child.pid))
    if time.monotonic()-start>a.seconds:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     raise TimeoutError('Renderer wall-time cap')
    time.sleep(1)
  result.update(renderer_exit_code=child.returncode,renderer_seconds=time.monotonic()-start,
                sampled_process_tree_peak_rss_bytes=peak)
  if child.returncode:raise RuntimeError('Renderer failed; preserve log and any completed views')
  sys.path.insert(0,str(source))
  from src.scene_observation import VIEW_ORDER,validate_scene_views
  paths=[out/f'step_006.{v}.png' for v in VIEW_ORDER]
  if not all(p.is_file() for p in paths):raise RuntimeError('Missing output view')
  # Execute the exact function body, avoiding src.runtime and Agent imports.
  # Its only Scene dependencies are the actual serialized state and metadata.
  original=source/'src/render_observation.py'
  manifest=json.loads((source/'manifest.json').read_text())
  expected=next(x['sha256'] for x in manifest['sources'] if x['path']=='src/render_observation.py')
  if sha(original)!=expected:raise ValueError('Annotation source changed')
  tree=ast.parse(original.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='annotate_live_topdown')
  ns={'Path':Path};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(original),'exec'),ns)
  actual=json.loads(state.read_text())
  scene=SimpleNamespace(scene_dir=Path(actual['scene_dir']),room_id=actual.get('room_id','room'),
                        to_state_dict=lambda:copy.deepcopy(actual))
  result['unannotated_topdown_sha256']=sha(paths[0])
  ns['annotate_live_topdown'](scene,paths[0])
  validate_scene_views(paths)
  from PIL import Image
  rows=[]
  for view,p in zip(VIEW_ORDER,paths):
   with Image.open(p) as im:
    rgb=im.convert('RGB');rows.append({'view':view,'path':str(p),'size':list(rgb.size),
       'bytes':p.stat().st_size,'sha256':sha(p),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
  result.update(status='native_five_views_rendered_and_annotated',images=rows,
                annotation_function_sha256=hashlib.sha256(ast.get_source_segment(original.read_text(),fn).encode()).hexdigest(),
                native_annotation_applied=True)
 except BaseException as e:
  result.update(status='failed',error=type(e).__name__+': '+str(e));raise
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
  result.update(wall_seconds=time.monotonic()-start,
    input_files_unchanged=all(sha(Path(p))==h for p,h in before.items()),
    training_admitted=False,original_pixels_equivalence='not established',full_scene_success=None)
  (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
