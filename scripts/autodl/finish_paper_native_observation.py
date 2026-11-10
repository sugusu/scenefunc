"""Annotate completed native views against their actual exported RoomScene.

No rerender or model call. The exact upstream annotation body is executed
without importing the Agent bootstrap. Restoring the RoomScene supplies its
real metadata rather than guessing omitted fields from the serialized dict.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

from native_state_override import apply_recorded_state

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['source','export','views','imported']:
  ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--receipt-name',default='annotation_result.json')
 ap.add_argument('--checkpoint-labels',action='store_true')
 a=ap.parse_args()
 if Path(a.receipt_name).name!=a.receipt_name:raise ValueError('Receipt basename only')
 receipt=a.views/a.receipt_name
 if receipt.exists():raise FileExistsError('Never annotate twice')
 if any(json.loads(p.read_text()).get('status')=='native_five_view_observation_completed'
        for p in a.views.glob('annotation_result*.json')):raise ValueError('Views already annotated')
 start=time.monotonic();result={'status':'started','renderer_calls':0,'model_calls':0}
 export=json.loads((a.export/'result.json').read_text())
 sys.path.insert(0,str(a.source.resolve()))
 try:
  render=json.loads((a.views/'result.json').read_text())
  if render.get('renderer_exit_code')!=0 or len(render.get('images',[]))!=5:
   raise ValueError('Renderer must finish successfully before annotation')
  if render.get('runtime_room_sha256')!=export['runtime_room_sha256']:
   raise ValueError('Rendered state differs from exported state')
  for row in render['images']:
   if sha(Path(row['path']))!=row['sha256']:raise ValueError('Unannotated render changed')
  from scenesmith.agent_utils.house import HouseScene
  from src.scene_observation import VIEW_ORDER,validate_scene_views
  from PIL import Image
  state=json.loads((a.imported/'house_state.json').read_text())
  meta=json.loads((a.imported/'manifest.json').read_text())
  if export.get('frozen_override'):
   record=export['frozen_override']
   state,proof=apply_recorded_state(state,meta['selected_room_id'],ledger=record['ledger'],unit=record['unit_id'])
   if proof!=record:raise ValueError('Frozen state override changed after export')
  house=HouseScene.from_state_dict(state,house_dir=a.imported.resolve())
  scene=house.rooms[meta['selected_room_id']]
  if a.checkpoint_labels:scene._scenefunc_structure_labels=True
  if canonical(scene.to_state_dict())!=export['runtime_room_sha256']:
   raise ValueError('Annotation state differs from exported state')
  original=a.source/'src/render_observation.py'
  manifest=json.loads((a.source/'manifest.json').read_text())
  expected=next(r['sha256'] for r in manifest['sources'] if r['path']=='src/render_observation.py')
  for patch in manifest.get('patches',[]):
   if patch['path']=='src/render_observation.py':expected=patch['after_sha256']
  if sha(original)!=expected:raise ValueError('Original annotation source changed')
  text=original.read_text();fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='annotate_live_topdown')
  ns={'Path':Path};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(original),'exec'),ns)
  paths=[a.views/f'step_006.{view}.png' for view in VIEW_ORDER]
  if not all(p.is_file() for p in paths):raise FileNotFoundError('Missing view')
  before={str(p):sha(p) for p in paths}
  result['unannotated_sha256']=before
  ns['annotate_live_topdown'](scene,paths[0]);validate_scene_views(paths)
  rows=[]
  for view,p in zip(VIEW_ORDER,paths):
   with Image.open(p) as im:
    rgb=im.convert('RGB');rows.append({'view':view,'path':str(p.resolve()),'size':list(rgb.size),
      'sha256':sha(p),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'bytes':p.stat().st_size})
  result.update(status='native_five_view_observation_completed',images=rows,
    unannotated_sha256=before,cardinal_views_unchanged=all(sha(p)==before[str(p)] for p in paths[1:]),
    source_room_sha256=export['source_room_sha256'],runtime_room_sha256=export['runtime_room_sha256'],
    actual_scene_dir=str(scene.scene_dir),original_annotation_source_sha256=expected,
    annotation_function_sha256=hashlib.sha256(ast.get_source_segment(text,fn).encode()).hexdigest())
 except BaseException as e:
  result.update(status='failed',error=type(e).__name__+': '+str(e));raise
 finally:
  result.update(seconds=time.monotonic()-start,training_admitted=False,
    original_pixel_equivalence='not established',full_scene_success=None)
  receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
