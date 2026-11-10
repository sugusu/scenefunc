"""Reconstruct the renderer actually fingerprinted by the old checkpoint.

The released repository alone is insufficient: the experiment applied four
published observation patches. Require exact final hashes, not visual guesses.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['source','patches','imported','export','out']:
  ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();shutil.copytree(a.source,a.out)
 manifest=json.loads((a.out/'manifest.json').read_text())
 checkpoint=json.loads((a.imported/'session_checkpoint.json').read_text())
 fingerprint=checkpoint['runtime_fingerprint'];rows=[]
 for name in ['cardinal_world_panels.patch','native_structure_labels.patch',
              'native_structure_views.patch','native_compact_labels.patch']:
  p=(a.patches/name).resolve()
  proc=subprocess.run(['patch','--batch','--forward','--fuzz=0','-p1','-i',str(p)],
                      cwd=a.out,text=True,capture_output=True)
  rows.append({'path':str(p),'sha256':sha(p),'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
  (a.out/'checkpoint_patch_application.json').write_text(json.dumps(rows,indent=2)+'\n')
  if proc.returncode:raise RuntimeError('Checkpoint patch failed')
 checked=[]
 for path in ['src/render_observation.py','src_critic/pil_annotate.py',
              'tools/render_five_views.py','tools/render_cardinal_views.py']:
  expected=fingerprint['designer/'+path];actual=sha(a.out/path)
  checked.append({'path':path,'actual_sha256':actual,'checkpoint_sha256':expected,'exact':actual==expected})
  if actual!=expected:raise ValueError('Renderer differs from frozen runtime: '+path)
  original=next(r['sha256'] for r in manifest['sources'] if r['path']==path)
  manifest['patches'].append({'path':path,'before_sha256':original,'after_sha256':actual,
                             'origin':'four published runs/920 observation patches; checkpoint verified'})
 (a.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 sys.path.insert(0,str(a.out.resolve()))
 from scenesmith.agent_utils.house import HouseScene
 state=json.loads((a.imported/'house_state.json').read_text())
 meta=json.loads((a.imported/'manifest.json').read_text())
 house=HouseScene.from_state_dict(state,house_dir=a.imported.resolve())
 scene=house.rooms[meta['selected_room_id']];scene._scenefunc_structure_labels=True
 export=json.loads((a.export/'result.json').read_text())
 if canonical(scene.to_state_dict())!=export['runtime_room_sha256']:raise ValueError('Scene changed')
 blend=(a.export/'step_006.blend').resolve()
 if sha(blend)!=export['blend_sha256']:raise ValueError('Blend changed')
 # Execute the exact sidecar-producing statement from the restored source;
 # skip the expensive, already completed native export and renderer call.
 original=a.out/'src/render_observation.py';tree=ast.parse(original.read_text())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='render_scene_views')
 statement=next(n for n in fn.body if isinstance(n,ast.If) and '_scenefunc_structure_labels' in ast.unparse(n.test))
 ns={'scene':scene,'blend':blend,'Path':Path}
 exec(compile(ast.Module(body=[statement],type_ignores=[]),str(original),'exec'),ns)
 sidecar=blend.with_suffix('.native_structure.json')
 result={'status':'exact_checkpoint_renderer_recovered','files':checked,
  'checkpoint_sha256':sha(a.imported/'session_checkpoint.json'),
  'native_structure_path':str(sidecar),'native_structure_sha256':sha(sidecar),
  'state_unchanged':canonical(scene.to_state_dict())==export['runtime_room_sha256'],
  'blend_unchanged':sha(blend)==export['blend_sha256'],
  'geometry_edits':0,'renderer_calls':0,'model_calls':0,
  'old_library_versions_reproduced':False,'original_pixel_equivalence':'not yet tested'}
 (a.out/'checkpoint_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))

if __name__=='__main__':main()
