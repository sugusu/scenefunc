"""Recover pinned renderer sources, retaining the author's integration patch.

Downloads source only. Does not install packages, load models or start services.
The original state-loader source tree is copied, never edited in place.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

SMITH = 'e888297a8b0a2f3acbb8634edf9def59cceab836'
DESIGNER = 'a0565e50ed7d32c0e09132b9311c0b9b91165405'
FILES = [
 'scenesmith/agent_utils/rendering.py',
 'scenesmith/agent_utils/drake_utils.py',
 'scenesmith/agent_utils/blender/__init__.py',
 'scenesmith/agent_utils/blender/server_manager.py',
 'scenesmith/agent_utils/blender/server_app.py',
 'scenesmith/agent_utils/blender/renderer.py',
 'scenesmith/agent_utils/blender/surface_utils.py',
] + ['scenesmith/agent_utils/blender/'+name+'.py' for name in (
 'annotations','camera_utils','canonicalization','coordinate_frame',
 'coordinate_grid_mixin','image_overlays','mesh_conversion','params',
 'render_dataclasses','render_settings','scene_setup_mixin','scene_utils',
 'standalone_server','surface_rendering_mixin','view_generation_mixin','wall_utils'
)] + ['scenesmith/agent_utils/mesh_utils.py','scenesmith/utils/network_utils.py',
      'scenesmith/utils/print_utils.py']
DESIGNER_FILES = ['integration/scenesmith.patch', 'scripts/export_scene_worker.py',
 'tools/render_five_views.py', 'tools/render_cardinal_views.py',
 'src_critic/blender_render.py', 'src_critic/pil_annotate.py',
 'src_critic/scene_io.py', 'src_critic/perturb.py',
 'src/render_observation.py', 'src/scene_observation.py']

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--extra-smith',action='append',default=[])
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 previous=Path('runs/autodl/20261110_native_state_runtime/source')
 manifest_path=a.out/'manifest.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'sources':[], 'patches':[]}
 known={r['path'] for r in manifest['sources']}
 def record(path,raw,**meta):
  dest=a.out/path;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and dest.read_bytes()!=raw:raise RuntimeError('Existing source differs: '+path)
  dest.write_bytes(raw)
  manifest['sources'].append(dict(path=path,bytes=len(raw),sha256=sha(raw),**meta))
  known.add(path)
 def fetch(repo,pin,path):
  if path in known:return
  if '..' in Path(path).parts or Path(path).is_absolute():raise ValueError(path)
  url=f'https://raw.githubusercontent.com/{repo}/{pin}/{path}'
  with urllib.request.urlopen(url,timeout=40) as r:raw=r.read(2*1024**2+1)
  if len(raw)>2*1024**2:raise RuntimeError('Unexpected source size')
  if shutil.disk_usage(a.out).free-len(raw)<5*1024**3:raise RuntimeError('Disk reserve')
  record(path,raw,url=url,pin=pin)
 try:
  for r in json.loads((previous/'manifest.json').read_text())['files']:
   if r['status']!='downloaded' or r['path'] in known:continue
   raw=(previous/r['path']).read_bytes()
   if sha(raw)!=r['sha256']:raise RuntimeError('Original source hash changed')
   record(r['path'],raw,url=r['url'],pin=SMITH)
  for path in dict.fromkeys(FILES+a.extra_smith):fetch('nepfaff/scenesmith',SMITH,path)
  for path in DESIGNER_FILES:fetch('ZJU-OmniAI/Scene-Designer',DESIGNER,path)
  # Select only integration patches for modules actually present. This is the
  # author's patch verbatim, not an equivalent rewrite of renderer behavior.
  full=(a.out/'integration/scenesmith.patch').read_text()
  blocks=[];included=[]
  # The released integration file uses ordinary unified diffs, not Git's
  # optional diff --git header. Require matching a/b paths before selecting.
  for block in re.split(r'(?m)(?=^--- a/)',full):
   if not block.startswith('--- a/'):continue
   lines=block.splitlines();path=lines[0][6:]
   if lines[1]!='+++ b/'+path:raise ValueError('Unexpected patch path pair')
   if (a.out/path).is_file() and path not in {x['path'] for x in manifest['patches']}:
    blocks.append(block);included.append(path)
  if blocks:
   selected=a.out/'selected_integration.patch';selected.write_text(''.join(blocks))
   before={p:sha((a.out/p).read_bytes()) for p in included}
   proc=subprocess.run(['patch','--batch','--forward','-p1','-i',str(selected.resolve())],
                       cwd=a.out,text=True,capture_output=True)
   (a.out/'patch.log').write_text(proc.stdout+proc.stderr)
   if proc.returncode:raise RuntimeError('Author integration patch did not apply cleanly')
   for p in included:manifest['patches'].append(dict(path=p,before_sha256=before[p],
      after_sha256=sha((a.out/p).read_bytes()),origin_patch_sha256=sha(full.encode())))
  manifest['status']='sources_ready'
 except BaseException as e:
  manifest['status']='incomplete';manifest['error']=type(e).__name__+': '+str(e);raise
 finally:
  manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({'status':manifest.get('status'),'sources':len(manifest['sources']),
                    'patched':len(manifest['patches'])}))

if __name__=='__main__':main()
