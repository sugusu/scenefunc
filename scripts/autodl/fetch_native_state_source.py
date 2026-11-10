"""Recover pinned SceneSmith state-loading modules only, with bounded downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.request

PIN='e888297a8b0a2f3acbb8634edf9def59cceab836'
FILES=['LICENSE','scenesmith/__init__.py','scenesmith/agent_utils/__init__.py',
 'scenesmith/utils/__init__.py','scenesmith/agent_utils/house.py',
 'scenesmith/agent_utils/room.py','scenesmith/agent_utils/sceneeval_exporter.py',
 'scenesmith/agent_utils/support_surface_extraction.py','scenesmith/utils/material.py',
 'scenesmith/utils/package_utils.py','scenesmith/utils/path_utils.py',
 'scenesmith/utils/geometry_utils.py','scenesmith/utils/sdf_utils.py']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);rows=[];total=0
 try:
  for name in FILES:
   if shutil.disk_usage(a.output).free<5*1024**3+1024**2:raise RuntimeError('Disk reserve')
   url=f'https://raw.githubusercontent.com/nepfaff/scenesmith/{PIN}/{name}'
   try:
    with urllib.request.urlopen(url,timeout=30) as r:raw=r.read(1024**2+1)
   except urllib.error.HTTPError as e:
    if e.code==404 and name.endswith('/__init__.py'):
     rows.append({'path':name,'status':'absent_upstream_namespace_package','url':url});continue
    raise
   if len(raw)>1024**2 or total+len(raw)>3*1024**2:raise RuntimeError('Source limit')
   p=a.output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);total+=len(raw)
   rows.append({'path':name,'status':'downloaded','url':url,'bytes':len(raw),
    'sha256':hashlib.sha256(raw).hexdigest()})
   print(name,len(raw),flush=True)
 finally:
  (a.output/'manifest.json').write_text(json.dumps({'pin':PIN,'files':rows,
   'downloaded_bytes':total,'source_edited':False,'model_weights_downloaded':False},indent=2)+'\n')

if __name__=='__main__':main()
