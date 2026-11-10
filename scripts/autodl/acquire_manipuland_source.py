"""Acquire pinned native small-object stage sources without models or datasets."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/autodl/20261110_native_manipuland_stage'
BASE=Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
PIN='e888297a8b0a2f3acbb8634edf9def59cceab836'

def main():
    OUT.mkdir(exist_ok=True)
    treepath=BASE/'native_room_bridge_20261104/source_tree.json'
    raw=treepath.read_bytes();tree=json.loads(raw)
    assert tree['sha']==PIN and tree['truncated'] is False
    items={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    paths=[
        'scenesmith/experiments/base_experiment.py',
        'scenesmith/experiments/indoor_scene_generation.py',
        'scenesmith/manipuland_agents/base_manipuland_agent.py',
        'scenesmith/manipuland_agents/stateful_manipuland_agent.py',
        'scenesmith/manipuland_agents/tools/manipuland_tools.py',
        'scenesmith/manipuland_agents/tools/vision_tools.py',
        'LICENSE']
    assert sum(items[p]['size'] for p in paths)<300000
    assert shutil.disk_usage(OUT).free>5*1024**3+sum(items[p]['size'] for p in paths)
    def fetch(p):
        dst=OUT/'source'/p;url=f'https://raw.githubusercontent.com/nepfaff/scenesmith/{PIN}/{p}'
        if dst.exists():data=dst.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=30) as r:data=r.read(items[p]['size']+1)
        assert len(data)==items[p]['size']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==items[p]['sha']
        if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
        return {'path':p,'url':url,'git_blob':items[p]['sha'],'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    with ThreadPoolExecutor(max_workers=4) as executor:rows=list(executor.map(fetch,paths))
    (OUT/'source_manifest.json').write_text(json.dumps({'pin':PIN,'tree_sha256':hashlib.sha256(raw).hexdigest(),'files':rows},indent=2)+'\n')
    print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows)}))

if __name__=='__main__':main()
