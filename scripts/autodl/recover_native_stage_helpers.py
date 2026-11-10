"""Fetch two missing small, pinned ORIGINAL downstream stage helpers.

Keep them outside the closed runtime/protocol. No model, service or training.
Their real import/function check uses the already qualified controller roots.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import urllib.request


def main(stage):
    stage=stage.resolve();out=stage/'next_stage_sources';out.mkdir(exist_ok=False)
    tree=json.loads((stage/'designer_tree.json').read_text())
    pin='a0565e50ed7d32c0e09132b9311c0b9b91165405'
    index={r['path']:r for r in tree['tree']};rows=[]
    for rel in ('src/room_summary.py','src/grpo_rollout.py'):
        r=index[rel];url=f'https://raw.githubusercontent.com/ZJU-OmniAI/Scene-Designer/{pin}/{rel}'
        assert r['size']<64*1024
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read(r['size']+1)
        assert len(data)==r['size']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==r['sha']
        p=out/Path(rel).name;p.write_bytes(data)
        rows.append(dict(path=str(p),original_path=rel,url=url,git_blob=r['sha'],
            bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (out/'source_manifest.json').write_text(json.dumps(dict(pin=pin,files=rows,
        source_read_scope='Acquired verified files; downloading is not full reading',
        current_closed_runtime_modified=False),indent=2)+'\n')
    print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),pin=pin)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    main(p.parse_args().stage)
