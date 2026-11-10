"""Fetch only the pinned, small author source files used by the route probe.

No repository clone, dependencies, datasets, models, or paid API. Sources are
read as data by the probe; this downloader never imports downloaded code.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

SOURCES = [
    ('Oscar-Yang11/ERVLA', '4e574e3f40a02678d958e092b41e91fb6b40a1de', [
        'README.md', 'src/data/datasets/json_dataset.py',
        'src/data/transforms/load.py', 'src/data/transforms/transform.py',
        'src/models/VLA/P2_Choice_KI.py', 'src/server/runtime/process_choice_nocot.py',
        'configs/data/libero/libero_all_qwen3vl_choice_CoT_fromPT.yaml',
        'configs/data/sources/libero_cot.yaml']),
    ('MichalZawalski/embodied-CoT', '1813ad76001f1e08095088f94a86c43fc0e457a3', [
        'README.md', 'LICENSE', 'prismatic/util/cot_utils.py',
        'prismatic/vla/datasets/datasets.py', 'prismatic/extern/hf/modeling_prismatic.py']),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    receipts = []
    for repo, pin, paths in SOURCES:
        for path in paths:
            if shutil.disk_usage(a.out).free < 5*1024**3+1024**2:
                raise RuntimeError('Data reserve would be crossed')
            url = 'https://api.github.com/repos/'+repo+'/contents/'+path+'?ref='+pin
            req = urllib.request.Request(url, headers={'User-Agent':'scene-research-readonly'})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read(2*1024**2+1)
            if len(raw)>2*1024**2:
                raise ValueError('Unexpected large source response')
            response = json.loads(raw)
            content = base64.b64decode(response['content'], validate=False)
            if len(content)>1024**2:
                raise ValueError('Unexpected large code file')
            # GitHub's blob id binds bytes independently of base64 transport.
            blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if blob != response['sha']:
                raise ValueError('Source Git blob mismatch')
            target = a.out/'source'/repo.split('/')[1]/path
            target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(content)
            receipts.append(dict(repository=repo,commit=pin,path=path,url=url,
                sha256=hashlib.sha256(content).hexdigest(),git_blob=blob,
                bytes=len(content),local_path=str(target),executed=False))
            (a.out/'source_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
            print(path,len(content),flush=True)


if __name__=='__main__':main()
