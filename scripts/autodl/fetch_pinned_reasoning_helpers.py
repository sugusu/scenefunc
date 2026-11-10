"""Reacquire reviewed prior code into a local research cache; never run/install it."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main(manifest,out):
    data=json.loads(manifest.read_text());out.mkdir(parents=True,exist_ok=False)
    receipts=[];total=0
    for item in data['files']:
        rel=Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe source path')
        if total+item['bytes']>500000:raise ValueError('Fixed source acquisition cap')
        with urllib.request.urlopen(item['URL'],timeout=30) as f:value=f.read(item['bytes']+1)
        assert len(value)==item['bytes']
        assert hashlib.sha256(value).hexdigest()==item['sha256']
        p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
        total+=len(value);receipts.append({'path':str(p),'sha256':item['sha256']})
    (out/'acquisition_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({'pin':data['pin'],'source_bytes':total,
                      'installed_or_executed':False,'cache':str(out)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.manifest.resolve(),a.out.resolve())
