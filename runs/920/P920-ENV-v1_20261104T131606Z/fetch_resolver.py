import hashlib,json,datetime,urllib.request
from pathlib import Path
root=Path('/data/wyh/scene_cvpr_20261020/design_decisions_920')
out=root/'resources/runtime_package_cache/resolver';out.mkdir(parents=True,exist_ok=True)
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
start=now();print(start,'fetch fixed resolver metadata/wheel',flush=True)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:19998','https':'http://127.0.0.1:19998'}))
meta_url='https://pypi.org/pypi/uv/0.9.13/json'
with opener.open(meta_url,timeout=45) as response:raw=response.read(2*1024*1024+1)
assert len(raw)<=2*1024*1024
meta=json.loads(raw);assert meta['info']['version']=='0.9.13'
rows=[r for r in meta['urls'] if r['filename'].endswith('-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')]
assert len(rows)==1;row=rows[0];assert row['size']<=50*1024*1024
assert row['url'].startswith('https://files.pythonhosted.org/')
with opener.open(row['url'],timeout=90) as response:data=response.read(50*1024*1024+1)
assert len(data)==row['size'];assert hashlib.sha256(data).hexdigest()==row['digests']['sha256']
p=out/row['filename'];assert p.name==row['filename']
with p.open('xb') as f:f.write(data)
(out/'pypi_uv_0.9.13.json').write_bytes(raw)
receipt={'started_at':start,'ended_at':now(),'metadata_url':meta_url,'metadata_sha256':hashlib.sha256(raw).hexdigest(),'wheel':str(p),'source_url':row['url'],'bytes':len(data),'sha256':row['digests']['sha256'],'status':'verified','exit_code':0}
(out/'FETCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
