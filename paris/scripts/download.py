"""Pinned allowlist downloader. Default is a no-network disk plan."""
import argparse, hashlib, json, os, shutil, urllib.error, urllib.request, urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ProviderRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  nxt=super().redirect_request(req,fp,code,msg,headers,newurl)
  if nxt and urllib.parse.urlparse(newurl).hostname!=urllib.parse.urlparse(req.full_url).hostname:
   nxt.remove_header('Authorization')
  return nxt
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def entries(kind,scope):
 if kind=='weights':
  x=read(ROOT/'config/weights.lock.json')
  return [dict(f,path='weights/'+f['path'],url=f"https://huggingface.co/{x['repository']}/resolve/{x['revision']}/{f['path']}") for f in x['files']]
 if kind=='upstream':return [dict(f,path='upstream/'+f['path']) for f in read(ROOT/'config/upstream.lock.json')['files']]
 result=[]
 for x in read(ROOT/'config/assets.lock.json')['rows']:
  if scope=='selected' and not x['selected_records']:continue
  result.append(dict(path='hssd-models/'+x['path'],bytes=x['bytes'],sha256=x['sha256'],url=f"https://huggingface.co/datasets/{x['repository']}/resolve/{x['revision']}/{x['path']}"))
  if x.get('frame'):
   f=x['frame'];result.append(dict(path='frames/'+x['id']+'.json',bytes=f['bytes'],sha256=f['sha256'],url=f"https://huggingface.co/datasets/{f['repository']}/resolve/{f['revision']}/{f['path']}"))
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['weights','assets','upstream']);ap.add_argument('--scope',choices=['full','selected'],default='full');ap.add_argument('--data',type=Path,required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--accept-hssd',action='store_true');ap.add_argument('--verify-only',action='store_true');a=ap.parse_args()
 files=entries(a.kind,a.scope);total=sum(x['bytes'] for x in files)
 print(json.dumps({'kind':a.kind,'scope':a.scope,'files':len(files),'total_bytes':total,'minimum_free_bytes_for_empty_cache':total+5*2**30,'network':a.execute,'no_duplicate_hf_cache':True}))
 if not a.execute and not a.verify_only:return
 if a.kind=='assets' and a.execute and not a.accept_hssd:raise SystemExit('Accept HSSD terms with your own authorized provider account, then pass --accept-hssd. No redistribution in Git.')
 a.data.mkdir(parents=True,exist_ok=True)
 for f in files:
  path=a.data/f['path'];assert path.resolve().is_relative_to(a.data.resolve());path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():
   assert path.stat().st_size==f['bytes'] and digest(path)==f['sha256'],f"Existing file mismatch: {f['path']}"
   continue
  if a.verify_only:raise SystemExit('Missing '+f['path'])
  part=path.with_suffix(path.suffix+'.part');offset=part.stat().st_size if part.exists() else 0
  if offset>f['bytes']:raise SystemExit('Oversized partial '+f['path'])
  if shutil.disk_usage(a.data).free < f['bytes']-offset+5*2**30:raise SystemExit('Would cross 5 GiB reserve')
  if offset<f['bytes']:
   headers={'Range':f'bytes={offset}-','User-Agent':'scenefunc-paris-reproduction'}
   token=os.environ.get('HF_TOKEN')
   if token and f['url'].startswith('https://huggingface.co/'):headers['Authorization']='Bearer '+token
   try:
    with urllib.request.build_opener(ProviderRedirect()).open(urllib.request.Request(f['url'],headers=headers),timeout=90) as response:
     if response.status==206:
      assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
     elif response.status==200:offset=0
     else:raise ValueError('Unexpected response status')
     with part.open('ab' if offset else 'wb') as out:
      while block:=response.read(1024*1024):
       if out.tell()+len(block)>f['bytes']:raise ValueError('Response exceeds pinned byte size')
       out.write(block)
   except urllib.error.HTTPError as e:raise SystemExit(f'{f["path"]}: HTTP {e.code}; check provider authorization/network. No credentials printed.')
  assert part.stat().st_size==f['bytes'] and digest(part)==f['sha256'],f"Hash/size mismatch: {f['path']}"
  part.replace(path)
 if a.kind=='weights':
  lock=read(ROOT/'config/weights.lock.json');idx=read(a.data/'weights/model.safetensors.index.json')
  assert set(idx['weight_map'].values())=={x['path'] for x in lock['files'] if x['kind']=='weights'}
 print(json.dumps({'verified':len(files),'bytes':total}))
if __name__=='__main__':main()
