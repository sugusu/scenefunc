"""CPU content integrity + package safety checks, no network/model calls."""
import ast,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 m=json.loads((ROOT/'release_manifest.json').read_text())
 for x in m['files']:
  p=ROOT/x['path'];b=p.read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],str(p)
  assert len(b)<10*1024**2
  assert p.suffix not in ['.glb','.blend','.safetensors','.key','.pem']
  if p.suffix=='.py':ast.parse(b)
  if p.suffix in ['.py','.json','.txt','.csv','.md']:
   t=b.decode();assert not re.search(r'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)',t),str(p)
   assert ('/'+'root/') not in t,str(p)
 print(json.dumps({'files_verified':len(m['files']),'syntax_and_content_scan':True,'network_calls':0,'model_calls':0}))
if __name__=='__main__':main()
