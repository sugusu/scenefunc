"""Archive one closed gate cell without rerunning a model or native edit."""
import argparse,json,hashlib,shutil
from pathlib import Path
from native_checkpoint import canonical_hash
p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();P=a.public.resolve();R=Path(__file__).resolve().parents[2];T=R.parent/'runtime'/P.name;A=a.arm
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
x=read(T/A/'RESULT.json');assert x['status']=='completed_pending_review';assert read(T/(A+'_receipt')/'receipt.json')['exit_code']==0
out=P/A/'evidence';out.mkdir(exist_ok=False);copies=[]
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(src)==sha(dst);copies.append(dict(source=str(src),destination=str(dst.relative_to(P)),sha256=sha(dst),bytes=dst.stat().st_size))
for f in (T/A).glob('*.json'):cp(f,out/f.name)
for rd in sorted((T/A).glob('round_*')):
 if not rd.is_dir():continue
 for rel in ['PLAN.json','request.json','REJECTION.json','receipt/receipt.json','receipt/stage.log','model/RESULT.json','model/ENCODING.json','model/raw_output.txt','model/output_ids.json']:
  if (rd/rel).exists():cp(rd/rel,out/rd.name/rel)
for rel in ['receipt.json','stage.log']:cp(T/(A+'_receipt')/rel,out/'execution'/rel)
for kind in ['restore','export']:
 result=read(T/(A+'_'+kind)/'RESULT.json');assert result['status'] in ['passed','geometry_export_roundtrip_checked']
 cp(T/(A+'_'+kind)/'RESULT.json',out/(kind.upper()+'_RESULT.json'))
 cp(T/(A+'_'+kind+'_receipt')/'receipt.json',out/(kind.upper()+'_RECEIPT.json'))
bundle=T/A/'final_checkpoint'
for name in ['manifest.json','house_state.json','runtime.json','task.json','session_checkpoint.json']:
 cp(bundle/name,out/'checkpoint'/name)
# Preserve every actual observation binding and image hash; publish final views.
views=[]
for f in sorted((T/A).glob('*_observation.json')):
 obs=read(f);bound=read(Path(obs['state_binding']));state=read(T/A/(f.name.replace('_observation','_state')))
 assert canonical_hash(state)==canonical_hash(bound['native_state'])
 assert len(obs['image_paths'])==5
 row=dict(observation=f.name,state_sha256=canonical_hash(state),images=[])
 for i,path in enumerate(obs['image_paths']):
  path=Path(path);row['images'].append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
  if f.name=='final_observation.json' and i in [0,1]:cp(path,P/A/'views'/path.name)
 cp(Path(obs['state_binding']),out/'bindings'/(f.stem+'.json'));views.append(row)
initial=read(T/'prepare/RESULT.json');assert x['initial_request_sha256']==initial['initial_request_sha256']
for rd in sorted((T/A).glob('round_*')):
 if rd.is_dir():
  encoding=read(rd/'model/ENCODING.json')
  for img in encoding['images']:assert sha(Path(img['origin']))==img['sha256']
row=dict(arm=A,status=x['status'],model_calls=x['model_calls'],native_calls=x['native_calls'],rejected=sum('rejection' in r for r in x['rounds']),actual_changes=sum(r.get('before_state_sha256')!=r.get('after_state_sha256') for r in x['rounds'] if 'reply' in r),explicit_done=x['stop_reason']=='model_done',gpu_subprocess_seconds=x['gpu_subprocess_seconds'],first_input_sha256=x['initial_request_sha256'],score=x['final_score'],checkpoint=x['final_checkpoint'],restore=read(T/(A+'_restore')/'RESULT.json'),export=read(T/(A+'_export')/'RESULT.json')['glb'])
write(P/A/'REVIEW_ROW.json',row);write(P/A/'ARCHIVE.json',dict(copies=copies,views=views,copy_hashes_verified=True,actual_five_image_bindings_verified=True,all_actual_model_image_bytes_verified=True,first_input_matches_preparation=True))
print(json.dumps(row,ensure_ascii=False,indent=2))
