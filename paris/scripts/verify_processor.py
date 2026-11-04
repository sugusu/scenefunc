"""CPU/offline processor and shard-index verification; does not load weights."""
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);a=p.parse_args()
 lock=json.loads((ROOT/'config/weights.lock.json').read_text());idx=json.loads((a.weights/'model.safetensors.index.json').read_text())
 assert set(idx['weight_map'].values())=={f['path'] for f in lock['files'] if f['kind']=='weights'}
 for f in lock['files']:
  q=a.weights/f['path'];assert q.stat().st_size==f['bytes'],f['path']
  if f['kind']!='weights':assert hashlib.sha256(q.read_bytes()).hexdigest()==f['sha256']
 from transformers import AutoProcessor
 from PIL import Image
 proc=AutoProcessor.from_pretrained(a.weights,local_files_only=True,trust_remote_code=False)
 image=Image.open(ROOT/'config/static/paris_A_inference_20261104/common_input_card.png').convert('RGB')
 msg=[{'role':'user','content':[{'type':'image'},{'type':'text','text':'Describe the empty room schematic.'}]}]
 text=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
 output=proc(text=[text],images=[image],return_tensors='pt')
 print(json.dumps({'processor_loaded_offline':True,'input_tokens':output['input_ids'].shape[1],'model_loaded':False,'shard_hash_scope':'Sizes checked here; download.py --verify-only checks all full hashes'}))
if __name__=='__main__':main()
