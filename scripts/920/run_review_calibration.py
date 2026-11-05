"""Bounded offline review calibration. No training, scene edits, or output retries."""
import argparse, fcntl, hashlib, json, os, re, socket, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def parse_answer(text):
 text=text.strip()
 if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4]
 answer=json.loads(text)
 if not isinstance(answer,dict):raise ValueError('Answer must be an object')
 assert set(answer['scores'])=={'Flow','Functionality','Overall'}
 assert all(v is None or type(v) in (int,float) and 1<=v<=10 for v in answer['scores'].values())
 assert isinstance(answer['missing_functions'],list) and len(set(answer['missing_functions']))==len(answer['missing_functions'])
 assert set(answer['missing_functions'])<={'living_room','bedroom','bathroom','kitchen','balcony'}
 assert isinstance(answer['issues'],list) and len(answer['issues'])<=3
 assert all(isinstance(x,dict) and all(isinstance(x[k],str) for k in ['region','evidence','consequence','certainty']) for x in answer['issues'])
 assert isinstance(answer['unknowns'],list) and all(isinstance(x,str) for x in answer['unknowns'])
 return answer

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode',choices=['encode','generate'],required=True)
 a=ap.parse_args();plan=read(a.plan);a.out.mkdir(parents=True,exist_ok=False);locks=[];start=time.monotonic()
 result={'status':'running','mode':a.mode,'started_at':now(),'plan_sha256':sha(a.plan),'model_calls':0,'training_updates':0,'scene_actions':0,'records':[]}
 def progress():
  save(a.out/'RESULT.json',result)
  status=read(plan['status_path']);status.update(status=result['status'],phase=a.mode,updated_at=now(),completed=len(result['records']) if a.mode=='generate' else 0,planned=len(plan['requests']),actual_model_calls=result['model_calls'],latest_verifiable_artifact=str(a.out/'RESULT.json'));save(plan['status_path'],status)
 print(now(),'begin bounded review',a.mode,flush=True);progress()
 try:
  for name,digest in plan['code_sha256'].items():assert sha(Path(__file__).parent/name)==digest,name
  assert sha(plan['encoder'])==plan['encoder_sha256']
  for name,digest in plan['model_configuration_sha256'].items():assert sha(Path(plan['model_directory'])/name)==digest,name
  os.environ.update(CUDA_VISIBLE_DEVICES=plan['gpu_uuid'] if a.mode=='generate' else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
  os.environ.pop('LD_LIBRARY_PATH',None)
  os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
  if a.mode=='generate':
   for path in plan['gpu_lock_paths']:
    lock=Path(path).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
   def query(q):return subprocess.check_output(['nvidia-smi','-i','0',q,'--format=csv,noheader,nounits'],text=True,timeout=15).strip()
   gpu=[x.strip() for x in query('--query-gpu=uuid,memory.used,memory.free').split(',')]
   required=17534247392+(plan['budget']['max_input_tokens']+plan['budget']['max_new_tokens'])*147456+4*1024**3
   assert gpu[0]==plan['gpu_uuid'] and int(gpu[1])<=512 and not query('--query-compute-apps=pid') and int(gpu[2])*1024**2>=required
   host=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
   assert host>=17534247392*2+8*1024**3
   result['gpu_before']={'uuid':gpu[0],'used_mib':int(gpu[1]),'free_mib':int(gpu[2]),'required_free_bytes':required,'host_available_bytes':host}
  def deny(*args,**kwargs):raise RuntimeError('Network disabled during offline review')
  socket.socket.connect=deny;socket.create_connection=deny
  sys.path.insert(0,str(Path(plan['encoder']).parent));from local_vlm import LocalQwenEngine
  import torch
  torch.set_num_threads(4);engine=LocalQwenEngine(plan['model_directory'],a.out/'unused_engine_calls',max_pixels=plan['budget']['max_pixels'],max_input_tokens=plan['budget']['max_input_tokens'],max_new_tokens=plan['budget']['max_new_tokens'])
  if a.mode=='encode':
   result['weight_stat']={}
   for row in plan['model_weight_files']:
    path=Path(plan['model_directory'])/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'];st=path.stat();result['weight_stat'][row['path']]=[st.st_size,st.st_mtime_ns]
   image_hashes={}
   for req in plan['requests']:
    assert sha(req['path'])==req['sha256'];directory=a.out/req['id'];directory.mkdir();inputs,receipt=engine.encode(read(req['path']));assert len(receipt['images'])==1
    torch.save(dict(inputs),directory/'inputs.pt');save(directory/'ENCODING.json',receipt)
    image_digest=hashlib.sha256(inputs['pixel_values'].numpy().tobytes()+inputs['image_grid_thw'].numpy().tobytes()).hexdigest()
    if req['record_id'] in image_hashes:assert image_digest==image_hashes[req['record_id']]
    image_hashes[req['record_id']]=image_digest
    result['records'].append({'id':req['id'],'record_id':req['record_id'],'arm':req['arm'],'input_tokens':receipt['input_tokens'],'tensor_path':str(directory/'inputs.pt'),'tensor_sha256':sha(directory/'inputs.pt'),'image_tensor_sha256':image_digest,'request_sha256':req['sha256']});progress()
   assert not torch.cuda.is_initialized();result.update(status='encoded_without_model',unique_images=len(image_hashes),pair_image_tensors_equal=True,cuda_initialized=False)
  else:
   enc=read(Path(plan['encoding_directory'])/'RESULT.json');assert enc['status']=='encoded_without_model' and enc['plan_sha256']==sha(a.plan)
   for name,expected in enc['weight_stat'].items():
    st=(Path(plan['model_directory'])/name).stat();assert [st.st_size,st.st_mtime_ns]==expected
   indexed={x['id']:x for x in enc['records']};torch.manual_seed(0);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.94)
   from transformers import Qwen3VLForConditionalGeneration
   t=time.monotonic();model=Qwen3VLForConditionalGeneration.from_pretrained(plan['model_directory'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True).eval();torch.cuda.synchronize();result['load_seconds']=time.monotonic()-t
   for req in plan['requests']:
    frozen=indexed[req['id']];assert sha(req['path'])==req['sha256']==frozen['request_sha256'];assert sha(frozen['tensor_path'])==frozen['tensor_sha256']
    directory=a.out/req['id'];directory.mkdir();row={'id':req['id'],'record_id':req['record_id'],'arm':req['arm'],'started_at':now(),'status':'running','input_tokens':frozen['input_tokens'],'request_sha256':req['sha256'],'model_calls':1}
    save(directory/'RESULT.json',row);result['model_calls']+=1;result['active_request']=req['id'];progress();print(now(),'generate',req['id'],flush=True)
    inputs=torch.load(frozen['tensor_path'],map_location='cpu',weights_only=True);inputs={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in inputs.items()};torch.cuda.reset_peak_memory_stats();t=time.monotonic()
    with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=plan['budget']['max_new_tokens'],do_sample=False,use_cache=True)
    torch.cuda.synchronize();ids=output[0,frozen['input_tokens']:].cpu().tolist();save(directory/'output_ids.json',ids)
    raw=engine.processor.tokenizer.decode(ids,skip_special_tokens=False);clean=engine.processor.tokenizer.decode(ids,skip_special_tokens=True);(directory/'raw_output.txt').write_text(raw);(directory/'text_output.txt').write_text(clean)
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or [];ended=bool(ids and ids[-1] in eos)
    row.update(ended_at=now(),generation_seconds=time.monotonic()-t,output_tokens=len(ids),ended_with_eos=ended,peak_reserved_bytes=torch.cuda.max_memory_reserved(),raw_sha256=sha(directory/'raw_output.txt'))
    try:
     if not ended:raise ValueError('Output reached cap without EOS')
     answer=parse_answer(clean);save(directory/'answer.json',answer);row['status']='completed'
    except (ValueError,KeyError,TypeError,AssertionError) as e:row.update(status='response_failure',failure=type(e).__name__+': '+str(e))
    save(directory/'RESULT.json',row);result['records'].append(row);result['active_request']=None;progress();del output,inputs
   result['status']='completed' if all(x['status']=='completed' for x in result['records']) else 'completed_with_response_failures'
 except BaseException as e:
  result.update(status='failed',failure=type(e).__name__+': '+str(e))
  if result.get('active_request'):
   failed_path=a.out/result['active_request']/'RESULT.json';failed=read(failed_path);failed.update(status='runtime_failure',ended_at=now(),failure=result['failure']);save(failed_path,failed);result['records'].append(failed)
  raise
 finally:
  result.update(ended_at=now(),elapsed_seconds=time.monotonic()-start);progress()
  for lock in locks:lock.close()
  print(now(),result['status'],'model_calls',result['model_calls'],flush=True)
if __name__=='__main__':main()
