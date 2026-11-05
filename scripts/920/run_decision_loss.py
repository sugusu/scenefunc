"""Bounded frozen forward loss; no optimizer, backward, generation or training."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from native_project import sha256


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--encoding',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.plan.read_text());enc=json.loads(a.encoding.read_text())
    a.out.mkdir(parents=True,exist_ok=False);pub=Path(cfg['public_directory'])
    now=lambda:datetime.now(timezone.utc).isoformat()
    result={'status':'running','started_at':now(),'plan_sha256':sha256(a.plan),
            'encoding_sha256':sha256(a.encoding),'rows':[],'planned_unique_forwards':30,
            'verification_forwards':0,'optimizer_updates':0,'training_authorized':False}
    locks=[]
    def progress():
        save(a.out/'RESULT.json',result)
        save(pub/'STATUS.json',{'run_id':cfg['run_id'],'question':cfg['question'],
             'stage':'frozen_forward_loss','status':result['status'],'updated_at':now(),
             'started_at':result['started_at'],'ended_at':result.get('ended_at'),
             'actual_exit_code':result.get('actual_exit_code'),
             'completed_planned':{'unique_source_forwards':len(result['rows']),'planned':30},
             'configuration_hashes':{'plan':sha256(a.plan),'encoding':sha256(a.encoding)},
             'latest_verifiable_artifact':str(a.out/'RESULT.json'),'failure_reason':result.get('error'),
             'next_step':'Report actual losses and source/task-format limitations; do not train.'})
        save(pub.parent/'latest.json',{'run_id':cfg['run_id'],'stage':'frozen_forward_loss',
             'status':result['status'],'updated_at':now(),'run_directory':cfg['runtime_directory'],
             'status_path':str(pub/'STATUS.json')})
    print(now(),'start frozen loss; no optimizer or generation',flush=True);progress()
    try:
        assert enc['plan_sha256']==sha256(a.plan) and len(enc['records'])==30
        for name,digest in cfg['scripts'].items():assert sha256(Path(__file__).parent/name)==digest,name
        for name,digest in cfg['model_configuration_sha256'].items():assert sha256(Path(cfg['model_directory'])/name)==digest
        for row in cfg['model_weight_files']:
            p=Path(cfg['model_directory'])/row['path']
            assert p.stat().st_size==row['bytes'] and sha256(p)==row['sha256']
        for p in cfg['gpu_lock_paths']:
            lock=Path(p).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        def query(q):return subprocess.check_output(['nvidia-smi','-i','0',q,'--format=csv,noheader,nounits'],text=True,timeout=15).strip()
        uuid,used,free=query('--query-gpu=uuid,memory.used,memory.free').split(',')
        assert uuid.strip()==cfg['gpu_uuid'] and int(used)<512 and int(free)>23000 and not query('--query-compute-apps=pid')
        result['initial_gpu']={'uuid':uuid.strip(),'used_mib':int(used),'free_mib':int(free)}
        os.environ.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        import torch
        import transformers
        from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration
        torch.set_num_threads(4);torch.manual_seed(0)
        start=time.monotonic()
        model=Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_directory'],local_files_only=True,
            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,attn_implementation='sdpa')
        model.eval();model.requires_grad_(False)
        result.update(load_seconds=time.monotonic()-start,torch_version=torch.__version__,transformers_version=transformers.__version__,
                      parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      dtype='bfloat16',attention='sdpa',use_cache=False)
        tokenizer=AutoTokenizer.from_pretrained(cfg['model_directory'],local_files_only=True)
        def selected(batch):
            inputs={k:v.to('cuda:0') for k,v in batch.items() if k!='labels'}
            labels=batch['labels'][0].to('cuda:0');pos=(labels!=-100).nonzero().flatten()
            assert len(pos) and pos.min()>0 and torch.equal(labels[pos],inputs['input_ids'][0,pos])
            model.model.rope_deltas=None
            logits=model(**inputs,logits_to_keep=pos-1,use_cache=False).logits[0]
            loss=torch.nn.functional.cross_entropy(logits.float(),labels[pos],reduction='none')
            return loss,labels[pos]
        with torch.inference_mode():
            # Check the selected-position implementation against the model's standard
            # causal loss, including a masked internal gap, on a short text fixture.
            tiny=tokenizer('Move a table while keeping the route to the door clear. Preserve every required object.',return_tensors='pt')
            tiny['labels']=tiny['input_ids'].clone();tiny['labels'][:,:4]=-100;tiny['labels'][:,8:11]=-100
            model.model.rope_deltas=None
            official=model(**{k:v.to('cuda:0') for k,v in tiny.items()},use_cache=False).loss.item()
            losses,_=selected(tiny);manual=losses.mean().item()
            result['verification_forwards']=2
            result['causal_loss_check']={'standard_loss':official,'selected_loss':manual,'absolute_difference':abs(official-manual),
                'tolerance':1e-5,'passed':abs(official-manual)<=1e-5,'scope':'short text fixture with masked prefix and internal gap'}
            assert result['causal_loss_check']['passed'];del losses,tiny
            progress()
            for record in enc['records']:
                path=Path(record['tensor_path']);assert sha256(path)==record['tensor_sha256']
                batch=torch.load(path,map_location='cpu',weights_only=True)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                losses,targets=selected(batch);torch.cuda.synchronize()
                assert torch.isfinite(losses).all() and targets.tolist()==record['target_ids']
                row={k:record[k] for k in ('id','decision_id','component','arm','family_id','source_decision','source_principle_in_reason','input_tokens','target_count','tensor_sha256')}
                row.update(nll_sum=losses.sum().item(),mean_nll=losses.mean().item(),token_nll=losses.tolist(),target_ids=targets.tolist(),
                    elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                result['rows'].append(row);save(a.out/(record['id']+'.json'),row)
                del losses,targets,batch
                progress();print(now(),record['id'],row['mean_nll'],flush=True)
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        result.update(status='completed',actual_exit_code=0,ended_at=now(),all_gradients_none=True)
    except BaseException as exc:
        result.update(status='failed',actual_exit_code=1,ended_at=now(),error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        progress()
        for lock in locks:lock.close()
    print(now(),'completed',len(result['rows']),'unique source forwards',flush=True)


if __name__=='__main__':main()
