"""Frozen final-adapter evaluation on common original action contexts.

No optimizer, backward, fitting, or generation. Training contexts and exact-action
teacher-forcing only: this is not design success, generalization or preference.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from native_project import sha256
from run_decision_loss import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False);pub=Path(cfg['public_directory']);now=lambda:datetime.now(timezone.utc).isoformat();locks=[]
    result=dict(status='running',started_at=now(),plan_sha256=sha256(a.plan),rows=[],planned_forwards=30,optimizer_updates=0,backward_calls=0,generation_calls=0,scene_actions=0)
    def progress():
        save(a.out/'RESULT.json',result)
        save(pub/'STATUS.json',dict(run_id=cfg['run_id'],question=cfg['question'],status=result['status'],updated_at=now(),started_at=result['started_at'],ended_at=result.get('ended_at'),first_log=result.get('first_log'),actual_exit_code=result.get('actual_exit_code'),completed=len(result['rows']),planned=30,configuration_sha256=result['plan_sha256'],input_sha256=cfg['encoding']['sha256'],latest_verifiable_artifact=str(a.out/'RESULT.json'),failure_reason=result.get('error'),next_step='Assess final same-input action likelihood before proposing further training; do not equate it with usable designs.'))
        save(pub.parent/'latest.json',dict(run_id=cfg['run_id'],stage='final_adapter_common_action_evaluation',updated_at=now(),run_directory=cfg['runtime_directory']))
    result['first_log']=now()+' begin frozen final-adapter common action evaluation';print(result['first_log'],flush=True);progress()
    try:
        for path,digest in cfg['code_sha256'].items():assert sha256(Path(path))==digest,path
        assert sha256(Path(cfg['encoding']['path']))==cfg['encoding']['sha256']
        enc=json.loads(Path(cfg['encoding']['path']).read_text());records=[x for x in enc['records'] if x['component']=='action' and x['arm']=='separate'];assert len(records)==10
        assert [x['id'] for x in records]==cfg['record_ids']
        for name,h in cfg['model_configuration_sha256'].items():assert sha256(Path(cfg['model_directory'])/name)==h
        for w in cfg['model_weight_files']:assert sha256(Path(cfg['model_directory'])/w['path'])==w['sha256']
        for adapter in cfg['adapters'].values():
            for name,h in adapter['sha256'].items():assert sha256(Path(adapter['directory'])/name)==h
        for path in cfg['gpu_lock_paths']:
            f=Path(path).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        def query(q):return subprocess.check_output(['nvidia-smi','-i','0',q,'--format=csv,noheader,nounits'],text=True,timeout=15).strip()
        uuid,used,free=query('--query-gpu=uuid,memory.used,memory.free').split(',')
        assert uuid.strip()==cfg['gpu_uuid'] and int(used)<512 and int(free)>23000 and not query('--query-compute-apps=pid')
        result['initial_gpu']=dict(uuid=uuid.strip(),used_mib=int(used),free_mib=int(free))
        os.environ.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        from peft import PeftModel,get_peft_model_state_dict,set_peft_model_state_dict
        from safetensors.torch import load_file
        torch.set_num_threads(4);torch.manual_seed(0)
        model=Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_directory'],local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,attn_implementation='sdpa')
        model=PeftModel.from_pretrained(model,cfg['adapters']['separate']['directory'],adapter_name='separate',is_trainable=False,autocast_adapter_dtype=True,local_files_only=True)
        model.load_adapter(cfg['adapters']['paired']['directory'],adapter_name='paired',is_trainable=False,autocast_adapter_dtype=True,local_files_only=True)
        # PEFT may copy saved FP32 values into newly created BF16 modules before
        # upcasting. Restore disk values after all adapter modules are FP32.
        load_differences={}
        for arm in ['separate','paired']:
            expected=load_file(str(Path(cfg['adapters'][arm]['directory'])/'adapter_model.safetensors'))
            actual=get_peft_model_state_dict(model,adapter_name=arm)
            assert set(actual)==set(expected)
            load_differences[arm]=dict(changed_tensors=sum(not torch.equal(actual[k].detach().cpu(),expected[k]) for k in actual),
                changed_elements=sum(int((actual[k].detach().cpu()!=expected[k]).sum()) for k in actual),
                max_abs_error=max((actual[k].detach().cpu()-expected[k]).abs().max().item() for k in actual),
                actual_dtypes=sorted({str(v.dtype) for v in actual.values()}))
            assert all(v.dtype==torch.float32 for v in actual.values())
            set_peft_model_state_dict(model,expected,adapter_name=arm)
        result['pre_restore_peft_load_differences']=load_differences
        progress()
        def check_adapters():
            checks={}
            for arm in ['separate','paired']:
                actual=get_peft_model_state_dict(model,adapter_name=arm);expected=load_file(str(Path(cfg['adapters'][arm]['directory'])/'adapter_model.safetensors'))
                assert set(actual)==set(expected)
                assert all(actual[k].dtype==expected[k].dtype and torch.equal(actual[k].detach().cpu(),expected[k]) for k in actual)
                checks[arm]=dict(tensors=len(actual),parameters=sum(v.numel() for v in actual.values()),dtype=sorted({str(v.dtype) for v in actual.values()}),disk_values_exact=True)
            return checks
        result['initial_adapter_checks']=check_adapters();progress()
        model.eval();model.requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
        for arm in cfg['arm_order']:
            if arm!='base':model.set_adapter(arm)
            model.eval();model.requires_grad_(False)
            core=model.get_base_model();ctx=model.disable_adapter() if arm=='base' else nullcontext()
            with ctx,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for record in records:
                    path=Path(record['tensor_path']);assert sha256(path)==record['tensor_sha256'];batch=torch.load(path,map_location='cpu',weights_only=True)
                    inputs={k:v.to('cuda:0') for k,v in batch.items() if k!='labels'};labels=batch['labels'][0].to('cuda:0');positions=(labels!=-100).nonzero().flatten()
                    assert len(positions)==record['target_count'] and positions.min()>0
                    assert labels[positions].tolist()==record['target_ids'] and torch.equal(labels[positions],inputs['input_ids'][0,positions])
                    core.model.rope_deltas=None;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                    logits=model(**inputs,logits_to_keep=positions-1,use_cache=False).logits[0]
                    losses=torch.nn.functional.cross_entropy(logits.float(),labels[positions],reduction='none');pred=logits.argmax(-1);torch.cuda.synchronize();assert torch.isfinite(losses).all()
                    row=dict(arm=arm,record_id=record['id'],family_id=record['family_id'],source_decision=record['source_decision'],tensor_sha256=record['tensor_sha256'],target_ids=record['target_ids'],input_tokens=record['input_tokens'],target_count=record['target_count'],nll_sum=losses.sum().item(),mean_nll=losses.mean().item(),token_nll=losses.tolist(),teacher_forced_top1_ids=pred.tolist(),teacher_forced_correct=int((pred==labels[positions]).sum()),elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    result['rows'].append(row);save(a.out/(arm+'_'+record['id']+'.json'),row);progress();print(now(),arm,record['id'],row['mean_nll'],flush=True)
                    del batch,inputs,labels,positions,logits,losses,pred
        assert len(result['rows'])==30 and all(p.grad is None and not p.requires_grad and p._version==versions[n] for n,p in model.named_parameters())
        result.update(final_adapter_checks=check_adapters(),all_parameters_frozen_and_versions_unchanged=True,status='completed',actual_exit_code=0,ended_at=now())
    except BaseException as exc:
        result.update(status='failed',actual_exit_code=1,ended_at=now(),error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
    finally:
        progress()
        for lock in locks:lock.close()
    print(now(),'completed 30 frozen action forwards, zero updates',flush=True)

if __name__=='__main__':main()
