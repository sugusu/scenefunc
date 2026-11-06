"""Finite furnished-room learning study; separate action and decision objectives.

Uses the verified official Qwen model/Trainer/offload path. Frozen historic pilot
files remain unchanged. Dry-run exercises Trainer scaling without backward.
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from decision_pilot_offload import DecisionTrainer, locked, single
from checkpoint_input_offload import install_checkpoint_input_offload
from native_project import sha256
from run_decision_loss import save

class EditExamples(Dataset):
    def __init__(self, records):
        self.records=records
        self.families=sorted({r['family_id'] for r in records})
        self.counts={f:{c:sum(r[c+'_tokens'] for r in records if r['family_id']==f)
                       for c in ('action','decision')} for f in self.families}
        if len(records)!=22 or len(self.families)!=6:raise ValueError('Frozen corpus count mismatch')
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i];b=torch.load(locked(r['tensor_path'],r['tensor_sha256']),map_location='cpu',weights_only=True)
        for c in ('action','decision'):
            mask=b[c+'_mask'];count=self.counts[r['family_id']][c]
            if int(mask.sum())!=r[c+'_tokens']:raise ValueError('Target mask count changed')
            b[c+'_weight']=len(self)/(len(self.families)*count) if count else 0.
        union=b['action_mask']|b['decision_mask']
        if not torch.equal(union,b['labels']!=-100) or bool((b['action_mask']&b['decision_mask']).any()):
            raise ValueError('Disjoint mask/label mismatch')
        return b

class EditTrainer(DecisionTrainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        x=dict(inputs);labels=x.pop('labels')[0]
        masks={c:x.pop(c+'_mask')[0] for c in ('action','decision')}
        weights={c:x.pop(c+'_weight') for c in masks}
        positions=(labels!=-100).nonzero().flatten()
        if not len(positions) or positions.min()<=0:raise ValueError('Unshiftable targets')
        output=model(**x,logits_to_keep=positions-1,use_cache=False)
        nll=torch.nn.functional.cross_entropy(output.logits[0].float(),labels[positions],reduction='none')
        if not bool(torch.isfinite(nll).all()):raise FloatingPointError('Nonfinite loss')
        terms={c:nll[masks[c][positions]].sum() for c in masks}
        loss=sum(terms[c]*weights[c] for c in masks)
        self.micro_log.append(dict(optimizer_step_before=int(self.state.global_step),
            **{c:dict(nll_sum=float(terms[c].detach()),tokens=int(masks[c].sum()),weight=float(weights[c])) for c in masks}))
        return (loss,output) if return_outputs else loss

def normalization_check(ds,out):
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('values',torch.tensor([0.]));self.config=SimpleNamespace()
        def forward(self,input_ids,attention_mask,logits_to_keep,use_cache=False):
            return SimpleNamespace(logits=torch.stack((torch.zeros_like(self.values),self.values),-1).unsqueeze(0))
    m=Constant();args=TrainingArguments(output_dir=str(out),use_cpu=True,report_to=[],
        per_device_train_batch_size=1,gradient_accumulation_steps=len(ds),remove_unused_columns=False)
    tr=EditTrainer(model=m,args=args,train_dataset=ds,data_collator=single)
    tr.current_gradient_accumulation_steps=len(ds)
    assert tr.accelerator.gradient_accumulation_steps==1
    actual=[];tr.accelerator.backward=lambda value,**kw:actual.append(float(value.detach()))
    expected=0.;rows=[]
    for i,r in enumerate(ds.records):
        real=ds[i];na,nd=r['action_tokens'],r['decision_tokens'];n=na+nd
        ma,md=.23+.03*i,.51+.02*i
        means=torch.tensor([md]*nd+[ma]*na);m.values=torch.log(torch.expm1(means))
        b=dict(input_ids=torch.zeros(1,n+1,dtype=torch.long),attention_mask=torch.ones(1,n+1,dtype=torch.long),labels=torch.zeros(1,n+1,dtype=torch.long))
        b['labels'][0,0]=-100
        b['decision_mask']=torch.tensor([[False]+[True]*nd+[False]*na]);b['action_mask']=torch.tensor([[False]+[False]*nd+[True]*na])
        for c in ('action','decision'):b[c+'_weight']=real[c+'_weight']
        tr.training_step(m,b)
        expected+=(na*ma*b['action_weight']+nd*md*b['decision_weight'])/len(ds)
        rows.append(dict(id=r['id'],input_tokens=int(real['input_ids'].shape[1]),action_tokens=na,decision_tokens=nd))
    error=abs(sum(actual)-expected)
    assert error<1e-5 and tr.optimizer is None and not torch.cuda.is_initialized()
    return dict(expected=expected,observed=sum(actual),absolute_error=error,rows=rows,
                optimizer_created=False,backward_executed=False,cuda_initialized=False)

def train_authorized(cfg,ds,arm,upstream,plan_path):
    import fcntl
    import subprocess
    from datetime import datetime,timezone
    from peft import LoraConfig,TaskType,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import set_seed
    out=Path(cfg['training_output_root'])/arm
    out.mkdir(parents=True,exist_ok=False)
    locks=[];now=lambda:datetime.now(timezone.utc).isoformat()
    result={'status':'running','arm':arm,'started_at':now(),'plan_sha256':sha256(plan_path),'optimizer_updates':0}
    save(out/'RUN.json',result)
    try:
        for path in cfg['gpu_lock_paths']:
            f=Path(path).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        line=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
        uuid,used,free=line.strip().split(',')
        if uuid.strip()!=cfg['gpu_uuid'] or int(used)>512 or int(free)<23000:raise RuntimeError('Authorized GPU0 unavailable')
        os.environ.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line and line.split()[1].isdigit()}
        if mem.get('MemAvailable',0)<cfg['minimum_host_available_bytes']:
            raise RuntimeError('Insufficient available host memory for checkpoint input copies')
        result['initial_host_available_bytes']=mem['MemAvailable']
        torch.set_num_threads(4);set_seed(cfg['seed'])
        model=upstream.Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_directory'],local_files_only=True,
            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(r=cfg['lora']['r'],lora_alpha=cfg['lora']['alpha'],lora_dropout=0.,
              bias='none',task_type=TaskType.CAUSAL_LM,target_modules=cfg['lora']['target_modules']))
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        if not trainable or any('lora_' not in n or '.language_model.' not in n for n in trainable):raise ValueError('Unexpected trainable base/vision weights')
        if sum(p.numel() for p in trainable.values())!=cfg['lora']['expected_parameters']:raise ValueError('LoRA count changed')
        initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
        model.enable_input_require_grads()
        args=TrainingArguments(output_dir=str(out/'checkpoints'),max_steps=cfg['updates_per_arm'],
            per_device_train_batch_size=1,gradient_accumulation_steps=len(ds),learning_rate=cfg['learning_rate'],
            lr_scheduler_type='constant',warmup_ratio=0.,weight_decay=0.,max_grad_norm=1.,optim='adafactor',
            bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
            save_strategy='steps',save_steps=1,save_total_limit=cfg['updates_per_arm'],save_only_model=False,
            logging_steps=1,report_to=[],seed=cfg['seed'],data_seed=cfg['seed'],dataloader_num_workers=0,
            remove_unused_columns=False,disable_tqdm=True)
        class Evidence(TrainerCallback):
            def on_train_begin(self,args,state,control,**kwargs):
                trainer.checkpoint_input_stats=install_checkpoint_input_offload(model,36)
                trainer._save_micro_progress('offload_installed_before_first_forward')
            def on_pre_optimizer_step(self,args,state,control,**kwargs):
                grads=[p.grad for p in trainable.values() if p.grad is not None]
                if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError('Missing/nonfinite adapter gradients; stop before optimizer')
            def on_step_end(self,args,state,control,**kwargs):
                result['optimizer_updates']=state.global_step;result['updated_at']=now();save(out/'RUN.json',result)
        trainer=EditTrainer(model=model,args=args,train_dataset=ds,data_collator=single,callbacks=[Evidence()])
        if trainer.accelerator.num_processes!=1 or trainer.accelerator.gradient_accumulation_steps!=1:raise ValueError('Unsupported accumulation/distributed setup')
        trainer.evidence_directory=out
        torch.cuda.reset_peak_memory_stats();trainer.train()
        if (trainer.backward_completed!=cfg['updates_per_arm']*len(ds) or
            trainer.checkpoint_input_stats['checkpoint_forward_calls']!=36*cfg['updates_per_arm']*len(ds)):
            raise ValueError('Checkpoint/backward budget mismatch')
        changed=[n for n,p in trainable.items() if not torch.equal(initial[n],p.detach().cpu())]
        if (trainer.state.global_step!=cfg['updates_per_arm'] or not changed or
            len(trainer.micro_log)!=cfg['updates_per_arm']*len(ds)):
            raise ValueError('Expected update or complete micro-example budget mismatch')
        trainer.save_model(str(out/'adapter'))
        saved=load_file(str(out/'adapter/adapter_model.safetensors'))
        before_reload={n:p.detach().cpu().clone() for n,p in get_peft_model_state_dict(model).items()}
        if set(saved)!=set(before_reload) or any(not torch.equal(saved[n],before_reload[n]) for n in saved):
            raise ValueError('Saved adapter differs from updated state')
        with torch.no_grad():
            for parameter in trainable.values():parameter.zero_()
        set_peft_model_state_dict(model,saved)
        restored=get_peft_model_state_dict(model)
        if any(not torch.equal(saved[n],restored[n].detach().cpu()) for n in saved):raise ValueError('Adapter reload mismatch')
        save(out/'MICRO_LOSS.json',trainer.micro_log)
        result.update(status='completed',ended_at=now(),exit_code=0,changed_adapter_tensors=len(changed),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),training_log=trainer.state.log_history,
                      adapter=str(out/'adapter'),same_process_adapter_save_zero_reload_exact=True,
                      backward_completed=trainer.backward_completed,checkpoint_input_stats=trainer.checkpoint_input_stats,
                      cold_process_optimizer_resume_tested=False,generalization_claim=False)
    except BaseException as exc:
        result.update(status='failed',ended_at=now(),exit_code=1,error=type(exc).__name__+': '+str(exc));raise
    finally:
        save(out/'RUN.json',result)
        for f in locks:f.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--arm',choices=['action_only','decision_action'],required=True)
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--out',type=Path);ap.add_argument('--authorization',type=Path);a=ap.parse_args()
    cfg=json.loads(a.plan.read_text())
    if cfg['purpose']!='furnished_room_decision_supervision' or cfg['updates_per_arm']!=8 or cfg['seconds_per_arm']!=7200:
        raise ValueError('Study budget changed')
    for path,digest in cfg['code_files'].items():locked(path,digest)
    enc=json.loads(locked(cfg['encoding']['path'],cfg['encoding']['sha256']).read_text())
    ds=EditExamples([r for r in enc['records'] if r['arm']==a.arm])
    if a.dry_run:
        if not a.out:raise ValueError('Dry-run output required')
        save(a.out,dict(status='CPU_actual_Trainer_scaling_passed',arm=a.arm,plan_sha256=sha256(a.plan),normalization=normalization_check(ds,a.out.parent/'fixture'),optimizer_updates=0));return
    if not a.authorization:raise PermissionError('User authorization evidence required')
    auth=json.loads(a.authorization.read_text())
    if auth.get('approved') is not True or auth.get('plan_sha256')!=sha256(a.plan) or not auth.get('user_messages'):
        raise PermissionError('Approval scope/hash mismatch')
    for row in cfg['model_weight_files']:locked(Path(cfg['model_directory'])/row['path'],row['sha256'])
    for name,digest in cfg['model_configuration_sha256'].items():locked(Path(cfg['model_directory'])/name,digest)
    # Import the actual pinned official SFT module (unmodified) for its model class.
    import importlib.util
    source=Path(cfg['upstream_source'])/'src';sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('edit_study_official_sft',source/'train_sft.py');up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
    train_authorized(cfg,ds,a.arm,up,a.plan)

if __name__=='__main__':main()
