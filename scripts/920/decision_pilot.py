"""Explicitly authorized LoRA mechanism pilot, dispatched by pinned official SFT CLI.

The dry-run never loads model weights or performs backward/updates. Professional
training admission is not inferred from this separate, scoped mechanism pilot.
"""
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, SequentialSampler
from transformers import Trainer, TrainingArguments, TrainerCallback
from native_project import sha256
from run_decision_loss import save


def locked(path, digest):
    path=Path(path)
    if sha256(path)!=digest:raise ValueError('Hash mismatch: '+str(path))
    return path


class TensorDecisions(Dataset):
    def __init__(self, records, sources):
        self.records=records;self.sources=sources
        self.counts={s:sum(x['target_count'] for x in records if x['source_decision']==s) for s in sources}
        if not all(self.counts.values()):raise ValueError('Empty source')
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records[index];path=locked(row['tensor_path'],row['tensor_sha256'])
        batch=torch.load(path,map_location='cpu',weights_only=True)
        if int((batch['labels']!=-100).sum())!=row['target_count']:raise ValueError('Changed targets')
        # Trainer divides by accumulation length M. The summed macro-step is
        # (1 / number_of_sources) * sum_s(total_NLL_s / total_target_tokens_s).
        batch['source_weight']=len(self.records)/(len(self.sources)*self.counts[row['source_decision']])
        return batch


def single(batch):
    if len(batch)!=1:raise ValueError('One pre-batched micro-example required')
    return batch[0]


class DecisionTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model_accepts_loss_kwargs=False
        self.micro_log=[]
    def _get_train_sampler(self,train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        inputs=dict(inputs);weight=inputs.pop('source_weight');labels=inputs.pop('labels')[0]
        positions=(labels!=-100).nonzero().flatten()
        if not len(positions) or positions.min()<=0:raise ValueError('Empty or unshiftable targets')
        outputs=model(**inputs,logits_to_keep=positions-1,use_cache=False)
        nll=torch.nn.functional.cross_entropy(outputs.logits[0].float(),labels[positions],reduction='sum')
        if not bool(torch.isfinite(nll)):raise FloatingPointError('Nonfinite supervised loss; no update')
        loss=nll*weight
        self.micro_log.append({'nll_sum':float(nll.detach()),'target_count':len(positions),'weight':float(weight),
                              'optimizer_step_before':int(self.state.global_step)})
        return (loss,outputs) if return_outputs else loss


def normalization_check(records, measured, directory):
    """Exercise actual Trainer.training_step normalization; intercept backward.

    A constant two-class fixture reproduces each measured micro-example mean NLL.
    No original model is loaded, no differentiable parameters or optimizer exist.
    """
    class ConstantModel(torch.nn.Module):
        def __init__(self):super().__init__();self.register_buffer('logit',torch.tensor(0.));self.config=SimpleNamespace()
        def forward(self,input_ids,attention_mask,logits_to_keep,use_cache=False):
            logits=torch.stack([torch.zeros_like(self.logit),self.logit]).expand(1,len(logits_to_keep),2)
            return SimpleNamespace(logits=logits)
    sources=sorted({x['source_decision'] for x in records});ds=TensorDecisions(records,sources)
    model=ConstantModel();args=TrainingArguments(output_dir=str(directory),use_cpu=True,report_to=[],
        per_device_train_batch_size=1,gradient_accumulation_steps=len(ds),remove_unused_columns=False)
    trainer=DecisionTrainer(model=model,args=args,train_dataset=ds,data_collator=single)
    trainer.current_gradient_accumulation_steps=len(ds)
    if trainer.accelerator.gradient_accumulation_steps!=1:raise ValueError('Unexpected second accumulation divisor')
    intercepted=[];trainer.accelerator.backward=lambda loss,**kw:intercepted.append(float(loss.detach()))
    expected=0.
    for record in records:
        count=record['target_count'];mean=measured[record['id']]['mean_nll']
        model.logit.fill_(float(torch.log(torch.expm1(torch.tensor(mean)))))
        batch={'input_ids':torch.zeros(1,count+1,dtype=torch.long),'attention_mask':torch.ones(1,count+1,dtype=torch.long),
               'labels':torch.zeros(1,count+1,dtype=torch.long),'source_weight':len(ds)/(len(sources)*ds.counts[record['source_decision']])}
        batch['labels'][0,0]=-100
        trainer.training_step(model,batch)
        expected+=mean*count/(len(sources)*ds.counts[record['source_decision']])
    error=abs(sum(intercepted)-expected)
    if error>1e-5 or trainer.optimizer is not None:raise ValueError('Normalization test failed')
    return {'expected_source_balanced_loss':expected,'actual_sum_passed_to_intercepted_backward':sum(intercepted),
            'absolute_error':error,'micro_examples':len(ds),'optimizer_created':False,'backward_executed':False,
            'scope':'Actual pinned Trainer.training_step and custom loss on constant synthetic logits, not real model training.'}


def validate(plan_path,arm,approval,dry):
    cfg=json.loads(Path(plan_path).read_text())
    if cfg['purpose']!='conditional_reconstruction_mechanism_pilot':raise ValueError('Wrong plan purpose')
    if arm not in ('paired','separate'):raise ValueError('Choose an arm')
    if (cfg['updates_per_arm']!=2 or cfg['lora']['r']!=8 or cfg['lora']['alpha']!=16 or
        cfg['learning_rate']!=2e-5 or cfg['seed']!=42):raise ValueError('Unreviewed pilot budget')
    if not dry:
        if not approval:raise PermissionError('Separate user authorization is required before any model load or training')
        auth=json.loads(Path(approval).read_text())
        if (auth.get('approved') is not True or auth.get('plan_sha256')!=sha256(Path(plan_path)) or
            auth.get('scope')!=cfg['purpose'] or not auth.get('user_message_reference')):
            raise PermissionError('Approval must record a real user instruction bound to this exact plan')
    for path,digest in cfg['code_files'].items():locked(path,digest)
    locked(cfg['encoding']['path'],cfg['encoding']['sha256']);locked(cfg['loss']['path'],cfg['loss']['sha256'])
    for x in cfg['model_weight_files']:locked(Path(cfg['model_directory'])/x['path'],x['sha256'])
    for name,digest in cfg['model_configuration_sha256'].items():locked(Path(cfg['model_directory'])/name,digest)
    return cfg


def pilot_entry(args,upstream):
    from datetime import datetime,timezone
    print(datetime.now(timezone.utc).isoformat(),'official pilot entry',args.decision_arm,
          'inspection' if args.dry_run else 'authorization check before training',flush=True)
    # Enforce authorization before importing/initializing actual model or optimizer.
    cfg=validate(args.decision_pilot_plan,args.decision_arm,args.decision_approval,args.dry_run)
    enc=json.loads(Path(cfg['encoding']['path']).read_text());actual=json.loads(Path(cfg['loss']['path']).read_text())
    records=[x for x in enc['records'] if x['component']=='reason' or x['arm']==args.decision_arm]
    if len(records)!=20:raise ValueError('Pilot requires the exact frozen twenty-example macro-batch')
    ds=TensorDecisions(records,sorted({x['source_decision'] for x in records}))
    if args.dry_run:
        if not args.decision_inspection_report:raise ValueError('Inspection report required')
        rows=[]
        for i,row in enumerate(records):
            batch=ds[i];rows.append({'id':row['id'],'source':row['source_decision'],'target_count':row['target_count'],
                'source_weight_before_trainer_division':batch['source_weight'],'input_tokens':batch['input_ids'].shape[1]})
        result={'status':'prepared_not_authorized','plan_sha256':sha256(Path(args.decision_pilot_plan)),
            'arm':args.decision_arm,'rows':rows,'source_target_counts':ds.counts,'updates_executed':0,
            'normalization':normalization_check(records,{x['id']:x for x in actual['rows']},Path(args.decision_inspection_report).parent/'constant_fixture'),
            'cuda_initialized':torch.cuda.is_initialized()}
        if result['cuda_initialized']:raise ValueError('Unexpected GPU use in inspection')
        save(Path(args.decision_inspection_report),result);print(json.dumps(result));return
    train_authorized(cfg,ds,args.decision_arm,upstream,Path(args.decision_pilot_plan))


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
            def on_pre_optimizer_step(self,args,state,control,**kwargs):
                grads=[p.grad for p in trainable.values() if p.grad is not None]
                if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError('Missing/nonfinite adapter gradients; stop before optimizer')
            def on_step_end(self,args,state,control,**kwargs):
                result['optimizer_updates']=state.global_step;result['updated_at']=now();save(out/'RUN.json',result)
        trainer=DecisionTrainer(model=model,args=args,train_dataset=ds,data_collator=single,callbacks=[Evidence()])
        if trainer.accelerator.num_processes!=1 or trainer.accelerator.gradient_accumulation_steps!=1:raise ValueError('Unsupported accumulation/distributed setup')
        torch.cuda.reset_peak_memory_stats();trainer.train()
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
                      cold_process_optimizer_resume_tested=False,generalization_claim=False)
    except BaseException as exc:
        result.update(status='failed',ended_at=now(),exit_code=1,error=type(exc).__name__+': '+str(exc));raise
    finally:
        save(out/'RUN.json',result)
        for f in locks:f.close()
