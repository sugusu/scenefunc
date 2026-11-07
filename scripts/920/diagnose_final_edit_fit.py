"""Bounded frozen-weight diagnostic; CPU preflight never loads a model.

Common action targets, explicitly oracle-conditioned targets, and free decoding
are distinct outputs. No training, scene edits, or room-quality scoring occurs.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import traceback


ARMS=('base','action_only','decision_action')
GENERATE_IDS=('paper_table_modes_build_08','paper_table_modes_build_09',
              'waterloo_edit_02','waterloo_edit_03',
              'waterloo_initial_work_stop','waterloo_joined_host_stop')


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def locked(p,h):
    if sha(p)!=h: raise ValueError('Frozen file changed: '+str(p))
    return Path(p)


def parse_output(raw):
    """Representation only: this never asserts tool or geometry feasibility."""
    text=raw.strip()
    for end in ('<|im_end|>','<|endoftext|>'):
        if text.endswith(end): text=text[:-len(end)].rstrip()
    matches=list(re.finditer(r'<(tool_call|done)>(.*?)</\1>',text,re.S))
    if len(matches)!=1 or matches[0].end()!=len(text): return None
    m=matches[0]
    if text.count('<tool_call>')+text.count('<done>')!=1: return None
    def pairs(xs):
        result={}
        for k,v in xs:
            if k in result: raise ValueError('Duplicate JSON key')
            result[k]=v
        return result
    def invalid(x): raise ValueError('Nonfinite JSON')
    def finite_float(x):
        value=float(x)
        if not math.isfinite(value): raise ValueError('Nonfinite JSON')
        return value
    try: value=json.loads(m[2],object_pairs_hook=pairs,parse_constant=invalid,parse_float=finite_float)
    except (ValueError,TypeError): return None
    if not isinstance(value,dict): return None
    if m[1]=='tool_call' and (set(value)!={'name','arguments'} or
            not isinstance(value['name'],str) or not isinstance(value['arguments'],dict)):return None
    return {'kind':m[1],'value':value}


def model_inputs(batch,prompt_tokens=None):
    result={k:v for k,v in batch.items() if k not in ('labels','action_mask','decision_mask')}
    if prompt_tokens is not None:
        allowed={'input_ids','attention_mask','mm_token_type_ids','pixel_values','image_grid_thw'}
        if not set(result)<=allowed:raise ValueError('Unreviewed input tensor shape')
        for k in ('input_ids','attention_mask','mm_token_type_ids'):
            if k in result:result[k]=result[k][:,:prompt_tokens]
    return result


def validate_plan(cfg):
    if (cfg['purpose']!='final_edit_fit_diagnostic' or cfg['arms']!=list(ARMS)
        or cfg['generate_ids']!=list(GENERATE_IDS) or cfg['planned_forwards']!=96
        or cfg['planned_generations']!=18 or cfg['maximum_process_seconds']!=1800
        or cfg['max_new_tokens']!=768 or cfg['training_updates']!=0):
        raise ValueError('Diagnostic scope changed')
    for p,h in cfg['code_sha256'].items():locked(p,h)
    enc=read(locked(cfg['encoding']['path'],cfg['encoding']['sha256']))
    for name,h in cfg['model_configuration_sha256'].items():locked(Path(cfg['model_directory'])/name,h)
    for record in cfg['adapters'].values():
        for name,h in record['sha256'].items():locked(Path(record['directory'])/name,h)
    return enc


def load_batches(enc):
    import torch
    records={arm:[r for r in enc['records'] if r['arm']==arm] for arm in ARMS[1:]}
    if any(len(v)!=24 for v in records.values()):raise ValueError('Expected 24 examples per arm')
    if len({r['id'] for r in records['action_only']})!=24:raise ValueError('Repeated example')
    result={};checks=[]
    for left,right in zip(records['action_only'],records['decision_action']):
        if left['id']!=right['id'] or left['prompt_tokens']!=right['prompt_tokens']:raise ValueError('Unpaired input')
        pair=[]
        for r in (left,right):
            b=torch.load(locked(r['tensor_path'],r['tensor_sha256']),map_location='cpu',weights_only=True)
            am,dm=b['action_mask'],b['decision_mask'];labels=b['labels'];boundary=r['prompt_tokens']
            if (int(am.sum())!=r['action_tokens'] or int(dm.sum())!=r['decision_tokens']
                or bool((am&dm).any()) or not torch.equal(am|dm,labels!=-100)
                or not bool((labels[:,:boundary]==-100).all())
                or not bool((labels[:,boundary:]!=-100).all())
                or not torch.equal(labels[labels!=-100],b['input_ids'][labels!=-100])
                or tuple(b['image_grid_thw'].shape)!=(5,3)
                or b['input_ids'].shape!=(1,r['input_tokens'])):
                raise ValueError('Input/mask invariant failed: '+r['id'])
            pair.append(b)
        if left['decision_tokens']!=0:raise ValueError('Common-action batch has gold reason')
        for key in ('input_ids','attention_mask'):
            if not torch.equal(pair[0][key][:,:left['prompt_tokens']],pair[1][key][:,:right['prompt_tokens']]):raise ValueError('Different prompts')
        for key in ('pixel_values','image_grid_thw'):
            if not torch.equal(pair[0][key],pair[1][key]):raise ValueError('Different views')
        if not torch.equal(pair[0]['labels'][pair[0]['action_mask']],pair[1]['labels'][pair[1]['action_mask']]):raise ValueError('Different actions')
        result[left['id']]={'common':(left,pair[0]),'oracle':(right,pair[1])}
        checks.append({'id':left['id'],'prompt_tokens':left['prompt_tokens'],'common_action_tokens':left['action_tokens'],'oracle_decision_tokens':right['decision_tokens']})
    if sum(x['oracle'][0]['decision_tokens']>0 for x in result.values())!=8:raise ValueError('Expected eight oracle examples')
    if not set(GENERATE_IDS)<=set(result):raise ValueError('Missing free-decoding example')
    return result,checks


def authorize(path,plan):
    if path is None:raise PermissionError('A new explicit diagnostic approval is required; previous evaluation ended.')
    a=read(path)
    if a.get('approved') is not True or a.get('plan_sha256')!=sha(plan) or not a.get('user_message'):
        raise PermissionError('Diagnostic authorization does not match this plan')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--dry-run',action='store_true');ap.add_argument('--authorization',type=Path)
    args=ap.parse_args();cfg=read(args.plan)
    if not args.dry_run:authorize(args.authorization,args.plan)
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.dry_run else cfg['gpu_uuid'],
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    args.out.mkdir(parents=True,exist_ok=False)
    result={'status':'running','started_at':now(),'plan_sha256':sha(args.plan),'forwards':[],
            'generations':[],'training_updates':0,'scene_edits':0,'dry_run':args.dry_run}
    locks=[]
    def progress():
        save(args.out/'RESULT.json',result)
        if not args.dry_run:
            pub=Path(cfg['public_directory'])
            save(pub/'STATUS.json',dict(run_id=cfg['run_id'],state=result['status'],question=cfg['question'],
                started_at=result['started_at'],ended_at=result.get('ended_at'),actual_exit_code=result.get('exit_code'),
                updated_at=now(),first_log=result.get('first_log'),completed=len(result['forwards'])+len(result['generations']),planned=114,
                configuration_sha256=result['plan_sha256'],input_sha256=cfg['encoding']['sha256'],
                latest_verifiable_artifact=str(args.out/'RESULT.json'),failure_reason=result.get('error'),
                next_step='Interpret fit and free decoding separately; no automatic training or inference extension.'))
            save(pub.parent/'latest.json',dict(run_id=cfg['run_id'],stage=result['status'],updated_at=now(),run_directory=cfg['runtime_directory']))
    result['first_log']=now()+' begin '+('CPU preflight' if args.dry_run else 'authorized frozen-weight diagnostic')
    print(result['first_log'],flush=True);progress()
    try:
        enc=validate_plan(cfg)
        import torch
        torch.set_num_threads(4)
        batches,checks=load_batches(enc);result['input_checks']=checks
        if args.dry_run:
            if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
            result.update(status='cpu_preflight_passed',exit_code=0,ended_at=now(),model_loaded=False,cuda_initialized=False)
            return
        import fcntl
        for p in cfg['gpu_lock_paths']:
            f=Path(p).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
        if gpu[0].strip()!=cfg['gpu_uuid'] or int(gpu[1])>512 or int(gpu[2])<23000:raise RuntimeError('GPU0 unavailable')
        for r in cfg['model_weight_files']:locked(Path(cfg['model_directory'])/r['path'],r['sha256'])
        from transformers import Qwen3VLForConditionalGeneration,AutoProcessor,StoppingCriteria,StoppingCriteriaList
        from peft import PeftModel,get_peft_model_state_dict,set_peft_model_state_dict
        from safetensors.torch import load_file
        torch.manual_seed(0);torch.cuda.set_per_process_memory_fraction(.94)
        processor=AutoProcessor.from_pretrained(cfg['model_directory'],local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_directory'],local_files_only=True,
            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True)
        model=PeftModel.from_pretrained(model,cfg['adapters']['action_only']['directory'],adapter_name='action_only',is_trainable=False,local_files_only=True,autocast_adapter_dtype=True)
        model.load_adapter(cfg['adapters']['decision_action']['directory'],adapter_name='decision_action',is_trainable=False,local_files_only=True,autocast_adapter_dtype=True)
        saved={arm:load_file(str(Path(cfg['adapters'][arm]['directory'])/'adapter_model.safetensors')) for arm in ARMS[1:]}
        # Restore after all adapter modules are upcast, avoiding PEFT second-load rounding.
        for arm in ARMS[1:]:
            actual=get_peft_model_state_dict(model,adapter_name=arm)
            if set(actual)!=set(saved[arm]) or any(v.dtype!=torch.float32 for v in actual.values()):raise ValueError('Unexpected adapter shape/dtype')
            set_peft_model_state_dict(model,saved[arm],adapter_name=arm)
        def check_adapters():
            for arm in ARMS[1:]:
                actual=get_peft_model_state_dict(model,adapter_name=arm)
                if set(actual)!=set(saved[arm]) or any(v.dtype!=saved[arm][k].dtype or not torch.equal(v.detach().cpu(),saved[arm][k]) for k,v in actual.items()):raise ValueError('Adapter differs from accepted disk values')
        check_adapters();model.eval();model.requires_grad_(False)
        versions={n:p._version for n,p in model.named_parameters()};core=model.get_base_model()
        for arm in ARMS:
            if arm!='base':model.set_adapter(arm)
            model.eval();model.requires_grad_(False)
            with (model.disable_adapter() if arm=='base' else nullcontext()),torch.inference_mode():
                for setting in ('common','oracle'):
                    for rid,pair in batches.items():
                        record,batch=pair[setting]
                        if setting=='oracle' and record['decision_tokens']==0:continue
                        inputs={k:v.to('cuda') for k,v in model_inputs(batch).items()}
                        labels=batch['labels'][0].to('cuda');positions=(labels!=-100).nonzero().flatten()
                        core.model.rope_deltas=None;torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                        logits=model(**inputs,logits_to_keep=positions-1,use_cache=False).logits[0]
                        nll=torch.nn.functional.cross_entropy(logits.float(),labels[positions],reduction='none')
                        if not bool(torch.isfinite(nll).all()):raise FloatingPointError('Nonfinite likelihood')
                        prediction=logits.argmax(-1);parts={}
                        for c in ('action','decision'):
                            mask=batch[c+'_mask'][0].to('cuda')[positions]
                            parts[c]={'tokens':int(mask.sum()),'nll_sum':float(nll[mask].sum()),'teacher_forced_correct':int((prediction[mask]==labels[positions][mask]).sum()),'token_nll':nll[mask].tolist()}
                        torch.cuda.synchronize()
                        row={'arm':arm,'setting':setting,'id':rid,'family_id':record['family_id'],'components':parts,'seconds':time.monotonic()-start,'tensor_sha256':record['tensor_sha256']}
                        result['forwards'].append(row);progress()
                        del inputs,labels,positions,logits,nll,prediction,parts
                for rid in GENERATE_IDS:
                    record,batch=batches[rid]['common'];boundary=record['prompt_tokens']
                    inputs={k:v.to('cuda') for k,v in model_inputs(batch,boundary).items()}
                    class Complete(StoppingCriteria):
                        def __call__(self,ids,scores,**kwargs):
                            return torch.tensor([parse_output(processor.decode(ids[0,boundary:].tolist(),skip_special_tokens=False)) is not None],device=ids.device)
                    core.model.rope_deltas=None;torch.manual_seed(0);start=time.monotonic()
                    output=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,use_cache=True,
                        stopping_criteria=StoppingCriteriaList([Complete()]))
                    ids=output[0,boundary:].tolist();raw=processor.decode(ids,skip_special_tokens=False)
                    target=processor.decode(batch['labels'][batch['action_mask']].tolist(),skip_special_tokens=False)
                    parsed=parse_output(raw);expected=parse_output(target)
                    if expected is None:raise ValueError('Target output cannot be parsed')
                    path=args.out/'raw'/arm/(rid+'.txt');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(raw)
                    row={'arm':arm,'id':rid,'input_tensor_sha256':record['tensor_sha256'],'prompt_tokens':boundary,'output_ids':ids,
                        'parsed':parsed,'expected':expected,'exact_target_structure':parsed==expected,
                        'raw_output':str(path),'raw_output_sha256':sha(path),'seconds':time.monotonic()-start,
                        'geometry_or_utility_verified':False,'target_not_in_generation_input':True}
                    result['generations'].append(row);progress();del inputs,output
                print(now(),arm,'complete',flush=True)
        check_adapters()
        if len(result['forwards'])!=96 or len(result['generations'])!=18:raise ValueError('Diagnostic count mismatch')
        if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('Frozen parameter invariant failed')
        result.update(status='completed',exit_code=0,ended_at=now(),adapters_disk_exact=True,all_parameters_frozen_unchanged=True)
    except BaseException as exc:
        result.update(status='failed',exit_code=1,ended_at=now(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
    finally:
        progress()
        for f in locks:f.close()


if __name__=='__main__':main()
