"""One offline Scene-Builder attempt with the shared strict JSON contract.

The caller owns GPU locks and the finite rollout budget. No retries or training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def complete_single_action(text):
    """Accept a complete single JSON tool block, never a partial/multi-action tail."""
    blocks=re.findall(r'<tool_call>(.*?)</tool_call>',text,re.DOTALL)
    if len(blocks)!=1 or text.count('<tool_call>')!=1 or not text.rstrip().endswith('</tool_call>'):
        return False
    try:call=json.loads(blocks[0])
    except json.JSONDecodeError:return False
    return isinstance(call,dict) and isinstance(call.get('name'),str) and isinstance(call.get('arguments'),dict)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--request',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['encode','generate'],required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    for name,digest in plan['code_sha256'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('Frozen code changed: '+name)
    if sha(plan['encoder'])!=plan['encoder_sha256']:raise ValueError('Encoder changed')
    for name,digest in plan['model_configuration_sha256'].items():
        if sha(Path(plan['model_directory'])/name)!=digest:raise ValueError('Fixed model configuration changed')
    dependency=plan['structured_output']
    if sha(dependency['installed_manifest'])!=dependency['installed_manifest_sha256']:raise ValueError('Dependency manifest changed')
    for item in json.loads(Path(dependency['installed_manifest']).read_text()):
        path=Path(dependency['dependency_directory'])/item['path']
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Dependency file changed: '+item['path'])
    def deny(*args,**kwargs):raise RuntimeError('Network disabled in offline model process')
    socket.socket.connect=deny;socket.create_connection=deny
    sys.path.insert(0,str(Path(plan['encoder']).parent))
    from local_vlm import LocalQwenEngine
    import torch
    engine=LocalQwenEngine(plan['model_directory'],a.out/'unused_engine_calls',
        max_pixels=plan['budget']['max_pixels'],max_input_tokens=plan['budget']['max_input_tokens'],
        max_new_tokens=plan['budget']['max_new_tokens'])
    now=lambda:datetime.now(timezone.utc).isoformat();started=time.monotonic()
    result={'started_at':now(),'mode':a.mode,'status':'running','request_sha256':sha(a.request),
        'plan_sha256':sha(a.plan),'model_calls':0,'training_updates':0,'paid_api_calls':0,
        'dtype':'bfloat16','attention':'sdpa','seed':0,'batch_size':1}
    write(a.out/'RESULT.json',result);print(now(),'begin',a.mode,flush=True)
    try:
        from shared_action_contract import ActionContract,context_from_request,schemas_from_observation
        request=json.loads(a.request.read_text())
        context=context_from_request(request)
        contract=ActionContract(schemas_from_observation({'system':request[0]['content']}),context)
        write(a.out/'ACTION_SCHEMA.json',contract.schema)
        inputs,encoding=engine.encode(request)
        if len(encoding['images'])!=5:raise ValueError('Exactly five actual scene views required')
        result['encoding']=encoding;write(a.out/'ENCODING.json',encoding)
        if a.mode=='generate':
            from transformers import Qwen3VLForConditionalGeneration
            torch.manual_seed(0);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.94)
            t=time.monotonic()
            model=Qwen3VLForConditionalGeneration.from_pretrained(plan['model_directory'],
                local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,
                device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
            adapter=plan.get('adapter')
            if adapter:
                for name,digest in adapter['sha256'].items():
                    if sha(Path(adapter['directory'])/name)!=digest:raise ValueError('Adapter hash changed')
                from peft import PeftModel
                model=PeftModel.from_pretrained(model,adapter['directory'],is_trainable=False,local_files_only=True,autocast_adapter_dtype=False).eval()
                result['adapter']=adapter
            else:
                result['adapter']=None
            for parameter in model.parameters():parameter.requires_grad_(False)
            if any(p.requires_grad for p in model.parameters()):raise ValueError('Inference model has trainable parameters')
            torch.cuda.synchronize();result['load_seconds']=time.monotonic()-t
            inputs={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            t=time.monotonic();result['model_calls']=1;write(a.out/'RESULT.json',result)
            from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
            setup=time.monotonic()
            tokenizer=engine.processor.tokenizer
            tokenizer_data=build_token_enforcer_tokenizer_data(tokenizer)
            prefix_function=contract.prefix_function(tokenizer_data)
            result['constraint_setup_seconds']=time.monotonic()-setup
            extra={'prefix_allowed_tokens_fn':prefix_function}
            t=time.monotonic()
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=plan['budget']['max_new_tokens'],do_sample=False,use_cache=True,**extra)
            torch.cuda.synchronize();result['generation_seconds']=time.monotonic()-t
            ids=output[0,encoding['input_tokens']:].cpu().tolist();write(a.out/'output_ids.json',ids)
            raw=engine.processor.decode(ids,skip_special_tokens=False)
            (a.out/'raw_output.txt').write_text(raw)
            eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else (eos or [])
            result.update(output_tokens=len(ids),ended_with_eos=bool(ids and ids[-1] in eos),
                raw_output_sha256=sha(a.out/'raw_output.txt'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if not ids or ids[-1]!=tokenizer.eos_token_id:
                raise RuntimeError('Incomplete constrained output retained without execution; no retry')
            action_text=engine.processor.decode(ids[:-1],skip_special_tokens=False)
            (a.out/'action_output.txt').write_text(action_text)
            validated=contract.validate(action_text,context)
            write(a.out/'VALIDATED_ACTION.json',validated)
            result.update(action_kind=validated['kind'],action_schema_sha256=validated['schema_sha256'],
                context_sha256=validated['context_sha256'],action_output_sha256=sha(a.out/'action_output.txt'))
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result.update(ended_at=now(),elapsed_seconds=time.monotonic()-started)
        write(a.out/'RESULT.json',result);print(now(),result['status'],flush=True)


if __name__=='__main__':main()
