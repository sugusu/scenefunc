"""Same Paris entry for CPU input preflight and one actual local GPU policy turn.
The first-turn artifact is separate from complete scene generation acceptance.
"""
import argparse, ast, hashlib, importlib.util, json, os, resource, socket, subprocess, sys, time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
P=Path(__file__).resolve().parent;W=P.parent
sys.path.insert(0,str(W/'native_local_inference_20261104'))
from local_vlm import LocalQwenEngine

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def limit(path):
    p=Path(path)
    return int(p.read_text()) if p.exists() and p.read_text().strip()!='max' else None

def parser():
    import re
    source=W/'native_execution_inventory_20261104/source/Designer/agent_rollout.py'
    nodes=[]
    for n in ast.parse(source.read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('_TOOL_CALL_RE','_THINK_RE') for t in n.targets):nodes.append(n)
        if isinstance(n,ast.FunctionDef) and n.name=='parse_assistant_text':nodes.append(n)
    ns={'re':re,'json':json};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    return ns['parse_assistant_text']

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phase',choices=['preflight','generate'],required=True);ap.add_argument('--arm',choices=['A','B'],default='A');ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    t=time.monotonic();attempts=[]
    def reject(*a,**k):attempts.append('connection');raise RuntimeError('Network is disabled for this local run')
    socket.socket.connect=reject;socket.create_connection=reject
    request=P/f'request_{args.arm}.json';messages=json.loads(request.read_text());prep=json.loads((P/'preparation.json').read_text())
    assert sha(request)==prep['arms'][args.arm]['request_sha256']
    for name,digest in prep['frozen_shared_sources'].items():assert sha(Path(name))==digest
    assert sha(P/'common_input_card.png')==prep['card']['raster_sha256']
    checkpoint=W/'critic_public_runtime_20261104/checkpoint'
    engine=LocalQwenEngine(checkpoint,out/'calls',max_pixels=65536,max_input_tokens=8192,max_new_tokens=2048)
    record={'phase':args.phase,'arm':args.arm,'request_sha256':sha(request),'checkpoint_revision':'63e235a0c0d6e5bc2f8f3485214df4457fd2e672','checkpoint':str(checkpoint),'batch_size':1,'dtype':'float16','max_new_tokens':2048,'max_input_tokens':8192,'max_pixels':65536,'seed':0,'attention':'eager','paid_api_calls':0,'training_runs':0,'model_forward':False,'complete_room_success':None,'status':'started'}
    try:
        inputs,encoded=engine.encode(messages);record['encoded']=encoded
        import torch
        torch.manual_seed(0)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
        memlimit=limit('/sys/fs/cgroup/memory/memory.limit_in_bytes') or limit('/sys/fs/cgroup/memory.max')
        record['resources']={'cuda_visible':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'cgroup_limit_bytes':memlimit,'cpu_quota_us':limit('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),'cpu_period_us':limit('/sys/fs/cgroup/cpu/cpu.cfs_period_us')}
        model=json.loads((W/'v100_runtime_20261104/checkpoint_memory_lower_bound.json').read_text());tc=model['text_config'];length=encoded['input_tokens']+2048
        # These are capacity calculations, not measured peak memory.
        kv=2*tc['num_hidden_layers']*tc['num_key_value_heads']*tc['head_dim']*length*2
        scores=tc['num_attention_heads']*encoded['input_tokens']**2*4
        record['capacity_estimate']={'dense_fp16_weights_bytes':model['hypothetical_dense_fp16_tensor_bytes'],'max_sequence_tokens_this_call':length,'kv_cache_bytes_at_max_output':kv,'one_fp32_prefill_attention_score_tensor_bytes':scores,'estimate_not_peak_measurement':True,'additional_memory':'vision activations, eager attention temporaries, CUDA context/workspaces, allocator and load buffers','recommended_instance':'one V100 32 GiB; at least 24 GiB system RAM, 4 CPU cores; batch 1 FP16; no concurrent CLIP/render process','inference_duration':'unmeasured; first call only, release when saved output and tool responses are verified'}
        assert engine._model is None and encoded['input_tokens']<=8192
        if args.phase=='preflight':
            record.update(status='cpu_input_preflight_passed',next_step='GPU first policy turn; no more unrelated prerequisites')
        else:
            if not torch.cuda.is_available():raise RuntimeError('GPU is required for this step; no CPU weight fallback')
            free,total=torch.cuda.mem_get_info();record['resources'].update(gpu_free_bytes=free,gpu_total_bytes=total,gpu_name=torch.cuda.get_device_name())
            if free<26*1024**3:raise RuntimeError('Less than the conservative 26 GiB free VRAM gate for this fixed first call')
            if memlimit and memlimit<24*1024**3:raise RuntimeError('System RAM below the conservative 24 GiB loading gate; this is not a measured minimum')
            result=engine.complete(messages,purpose=f'paris_{args.arm}_first_furniture_turn')
            (out/'response_raw.txt').write_text(result['raw']);(out/'response_text.txt').write_text(result['text'])
            _,calls=parser()(result['raw']);dump(out/'parsed_tool_calls.json',calls)
            record.update(model_forward=True,model_receipt_dir=result['receipt_dir'],parsed_tool_call_count=len(calls))
            tool_python='@LEGACY@/envs/hsm_cpu/bin/python'
            env=dict(os.environ,PYTHONPATH='@GEOMETRY_DEPS@',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            run=subprocess.run([tool_python,str(P/'first_tools.py'),'--calls',str(out/'parsed_tool_calls.json'),'--output',str(out/'tools')],env=env,capture_output=True,text=True,timeout=300)
            (out/'tools.stdout').write_text(run.stdout);(out/'tools.stderr').write_text(run.stderr)
            record.update(tool_process_exit=run.returncode,status='first_model_turn_saved',full_generation=False)
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-t,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,network_connection_attempts=len(attempts))
        dump(out/'receipt.json',record)
        print(json.dumps(record,ensure_ascii=False))

if __name__=='__main__':main()
