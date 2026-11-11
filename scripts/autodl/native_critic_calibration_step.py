"""One offline public-SFT Critic contract probe; no retry or LoRA claim."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import resource
import socket
import sys
import time
from cpu_continuation import read,dump,sha,resource_snapshot

W=Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
SOURCE=W.parent/'systematic_scene_literature_20261029/scene_designer_context_review_20261102/code/src_critic/grpo_advantage.py'


def native_parser():
    nodes=[]
    for n in ast.parse(SOURCE.read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['_ANS_RE','_LABELS'] for t in n.targets):nodes.append(n)
        if isinstance(n,ast.FunctionDef) and n.name=='parse_ranking':nodes.append(n)
    scope={'re':re};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),scope)
    return scope['parse_ranking']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    attempts=[]
    def reject(*args,**kwargs):
        attempts.append(True);raise RuntimeError('Offline calibration forbids network')
    socket.socket.connect=reject;socket.create_connection=reject
    start=time.monotonic();receipt=dict(status='started',request_sha256=sha(a.request),paid_API_calls=0,training=False,
          source_parser_sha256=sha(SOURCE),max_pixels=16777216,max_input_tokens=8192,max_new_tokens=2048,
          dtype='float16',weights='public joint SFT, not v10 adapter or Scene-GRPO policy',model_calls=0)
    try:
        sys.path.insert(0,str(W/'paris_local_run_results_20261104'))
        from local_vlm_sdpa import LocalQwenEngine
        import torch
        torch.manual_seed(0)
        assert torch.cuda.is_available()
        snap=resource_snapshot(a.output);free,total=torch.cuda.mem_get_info()
        assert int(snap['memory_limit_bytes'])>=24*1024**3 and free>=26*1024**3
        receipt['resources']=dict(snap,gpu_free_bytes=free,gpu_total_bytes=total)
        # Keep the saved processor's native 16M maximum; these fixed benchmark
        # images are 1024 square and are not downsampled to the policy's 256 cap.
        engine=LocalQwenEngine(W/'critic_public_runtime_20261104/checkpoint',a.output/'calls',
                              max_pixels=16777216,max_input_tokens=8192,max_new_tokens=2048)
        request=read(a.request);K=sum(x.get('type')=='image' for m in request for x in m['content'])
        assert len(request)==1 and request[0]['role']=='user' and K in [2,4]
        result=engine.complete(request,purpose='native_critic_contract_qualification')
        (a.output/'raw_output.txt').write_text(result['raw']);(a.output/'text_output.txt').write_text(result['text'])
        rank=native_parser()(result['text'],K)
        dump(a.output/'parsed_ranking.json',dict(ranking=rank,parse_valid=rank is not None))
        receipt.update(status='completed',model_calls=1,ranking_parse_valid=rank is not None,engine_receipt=result['receipt_dir'])
    except Exception as e:
        receipt.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       network_connection_attempts=len(attempts))
        dump(a.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
