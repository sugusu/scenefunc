"""Closed-input likelihood audit; no generation, gradients, retries or new scenes.

Uses actual upstream no-thinking SFT message serialization and the frozen local
checkpoint. Six sequential teacher-forced forwards score two legal custom tool
bindings in each of three saved contexts. This is not a policy evaluation or a
new learning method. Raw program likelihood is not a feasibility probability.
"""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
from types import SimpleNamespace
from typing import Any

W=Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
R=Path(__file__).resolve().parents[2]
SFT=W/'autodl_continuation_20261110/training_source_review_20261110/upstream/src/sft_dataset.py'
IMPL=W/'inference_autodl_20261103/runtime/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py'

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def native_build():
    fn=[n for n in ast.parse(SFT.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_build_messages']
    assert len(fn)==1
    env={'Any':Any,'StepSample':SimpleNamespace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=fn,type_ignores=[])),str(SFT),'exec'),env)
    return env['_build_messages']

def candidates(messages):
    user=messages[1]['content'];text=user[-1]['text'];assert text.startswith('## Task\n')
    task,memory=text[len('## Task\n'):].split('\n\n## Memory\n',1)
    s=SimpleNamespace(stage='furniture',system=messages[0]['content'],task_context=task,memory_str=memory,
        scene_images=[x['image'] for x in user if x['type']=='image'],image_path=None,assistant_content='')
    out={}
    for name,surface,seat in [('shared','meal_table_0','meal_chair_n_0'),('independent','new_work_table','new_work_chair')]:
        args=dict(work_surface=surface,work_seat=seat,meal_surface='meal_table_0',
            meal_seats=['meal_chair_n_0','meal_chair_s_0'],material_persists_during_meal=True,resident_works_while_meal=False)
        s.tool_calls=[dict(name='allocate_resources',arguments=args)]
        full,prompt=native_build()(s,no_thinking=True)
        assert prompt==messages,'Native SFT prefix must match the actual request exactly'
        out[name]=full
    return out

def prepare(root,previous):
    root.mkdir(exist_ok=False);old=read(previous/'model_probe.json');assert old['status']=='completed'
    contexts=[]
    for x in old['attempts']:
        p=Path(x['receipt_dir'])/'request.json';m=read(p)
        full=candidates(m)
        name=f"{x['task']}_turn_{x['turn']}"
        dump(root/(name+'_request.json'),m)
        for c,msg in full.items():dump(root/(name+'_'+c+'_sft.json'),msg)
        contexts.append(dict(id=name,task=x['task'],turn=x['turn'],source_request=str(p),source_sha256=sha(p),
            source_generation_receipt=str(Path(x['receipt_dir'])/'receipt.json'),
            feedback_contains_optimal_count=x['turn']>0,first_generated_plan=x['feedback']['plan']))
    assert len(contexts)==3
    dump(root/'protocol.json',dict(contexts=contexts,source_sft=str(SFT),source_sft_sha256=sha(SFT),
        source_model_forward=str(IMPL),source_model_forward_sha256=sha(IMPL),
        max_teacher_forwards=6,total_process_cap_seconds=360,per_forward_alarm_seconds=100,
        checkpoint=str(W/'critic_public_runtime_20261104/checkpoint'),
        checkpoint_revision='63e235a0c0d6e5bc2f8f3485214df4457fd2e672',
        saved_processor_max_pixels=16777216,max_input_tokens=24576,
        logits_to_keep='Target length+1; never full17600position vocabulary tensor',
        use_cache=False,do_sample=False,training=False,generation=False,paid_API_calls=0,
        scope='Existing custom two-skill allocation contexts on one exposed furnished room; not original native Actor API',
        measures=['Full target log probability','Per-token mean log probability','First differing target token odds on an identical prefix'],
        constraints=['No altered source requests','No new generated requests or quality retry','No parameter update','No inference from normalized program odds to true plan probability'],
        maximum_memory_limit_bytes=26843545600))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--previous',type=Path);ap.add_argument('--prepare',action='store_true');ap.add_argument('--encode-only',action='store_true');a=ap.parse_args()
    root=a.root.resolve()
    if a.prepare:prepare(root,a.previous.resolve());return
    protocol=read(root/'protocol.json')
    assert sha(SFT)==protocol['source_sft_sha256'] and sha(IMPL)==protocol['source_model_forward_sha256']
    for x in protocol['contexts']:assert sha(x['source_request'])==x['source_sha256']
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    network=[]
    def reject(*_,**__):network.append(True);raise RuntimeError('Offline audit forbids network')
    socket.socket.connect=reject;socket.create_connection=reject
    sys.path.insert(0,str(W/'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine
    import torch
    engine=LocalQwenEngine(Path(protocol['checkpoint']),root/'unused_generation_directory',max_pixels=16777216,max_input_tokens=24576,max_new_tokens=1)
    encoding=[]
    for ctx in protocol['contexts']:
        request=read(root/(ctx['id']+'_request.json'));prompt,prompt_receipt=engine.encode(request);P=prompt['input_ids'].shape[1]
        targets={}
        for name in ['shared','independent']:
            full=read(root/(ctx['id']+'_'+name+'_sft.json'))
            normalized,images,evidence=engine.normalize_messages(full)
            text=engine.processor.apply_chat_template(normalized,tokenize=False,add_generation_prompt=False)
            inputs=engine.processor(text=[text],images=images,return_tensors='pt')
            assert torch.equal(inputs['input_ids'][:,:P],prompt['input_ids'])
            assert inputs['input_ids'].shape[1]<=24576
            ids=inputs['input_ids'][0,P:].tolist();assert ids
            targets[name]=dict(token_ids=ids,target_tokens=len(ids),serialized_assistant=full[-1],
                full_token_count=inputs['input_ids'].shape[1],chat_template_sha256=hashlib.sha256(text.encode()).hexdigest())
            del inputs
        common=0
        for l,r in zip(targets['shared']['token_ids'],targets['independent']['token_ids']):
            if l!=r:break
            common+=1
        assert common<min(x['target_tokens'] for x in targets.values())
        encoding.append(dict(context=ctx['id'],prompt_tokens=P,prompt_encoding=prompt_receipt,targets=targets,
            common_prefix_tokens=common,first_differing_decoded={n:engine.processor.tokenizer.decode([v['token_ids'][common]]) for n,v in targets.items()}))
        del prompt
    dump(root/'encoding_preflight.json',dict(rows=encoding,all_prompt_prefixes_exact=True,weights_loaded=False,model_forwards=0))
    if a.encode_only:return
    if (root/'started.json').exists():raise FileExistsError('No implicit audit rerun')
    lock=(root/'exclusive_model.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();dump(root/'started.json',dict(pid=os.getpid(),time_unix=time.time(),no_retry=True))
    scores=[];attempts=0
    def expired(*_):raise TimeoutError('Bounded audit forward expired, no retry')
    signal.signal(signal.SIGALRM,expired)
    try:
        torch.manual_seed(0);signal.alarm(100);engine._load_model();signal.alarm(0)
        from torch.nn.attention import sdpa_kernel,SDPBackend
        for entry in encoding:
            perctx=[];common=entry['common_prefix_tokens']
            for name in ['shared','independent']:
                assert attempts<6;attempts+=1;t=time.monotonic();signal.alarm(100)
                full=read(root/(entry['context']+'_'+name+'_sft.json'));norm,images,_=engine.normalize_messages(full)
                text=engine.processor.apply_chat_template(norm,tokenize=False,add_generation_prompt=False)
                assert hashlib.sha256(text.encode()).hexdigest()==entry['targets'][name]['chat_template_sha256']
                inputs=engine.processor(text=[text],images=images,return_tensors='pt').to('cuda:0')
                T=entry['targets'][name]['target_tokens'];target=inputs['input_ids'][:,-T:]
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    out=engine._model(**inputs,use_cache=False,logits_to_keep=T+1)
                    logits=out.logits[:,:-1,:].float()
                    assert logits.shape[1]==T
                    per_token=logits.log_softmax(-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)[0]
                    ids={n:x['token_ids'][common] for n,x in entry['targets'].items()}
                    decision={n:float(logits[0,common,i]-logits[0,common].logsumexp(-1)) for n,i in ids.items()}
                    values=per_token.cpu().tolist()
                torch.cuda.synchronize();signal.alarm(0)
                row=dict(context=entry['context'],candidate=name,target_tokens=T,per_token_logp=values,
                    sum_logp=sum(values),mean_logp=sum(values)/T,
                    first_different_token_logp=decision,prompt_tokens=entry['prompt_tokens'],
                    full_logit_positions_returned=T+1,elapsed_seconds=time.monotonic()-t,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    no_grad=True,use_cache=False,source_messages_unchanged=True)
                scores.append(row);perctx.append(row)
                dump(root/'likelihood_scores.json',dict(status='running',rows=scores,teacher_forwards=attempts))
                del logits,out,inputs,target,per_token;torch.cuda.empty_cache()
            # Equal prefixes must give equal first-decision odds, independently
            # of subsequent teacher-forced tokens (causal attention check).
            error=max(abs(perctx[0]['first_different_token_logp'][n]-perctx[1]['first_different_token_logp'][n]) for n in ['shared','independent'])
            dump(root/(entry['context']+'_causal_check.json'),dict(same_prefix_token_logp_max_difference=error,
                within_1e_2=error<=1e-2,comparison_scope='Numerical causal-prefix consistency, not a policy quality result'))
        dump(root/'likelihood_scores.json',dict(status='completed',rows=scores,teacher_forwards=attempts,
            model_generate_calls=0,training_updates=0,paid_API_calls=0,network_attempts=len(network),
            wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    except BaseException as e:
        dump(root/'likelihood_scores.json',dict(status='failed',rows=scores,teacher_forwards=attempts,
            error=f'{type(e).__name__}: {e}',wall_seconds=time.monotonic()-start,no_retry=True))
        raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
