"""Real CPU multimodal scoring inputs for report/echo/generic controls.

No model instance or teacher probabilities. All variants score the exact same
archived response, with the same original images and separately logged teacher
context. Token ownership masks do not assign failure responsibility.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time

from native_feedback_controls import MODES,feedback_context,ordered_call_token_spans,span_loss_mask
from native_feedback_batch import make_scoring_inputs
from native_phase_torch_backend import encode_native_phase_inputs


def sha(data):return hashlib.sha256(data).hexdigest()


def main(repo,out,checkpoint):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    pdir=repo/'runs/autodl/20261111_closed_loop_data/feedback_material_v2'
    packets=json.loads((pdir/'offline_feedback_packets.json').read_text())
    provenance=json.loads((pdir/'result.json').read_text())
    sys.path.insert(0,str(repo/'paris/src/paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine
    from transformers import AutoConfig
    import torch
    assert not torch.cuda.is_initialized()
    config=AutoConfig.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    class ProcessorOnly(LocalQwenEngine):
        def _load_model(self):raise RuntimeError('No model weight load allowed')
        def complete(self,*a,**kw):raise RuntimeError('No generation allowed')
    needed={p['original_request']['archive_path'] for p in packets}
    for p in packets:
        needed.update(v['archive_path'] for v in p['input_images'])
        needed.add(str(Path(p['original_request']['archive_path']).parent/'receipt.json'))
    records=[];totals={m:dict(prompt_tokens=0,response_tokens=0,feedback_presence_tokens=0) for m in MODES}
    with tempfile.TemporaryDirectory(prefix='native_feedback_controls_',dir='/dev/shm') as temp:
        root=Path(temp);selected={};wanted={};verified=set()
        with tarfile.open(provenance['source_archive'],'r|xz') as tar:
            for member in tar:
                if member.name=='paths.json':
                    index=json.load(tar.extractfile(member));source={r['path']:r for r in index['files']}
                    assert needed<=set(source)
                    for path in needed:wanted.setdefault(source[path]['sha256'],[]).append(path)
                elif member.name.startswith('blobs/') and member.name[6:] in wanted:
                    data=tar.extractfile(member).read();digest=member.name[6:]
                    assert sha(data)==digest;verified.add(digest)
                    for path in wanted[digest]:
                        assert len(data)==source[path]['bytes']
                        if path.endswith('.json'):selected[path]=json.loads(data)
                        else:
                            f=root/digest;f.write_bytes(data);selected[path]=f
        assert verified==set(wanted)
        engine=ProcessorOnly(checkpoint,root/'processor_only',max_pixels=65536,max_input_tokens=16384,max_new_tokens=1024)
        tok=engine.processor.tokenizer
        for packet in packets:
            request=packet['original_request']['archive_path'];messages=copy.deepcopy(selected[request])
            old=selected[str(Path(request).parent/'receipt.json')]
            for message in messages:
                if not isinstance(message['content'],list):continue
                for item in message['content']:
                    if item.get('type')=='image':
                        key='image' if 'image' in item else 'path'
                        item[key]=str(selected[Path(item[key]).relative_to(index['original_root']).as_posix()])
                    elif item.get('type')=='image_url':
                        item['image_url']['url']=str(selected[Path(item['image_url']['url']).relative_to(index['original_root']).as_posix()])
            ids=packet['exact_output_token_ids'];response=torch.tensor([ids],dtype=torch.long)
            spans=ordered_call_token_spans(tok,ids,packet['source_tool_calls'])
            call_mask=span_loss_mask(len(ids),spans,list(range(len(spans))))
            reference_image_tensors=None
            for mode in MODES:
                extra,meta=feedback_context(packet['source_tool_calls'],packet['original_feedback'],mode)
                teacher=copy.deepcopy(messages)+([] if extra is None else [extra])
                inputs,encoded=encode_native_phase_inputs(engine.processor,engine.normalize_messages,
                    teacher,assistant_prefix='',max_context_tokens=16384)
                image_hashes={}
                for key in ('pixel_values','image_grid_thw'):
                    if key in inputs:
                        t=inputs[key].contiguous().cpu();image_hashes[key]=dict(shape=list(t.shape),sha256=sha(t.numpy().tobytes()))
                if reference_image_tensors is None:reference_image_tensors=image_hashes
                assert image_hashes==reference_image_tensors
                assert [r['sha256'] for r in encoded['images']]==[r['sha256'] for r in packet['input_images']]
                if mode=='no_extra_context':
                    assert encoded['prompt_sha256']==old['chat_template_sha256']
                    assert encoded['input_tokens']==old['input_tokens']
                native,nmeta=make_scoring_inputs(inputs,response,config=config)
                prefix=nmeta['prompt_tokens'];p=native['position_ids'].contiguous().cpu()
                # Each selected output token j is predicted at causal logit
                # position prefix+j-1. Unselected causal inputs stay present.
                call_logits=[prefix+j-1 for j,on in enumerate(call_mask) if on]
                assert len(call_logits)==sum(s['token_count'] for s in spans)
                assert native['input_ids'][0,-len(ids):].tolist()==ids
                rec=dict(source_id=packet['id'],condition=mode,input_tokens=prefix,response_tokens=len(ids),
                    prompt_sha256=encoded['prompt_sha256'],original_response_sha256=sha(json.dumps(ids).encode()),
                    images=encoded['images'],image_tensors=image_hashes,images_exactly_same_across_conditions=True,
                    last_EOS_causal_logit_position=prefix+len(ids)-2,
                    all_response_scoring_positions=[prefix-1,prefix+len(ids)-1],
                    call_ownership_causal_logit_positions=call_logits,call_ownership_tokens=len(call_logits),
                    full_response_preserved_even_outside_call_mask=True,
                    mRoPE_shape=list(p.shape),mRoPE_sha256=sha(p.numpy().tobytes()),
                    original_library_mRoPE_executed=True,no_model_instantiated=True,
                    hindsight_feedback_exists_mask=meta['hindsight_distillation_mask'],
                    mask_is_not_teacher_correctness_or_action_blame=True,
                    independent_failure_target_selection=None,teacher_logits=None,
                    source_is_historical_greedy=True,training_admitted=False)
                records.append(rec)
                totals[mode]['prompt_tokens']+=prefix;totals[mode]['response_tokens']+=len(ids)
                totals[mode]['feedback_presence_tokens']+=len(ids)*meta['hindsight_distillation_mask']
            assert engine._model is None
            print(json.dumps(dict(id=packet['id'],conditions=len(MODES),source_calls=len(spans))),flush=True)
    assert len(records)==95 and not torch.cuda.is_initialized()
    (out/'input_receipts.json.gz').write_bytes(gzip.compress((json.dumps(records,indent=2)+'\n').encode(),mtime=0))
    text_config=getattr(config,'text_config',config);vocab=int(text_config.vocab_size)
    fp32_all=sum(r['response_tokens'] for r in records)*vocab*4
    result=dict(status='five_feedback_information_contexts_actual_CPU_processor_and_mRoPE',
        source_policy_observations=19,source_image_references=55,conditions=list(MODES),actual_processor_encodes=95,
        original_no_extra_prompt_hashes_and_lengths_reproduced=19,
        original_image_tensors_shared_across_conditions=True,same_response_sequences_under_all_conditions=True,
        maximum_combined_prompt_response_tokens=max(r['input_tokens']+r['response_tokens'] for r in records),
        token_totals_by_condition=totals,per_call_causal_logit_positions_bound=True,
        independent_task_correctness_labels=0,teacher_distributions_computed=0,
        source_archive_sha256=provenance['source_archive_sha256'],
        vocab_size=vocab,full_FP32_all_teacher_logits_theoretical_bytes=fp32_all,
        logits_bytes_are_arithmetic_not_measured_peak_memory=True,full_logits_saved=False,
        model_weights_loaded=False,local_VLM_forwards=0,new_model_generations=0,
        training_updates=0,paid_API_calls=0,new_native_SDK_calls=0,new_complete_scene_successes=0,
        CUDA_initialized=False,seconds=time.monotonic()-tick,
        feedback_control_code_sha256=sha((repo/'scripts/autodl/native_feedback_controls.py').read_bytes()))
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.out.resolve(),a.checkpoint.resolve())
