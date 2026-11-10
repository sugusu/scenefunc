"""CPU processor/mRoPE execution on actual native stage/batch observations.

Runs the SAME encoder helper used by NativePhaseTorchBackend, with no model
instantiation, generate, optimizer, SDK dispatch or renderer. Archived responses
are scoring-alignment probes, never reclassified as new or expert trajectories.
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

from decision_field_routes import route_decision_text
from native_feedback_batch import make_scoring_inputs
from native_multistage_phase import prepare_phase_messages, fingerprint
from native_phase_torch_backend import encode_native_phase_inputs
from routed_multimodal import frame_completed_decision


def sha(data):return hashlib.sha256(data).hexdigest()


def main(repo,out,checkpoint):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    packet_dir=repo/'runs/autodl/20261111_closed_loop_data/feedback_material_v2'
    packets=json.loads((packet_dir/'offline_feedback_packets.json').read_text())
    provenance=json.loads((packet_dir/'result.json').read_text())
    engine_source=repo/'paris/src/paris_local_run_results_20261104'
    sys.path.insert(0,str(engine_source))
    from local_vlm_sdpa import LocalQwenEngine
    from transformers import AutoConfig
    import torch
    assert not torch.cuda.is_initialized()
    config=AutoConfig.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    class ProcessorOnlyEngine(LocalQwenEngine):
        def _load_model(self):raise RuntimeError('No model load allowed')
        def complete(self,*a,**kw):raise RuntimeError('No generation allowed')
    needed={p['original_request']['archive_path'] for p in packets}
    for packet in packets:
        needed.update(v['archive_path'] for v in packet['input_images'])
        needed.add(str(Path(packet['original_request']['archive_path']).parent/'receipt.json'))
    conditions=[('native_joint','full_record'),('action_direct','full_record'),
                ('decision_independent','full_record'),('decision_conditioned','full_record'),
                ('decision_conditioned','without_principle_field'),('decision_conditioned','plan_field_only')]
    records=[];unique_encodes=0;baseline_matches=0
    with tempfile.TemporaryDirectory(prefix='native_phase_inputs_',dir='/dev/shm') as temp:
        root=Path(temp);selected={};wanted={};verified=set()
        with tarfile.open(provenance['source_archive'],'r|xz') as tar:
            for member in tar:
                if member.name=='paths.json':
                    index=json.load(tar.extractfile(member));sources={r['path']:r for r in index['files']}
                    assert needed<=set(sources)
                    for path in needed:wanted.setdefault(sources[path]['sha256'],[]).append(path)
                elif member.name.startswith('blobs/') and member.name[6:] in wanted:
                    data=tar.extractfile(member).read();digest=member.name[6:]
                    assert sha(data)==digest;verified.add(digest)
                    for path in wanted[digest]:
                        assert len(data)==sources[path]['bytes']
                        if path.endswith('.json'):selected[path]=json.loads(data)
                        else:
                            target=root/digest;target.write_bytes(data);selected[path]=target
        assert verified==set(wanted)
        engine=ProcessorOnlyEngine(checkpoint,root/'no_model',max_pixels=65536,max_input_tokens=16384,max_new_tokens=1024)
        tok=engine.processor.tokenizer;close=tok.convert_tokens_to_ids('</think>')
        for packet in packets:
            request=packet['original_request']['archive_path'];messages=copy.deepcopy(selected[request])
            original=selected[str(Path(request).parent/'receipt.json')]
            for message in messages:
                if not isinstance(message['content'],list):continue
                for item in message['content']:
                    if item.get('type')=='image':
                        key='image' if 'image' in item else 'path'
                        item[key]=str(selected[Path(item[key]).relative_to(index['original_root']).as_posix()])
                    elif item.get('type')=='image_url':
                        key=Path(item['image_url']['url']).relative_to(index['original_root']).as_posix()
                        item['image_url']['url']=str(selected[key])
            ids=packet['exact_output_token_ids'];assert ids.count(close)==1
            split=ids.index(close)+1;draft=tok.decode(ids[:split],skip_special_tokens=False)
            cache={};image_reference=None;count=0
            for condition,route in conditions:
                phases=['joint'] if condition=='native_joint' else ['action'] if condition=='action_direct' else ['decision','action']
                for phase in phases:
                    prefix='';route_meta=dict(status='not_consumed')
                    if phase=='action' and condition=='decision_conditioned':
                        routed,route_meta=route_decision_text(draft,route);prefix=frame_completed_decision(routed)
                    prompt=prepare_phase_messages(messages,phase)
                    key=fingerprint([prompt,prefix])
                    target=ids if phase=='joint' else ids[:split] if phase=='decision' else ids[split:]
                    if key not in cache:
                        inputs,encoded=encode_native_phase_inputs(engine.processor,engine.normalize_messages,
                            prompt,assistant_prefix=prefix,max_context_tokens=16384)
                        image_hashes={}
                        for name in ['pixel_values','image_grid_thw']:
                            if name in inputs:
                                t=inputs[name].contiguous().cpu()
                                image_hashes[name]=dict(shape=list(t.shape),dtype=str(t.dtype),sha256=sha(t.numpy().tobytes()))
                        if image_reference is None:image_reference=image_hashes
                        else:assert image_hashes==image_reference
                        assert [v['sha256'] for v in encoded['images']]==[v['sha256'] for v in packet['input_images']]
                        cache[key]=(inputs,encoded,image_hashes);unique_encodes+=1
                    inputs,encoded,image_hashes=cache[key]
                    response=torch.tensor([target],dtype=torch.long)
                    native,meta=make_scoring_inputs(inputs,response,config=config)
                    pos=native['position_ids'].contiguous().cpu()
                    assert native['input_ids'][0,-len(target):].tolist()==target
                    if phase=='joint':
                        assert encoded['prompt_sha256']==original['chat_template_sha256']
                        assert encoded['input_tokens']==original['input_tokens'];baseline_matches+=1
                    records.append(dict(source_id=packet['id'],condition=condition,route=route,phase=phase,
                        input_tokens=encoded['input_tokens'],response_tokens=len(target),
                        prompt_sha256=encoded['prompt_sha256'],images=encoded['images'],image_tensors=image_hashes,
                        all_conditions_same_original_image_bytes_and_tensors=True,
                        source_response_is_alignment_probe_not_new_sample=True,
                        response_sha256=sha(json.dumps(target).encode()),field_routing=route_meta,
                        mRoPE=dict(shape=list(pos.shape),sha256=sha(pos.numpy().tobytes()),
                                   original_library_method_executed=True,model_instantiated=False),
                        first_causal_logit_position=meta['prompt_tokens']-1,
                        last_causal_logit_position=meta['prompt_tokens']+len(target)-2,
                        no_truncation=True,training_admitted=False))
                    count+=1
            print(json.dumps(dict(source_id=packet['id'],phase_records=count,unique_encodes=len(cache))),flush=True)
            assert engine._model is None
    assert len(records)==190 and baseline_matches==19 and not torch.cuda.is_initialized()
    (out/'multistage_input_receipts.json.gz').write_bytes(gzip.compress((json.dumps(records,indent=2)+'\n').encode(),mtime=0))
    result=dict(status='actual_native_phase_backend_encoder_and_library_mRoPE_executed_CPU_only',
        original_policy_observations=19,source_image_references=55,conditions=6,
        phase_input_records=len(records),unique_actual_processor_encodes=unique_encodes,
        joint_original_native_prompt_hash_and_length_reproduced=baseline_matches,
        floor_zero_view_and_furniture_wall_five_view_inputs=True,
        declared_ceiling_actual_observations=0,separate_surface_workflow_encoded=False,
        actual_saved_D_schema_routing='Plain source thoughts pass through; routing does not invent source-principle records',
        maximum_prompt_plus_response_tokens=max(r['input_tokens']+r['response_tokens'] for r in records),
        actual_library_mRoPE_scoring_inputs=len(records),model_backend_forward_executed=False,
        old_source_responses_new_prefix_equivalence_claim=False,
        model_weights_loaded=False,local_VLM_forwards=0,model_generate_requests=0,
        training_updates=0,paid_API_calls=0,new_native_tool_calls=0,new_scene_successes=0,
        CUDA_initialized=False,seconds=time.monotonic()-tick,
        source_archive_sha256=provenance['source_archive_sha256'],
        encoder_helper_source_sha256=sha((repo/'scripts/autodl/native_phase_torch_backend.py').read_bytes()))
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.out.resolve(),a.checkpoint.resolve())
