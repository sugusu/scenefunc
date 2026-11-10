"""Audit fixed native reports, action echoes and exact per-call token ownership.

No generated labels, model, optimizer or simulated task scores. Exact literal
overlap is NOT evidence of a model shortcut; it identifies a necessary control.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import time

from native_feedback_controls import MODES,feedback_context,ordered_call_token_spans,span_loss_mask


def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(data):return hashlib.sha256(data).hexdigest()


def leaves(value,path=()):
    if isinstance(value,dict):
        for key,child in value.items():yield from leaves(child,path+(key,))
    elif isinstance(value,list):
        for key,child in enumerate(value):yield from leaves(child,path+(key,))
    else:yield path,value


def main(repo,out,checkpoint):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    source=repo/'runs/autodl/20261111_closed_loop_data/feedback_material_v2/offline_feedback_packets.json'
    original=source.read_bytes();packets=json.loads(original)
    from transformers import AutoTokenizer
    import torch
    assert not torch.cuda.is_initialized()
    tok=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    rows=[];operations=[];contexts=[];categories=collections.Counter();context_tokens=collections.Counter()
    for packet in packets:
        ids=packet['exact_output_token_ids'];calls=packet['source_tool_calls'];feedback=packet['original_feedback']
        assert tok.decode(ids,skip_special_tokens=False)==packet['raw_proposal']
        spans=ordered_call_token_spans(tok,ids,calls)
        full_mask=span_loss_mask(len(ids),spans,list(range(len(spans))))
        union=[False]*len(ids)
        for span,report in zip(spans,feedback):
            single=span_loss_mask(len(ids),spans,[span['call_index']])
            assert sum(single)==span['token_count'] and not any(a and b for a,b in zip(union,single))
            union=[a or b for a,b in zip(union,single)]
            output=report['output'];categories[report['category']]+=1
            arg_leaves=list(leaves(span['submitted_call']['arguments']));out_leaves=list(leaves(output))
            matches=[]
            for path,value in arg_leaves:
                if value is None or isinstance(value,bool) or value=='' or (isinstance(value,(int,float)) and value==0):continue
                hits=[list(op) for op,ov in out_leaves if type(ov)==type(value) and ov==value]
                if hits:matches.append(dict(argument_path=list(path),value=value,exact_output_leaf_paths=hits))
            operations.append(dict(source_id=packet['id'],**span,
                corresponding_output=report,exact_argument_leaf_overlaps=matches,
                echo_overlap_is_not_proof_of_copying=True,
                output_exists_not_failure_relevance=True,
                native_report_category=report['category'],
                independent_operation_correctness=None,selected_for_training=False,
                response_length=len(ids),full_original_response_kept_as_causal_input=True))
        assert union==full_mask
        for mode in MODES:
            extra,meta=feedback_context(calls,feedback,mode)
            if extra:
                payload=json.loads(extra['content'].split('\n',1)[1])
                if 'submitted_calls' in payload:assert payload['submitted_calls']==calls
                if 'reported_native_outputs' in payload:assert payload['reported_native_outputs']==feedback
            n=0 if extra is None else len(tok.encode(extra['content'],add_special_tokens=False))
            context_tokens[mode]+=n
            contexts.append(dict(source_id=packet['id'],mode=mode,message=extra,metadata=meta,
                                 supplementary_text_tokens=n,original_response_sha256=sha(json.dumps(ids).encode()),
                                 original_task_and_images_unchanged=True))
        rows.append(dict(source_id=packet['id'],response_tokens=len(ids),calls=len(spans),
                         call_span_tokens=sum(full_mask),other_response_tokens=len(ids)-sum(full_mask),
                         all_original_calls_bound=True,
                         actual_failed_or_successful_scene_label=None,
                         full_feedback_loss_presence_mask=int(bool(feedback)),
                         failure_relevance_selection_made=False))
    assert len(rows)==19 and len(operations)==35 and len(contexts)==95
    assert source.read_bytes()==original and not torch.cuda.is_initialized()
    dump(out/'call_token_ownership.json',operations)
    dump(out/'response_span_summary.json',rows)
    data=(json.dumps(contexts,ensure_ascii=False,indent=2)+'\n').encode()
    (out/'prospective_teacher_context_controls.json.gz').write_bytes(gzip.compress(data,mtime=0))
    result=dict(status='fixed_actual_native_reports_bound_to_calls_and_five_teacher_context_controls',
        source_policy_turns=19,source_calls_and_outputs=35,conditions=list(MODES),teacher_context_records=95,
        full_response_tokens=sum(r['response_tokens'] for r in rows),
        exact_tool_call_span_tokens=sum(r['call_span_tokens'] for r in rows),
        noncall_response_tokens=sum(r['other_response_tokens'] for r in rows),
        original_call_tokens_covered_exactly_once=True,
        tool_names_reported_again=sum(o['name']==o['corresponding_output']['name'] for o in operations),
        calls_with_nontrivial_exact_argument_leaf_overlap=sum(bool(o['exact_argument_leaf_overlaps']) for o in operations),
        exact_nontrivial_argument_leaf_matches=sum(len(o['exact_argument_leaf_overlaps']) for o in operations),
        overlap_excludes_null_bool_empty_and_numeric_zero=True,
        overlap_does_not_prove_model_uses_or_benefits_from_copying=True,
        feedback_categories=dict(categories),supplementary_text_tokens_by_condition=dict(context_tokens),
        feedback_present_turns=sum(r['full_feedback_loss_presence_mask'] for r in rows),
        independent_correctness_or_causal_blame_labels_added=0,
        actual_teacher_logits_computed=False,training_examples_admitted=0,
        multimodal_image_encoding_in_this_audit=False,followup_multimodal_encoding_required=True,
        new_native_SDK_calls=0,model_weights_loaded=False,local_VLM_forwards=0,
        model_generate_requests=0,training_updates=0,paid_API_calls=0,CUDA_initialized=False,
        seconds=time.monotonic()-tick,source_sha256=sha(original))
    dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.out.resolve(),a.checkpoint.resolve())
