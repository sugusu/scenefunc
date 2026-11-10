"""Execute original request/serializer helpers, never annotator model calls.

Uses the fixed eight actual floor policy turns (the text-only native stage).
Mapping archived observations to the published SFT sample fields is explicit;
this is not a claim that native rollout and SFT prompts were byte-identical.
Original author source stays in its internal RAM directory, not this repo.
"""
import argparse
import ast
import base64
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any


def digest(data):return hashlib.sha256(data).hexdigest()


def load_functions(path,names,namespace):
    text=path.read_text();tree=ast.parse(text)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert [n.name for n in functions]==names
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return [dict(name=n.name,first_line=n.lineno,last_line=n.end_lineno,
                 function_sha256=digest(ast.get_source_segment(text,n).encode())) for n in functions]


def user_text(messages):
    texts=[]
    for m in messages:
        if m['role']!='user':continue
        c=m['content']
        if isinstance(c,str):texts.append(c)
        else:texts.extend(x['text'] for x in c if x.get('type')=='text')
    return '\n'.join(texts)


def main(repo,root,out):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    source_manifest=repo/'runs/autodl/20261111_decision_source_value/SceneDesigner_read_sources.json'
    source=json.loads(source_manifest.read_text())
    for row in source['source_files']:
        assert digest((root/row['path']).read_bytes())==row['sha256']
    namespace=dict(json=json,Counter=Counter,Path=Path,base64=base64,Any=Any,
                   REPO=root,StepSample=SimpleNamespace)
    # VIEW_NOTE is a literal constant in the original source. Other module
    # imports/API globals are deliberately not executed.
    rewrite=root/'tools/rewrite_cot_gpt.py'
    for node in ast.parse(rewrite.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VIEW_NOTE' for t in node.targets):
            namespace['VIEW_NOTE']=ast.literal_eval(node.value)
    original=[]
    original+=load_functions(rewrite,['b64img','turn_images','build_turn_request'],namespace)
    original+=load_functions(root/'src/sft_dataset.py',['_build_messages'],namespace)
    trace=repo/'runs/autodl/20261110_complete_native_runtime/review_traces/floor_trace.json'
    turns=json.loads(trace.read_text());assert len(turns)==8
    results=[]
    for row in turns:
        obs=row['observation'];prop=row['proposal']
        assert not obs['image_paths']
        target=dict(role='assistant',content=prop['think'],tool_calls=prop['tool_calls'])
        # No current-result or future marker belongs to the frozen before-state
        # memory. The probe tests this helper's consumption of those fields.
        request=namespace['build_turn_request']([target],0,obs['room_prompt'],obs['system'],False,
                    memory_str=obs['memory_str'],stage='floor_plan')
        text=request[0]['text']
        expected=json.dumps(prop['tool_calls'] or [],ensure_ascii=False)
        assert f'ANNOTATION TARGET TOOL CALLS:\n{expected}\n' in text
        probes=[]
        for kind in ['prior_assistant_thought','current_results_argument','future_turn']:
            marker='__OFFLINE_BOUNDARY_'+kind.upper()+'__'
            local=[copy.deepcopy(target)];kw={}
            if kind=='prior_assistant_thought':local[0]['content']=marker
            if kind=='current_results_argument':kw['results_after']=[dict(output=marker)]
            if kind=='future_turn':local.append(dict(role='tool_results',results=[dict(output=marker)]))
            changed=namespace['build_turn_request'](local,0,obs['room_prompt'],obs['system'],False,
                        memory_str=obs['memory_str'],stage='floor_plan',**kw)
            assert changed==request and marker not in json.dumps(changed)
            probes.append(dict(changed_field=kind,request_identical=True,marker_not_consumed=True))
        sample=SimpleNamespace(stage='floor_plan',task_context=obs['room_prompt'],memory_str=obs['memory_str'],
                               system=obs['system'],scene_images=[],image_path=None,
                               assistant_content=prop['think'],tool_calls=prop['tool_calls'])
        full,prompt=namespace['_build_messages'](sample,no_thinking=False)
        action_only,prompt_ao=namespace['_build_messages'](sample,no_thinking=True)
        assert prompt==prompt_ao and full[-1].get('tool_calls')==action_only[-1].get('tool_calls')
        assert action_only[-1]['content']==''
        policy_text=user_text(prompt)
        assert 'POLICY USER MESSAGE:\n'+policy_text+'\n\nANNOTATION TARGET TOOL CALLS:' in text
        results.append(dict(id='floor:'+str(row['turn']),target_action_count=len(prop['tool_calls']),
                            original_constructor_request_sha256=digest(json.dumps(request,ensure_ascii=False).encode()),
                            source_target_calls_visible_to_annotator=True,
                            source_current_result_visible=False,source_future_visible=False,
                            helper_checks=probes,
                            original_SFT_loader_same_action_with_or_without_thinking=True,
                            annotation_task_memory_matches_original_SFT_wrapper=True,
                            annotation_wrapper_equals_recorded_native_user_text=policy_text==user_text(obs['messages']),
                            images=0,new_annotations_generated=0))
    result=dict(status='original_annotation_and_SFT_helpers_exercised_without_model_calls',
                source_pin=source['source_pin'],original_functions=original,
                fixed_actual_floor_turns=8,boundary_checks=24,rows=results,
                original_SFT_prompt_wrapper_matches_annotation=all(x['annotation_task_memory_matches_original_SFT_wrapper'] for x in results),
                recorded_native_user_text_matches_published_SFT_wrapper=sum(x['annotation_wrapper_equals_recorded_native_user_text'] for x in results),
                no_thinking_preserves_target_tool_calls=True,source_target_action_is_annotation_privilege=True,
                current_or_future_feedback_injected_into_constructor=False,
                no_full_memory_builder_or_multimodal_annotation_reproduction=True,
                author_source_redistributed=False,generated_expert_rationales=0,
                local_VLM_forwards=0,paid_API_calls=0,training_updates=0,native_tool_calls=0,
                source_trace_sha256=digest(trace.read_bytes()),seconds=time.monotonic()-tick,
                scope='Original pure helper bodies, eight text-only archived floor observations mapped to SFT fields; not the annotator/reviewer models or full pipeline.')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','original_functions']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.root.resolve(),a.out.resolve())
