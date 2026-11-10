"""CPU replay of 19 real source outputs through a common stage/batch policy.

Not model inference, native SDK execution, geometry replay or a method-quality
comparison. Original ToolEnvironment.step and memory handlers consume saved
results via an explicitly recorded dispatch substitute. The policy API uses
the same original observation; no answers are loaded by production policy.
"""
import argparse
import ast
import asyncio
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

from native_multistage_phase import (NativeMultiStagePolicy,NativePhaseFailure,
    classify_native_batch,execute_native_batch)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):
    data=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    path.write_bytes(gzip.compress(data,mtime=0) if path.suffix=='.gz' else data)


class ArchivedBackend:
    execution_kind='archived_sequence_replay'
    def __init__(self,tokenizer,eos_ids,packet):
        self.tokenizer=tokenizer;self.eos_ids=eos_ids;self.packet=packet
        self.requests=[]
    def generate(self,messages,*,assistant_prefix,phase,token_limit,seconds_limit):
        ids=self.packet['exact_output_token_ids']
        close=self.tokenizer.convert_tokens_to_ids('</think>')
        assert ids.count(close)==1
        split=ids.index(close)+1
        answer=ids if phase=='joint' else ids[:split] if phase=='decision' else ids[split:]
        row=dict(phase=phase,messages=messages,assistant_prefix=assistant_prefix,
                 output_token_limit=token_limit,source_token_interval=[0,len(ids)] if phase=='joint'
                 else [0,split] if phase=='decision' else [split,len(ids)])
        self.requests.append(row)
        return dict(output_ids=answer,input_tokens=0,execution_kind=self.execution_kind,
                    actual_encoded_input_tokens=0,source_span=row['source_token_interval'],
                    no_probability_equivalence_claim=True,
                    note='Actual original token span, not sampled under this new phase/context')


def original_functions(root):
    p=root/'Designer/agent_rollout.py';source=p.read_text();tree=ast.parse(source)
    nodes=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('_TOOL_CALL_RE','_THINK_RE') for x in n.targets):nodes.append(n)
        if isinstance(n,ast.FunctionDef) and n.name=='parse_assistant_text':nodes.append(n)
    ns=dict(re=re,json=json)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    p2=root/'src/agent_rollout_decoupled.py';tree=ast.parse(p2.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ToolEnvironment')
    step=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='step')
    step_ns=dict(json=json,sys=sys,traceback=traceback,dispatch=None)
    exec(compile(ast.Module(body=[step],type_ignores=[]),str(p2),'exec'),step_ns)
    mp=root/'src/memory.py';spec=importlib.util.spec_from_file_location('original_stage_memory_replay_only',mp)
    mem=importlib.util.module_from_spec(spec);spec.loader.exec_module(mem)
    return ns['parse_assistant_text'],step_ns,mem,dict(
        parser=dict(path=str(p),sha256=sha(p)),
        original_batch_step=dict(path=str(p2),sha256=sha(p2),first_line=step.lineno,last_line=step.end_lineno),
        memory=dict(path=str(mp),sha256=sha(mp)))


async def main(repo,out,checkpoint):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    source=repo/'runs/autodl/20261110_complete_native_runtime'
    runtime=source/'runtime'
    parser,step_ns,memory,source_hashes=original_functions(runtime)
    packet_path=repo/'runs/autodl/20261111_closed_loop_data/feedback_material_v2/offline_feedback_packets.json'
    packets={p['id']:p for p in json.loads(packet_path.read_text())}
    traces={s:json.loads((source/f'review_traces/{s}_trace.json').read_text()) for s in ('floor','furniture','wall')}
    protocol=json.loads((source/'trial_protocol_v2.json').read_text())
    initial_call=json.loads(Path(protocol['initial_floor_proposal']).read_text())['tool_calls'][0]
    initial_output=json.loads(Path(protocol['initial_floor_tool_results']).read_text())[0]['output']
    assert sha(Path(protocol['initial_floor_proposal']))==protocol['initial_floor_proposal_sha256']
    assert sha(Path(protocol['initial_floor_tool_results']))==protocol['initial_floor_tool_results_sha256']
    from transformers import AutoTokenizer
    import torch
    assert not torch.cuda.is_initialized()
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    gc=json.loads((checkpoint/'generation_config.json').read_text());eos=gc['eos_token_id']
    eos=eos if isinstance(eos,list) else [eos]
    conditions=[('native_joint','full_record'),('action_direct','full_record'),
                ('decision_independent','full_record'),('decision_conditioned','full_record'),
                ('decision_conditioned','without_principle_field'),('decision_conditioned','plan_field_only')]
    cohort=[];details=[]
    # The original batch method resolves dispatch via this namespace; no SDK,
    # socket or geometry execution occurs inside the substitute.
    active={}
    async def recorded_dispatch(tool_map,name,args):
        k=active['offset'];row=active['source'];expected=row['proposal']['tool_calls'][k]
        assert name==expected['name'] and args==expected['arguments']
        result=row['results'][k];assert name==result['name']
        active['offset']+=1
        output=copy.deepcopy(result['output'])
        return json.dumps(output,ensure_ascii=False) if isinstance(output,(dict,list)) else output
    step_ns['dispatch']=recorded_dispatch
    for condition,route in conditions:
        method=condition+':'+route;counts=dict(policy_proposals=0,replay_backend_requests=0,
            actual_model_generate_requests=0,original_batch_method_invocations=0,
            saved_dispatch_outputs_consumed=0,pre_action_memory_texts_equal=0,
            proposed_stops=0,reference_tokens_replayed=0)
        for short,trace in traces.items():
            stage=trace[0]['observation']['stage']
            env=SimpleNamespace(stage=stage,step_idx=trace[0]['observation']['step_idx'],
                                mem=memory.empty_memory(stage),tool_map={},_handler=memory.HANDLERS[stage])
            env.step=MethodType(step_ns['step'],env)
            if stage=='floor_plan':env._handler(env.mem,initial_call['name'],initial_call['arguments'],initial_output)
            for row in trace:
                obs=row['observation'];key=short+':'+str(row['turn']);packet=packets[key]
                assert memory.render_memory(stage,env.mem)==obs['memory_str']
                counts['pre_action_memory_texts_equal']+=1
                assert env.step_idx==obs['step_idx']
                backend=ArchivedBackend(tokenizer,eos,packet)
                policy=NativeMultiStagePolicy(backend,native_parser=parser,condition=condition,
                    field_route=route,total_output_tokens=1024,decision_tokens=512,action_tokens=512,
                    seconds=10,max_calls_per_batch=64)
                proposal=policy.generate(obs)
                assert proposal['tool_calls']==row['proposal']['tool_calls']
                assert proposal['full_scene_success'] is None
                counts['policy_proposals']+=1
                rec=proposal['transport_receipt']
                counts['replay_backend_requests']+=rec['backend_requests']
                counts['actual_model_generate_requests']+=rec['actual_model_generate_requests']
                counts['reference_tokens_replayed']+=rec['generated_or_replayed_output_tokens']
                for phase_row in rec['phases']:
                    if phase_row['phase']=='action' and condition=='decision_conditioned':
                        routing=phase_row['field_routing']['status']
                        label='generated_record_schema_matched' if routing=='structured_record_routed' else 'generated_record_common_passthrough'
                        counts[label]=counts.get(label,0)+1
                active.update(source=row,offset=0)
                returned,execution=await execute_native_batch(env,proposal,remaining_native_calls=64)
                assert returned==row.get('results',[])
                if proposal['tool_calls']:
                    assert active['offset']==len(proposal['tool_calls'])
                    counts['original_batch_method_invocations']+=1
                    counts['saved_dispatch_outputs_consumed']+=len(returned)
                else:counts['proposed_stops']+=1
                details.append(dict(method=method,source_id=key,
                                    exact_source_call_arguments_and_order_preserved=True,
                                    original_memory_before_action_equal=True,
                                    policy=rec,execution=execution,
                                    dispatcher_kind='saved_outputs_only_no_SDK'))
        assert counts['policy_proposals']==19 and counts['saved_dispatch_outputs_consumed']==35
        assert counts['actual_model_generate_requests']==0
        cohort.append(dict(method=method,**counts))
    # Small safety/accounting checks tied to actual outputs, not a new scene
    # task set or a collection of hand-written agent answers.
    failures=[]
    p=packets['furniture:1'];row=traces['furniture'][1];backend=ArchivedBackend(tokenizer,eos,p)
    policy=NativeMultiStagePolicy(backend,native_parser=parser,condition='native_joint',
         total_output_tokens=1024,decision_tokens=512,action_tokens=512,seconds=10,max_calls_per_batch=64)
    proposal=policy.generate(row['observation'])
    forbidden_calls=[]
    async def must_not_dispatch(calls):forbidden_calls.append(calls);raise AssertionError('Must not dispatch')
    env=SimpleNamespace(stage='furniture',step_idx=row['observation']['step_idx'],step=must_not_dispatch)
    for reason,remaining,delta in [('insufficient_batch_budget',3,0),('stale_step_counter',64,1)]:
        env.step_idx=row['observation']['step_idx']+delta
        try:await execute_native_batch(env,proposal,remaining_native_calls=remaining)
        except ValueError:failures.append(dict(case=reason,rejected_before_dispatch=True))
        else:raise AssertionError('Unsafe batch admitted')
    assert not forbidden_calls
    bad=classify_native_batch(tokenizer,p['exact_output_token_ids'][:-1],phase='joint',eos_ids=eos,native_parser=parser)
    assert bad['kind']=='invalid'
    failures.append(dict(case='actual_response_without_terminal_EOS',invalid_not_STOP=True))
    raw=tokenizer.decode(p['exact_output_token_ids'][:-1],skip_special_tokens=False)
    for tag in ('<tool_call','</tool_call'):
        # Serialization corruption of one actual source response, not a new
        # scene/example and not an invented model decision.
        broken=raw.split('<tool_call>',1)[0]+tag
        corrupt=tokenizer.encode(broken,add_special_tokens=False)+[eos[0]]
        parsed=classify_native_batch(tokenizer,corrupt,phase='joint',eos_ids=eos,native_parser=parser)
        assert parsed['kind']=='invalid'
        failures.append(dict(case='truncated_tag_'+tag,invalid_not_STOP=True))
    tiny=NativeMultiStagePolicy(backend,native_parser=parser,condition='native_joint',
         total_output_tokens=1,decision_tokens=1,action_tokens=1,seconds=10,max_calls_per_batch=64)
    try:tiny.generate(row['observation'])
    except NativePhaseFailure as exc:
        assert exc.receipt['actual_model_generate_requests']==0
        assert exc.receipt['generated_or_replayed_output_tokens']==len(p['exact_output_token_ids'])
        failures.append(dict(case='over_cap_saved_output',rejected_not_silently_truncated=True,
                             original_replayed_length_accounted=True))
    else:raise AssertionError('Oversized output admitted')
    async def interrupted(calls):raise KeyboardInterrupt('synthetic accounting interruption, no SDK')
    env.step_idx=row['observation']['step_idx'];env.step=interrupted
    try:await execute_native_batch(env,proposal,remaining_native_calls=64)
    except KeyboardInterrupt as exc:
        assert exc.native_batch_receipt['reserved_calls']==4 and exc.native_batch_receipt['actual_attempted_calls'] is None
        failures.append(dict(case='interrupted_batch_accounting',reserved_calls=4,actual_attempts=None))
    else:raise AssertionError('Missing interruption')
    assert len(failures)==7 and not torch.cuda.is_initialized()
    dump(out/'all_replay_receipts.json.gz',details)
    dump(out/'cohort_accounting.json',cohort)
    dump(out/'failure_accounting.json',failures)
    dump(out/'original_source_manifest.json',source_hashes)
    result=dict(status='actual_saved_native_batches_and_original_memory_method_replayed_CPU_only',
        source_policy_turns=19,source_tool_results=35,observed_native_stages=['floor_plan','furniture','wall'],
        declared_ceiling_stage_actual_outputs=0,real_surface_workflow_replayed=False,
        conditions=len(cohort),transport_proposals=sum(c['policy_proposals'] for c in cohort),
        original_batch_method_replays=sum(c['original_batch_method_invocations'] for c in cohort),
        saved_dispatch_output_replays=sum(c['saved_dispatch_outputs_consumed'] for c in cohort),
        reconstructed_pre_action_memory_text_matches=sum(c['pre_action_memory_texts_equal'] for c in cohort),
        all_source_calls_preserved=True,all_source_partial_failures_preserved=True,
        new_model_generate_requests=0,training_updates=0,paid_API_calls=0,
        new_native_SDK_calls=0,new_rendering=0,new_complete_scene_successes=0,
        original_dispatcher_executed=False,original_memory_handlers_executed=True,
        phase_outputs='Original joint-generated token spans partitioned for transport replay only',
        no_new_context_sampling_or_probability_equivalence_claim=True,
        does_not_prove=['Model backend forward correctness or memory fit','Learned field-routing effect',
                       'Actual multi-stage scene generation/handover','Ceiling/surface native policy execution',
                       'Physical validity or source proposal quality'],
        safety_accounting_checks=len(failures),CUDA_initialized=False,seconds=time.monotonic()-tick,
        source_packets_sha256=sha(packet_path))
    dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();asyncio.run(main(a.repo.resolve(),a.out.resolve(),a.checkpoint.resolve()))
