"""Compile saved real calls into a symbolic plan and replay argument binding only.

This is NOT a generated plan or a new simulator rollout. It qualifies the dynamic
ID interface a fair whole-plan control needs before asking a model to predict one.
"""
import argparse
import hashlib
import json
from pathlib import Path
from native_trajectory_references import (abstract_observed_ids, observed_id_paths,
                                         resolve_arguments)


def main(repo, out):
    out.mkdir(parents=True, exist_ok=False)
    p=repo/'runs/autodl/20261110_complete_native_runtime/review_traces'
    plans=[];counts=[]
    for stage in ('floor','furniture','wall'):
        path=p/f'{stage}_trace.json';traces=json.loads(path.read_text())
        known={};outputs=[];calls=[];resolved_count=0
        for turn in traces:
            proposal=turn['proposal']
            for call, result in zip(proposal.get('tool_calls',[]),turn.get('results',[])):
                assert call['name']==result['name']
                arguments=call.get('arguments',{})
                symbolic=abstract_observed_ids(arguments,known)
                resolved=resolve_arguments(symbolic,outputs)
                assert resolved==arguments
                resolved_count += int(symbolic != arguments)
                calls.append(dict(name=call['name'],arguments=symbolic,
                                  source_turn=turn['turn']))
                output=result['output']
                for value,output_path in observed_id_paths(output):
                    known.setdefault(value, {'$result':{'call':len(outputs),'path':output_path}})
                outputs.append(output)
        counts.append(dict(stage=stage,calls=len(calls),calls_with_observed_ID_references=resolved_count,
                           successful_output_ID_values=len(known),
                           recorded_policy_stop=any(t['status']=='native_policy_stop' for t in traces)))
        plans.append(dict(stage=stage,mode='archived_arguments_only_not_a_policy_prediction',
                          calls=calls,source_trace=str(path.relative_to(repo)),
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          terminal_success_added=False))
    # Negative controls: a caller cannot retrieve future/failing outputs or run code.
    rejected=[]
    bad=[({'$result':{'call':0,'path':[]}},[]),
         ({'$result':{'call':0,'path':['object_id']}},[{'success':False,'object_id':'not_created'}]),
         ({'$result':{'call':0,'path':['__class__']}},[{'success':True,'object_id':'x'}])]
    for ref,previous in bad:
        try:resolve_arguments(ref,previous)
        except ValueError as exc:rejected.append(str(exc))
        else:raise AssertionError('Unsafe reference accepted')
    assert resolve_arguments('__import__("os").system("false")',[])=='__import__("os").system("false")'
    (out/'archived_symbolic_plans.json').write_text(json.dumps(plans,indent=2)+'\n')
    result=dict(status='archived_actual_arguments_replayed_exactly',stages=counts,
                total_calls=sum(x['calls'] for x in counts),
                referenced_calls=sum(x['calls_with_observed_ID_references'] for x in counts),
                invalid_reference_rejections=rejected,model_forwards=0,
                native_tool_executions=0,training_updates=0,paid_API_calls=0,
                new_generated_rooms=0,method_quality_claim=False,
                limitation='No predicted whole plan executed; native tool types differ from original SceneWeaver.',
                qualification='ID substitution only. Mutable tool-state reset separately qualified for rescale only.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.out.resolve())
