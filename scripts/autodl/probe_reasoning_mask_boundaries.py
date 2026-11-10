"""Execute reviewed upstream mask helpers on existing immutable phase components.

No model, backward pass, optimizer or native scene edits. The leading prompt is
an explicit synthetic boundary harness, not a rendered/multimodal model input.
Actual assistant-prefix and target token IDs are reused without retokenization.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch


def main(repo, stage, source):
    out=stage/'mask_boundary_v1';out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((stage/'faithful_cot_source/targeted_source_manifest.json').read_text())
    expected=next(x['sha256'] for x in manifest['files'] if x['path']=='train/verl/utils/cot_masking.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    tree=ast.parse(source.read_text())
    names={'_find_subsequence_in_sequence','find_special_token_positions','create_cot_attention_mask'}
    selected=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in selected}==names
    namespace={'torch':torch,'Optional':Optional,'Tuple':Tuple,'Dict':Dict}
    # Only the three completely reviewed standalone tensor helpers are executed.
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    find=namespace['find_special_token_positions'];build=namespace['create_cot_attention_mask']
    checkpoint=Path('/root/autodl-tmp/agent_method_preparation_20261027/work/critic_public_runtime_20261104/checkpoint')
    token_cfg=checkpoint/'tokenizer_config.json';config=json.loads(token_cfg.read_text())
    close=int(next(k for k,v in config['added_tokens_decoder'].items() if v['content']=='</think>'))
    data=repo/'runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl'
    components=[x for x in map(json.loads,data.read_text().splitlines())
                if x['arm']=='paired_phase_v3' and x['stage']=='action']
    assert len(components)==24
    records=[];witness=None
    for row in components:
        prefix=row['assistant_prefix_token_ids'];target=row['target_token_ids']
        # A literal format-instruction marker in the leading prompt is synthetic.
        leading=[11,close,12,13];ids=torch.tensor([leading+prefix+target],device='cpu')
        response_length=len(target)
        pp,pc,pa=find(ids,response_length,end_think_token_id=close)
        sp,sc,sa=find(ids,response_length,
                    end_think_delimiter_ids=[torch.tensor([close],device='cpu')])
        raw_record=dict(unit_id=row['unit_id'],prefix_tokens=len(prefix),target_tokens=len(target),
             prefix_has_closing_delimiter=close in prefix,
             special_mode_prompt_answer_overlap=int((pp&pa).sum()),
             substring_mode_CoT_positions=int(sc.sum()),
             supplied_response_length=response_length,
             assistant_prefix_counted_as_prompt_in_component_API=True,
             premise='Actual archived prefix/target IDs; explicit four-token synthetic leading prompt. No images expanded or model run.')
        if close in prefix:
            # This is what unchanged substring-mode code would see if an action-
            # only training component were passed with its target response_length.
            mask=build(ids,response_length,
                 end_think_delimiter_ids=[torch.tensor([close],device='cpu')])[0,0]
            d_positions=list(range(len(leading),len(leading)+len(prefix)))
            a_query=len(leading)+len(prefix)
            raw_record['all_prefilled_D_blocked_from_A_query']=bool(torch.isneginf(mask[a_query,d_positions]).all())
            assert raw_record['all_prefilled_D_blocked_from_A_query']
            # Supplying a combined assistant boundary makes the delimiter visible,
            # but this is not the existing two-pass component boundary.
            combined_length=len(prefix)+len(target)
            gp,gc,ga=find(ids,combined_length,
                 end_think_delimiter_ids=[torch.tensor([close],device='cpu')])
            raw_record['combined_assistant_CoT_positions']=int(gc.sum())
            assert gc.sum()>0 and not (gp&ga).any()
            if witness is None:
                correct_mask=build(ids,combined_length,
                  end_think_delimiter_ids=[torch.tensor([close],device='cpu')])[0,0]
                prompt_index=0;decision_index=len(leading);answer_index=ids.shape[1]-1
                witness=dict(unit_id=row['unit_id'],
                    prompt_index=prompt_index,decision_index=decision_index,answer_index=answer_index,
                    direct_prompt_to_answer_blocked=bool(torch.isneginf(correct_mask[answer_index,prompt_index])),
                    prompt_to_decision_representation_allowed=bool(torch.isfinite(correct_mask[decision_index,prompt_index])),
                    decision_representation_to_answer_allowed=bool(torch.isfinite(correct_mask[answer_index,decision_index])),
                    interpretation='A two-layer hidden-representation route remains. This is intended mediation, not proof that the visible D text is complete or correct; no measured model effect.')
                assert witness['direct_prompt_to_answer_blocked']
                assert witness['prompt_to_decision_representation_allowed']
                assert witness['decision_representation_to_answer_allowed']
        else:
            assert int(sc.sum())==0
        records.append(raw_record)
    result=dict(status='original_reviewed_mask_helpers_executed_on_archived_component_IDs',
                source_pin=manifest['pin'],source_sha256=expected,
                tokenizer_config_sha256=hashlib.sha256(token_cfg.read_bytes()).hexdigest(),
                component_source_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                components=len(records),components_with_prefilled_D=sum(x['prefix_has_closing_delimiter'] for x in records),
                no_D_components=sum(not x['prefix_has_closing_delimiter'] for x in records),
                special_token_mode_overlapping_boundaries=sum(x['special_mode_prompt_answer_overlap']>0 for x in records),
                substring_action_only_mode_erases_prefilled_D=sum(x.get('all_prefilled_D_blocked_from_A_query',False) for x in records),
                paper_experiments_use_substring_mode=True,
                not_a_claim_of_published_experiment_failure=True,
                first_action_token_prediction_note='Causal likelihood is evaluated at the previous token. These are attention-query boundary diagnostics, not a forward/gradient faithfulness measurement.',
                records=records,hidden_state_path_witness=witness,
                model_forwards=0,backward_calls=0,training_updates=0,paid_API_calls=0)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','hidden_state_path_witness')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve(),a.source.resolve())
