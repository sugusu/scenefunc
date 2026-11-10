"""Audit all fixed design targets and prepare equal-supervision field controls.

No model/optimization, new experts, modified old labels or scene simulation.
New inputs are prospective TRAINING text; original targets/coefficients stay
exactly fixed. These files are not an oracle inference policy.
"""
import argparse
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

from decision_field_routes import MODES, route_decision_text


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main(repo, out, checkpoint):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    source=repo/'runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl'
    rows=[x for x in map(json.loads,source.read_text().splitlines()) if x['arm']=='paired_phase_v3']
    actions={r['unit_id']:r for r in rows if r['stage']=='action'}
    decisions={r['unit_id']:r for r in rows if r['stage']=='decision'}
    assert len(rows)==34 and len(actions)==24 and len(decisions)==10
    from transformers import AutoTokenizer
    import torch
    assert not torch.cuda.is_initialized()
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    info=[];prospective=[];counts=collections.Counter();token_totals=collections.Counter();masses={}
    for mode in MODES:
        masses[mode]=0.
        for row in rows:
            revised=copy.deepcopy(row)
            revised['arm']='field_control_'+mode
            if row['stage']=='action':
                prefix,meta=route_decision_text(row['assistant_prefix'],mode)
                revised['assistant_prefix']=prefix
                revised['assistant_prefix_token_ids']=tokenizer.encode(prefix,add_special_tokens=False)
                revised['field_routing']=meta
                counts[mode+':'+meta['status']]+=1
                base=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
                before=tokenizer.encode(base+prefix,add_special_tokens=False)
                after=tokenizer.encode(base+prefix+row['target_text'],add_special_tokens=False)
                assert after==before+row['target_token_ids']
                revised['text_prefix_tokens']=len(before)
                token_totals[mode]+=len(before)
            else:
                revised['field_routing']=dict(status='full_original_D_supervision_unchanged',changed=False)
            for field in ['target_text','target_token_ids','source_token_origins','token_objective_weights','source_request_sha256']:
                assert revised[field]==row[field]
            revised['image_encoding_checked']=False
            revised['training_text_only']=True
            revised['deployment_gold_D_allowed']=False
            prospective.append(revised)
            masses[mode]+=sum(row['token_objective_weights'])
    assert len(set(round(v,12) for v in masses.values()))==1
    for ident,action in actions.items():
        d=decisions.get(ident)
        routed,meta=route_decision_text(action['assistant_prefix'],'plan_field_only')
        if meta['status']!='structured_record_routed':
            continue
        value=json.loads(d['target_text'].split('<think>',1)[1].split('</think>',1)[0])
        tool=json.loads(action['target_text'].split('<tool_call>',1)[1].split('</tool_call>',1)[0])
        references=value['proposed_decision']['object_or_asset_references']
        args=tool['arguments']
        referent=args.get('object_id',args.get('asset_id'))
        assert isinstance(references,str) and references==referent
        pose_fields={k:v for k,v in args.items() if k in ['x','y','z','yaw','roll','pitch']}
        numeric_leaves=[]
        def walk(obj,path=()):
            if isinstance(obj,dict):
                for k,v in obj.items():walk(v,path+(k,))
            elif isinstance(obj,list):
                for k,v in enumerate(obj):walk(v,path+(k,))
            elif isinstance(obj,(int,float)) and not isinstance(obj,bool):
                numeric_leaves.append(dict(path=list(path),value=obj))
        walk(value)
        info.append(dict(unit_id=ident,family_id=action['family_id'],
                         source_decision_id=value['principle_applicability']['source_decision_id'],
                         proposed_reference=references,action_referent=referent,
                         exact_referent_in_decision=True,
                         literal_tool_name_in_decision=tool['name'] in d['target_text'],
                         action_pose_fields=pose_fields,typed_numeric_leaves_in_D=numeric_leaves,
                         rationale_origin='Local state/action application of a source principle, not original designer tool trace',
                         cannot_conclude=['No claim these coordinates are implied by the source',
                                         'No claim copied reference explains all action behavior',
                                         'No model has been run under these field controls']))
    assert len(info)==8
    with (out/'prospective_field_components.jsonl').open('w') as f:
        for r in prospective:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    dump(out/'decision_information_audit.json',info)
    with (out/'decision_information.csv').open('w',newline='') as f:
        fields=['unit_id','family_id','source_decision_id','proposed_reference','action_referent',
                'exact_referent_in_decision','literal_tool_name_in_decision','typed_numeric_leaf_count','action_pose_field_count']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in info:
            w.writerow({k:r[k] if k in r else len(r['typed_numeric_leaves_in_D']) if k=='typed_numeric_leaf_count'
                        else len(r['action_pose_fields']) for k in fields})
    result=dict(status='all_fixed_original_D_and_action_supervision_retained_across_field_routes',
                source_sha256=sha(source),fixed_units=24,full_original_D_components=10,
                structured_sourced_D_components=8,common_other_schema_terminal_D_components=2,
                no_D_units=14,components_per_arm=34,arms=list(MODES),new_components=len(prospective),
                referents_exactly_revealed_by_structured_D=sum(r['exact_referent_in_decision'] for r in info),
                exact_literal_tool_names_revealed=sum(r['literal_tool_name_in_decision'] for r in info),
                action_pose_fields_in_these_eight=sum(len(r['action_pose_fields']) for r in info),
                typed_numeric_leaves_in_these_D=sum(len(r['typed_numeric_leaves_in_D']) for r in info),
                original_supervision_mass_by_arm=masses,action_text_prefix_tokens_by_arm=dict(token_totals),
                route_counts=dict(counts),source_targets_and_coefficients_identical=True,
                all_action_causal_token_boundaries_verified=True,new_supervision_annotations=0,
                no_exact_compute_equivalence_claim=True,
                input_scope='Tokenizer/text boundary only, no image reconstruction or model execution in this audit',
                model_weights_loaded=False,local_VLM_forwards=0,paid_API_calls=0,training_updates=0,
                CUDA_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-tick,
                conclusion='Full-D versus plan/no-principle routing separates extra explanatory conditioning from explicit plan/referent information; benefit still requires own-generated D and complete native trajectories, not gold-prefix scores.')
    dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.out.resolve(),a.checkpoint.resolve())
