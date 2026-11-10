"""Prospective boundary-aligned phase material; no models or training.

Old targets are immutable. Only serializer whitespace AFTER a phase stop is
excluded, with every source index/weight recorded. Retained coefficients are
not renormalized. Four matched arms retain the same D/A supervision; a native
joint training path is deliberately not declared qualified by this file.
"""
import argparse
import collections
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time

from decision_field_routes import route_decision_text
from native_multistage_phase import prepare_phase_messages
from phase_objective import PHASE_MODE
from routed_multimodal import frame_completed_decision


def sha(data): return hashlib.sha256(data).hexdigest()
def dump(path, value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main(repo, out, checkpoint):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    source=repo/'runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl'
    rows=[r for r in map(json.loads,source.read_text().splitlines()) if r['arm']=='paired_phase_v3']
    from transformers import AutoTokenizer
    import torch
    assert not torch.cuda.is_initialized()
    tok=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
    gc=json.loads((checkpoint/'generation_config.json').read_text());eos=gc['eos_token_id']
    eos=eos if isinstance(eos,list) else [eos]
    close=tok.convert_tokens_to_ids('</think>')
    decisions={r['unit_id']:r for r in rows if r['stage']=='decision'}
    modes=[('decision_independent','full_record'),('decision_conditioned','full_record'),
           ('decision_conditioned','without_principle_field'),('decision_conditioned','plan_field_only')]
    derived=[];boundaries=[];arm_stats=[]
    for condition,route in modes:
        key=condition+':'+route; mass=0.;removed_mass=0.;kept_count=0;removed_count=0
        kinds=collections.Counter()
        for original in rows:
            row=copy.deepcopy(original);ids=original['target_token_ids'];stage=original['stage']
            stops=[close] if stage=='decision' else eos
            positions=[i for i,t in enumerate(ids) if t in stops]
            assert len(positions)==1
            end=positions[0]+1
            suffix=ids[end:]
            assert not suffix or (tok.decode(suffix,skip_special_tokens=False).isspace()
                                  and all(t not in eos+[close] for t in suffix))
            # Whitespace BEFORE the stop remains part of the actual target.
            kept=ids[:end];text=tok.decode(kept,skip_special_tokens=False)
            assert tok.encode(text,add_special_tokens=False)==kept
            old_suffix=PHASE_MODE[stage]
            assert row['messages'][0]['content'].endswith(old_suffix)
            base_messages=copy.deepcopy(row['messages'])
            base_messages[0]['content']=base_messages[0]['content'][:-len(old_suffix)]
            row['messages']=prepare_phase_messages(base_messages,stage)
            assert row['messages'][1:]==original['messages'][1:]
            if stage=='action' and condition=='decision_conditioned':
                old_d=decisions.get(row['unit_id'])
                if old_d:
                    d_ids=old_d['target_token_ids']
                    d_text=tok.decode(d_ids[:d_ids.index(close)+1],skip_special_tokens=False)
                    d_text,meta=route_decision_text(d_text,route)
                    prefix=frame_completed_decision(d_text)
                else:
                    prefix='';meta=dict(status='no_decision_common',changed=False)
            else:
                prefix='';meta=dict(status='not_conditioned_on_D',changed=False)
            prompt=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)+prefix
            before=tok.encode(prompt,add_special_tokens=False)
            assert tok.encode(prompt+text,add_special_tokens=False)==before+kept
            row.update(arm='native_phase_v4:'+key,target_text=text,target_token_ids=kept,
                       target_sha256=sha(json.dumps(kept).encode()),
                       source_token_origins=original['source_token_origins'][:end],
                       token_objective_weights=original['token_objective_weights'][:end],
                       assistant_prefix=prefix,assistant_prefix_token_ids=tok.encode(prefix,add_special_tokens=False),
                       text_prefix_tokens=len(before),field_routing=meta,
                       new_phase_prompt_sha256=sha(json.dumps(row['messages'],ensure_ascii=False).encode()),
                       original_component_target_sha256=original['target_sha256'],
                       image_encoding_checked=False,training_admitted=False,
                       own_generated_decision_required_at_deployment=True)
            row.pop('host_separator_tokens_added',None)
            row['host_prefix_whitespace_is_not_a_target']=True
            derived.append(row)
            excluded=[dict(source_index=i,token_id=ids[i],
                           source_origin=original['source_token_origins'][i],
                           original_coefficient=original['token_objective_weights'][i])
                      for i in range(end,len(ids))]
            boundary=dict(arm=key,unit_id=row['unit_id'],stage=stage,
                          original_tokens=len(ids),emitted_target_tokens=len(kept),
                          excluded_post_stop_whitespace=excluded,
                          excluded_text=tok.decode(suffix,skip_special_tokens=False),
                          retained_coefficients_unchanged=True,source_prefix_unchanged=True,
                          phase_prompt_changed=True)
            boundaries.append(boundary)
            mass+=sum(row['token_objective_weights']);removed_mass+=sum(x['original_coefficient'] for x in excluded)
            kept_count+=len(kept);removed_count+=len(excluded);kinds[stage]+=1
        arm_stats.append(dict(arm=key,components=dict(kinds),target_tokens=kept_count,
                              removed_host_whitespace_tokens=removed_count,
                              retained_coefficient_mass=mass,excluded_coefficient_mass=removed_mass))
    assert len(derived)==136 and all(s['target_tokens']==arm_stats[0]['target_tokens'] for s in arm_stats)
    assert len({round(s['retained_coefficient_mass'],12) for s in arm_stats})==1
    # Ordinary source-index equality, not just equality of scalar total mass.
    def identity(rs):
        return sorted((r['unit_id'],r['stage'],json.dumps(o,sort_keys=True),t,w)
                      for r in rs for o,t,w in zip(r['source_token_origins'],r['target_token_ids'],r['token_objective_weights'],strict=True))
    reference=identity(derived[:34])
    for i in (34,68,102): assert identity(derived[i:i+34])==reference
    # Strong baselines use the same immutable source tokens. Joint generation
    # must still emit an inter-phase separator, but that serializer token is
    # not a loss position in ANY matched arm. A caller masks zero coefficients
    # to labels=-100; removing the token from causal input would change context.
    baselines=[];baseline_stats=[]
    for condition in ('native_joint','action_direct'):
        mass=0.;positive_count=0;zero_count=0
        for action in (r for r in rows if r['stage']=='action'):
            parts=([decisions[action['unit_id']]] if condition=='native_joint' and action['unit_id'] in decisions else [])+[action]
            ids=[];origins=[];weights=[]
            for part in parts:
                stop=close if part['stage']=='decision' else next(t for t in part['target_token_ids'] if t in eos)
                end=part['target_token_ids'].index(stop)+1
                keep=len(part['target_token_ids']) if part['stage']=='decision' else end
                for i in range(keep):
                    ids.append(part['target_token_ids'][i])
                    origins.append(part['source_token_origins'][i])
                    weights.append(part['token_objective_weights'][i] if i<end else 0.)
            row=copy.deepcopy(action);old_suffix=PHASE_MODE['action']
            assert row['messages'][0]['content'].endswith(old_suffix)
            base=copy.deepcopy(row['messages']);base[0]['content']=base[0]['content'][:-len(old_suffix)]
            row['messages']=prepare_phase_messages(base,'joint' if condition=='native_joint' else 'action')
            text=tok.decode(ids,skip_special_tokens=False)
            prompt=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
            before=tok.encode(prompt,add_special_tokens=False)
            assert tok.encode(prompt+text,add_special_tokens=False)==before+ids
            assert ids[-1] in eos and sum(t in eos for t in ids)==1
            row.update(arm='native_phase_v4:'+condition,stage='joint' if condition=='native_joint' else 'action',
                       target_text=text,target_token_ids=ids,target_sha256=sha(json.dumps(ids).encode()),
                       source_token_origins=origins,token_objective_weights=weights,
                       response_loss_mask=[w>0 for w in weights],
                       assistant_prefix='',assistant_prefix_token_ids=[],text_prefix_tokens=len(before),
                       new_phase_prompt_sha256=sha(json.dumps(row['messages'],ensure_ascii=False).encode()),
                       image_encoding_checked=False,training_admitted=False,
                       zero_coefficient_response_tokens_must_be_labels_minus_100=True)
            row.pop('host_separator_tokens_added',None)
            baselines.append(row);mass+=sum(weights);positive_count+=sum(w>0 for w in weights);zero_count+=sum(w==0 for w in weights)
        baseline_stats.append(dict(arm=condition,components=24,positive_target_tokens=positive_count,
                                   masked_interphase_separator_tokens=zero_count,coefficient_mass=mass))
    def positive_identity(rs):
        return sorted((r['unit_id'],json.dumps(o,sort_keys=True),t,w)
                      for r in rs for o,t,w in zip(r['source_token_origins'],r['target_token_ids'],r['token_objective_weights'],strict=True) if w>0)
    assert positive_identity(baselines[:24])==positive_identity(derived[:34])
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in derived).encode()
    (out/'prospective_native_phase_v4.jsonl.gz').write_bytes(gzip.compress(data,mtime=0))
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in baselines).encode()
    (out/'prospective_native_baselines_v4.jsonl.gz').write_bytes(gzip.compress(data,mtime=0))
    dump(out/'boundary_exclusions.json',boundaries)
    result=dict(status='new_shared_native_phase_prompts_and_stop_boundaries_materialized_text_only',
        source_sha256=sha(source.read_bytes()),source_components=34,source_units=24,
        source_D_components=10,source_missing_D_units=14,arms=arm_stats,
        fixed_retained_source_token_identities_equal=True,retained_coefficients_not_renormalized=True,
        old_targets_or_files_changed=False,old_objective_exactly_preserved=False,
        reason_old_objective_changed='Uniform exclusion of serializer whitespace after a deployment stop; see source-index exclusion ledger',
        baselines=baseline_stats,joint_positive_source_token_identities_match_phase_arms=True,
        native_multicall_or_floor_training_examples_added=0,
        joint_training_objective_ported='Text/material only; zero-weight separators require explicit label masking before trainer integration',complete_source_images_restored=False,
        no_claim_new_training_or_complete_scene_data=True,
        local_VLM_forwards=0,model_weights_loaded=False,training_updates=0,paid_API_calls=0,
        CUDA_initialized=False,seconds=time.monotonic()-tick)
    dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.out.resolve(),a.checkpoint.resolve())
