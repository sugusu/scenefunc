"""Audit decision-stage coverage; repartition existing targets without adding text.

Prospective revision only: no model, training, or old dataset modifications.
Original per-token coefficients are retained even when a token changes stage.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import resource
import time

from audit_material_routing import MODE, digest, token_hash, write, tagged_request
from routed_multimodal import frame_completed_decision
from phase_objective import phase_messages


def raw_messages(row):
    messages = copy.deepcopy(row['messages'])
    suffix = MODE[row['component']]
    if not messages[0]['content'].endswith(suffix):
        raise ValueError('Unknown source routing prompt')
    messages[0]['content'] = messages[0]['content'][:-len(suffix)]
    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    tick = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    source_bytes = args.source.read_bytes()
    source = [json.loads(line) for line in source_bytes.decode().splitlines()]
    chosen = [r for r in source if r['arm'] == 'paired_tagged']
    indexed = defaultdict(dict)
    for row in chosen:
        indexed[row['unit_id']][row['component']] = row
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    close = tokenizer.convert_tokens_to_ids('</think>')
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    source_coverage = Counter()
    revised, units = [], []
    for ident, components in indexed.items():
        action = components['action']
        decision = components.get('decision')
        kind = action['target_kind']
        source_coverage[('with_decision_' if decision else 'without_decision_') + kind] += 1
        old_components = ([decision] if decision else []) + [action]
        ids, provenance, weights = [], [], []
        for part in old_components:
            for i, tok in enumerate(part['target_token_ids']):
                ids.append(tok)
                provenance.append({'component': part['component'], 'index': i})
                weights.append(part['target_component_weight'])
        # Keep the original eight D targets exactly, including their separator.
        # Terminal text is split at </think>; EOS remains an action target.
        # The host supplies one separator newline to a completed D prefix.
        if decision:
            split = len(decision['target_token_ids'])
        elif close in ids:
            if ids.count(close) != 1 or ids[0] != tokenizer.convert_tokens_to_ids('<think>'):
                raise ValueError('Noncanonical source decision')
            split = ids.index(close) + 1
        else:
            split = 0
        stages = [('decision', 0, split), ('action', split, len(ids))]
        old_mass = sum(weights)
        new_mass = 0.
        for stage, begin, finish in stages:
            if begin == finish:
                continue
            target = ids[begin:finish]
            text = tokenizer.decode(target, skip_special_tokens=False)
            if tokenizer.encode(text, add_special_tokens=False) != target:
                raise ValueError('Target roundtrip differs')
            if stage == 'decision' and target[-1] != close and target[-2:] != [close, 198]:
                raise ValueError('D stage does not end at the host stop token')
            if kind == 'terminal_control' and stage == 'action' and target != [end, 198]:
                raise ValueError('Terminal continuation must remain the exact old EOS/newline')
            messages = phase_messages(raw_messages(action), stage)
            base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for arm in ('separate_phase_v3', 'paired_phase_v3'):
                prefix = frame_completed_decision(tokenizer.decode(ids[:split], skip_special_tokens=False)) if split and stage == 'action' and arm == 'paired_phase_v3' else ''
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                observed = tokenizer.encode(base + prefix, add_special_tokens=False)
                actual = tokenizer.encode(base + prefix + text, add_special_tokens=False)
                if actual != observed + target:
                    raise ValueError('Phase tokenization boundary differs')
                revised.append({'unit_id':ident, 'family_id':action['family_id'], 'target_kind':kind,
                                'arm':arm, 'stage':stage, 'messages':messages,
                                'assistant_prefix':prefix, 'assistant_prefix_token_ids':prefix_ids,
                                'host_separator_tokens_added':len(prefix_ids)-split if prefix else 0,
                                'target_text':text, 'target_token_ids':target, 'target_sha256':token_hash(target),
                                'source_token_origins':provenance[begin:finish],
                                'token_objective_weights':weights[begin:finish],
                                'source_request_sha256':action['request_source_sha256'],
                                'decision_stop_token_id':close if stage=='decision' else None,
                                'text_prefix_tokens':len(observed), 'image_encoding_checked':False})
            new_mass += sum(weights[begin:finish])
        if abs(old_mass-new_mass) > 1e-12:
            raise ValueError('Token weighting changed during repartition')
        units.append({'unit_id':ident,'family_id':action['family_id'],'target_kind':kind,
                      'old_decision_tokens':len(decision['target_token_ids']) if decision else 0,
                      'old_action_tokens':len(action['target_token_ids']),
                      'new_decision_tokens':split,'new_action_tokens':len(ids)-split,
                      'original_full_target_sha256':token_hash(ids),
                      'source_tokens_covered_once':True,'inherited_weight_mass':old_mass,
                      'new_annotation_created':False})
    # Compare inherited coefficients with a tempting but different renormalization.
    family_counts = defaultdict(Counter)
    for r in revised:
        if r['arm']=='paired_phase_v3':family_counts[r['family_id']][r['stage']] += len(r['target_token_ids'])
    weight_changes = []
    for r in revised:
        if r['arm']!='paired_phase_v3':continue
        naive = 1/(6*family_counts[r['family_id']][r['stage']])
        ratios = [naive/w for w in r['token_objective_weights']]
        if any(abs(x-1)>1e-12 for x in ratios):
            weight_changes.append({'unit_id':r['unit_id'],'stage':r['stage'],
                                   'naive_to_inherited_weight_min':min(ratios),
                                   'naive_to_inherited_weight_max':max(ratios)})
    with (args.output/'prospective_phase_components.jsonl').open('w') as handle:
        for r in revised:handle.write(json.dumps(r,ensure_ascii=False)+'\n')
    result = {'source_units':len(indexed),'source_coverage':dict(source_coverage),
              'old_decision_stage_stop_examples':0,
              'revised_decision_stage_stop_examples':sum(u['target_kind']=='terminal_control' and u['new_decision_tokens']>0 for u in units),
              'revised_material_per_arm':{s:sum(len(r['target_token_ids']) for r in revised if r['arm']=='paired_phase_v3' and r['stage']==s) for s in ('decision','action')},
              'revised_components_per_arm':len(revised)//2,
              'same_original_supervised_token_count':sum(u['old_decision_tokens']+u['old_action_tokens'] for u in units),
              'units':units,'naive_renormalization_changes':weight_changes,
              'full_pass_inherited_coefficient_mass':sum(u['inherited_weight_mass'] for u in units),
              'remaining_unannotated_decision_states':[u['unit_id'] for u in units if u['new_decision_tokens']==0],
              'scope':'Repartitioned existing targets; no new annotations, learned behavior, or performance evidence. Missing D labels on 14 edit states remain explicit.',
              'phase_revision':'v3: action instruction explicitly permits EOS-only STOP; v2 prose instruction did not match the repartitioned terminal target.',
              'source_sha256':digest(source_bytes),'model_calls':0,'optimizer_updates':0,
              'elapsed_seconds':time.perf_counter()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write(args.output/'phase_audit.json',result)
    print(json.dumps({k:result[k] for k in ('source_coverage','revised_material_per_arm','revised_components_per_arm','same_original_supervised_token_count','full_pass_inherited_coefficient_mass','elapsed_seconds')}))


if __name__=='__main__':main()
