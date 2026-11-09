"""Audit project and principle overlap; exercise explicit knowledge-card removal.

This is data preparation, not a model benchmark. All six held-out-family
views are computed, without choosing a fold from outcome scores. Original
inputs and annotations remain read-only. No new labels or actions are made.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import subprocess
import time

from audit_material_routing import PIN, digest, write

REGISTRY='research/920/decision_packets_v1/SOURCED_DECISIONS.json'
MARKER='CURRENT NATIVE STATE:\n'


def structural_signature(context):
    objects=context['objects']
    if isinstance(objects,dict):objects=list(objects.values())
    def numeric(value):
        if isinstance(value,(float,int)) and not isinstance(value,bool):return float(value)
        if isinstance(value,list):return [numeric(v) for v in value]
        if isinstance(value,dict):return {k:numeric(v) for k,v in value.items()}
        return value
    selected=[numeric({k:o[k] for k in ('object_type','bbox_min','bbox_max','transform')})
              for o in objects if o['object_type'] in ('wall','door','window','wall_mounted')]
    if not selected:raise ValueError('No structural records; cannot assert template separation')
    canonical=sorted(selected,key=lambda s:json.dumps(s,sort_keys=True))
    return {'sha256':digest(json.dumps(canonical,sort_keys=True).encode()),
            'structural_record_count':len(canonical),
            'scope':'Exact numerical structural bounds/transforms ignoring IDs; not mesh, texture, whole-scene or general geometric equivalence.'}


def exclude_source_cards(messages, excluded_ids, card_texts):
    """Change the structured current-state knowledge field, preserving other data.

    A residual literal identifier/text is reported, never silently deleted.
    This cannot certify absence of paraphrased source information or pretraining.
    """
    result=copy.deepcopy(messages);changes=[];states=0
    for message in result:
        if not isinstance(message.get('content'),list):continue
        for part in message['content']:
            text=part.get('text','')
            if MARKER not in text:continue
            left,right=text.split(MARKER,1)
            state,end=json.JSONDecoder().raw_decode(right)
            old=copy.deepcopy(state)
            cards=state.get('available_source_knowledge')
            if not isinstance(cards,list):raise ValueError('Missing explicit source-card field')
            kept=[k for k in cards if k['decision_id'] not in excluded_ids]
            removed=[k['decision_id'] for k in cards if k['decision_id'] in excluded_ids]
            state['available_source_knowledge']=kept
            expect=copy.deepcopy(old);expect['available_source_knowledge']=kept
            assert state==expect
            part['text']=left+MARKER+json.dumps(state,ensure_ascii=False,sort_keys=True)+right[end:]
            changes.extend(removed);states+=1
    if states!=1:raise ValueError('Expected exactly one current-state block')
    # Parse back, preserving all non-state prefix and suffix text exactly.
    original_copy=copy.deepcopy(messages);new_copy=copy.deepcopy(result)
    for original,new in zip(original_copy,new_copy):
        if not isinstance(original.get('content'),list):continue
        for a,b in zip(original['content'],new['content']):
            if MARKER not in a.get('text',''):continue
            la,ra=a['text'].split(MARKER,1);lb,rb=b['text'].split(MARKER,1)
            sa,ea=json.JSONDecoder().raw_decode(ra);sb,eb=json.JSONDecoder().raw_decode(rb)
            assert la==lb and ra[ea:]==rb[eb:]
            sa.pop('available_source_knowledge');sb.pop('available_source_knowledge')
            assert sa==sb
            a['text']=b['text']='{CHECKED_STATE_BLOCK}'
    assert original_copy==new_copy
    encoded=json.dumps(result,ensure_ascii=False,sort_keys=True)
    residual=[s for s in sorted(excluded_ids|set(card_texts)) if s and s in encoded]
    return result,{'removed_card_ids':changes,'residual_exact_source_markers':residual,
                   'all_nonknowledge_context_and_message_fields_preserved':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();sources={}
    def read(path):
        raw=subprocess.check_output(['git','show',PIN+':'+path]);sources[path]=digest(raw)
        return json.loads(raw)
    coverage_path=Path('runs/autodl/20261110_conditional_evidence/actual/condition_coverage.json')
    coverage=json.loads(coverage_path.read_text())
    registry=read(REGISTRY)['records'];lookup={r['decision_id']:r for r in registry}
    reasons={r['unit_id']:r for r in coverage['sourced_decisions']}
    requests={};rows=[];application=defaultdict(list);knowledge={}
    for unit in coverage['rows']:
        inp=read(unit['input_path']);request=read(unit['request_path'])
        assert sources[unit['input_path']]==unit['input_sha256']
        assert sources[unit['request_path']]==unit['request_sha256']
        cards=inp['context']['available_source_knowledge']
        for card in cards:
            ident=card['decision_id'];assert ident in lookup
            if ident in knowledge:assert knowledge[ident]['card']==card
            knowledge[ident]={'card':card,'source_family':lookup[ident]['family_id'],
                             'category':lookup[ident]['knowledge_category']}
        r={'unit_id':unit['unit_id'],'family_id':unit['family_id'],
           'visible_principle_ids':[k['decision_id'] for k in cards],
           'has_sourced_application_target':unit['unit_id'] in reasons,
           'request_path':unit['request_path'],'input_path':unit['input_path'],
           'structural_signature':structural_signature(inp['context'])}
        if unit['unit_id'] in reasons:
            pid=reasons[unit['unit_id']]['reason']['principle_applicability']['source_decision_id']
            r['supervised_principle_id']=pid;application[pid].append(unit['unit_id'])
        rows.append(r);requests[unit['unit_id']]=request
    byid={r['unit_id']:r for r in rows};families=sorted({r['family_id'] for r in rows})
    shell_families=defaultdict(set)
    for r in rows:shell_families[r['structural_signature']['sha256']].add(r['family_id'])
    parent={f:f for f in families}
    def root(f):
        while parent[f]!=f:f=parent[f]
        return f
    for fs in shell_families.values():
        fs=sorted(fs)
        for f in fs[1:]:parent[root(f)]=root(fs[0])
    clusters=defaultdict(list)
    for f in families:clusters[root(f)].append(f)
    groups=[('source_family',[f]) for f in families]+[
        ('source_and_structural_template',sorted(fs)) for fs in clusters.values()]
    folds=[];rewrite_rows=[]
    for kind,held_families in groups:
        fold_id=kind+':'+','.join(held_families)
        train=[r for r in rows if r['family_id'] not in held_families]
        held=[r for r in rows if r['family_id'] in held_families]
        excluded={p for p,k in knowledge.items() if k['source_family'] in held_families}
        texts={knowledge[p]['card']['principle'] for p in excluded}
        train_app={r['supervised_principle_id'] for r in train if r['has_sourced_application_target']}
        fold={'fold_id':fold_id,'split_kind':kind,'held_families':held_families,
              'train_unit_ids':[r['unit_id'] for r in train],
              'held_unit_ids':[r['unit_id'] for r in held],
              'source_excluded_card_ids':sorted(excluded),
              'retained_card_ids':sorted(set(knowledge)-excluded),
              'held_application_units_with_same_principle_training_application':[
                  r['unit_id'] for r in held if r.get('supervised_principle_id') in train_app],
              'held_application_units':[r['unit_id'] for r in held if r['has_sourced_application_target']],
              'unmodified_mode':'All original knowledge cards remain visible; source-project exposed',
              'filtered_mode':'Exclude held-family cards in BOTH train and evaluation; current structured card field only',
              'scope':'Retrospective development partition, not a newly sealed test set; no labels changed or training launched.'}
        train_shells={r['structural_signature']['sha256'] for r in train}
        held_shells={r['structural_signature']['sha256'] for r in held}
        fold['shared_structural_signatures']=sorted(train_shells&held_shells)
        if kind=='source_and_structural_template':assert not fold['shared_structural_signatures']
        for row in rows:
            canonical,canonical_receipt=exclude_source_cards(requests[row['unit_id']],set(),set())
            assert not canonical_receipt['removed_card_ids']
            changed,receipt=exclude_source_cards(requests[row['unit_id']],excluded,texts)
            canonical_filtered,_=exclude_source_cards(canonical,excluded,texts)
            assert canonical_filtered==changed
            rewrite_rows.append(dict(fold_id=fold_id,unit_id=row['unit_id'],
                membership='held' if row['family_id'] in held_families else 'train',
                original_request_sha256=sources[row['request_path']],
                canonical_all_cards_sha256=digest(json.dumps(canonical,ensure_ascii=False,sort_keys=True).encode()),
                canonicalization_changed_original_messages=canonical!=requests[row['unit_id']],
                paired_canonicalization_verified=True,
                transformed_canonical_sha256=digest(json.dumps(changed,ensure_ascii=False,sort_keys=True).encode()),
                **receipt))
        folds.append(fold)
    applications=[dict(principle_id=p,source_family=knowledge[p]['source_family'],
        application_units=ids,application_families=sorted({byid[u]['family_id'] for u in ids}))
        for p,ids in sorted(application.items())]
    result={'pin':PIN,'source_sha256':sources,'local_source_sha256':{str(coverage_path):digest(coverage_path.read_bytes())},
        'units':rows,'knowledge':knowledge,'applications':applications,'folds':folds,
        'structural_signature_families':{k:sorted(v) for k,v in shell_families.items()},
        'counts':{'units':len(rows),'families':len(families),'visible_principles':len(knowledge),
            'sourced_application_targets':len(reasons),'supervised_principles':len(application),
            'principles_applied_in_multiple_families':sum(len(r['application_families'])>1 for r in applications),
            'units_with_all_cards':sum(set(r['visible_principle_ids'])==set(knowledge) for r in rows),
            'distinct_structural_signatures':len(shell_families),
            'source_and_template_components':len(clusters),
            'family_only_folds_with_shared_structural_signature':sum(
                bool(f['shared_structural_signatures']) for f in folds if f['split_kind']=='source_family'),
            'request_rewrites_exercised':len(rewrite_rows),
            'rewrites_with_residual_source_markers':sum(bool(r['residual_exact_source_markers']) for r in rewrite_rows)},
        'conclusion':'Knowledge text exposure and supervised application exposure are different. Existing family holdouts have no same-principle application example in another family; all original requests nevertheless expose all six knowledge cards. These are development diagnostics, not evidence of held-out scene performance.',
        'serialization_control':'Any future knowledge ablation must use canonicalized all-card requests as its comparator. Original unmodified text is a separate archival baseline; mixing formatting and information changes would confound the ablation.',
        'model_calls':0,'training_updates':0,'scene_edits':0,'wall_seconds':time.perf_counter()-tick}
    write(args.output/'coverage.json',result);write(args.output/'request_rewrites.json',rewrite_rows)
    print(json.dumps(result['counts']))


if __name__=='__main__':main()
