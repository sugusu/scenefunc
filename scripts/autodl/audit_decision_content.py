"""Inspect fixed source reasons and prepare oracle-prefix diagnostics.

No learned policy, new labels, scene edits or training. BM25 only ranks the
already supplied knowledge. The plan reader consumes a teacher plan and cannot
be reported as an autonomous decision policy.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import time

from audit_material_routing import digest, write
from audit_decision_phase_support import raw_messages
from phase_objective import phase_messages
from routed_multimodal import frame_completed_decision


def tokens(text):
    out = []
    for part in re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', text.lower()):
        if re.fullmatch(r'[\u4e00-\u9fff]+', part):
            out.extend(part[i:i+2] for i in range(len(part)-1))
            if len(part) == 1: out.append(part)
        else: out.append(part)
    return out


def rank_knowledge(context):
    # Never read a reason, target action, reference ID or evaluation label here.
    query = '\n'.join(str(context.get(k) or '') for k in ('design_request', 'current_requested_mode'))
    query += '\n'+str(context.get('current_activity_request', {}).get('request', ''))
    knowledge = context.get('available_source_knowledge', [])
    docs = [Counter(tokens(str(k.get('condition', ''))+' '+str(k.get('principle', '')))) for k in knowledge]
    n = len(docs); avg = sum(sum(d.values()) for d in docs)/max(n, 1)
    q = sorted(set(tokens(query))); ranked = []
    for k, d in zip(knowledge, docs):
        score = 0.
        for t in q:
            freq = d[t]
            if not freq: continue
            df = sum(t in doc for doc in docs)
            idf = math.log(1+(n-df+.5)/(df+.5))
            score += idf*freq*2.2/(freq+1.2*(.25+.75*sum(d.values())/avg))
        ranked.append({'decision_id':k['decision_id'], 'score':score})
    ranked.sort(key=lambda r:(-r['score'],r['decision_id']))
    return query, ranked


def read_explicit_plan(plan, context):
    """A post-hoc literal verb/ID reader, not a general language planner."""
    description = plan.get('description', '')
    match = re.search('增加|新增|移动|移到|移开|移除|删除', description)
    if not match: return None
    verb = match.group()
    name = ('add_furniture_to_scene_tool' if verb in ('增加', '新增') else
            'remove_furniture_tool' if verb in ('移除', '删除') else 'move_furniture_tool')
    key = 'asset_id' if name.startswith('add_') else 'object_id'
    ref = plan.get('object_or_asset_references')
    objects = context.get('objects', {})
    object_ids = (set(objects) if isinstance(objects, dict)
                  else {obj['object_id'] for obj in objects})
    ids = ({a['asset_id'] for a in context.get('asset_catalog', [])} if key=='asset_id'
           else object_ids)
    if not isinstance(ref, str) or ref not in ids: return None
    return {'name':name,'arguments':{key:ref}}


def make_prefix(value):
    return frame_completed_decision('<think>\n'+json.dumps(value,ensure_ascii=False)+'\n</think>')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--coverage',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    (args.output/'requests').mkdir()
    material=[json.loads(l) for l in args.source.read_text().splitlines()]
    chosen={r['unit_id']:r for r in material if r['arm']=='paired_tagged' and r['component']=='decision'}
    actions={r['unit_id']:r for r in material if r['arm']=='paired_tagged' and r['component']=='action'}
    coverage=json.loads(args.coverage.read_text()); source_rows={r['unit_id']:r for r in coverage['rows']}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    rows=[];variants=[];sources={str(args.source):digest(args.source.read_bytes()),str(args.coverage):digest(args.coverage.read_bytes())}
    for ident,r in chosen.items():
        a=actions[ident];original=source_rows[ident]
        b=subprocess.check_output(['git','show',coverage['pin']+':'+original['input_path']])
        assert digest(b)==original['input_sha256']
        sources[coverage['pin']+':'+original['input_path']]=digest(b)
        c=json.loads(b)['context']
        d=json.loads(r['target_text'].split('<think>',1)[1].split('</think>',1)[0])
        call=json.loads(a['target_text'].split('<tool_call>',1)[1].split('</tool_call>',1)[0])
        principle=d['principle_applicability'];ref=call['arguments'].get('object_id',call['arguments'].get('asset_id'))
        query,ranked=rank_knowledge(c)
        # Only now consult the teacher's selected principle for descriptive recall.
        gold_rank=next((i+1 for i,x in enumerate(ranked) if x['decision_id']==principle['source_decision_id']),None)
        public='\n'.join(item['text'] for m in a['messages'] if m['role']=='user'
                          for item in m['content'] if 'text' in item)
        exact_entries=[k for k in c.get('available_source_knowledge',[])
                       if k.get('decision_id')==principle['source_decision_id'] and k.get('principle')==principle['sourced_principle']]
        plan=read_explicit_plan(d['proposed_decision'],c)
        plan_match=bool(plan and plan['name']==call['name'] and
                        all(call['arguments'].get(k)==v for k,v in plan['arguments'].items()))
        rest=sorted(set(call['arguments'])-set(plan['arguments'])) if plan_match else sorted(call['arguments'])
        row={'unit_id':ident,'family_id':r['family_id'],'source_decision_id':principle['source_decision_id'],
             'source_principle_already_in_input':bool(exact_entries) and principle['sourced_principle'] in public,
             'selected_id_already_in_input':principle['source_decision_id'] in public,
             'exact_target_id_in_plan':d['proposed_decision']['object_or_asset_references']==ref,
             'plan_reader_matches_tool_and_id':plan_match,'plan_reader_output':plan,
             'remaining_action_parameters':rest,'native_executed':False,
             'declared_task':c.get('design_request'),'query':query,'bm25':ranked,'teacher_principle_rank':gold_rank,
             'decision_weight':r['target_component_weight'],'action_weight':a['target_component_weight'],
             'weight_ratio_D_to_A':r['target_component_weight']/a['target_component_weight'],
             'scope':'Descriptive teacher-field audit; target agreement is not a valid-design score.'}
        rows.append(row)
        input_path=args.output/'requests'/f'{ident}.json'
        write(input_path,{'unit_id':ident,'messages':phase_messages(raw_messages(a),'action'),
                         'source_request_sha256':a['request_source_sha256']})
        prefix_values={
            'no_decision':'',
            'selected_principle_only':make_prefix({'principle_applicability':principle}),
            'proposed_plan_only':make_prefix({'proposed_decision':d['proposed_decision']}),
            'condition_principle_limits':make_prefix({k:v for k,v in d.items() if k!='proposed_decision'}),
            'full_decision':frame_completed_decision(r['target_text']),
        }
        messages=json.loads(input_path.read_text())['messages']
        base=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        for name,prefix in prefix_values.items():
            before=tokenizer.encode(base+prefix,add_special_tokens=False)
            after=tokenizer.encode(base+prefix+a['target_text'],add_special_tokens=False)
            assert after==before+a['target_token_ids']
            variants.append({'unit_id':ident,'variant':name,'request_path':str(input_path),
                             'request_sha256':digest(input_path.read_bytes()),'assistant_prefix':prefix,
                             'prefix_tokens':len(tokenizer.encode(prefix,add_special_tokens=False)),
                             'target_text':a['target_text'],'target_token_ids':a['target_token_ids'],
                             'target_sha256':a['target_sha256'],'component':'action',
                             'evaluation_only_oracle_prefix':True,'training_admission':False,
                             'not_policy_output':True})
    (args.output/'oracle_prefixes.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in variants))
    counts={'reason_states':len(rows),'source_principle_already_in_input':sum(r['source_principle_already_in_input'] for r in rows),
            'exact_target_id_in_plan':sum(r['exact_target_id_in_plan'] for r in rows),
            'teacher_plan_tool_and_id_matches':sum(r['plan_reader_matches_tool_and_id'] for r in rows),
            'full_action_defined_by_plan':sum(r['plan_reader_matches_tool_and_id'] and not r['remaining_action_parameters'] for r in rows),
            'bm25_top1_teacher_selection':sum(r['teacher_principle_rank']==1 for r in rows),
            'bm25_top2_teacher_selection':sum(r['teacher_principle_rank'] in (1,2) for r in rows),
            'oracle_prefix_components':len(variants)}
    result={'counts':counts,'rows':rows,'source_sha256':sources,'source_pin':coverage['pin'],
            'bm25_contract':{'k1':1.2,'b':.75,'tokens':'Chinese character bigrams; alphanumeric words; unique query terms',
                             'query_fields':['design_request','current_requested_mode','current_activity_request.request'],
                             'candidate_fields':['condition','principle'],'weights_or_parameters_tuned':False},
            'scope':'Teacher plan reader and supplied-knowledge retrieval only. Neither places objects nor measures learning/generalization. Prefix ablations are oracle diagnostics, with unequal input lengths, not deployment arms.',
            'model_calls':0,'training_updates':0,'scene_edits':0,'wall_seconds':time.perf_counter()-tick}
    write(args.output/'audit.json',result)
    fields=['unit_id','family_id','source_principle_already_in_input','exact_target_id_in_plan','plan_reader_matches_tool_and_id','remaining_action_parameters','teacher_principle_rank','weight_ratio_D_to_A']
    with (args.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows: writer.writerow({k:r[k] for k in fields})
    print(json.dumps(counts))


if __name__=='__main__':main()
