"""CPU-only audit of actual over-budget context and matched G/U representations."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

from compact_catalog_context import pack_context,restore_catalog,OBJECT_FIELDS,geometry_feedback,use_feedback,canonical_hash


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    for x in plan['inputs'].values():
        if sha(Path(x['path']))!=x['sha256']:raise ValueError('Frozen input changed')
    for name,digest in plan['code_sha256'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('Frozen packing code changed')
    def deny(*args,**kwargs):raise RuntimeError('Network disabled for CPU encoding audit')
    socket.socket.connect=deny;socket.create_connection=deny;os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path.insert(0,str(Path(plan['inputs']['encoder']['path']).parent))
    from local_vlm import LocalQwenEngine
    original=json.loads(Path(plan['inputs']['failed_request']['path']).read_text())
    user=original[1]['content'];text=user[-1]['text'];body,memory=text.split('\n\n## Memory\n',1)
    brief,raw_context=body.split('\nCURRENT NATIVE STATE:\n',1);source=json.loads(raw_context)
    source['prior_tool_receipts']=json.loads(memory)['recent_actual_batches']
    packed,history=pack_context(source);audit=json.loads(Path(plan['inputs']['audit']['path']).read_text())
    state=json.loads(Path(plan['inputs']['native_state']['path']).read_text())['rooms']['geneva_development']
    checks={
        'actual_object_geometry_and_identity_preserved':packed['objects']=={k:{f:v[f] for f in OBJECT_FIELDS} for k,v in state['objects'].items()},
        'catalog_roundtrip_exact':restore_catalog(packed)==source['asset_catalog'],
        'request_and_unknowns_exact':all(packed[k]==source[k] for k in ('design_request','original_request','request_provenance','unknowns','asset_bindings','unavailable_instance_ids')),
        'all_current_objects_retained':set(packed['objects'])==set(source['objects']),
        'all_arguments_retained':all(new['tool_calls']==old['tool_calls'] for new,old in zip(history['recent_actual_batches'],source['prior_tool_receipts'])),
    }
    for new,old in zip(history['recent_actual_batches'],source['prior_tool_receipts']):
        for nr,orr in zip(new['results'],old['results']):
            if nr['name']=='get_current_scene_state' and nr['output'].get('historical_details_omitted'):
                checks['query_hash_'+str(new['step_idx'])]=nr['output']['full_query_result_sha256']==canonical_hash(orr['output'])
                checks['query_ids_'+str(new['step_idx'])]=nr['output']['queried_object_ids']==[x['object_id'] for x in orr['output']['objects']]
            else:checks['other_receipt_'+str(new['step_idx'])]=nr==orr
    if not all(checks.values()):raise ValueError('Current facts or receipt contract lost')
    # Higher ceiling ONLY to measure the already-failed original input on CPU.
    # Actual policy plans retain the prior 12288-token hard inference limit.
    engine=LocalQwenEngine(plan['model_directory'],a.out/'unused',max_pixels=65536,max_input_tokens=20000,max_new_tokens=1024)
    result={'started_at':datetime.now(timezone.utc).isoformat(),'checks':checks,'model_calls':0,'training_updates':0,'encoder':{}}
    _,encoding=engine.encode(original);result['encoder']['original']=encoding
    for arm in ('G','U'):
        feedback={'geometry':geometry_feedback(audit)}
        if arm=='U':feedback['use']=use_feedback(audit)
        messages=copy.deepcopy(original)
        messages[1]['content'][-1]['text']=brief+'\nCURRENT NATIVE STATE:\n'+json.dumps(packed,ensure_ascii=False,separators=(',',':'))+'\n\n## Memory\n'+json.dumps(history,ensure_ascii=False,separators=(',',':'))+'\n\n## Independent current-state feedback\n'+json.dumps(feedback,ensure_ascii=False,separators=(',',':'))
        write(a.out/(arm+'_request.json'),messages)
        _,encoded=engine.encode(messages);result['encoder'][arm]=encoded
        if encoded['input_tokens']>plan['policy_max_input_tokens']:raise ValueError('Packed input still exceeds policy ceiling')
        if encoded['images']!=encoding['images']:raise ValueError('Actual five views changed')
    result.update(status='passed',ended_at=datetime.now(timezone.utc).isoformat(),policy_max_input_tokens=plan['policy_max_input_tokens'],
        fact_scope='Same current typed object geometry, IDs, asset bindings, catalog content, task and unknowns as prior baseline projection. Past query details explicitly summarized, full receipts unchanged.',
        note='No claim of equal G/U input length or FLOPs; use information is the intended treatment.')
    write(a.out/'RESULT.json',result)
    print(json.dumps({'status':result['status'],'tokens':{k:v['input_tokens'] for k,v in result['encoder'].items()},'checks':len(checks)}),flush=True)


if __name__=='__main__':main()
