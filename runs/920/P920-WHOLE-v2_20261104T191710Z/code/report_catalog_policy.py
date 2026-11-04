"""Summarize recorded policy costs and independently checked partial-room facts.

Schema mismatch review is explicitly post hoc; no scene or prompt is changed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import jsonschema


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.runtime;rollout=json.loads((root/'rollout/RESULT.json').read_text())
    audit=json.loads((root/'FINAL_AUDIT.json').read_text())
    models=[json.loads(p.read_text()) for p in sorted((root/'rollout').glob('round_*/model/RESULT.json'))]
    forwards=[x for x in models if x['model_calls']]
    obs=json.loads((root/'prepare/observation.json').read_text())
    schemas,_=json.JSONDecoder().raw_decode(obs['system'].split('Available executable tools:\n',1)[1])
    schemas={x['name']:x['parameters'] for x in schemas}
    mismatches=[];tools=[]
    for row in rollout['rounds']:
        for call in row['calls']:
            tools.append(call['name'])
            errors=list(jsonschema.Draft202012Validator(schemas[call['name']]).iter_errors(call.get('arguments',{})))
            if errors:mismatches.append({'turn':row['turn'],'tool':call['name'],
                'errors':[x.message for x in errors],
                'native_success':row['reply']['results'][0]['output'].get('success')})
    result={'recorded_at':datetime.now(timezone.utc).isoformat(),'run_id':root.name,
        'rollout_status':rollout['status'],'actual_model_calls':sum(x['model_calls'] for x in models),
        'failed_pre_generation_attempts':sum(x['status']=='failed' and not x['model_calls'] for x in models),
        'actual_tool_calls':rollout['tool_calls'],'tool_counts':dict(Counter(tools)),
        'output_tokens':sum(x.get('output_tokens',0) for x in forwards),
        'generation_seconds':sum(x.get('generation_seconds',0) for x in forwards),
        'model_loading_seconds':sum(x.get('load_seconds',0) for x in forwards),
        'peak_allocated_bytes':max(x.get('peak_allocated_bytes',0) for x in forwards),
        'successful_input_token_range':[min(x['encoding']['input_tokens'] for x in forwards),max(x['encoding']['input_tokens'] for x in forwards)],
        'failure_reasons':[x.get('error') for x in models if x['status']=='failed'],
        'last_completed_checkpoint':rollout['rounds'][-1]['checkpoint'],
        'furniture_count':len(audit['inventory']),'actual_furniture':[x['object_id'] for x in audit['inventory']],
        'intersection_pairs':[x['objects'] for x in audit['positive_aabb_pairs'] if x['triangle_surface_intersection_detected']],
        'outside_room_count':sum(x['outside_nominal_room_xy'] for x in audit['bounds']),
        'immutable_structure_unchanged':all(audit['immutable_records_unchanged'].values()),
        'largest_empty_rectangle_m2':audit['free_floor']['largest_grid_empty_rectangle_m2'],
        'conditional_access':[{ 'square_body_width_m':s['square_body_width_m'],
            'reachable_of_hypothetical_approach_points':[sum(v['reachable_from_entry'] for v in s['targets'].values()),len(s['targets'])],
            'unreachable':[k for k,v in s['targets'].items() if not v['reachable_from_entry']]}
            for s in audit['access']['scenarios']],
        'post_hoc_schema_review':{'mismatches':mismatches,'jsonschema_version':importlib.metadata.version('jsonschema'),
            'scope':'Validate recorded arguments against advertised schemas; no replay, new scoring thresholds or model change.'},
        'whole_room_accepted':False,'professional_review':False,'training_updates':0,
        'interpretation':'Partial baseline only: visible mesh intersections, missing functional inventory and unknown privacy/installation. Empty area or native tool success cannot establish usable dwelling or learned design improvement.',
        'sources':{str(p.relative_to(root)):sha(p) for p in [root/'rollout/RESULT.json',root/'FINAL_AUDIT.json',root/'prepare/observation.json']},
        'report_script_sha256':sha(Path(__file__))}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['actual_model_calls','actual_tool_calls','furniture_count','intersection_pairs','whole_room_accepted']}))


if __name__=='__main__':main()
