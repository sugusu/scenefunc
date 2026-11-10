"""Combine saved policy/tool ledgers without claiming full runtime coverage.

No inference or tool execution. A missing tool ledger means unknown, not zero.
Timings overlap stage lifecycle durations, so only measured model durations are
summed; the outer trial clock is reported separately. Fixture use is explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(policy_path, tool_path=None, *, fixture=False):
    def read(path):
        p=Path(path);raw=p.read_bytes()
        return json.loads(raw),{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()}
    policy,p_source=read(policy_path)
    tools,t_source=read(tool_path) if tool_path is not None else (None,None)
    p_rows=policy['calls'];t_rows=[] if tools is None else tools['records']
    started=[r for r in t_rows if r['generation_started']]
    def tokens(rows,key):
        values=[r.get(key) for r in rows]
        if any(v is not None and (type(v) is not int or v<0) for v in values):
            raise ValueError('Invalid token count')
        return {'known':sum(v for v in values if v is not None),
                'complete':all(v is not None for v in values)}
    p_in=tokens(p_rows,'input_tokens');p_out=tokens(p_rows,'output_tokens')
    t_in=tokens(started,'input_tokens');t_out=tokens(started,'output_tokens')
    observed=p_rows+started
    all_known=tools is not None and all(x['complete'] for x in (p_in,p_out,t_in,t_out))
    return {
        'source_ledgers':[p_source]+([t_source] if t_source else []),
        'fixture_only':fixture,
        'policy_generate_attempts':len(p_rows),'tool_generate_attempts':len(started) if tools is not None else None,
        'tool_requests_rejected_before_generation':sum(not r['generation_started'] for r in t_rows) if tools is not None else None,
        'measured_generate_attempts':len(observed),
        'known_input_tokens':p_in['known']+t_in['known'],
        'known_output_tokens':p_out['known']+t_out['known'],
        'tokens_complete_for_supplied_policy_and_tool_ledgers':all_known,
        'failed_generate_attempts':sum(r['status']=='raised' for r in observed),
        'unfinished_generate_attempts':sum(r['status']=='started' for r in observed),
        'policy_generate_seconds_known':sum(r.get('wall_seconds',0) for r in p_rows),
        'tool_serial_processing_seconds_known':sum(r.get('wall_seconds',0) for r in t_rows),
        'tool_queue_seconds_known':sum(r.get('queue_seconds',0) for r in t_rows),
        'trial_wall_seconds':policy.get('trial',{}).get('wall_seconds'),
        'full_runtime_cost_complete':False,
        'language_model_calls_if_fixture':0 if fixture else None,
        'limits':[
            'Model processing time is not isolated GPU kernel time.',
            'Tool processing includes encoding; no throughput comparison follows from this total.',
            'Do not add policy/tool time to trial wall time: they overlap.',
            'Other SDK clients, renderers, geometry, retrieval and asset preprocessing need separate evidence.',
            'Source ledgers must belong to the same trial; this collector does not prove their provenance.',
            'No monetary cost inferred from tokens; local and remote pricing are not interchangeable.'
        ]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--tool',type=Path)
    p.add_argument('--fixture',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=summarize(a.policy,a.tool,fixture=a.fixture)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
