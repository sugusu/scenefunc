"""Identify the actual intervention and prepare an optional shared-support control.

Existing arms both bind every available reason to the correct scene. Their
difference is whether action prediction reads that reason. This audit derives
that distinction from the actual components. The optional mixed-context arm
splits existing action-token weight between with/without-prefix copies; it is
an ordinary context-dropout diagnostic, not a novel objective or trained model.
"""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import time

from audit_material_routing import digest, write

MATERIAL=Path('runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl')


def optional_shared_support(components, conditioned_fraction=0.5):
    if not 0<conditioned_fraction<1:raise ValueError('Both contexts need positive mass')
    out=[]
    for c in components:
        if c['arm']!='paired_phase_v3':continue
        split=c['stage']=='action' and bool(c['assistant_prefix'])
        versions=[('reason_conditioned',conditioned_fraction),('no_reason',1-conditioned_fraction)] if split else [('unchanged',1.)]
        for variant,factor in versions:
            row=copy.deepcopy(c)
            row['arm']='mixed_context_diagnostic_v1';row['variant']=variant
            row['component']=row['stage']
            row['source_component_arm']=c['arm']
            row['token_objective_weights']=[w*factor for w in c['token_objective_weights']]
            row['original_weight_fraction']=factor
            row['training_authorized']=False
            if variant=='no_reason':
                row['assistant_prefix']='';row['assistant_prefix_token_ids']=[]
                row['host_separator_tokens_added']=0
                row.pop('text_prefix_tokens',None)  # Must be re-encoded, not guessed.
            out.append(row)
    return out


def mass_by_source(records):
    out=defaultdict(float)
    for c in records:
        for origin,w,tok in zip(c['source_token_origins'],c['token_objective_weights'],c['target_token_ids']):
            out[(c['unit_id'],origin['component'],origin['index'],tok)]+=w
    return dict(out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();rows=[json.loads(s) for s in MATERIAL.read_text().splitlines()]
    index={(r['unit_id'],r['stage'],r['arm']):r for r in rows}
    pairs=[]
    for ident,stage,arm in index:
        if arm!='paired_phase_v3':continue
        p=index[ident,stage,arm];s=index[ident,stage,'separate_phase_v3']
        for key in ('messages','target_token_ids','token_objective_weights','source_token_origins','source_request_sha256'):
            assert p[key]==s[key],(ident,stage,key)
        pairs.append(dict(unit_id=ident,stage=stage,family_id=p['family_id'],
            same_scene_bound_messages=True,same_targets=True,same_target_coefficients=True,
            separate_prefix_tokens=len(s['assistant_prefix_token_ids']),
            paired_prefix_tokens=len(p['assistant_prefix_token_ids']),
            only_reason_prefix_differs=s['assistant_prefix']!=p['assistant_prefix'],
            target_tokens=len(p['target_token_ids']),original_mass=sum(p['token_objective_weights'])))
    mixed=optional_shared_support(rows)
    old=mass_by_source([r for r in rows if r['arm']=='paired_phase_v3']);new=mass_by_source(mixed)
    assert old.keys()==new.keys()
    maxerr=max(abs(old[k]-new[k]) for k in old)
    assert maxerr<1e-15
    with (args.output/'optional_mixed_components.jsonl').open('w') as handle:
        for r in mixed:handle.write(json.dumps(r,ensure_ascii=False)+'\n')
    actions=[r for r in pairs if r['stage']=='action'];decisions=[r for r in pairs if r['stage']=='decision']
    result={'source_path':str(MATERIAL),'source_sha256':digest(MATERIAL.read_bytes()),'pairs':pairs,
        'counts':{'source_units':len(actions),'decision_components_per_arm':len(decisions),
            'action_components_per_arm':len(actions),
            'reason_conditioned_action_units':sum(r['only_reason_prefix_differs'] for r in actions),
            'identical_action_context_units':sum(not r['only_reason_prefix_differs'] for r in actions),
            'identical_scene_bound_decision_components':sum(not r['only_reason_prefix_differs'] for r in decisions),
            'optional_mixed_components':len(mixed),'distinct_original_target_tokens':len(old)},
        'optional_control_max_token_mass_error':maxerr,
        'identified_intervention':'Explicit action conditioning on a scene-bound reason versus auxiliary reason prediction; both original arms retain correct reason-to-scene associations.',
        'cannot_identify':'Pure correct-vs-incorrect data pairing effect, unseen-project quality, or runtime reason value separately from training conditioning.',
        'missing_reason_scope':'14 action states have no supervised D; both arms use direct A there. Do not describe all 24 units as fully paired.',
        'optional_control_scope':'Same-model with/without-generated-reason diagnostic has both training contexts supported for the ten annotated units. It does not remove teacher-to-generated reason shift or make all missing-reason states supported.',
        'optional_control_compute':'Extra action-prefix copies add forward cost despite unchanged per-token target mass. Report real compute; do not equate equal coefficients with equal wall time.',
        'deployment':'A future within-model comparison must use current observations and its own generated reason, share tools and track total tokens and cost. No supplied teacher prefix for primary evaluation.',
        'optional_control_is_new_method':False,'training_authorized':False,'model_calls':0,
        'training_updates':0,'scene_edits':0,'wall_seconds':time.perf_counter()-tick}
    write(args.output/'attribution.json',result)
    print(json.dumps(result['counts']))


if __name__=='__main__':main()
