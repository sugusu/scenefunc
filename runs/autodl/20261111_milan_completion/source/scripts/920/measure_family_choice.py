"""Measure frozen visual-contact and access indicators on actual native outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from native_project import sha256
from preflight_sourced_trajectory import boxes
from native_checkpoint import canonical_hash


def hit_segment(origin, target, lo, hi):
    direction = target-origin
    first,last = 0.,1.
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < lo[axis] or origin[axis] > hi[axis]: return False
        else:
            ends = sorted([(lo[axis]-origin[axis])/direction[axis],(hi[axis]-origin[axis])/direction[axis]])
            first=max(first,ends[0]);last=min(last,ends[1])
            if first>last: return False
    return first<1-1e-8 and last>1e-8


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];public=repo/'runs/920'/a.run;runtime=repo.parent/'runtime'/a.run
    lock=json.loads((public/'MEASUREMENT_LOCK.json').read_text())
    assert sha256(Path(__file__))==lock['code_sha256']
    assert sha256(public/'CHOICE_PLAN.json')==lock['choice_plan_sha256']
    spec=json.loads((public/'CHOICE_PLAN.json').read_text()); candidates={}
    for label in spec['candidates']:
        native=runtime/label/'native'
        initial=json.loads((native/'initial_state.json').read_text())
        state=json.loads((native/'final_state.json').read_text())
        access=json.loads((native/'CONDITIONAL_ACCESS.json').read_text())
        envelopes={k:(lo,hi) for k,(lo,hi) in boxes(state).items() if state['objects'][k]['object_type']=='furniture'}
        rays=[]
        for height in spec['sight']['heights_m']:
            origin=np.array([*spec['sight']['observer_xy'],height])
            for xy in spec['sight']['target_xy']:
                target=np.array([*xy,height])
                blockers=[oid for oid,(lo,hi) in envelopes.items() if hit_segment(origin,target,lo,hi)]
                rays.append(dict(height_m=height,origin=origin.tolist(),target=target.tolist(),blockers=blockers,clear=not blockers))
        fixed={k:v for k,v in initial['objects'].items() if k!='recon_tall_storage_0'}
        unchanged=all(state['objects'][k]==v for k,v in fixed.items())
        ids=set(state['objects'])==set(initial['objects'])
        dimensions=all(state['objects'][k]['bbox_min']==v['bbox_min'] and state['objects'][k]['bbox_max']==v['bbox_max'] for k,v in initial['objects'].items())
        require_access={str(s['square_body_width_m']):[n for n,t in s['targets'].items() if not t['reachable_from_entry']] for s in access['scenarios']}
        candidates[label]=dict(initial_state_sha256=canonical_hash(initial),final_state_sha256=canonical_hash(state),
            clear_rays=sum(x['clear'] for x in rays),total_rays=len(rays),rays=rays,
            failed_access_targets_by_width=require_access,all_declared_access=not any(require_access.values()),
            all_ids_preserved=ids,all_dimensions_preserved=dimensions,other_objects_exact=unchanged,
            furniture_count=len(envelopes),actual_storage_transform=state['objects']['recon_tall_storage_0']['transform'])
        assert ids and dimensions and unchanged
    assert len({c['initial_state_sha256'] for c in candidates.values()})==1
    eligible=[k for k,v in candidates.items() if v['all_declared_access']]
    rankings={}
    for task,larger in [('shared_visual_contact',True),('reduced_visual_distraction',False)]:
        rankings[task]=dict(eligible_candidates=eligible,criterion='declared sight-ray count after conditional access gate',
            preferred_by_proxy=([k for k in eligible if candidates[k]['clear_rays']==
                (max if larger else min)(candidates[z]['clear_rays'] for z in eligible)] if eligible else []),
            professional_label=False)
    result=dict(status='actual_native_choice_indicators_measured',ended_at=datetime.now(timezone.utc).isoformat(),
        candidates=candidates,conditional_proxy_rankings=rankings,source_families=1,independent_layouts_or_projects=False,
        native_edits=2,model_calls=0,training_updates=0,
        limits=spec['unknowns']+['Endpoints only: actual physical motion unrecorded.',
            'Two candidates are not exhaustive; different layouts can satisfy the same task.',
            'The contrast request and all quantitative thresholds/observers are researcher constructs, not professional feedback.'])
    (public/'MEASUREMENT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:dict(clear_rays=v['clear_rays'],failed_access=v['failed_access_targets_by_width']) for k,v in candidates.items()}))


if __name__=='__main__':main()
