"""Exact interval test for fixed-orientation translating AABBs; retain broad warnings."""
import argparse,json
from pathlib import Path
import numpy as np
from preflight_sourced_trajectory import boxes
from native_project import sha256

def collision_interval(old,new,static):
    delta=new[0]-old[0];assert np.allclose(new[1]-old[1],delta,atol=1e-8,rtol=0),'Changing orientation/size is outside fixed-AABB translation test'
    low,high=0.,1.
    for axis in range(3):
        v=delta[axis]
        if abs(v)<1e-12:
            if min(old[1][axis],static[1][axis])-max(old[0][axis],static[0][axis])<=1e-6:return None
        else:
            a=(static[0][axis]+1e-6-old[1][axis])/v;b=(static[1][axis]-1e-6-old[0][axis])/v
            low=max(low,min(a,b));high=min(high,max(a,b))
            if high<=low:return None
    return [float(low),float(high)] if high>low else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--states',type=Path,required=True);ap.add_argument('--phase',choices=['preflight','native'],required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();plan=json.loads(a.plan.read_text());rows=[]
    def state(i):return json.loads((a.states/(f'proposed_step_{i:02d}.json' if a.phase=='preflight' else f'step_{i:02d}_after.json')).read_text())
    for i,move in enumerate(plan['moves'],7):
        before,after=state(i-1),state(i);old,new=boxes(before),boxes(after);oid=move['object_id'];hits=[]
        assert before['objects'][oid]['transform']['rotation_wxyz']==after['objects'][oid]['transform']['rotation_wxyz'],'Rotation change requires another model'
        for k,obj in before['objects'].items():
            if k==oid or obj['object_type']!='furniture':continue
            interval=collision_interval(old[oid],new[oid],old[k])
            if interval is not None:hits.append({'object':k,'positive_overlap_time_interval':interval})
        rows.append({'step':i,'object':oid,'interval_conflicts':hits})
    result={'phase':a.phase,'mode_plan_sha256':sha256(a.plan),'rows':rows,'all_fixed_orientation_translation_envelopes_clear':all(not x['interval_conflicts'] for x in rows),'scope':'Continuous AABB translation against other furniture; no rotated sweep, human pushing, articulation, wall/door operating or mechanical validation. Broad swept-union warnings remain separately recorded.'}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
