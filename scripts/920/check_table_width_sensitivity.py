"""Analytic envelope perturbation, never a resized native scene or label."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from measure_free_floor import footprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());source=Path(plan['source_state'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_state_sha256']
    state=json.loads(source.read_text());rows=[]
    for factor in plan['width_factors']:
        objects=[copy.deepcopy(state['objects'][x]) for x in plan['table_ids']]
        for obj in objects:
            lo,hi=obj['bbox_min'][0],obj['bbox_max'][0];center=(lo+hi)/2
            obj['bbox_min'][0]=center+(lo-center)*factor
            obj['bbox_max'][0]=center+(hi-center)*factor
        a_poly,b_poly=[footprint(x) for x in objects]
        gap=a_poly.distance(b_poly);overlap=a_poly.intersection(b_poly).area
        rows.append({'table_width_factor':factor,'gap_m':gap,'overlap_area_m2':overlap,
            'fixed_pose_seam_condition':overlap<=1e-8 and gap<=plan['max_acceptable_gap_m']})
    result={'recorded_at':datetime.now(timezone.utc).isoformat(),'plan':plan,'rows':rows,
        'conclusion':'These recorded coordinates depend on the chosen table widths; a transferable decision must adapt positions to actual geometry. This does not reject the source strategy.',
        'supervised_examples_created':0,'native_calls':0,'images_created':0,'original_state_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_state_sha256']}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
