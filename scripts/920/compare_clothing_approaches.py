"""Compare actual approach faces of one rail and one planned static wardrobe."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from native_project import sha256,validate_bundle
from inspect_native_mesh_contacts import world_transform
from measure_room_access import measure


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();p=json.loads(a.plan.read_text())
    for f,h in p['code_sha256'].items():assert sha256(Path(__file__).parent/f)==h
    bundle=Path(p['bundle']);assert sha256(bundle/'manifest.json')==p['manifest_sha256']
    manifest,house,_,_=validate_bundle(bundle);rid=manifest['selected_room_id'];room=bundle/('room_'+rid)
    print(datetime.now(timezone.utc).isoformat(),'measure distinct clothing access faces; no native edits',flush=True)
    proposed=Path(p['proposed_state']);assert sha256(proposed)==p['proposed_state_sha256']
    states={'open_rail':house['rooms'][rid],'closed_wardrobe':json.loads(proposed.read_text())};results={}
    for label,state in states.items():
        oid=p['objects'][label]['id'];obj=state['objects'][oid];lo,hi=np.array(obj['bbox_min']),np.array(obj['bbox_max']);matrix=world_transform(obj);strips=[]
        for width in p['measurement']['square_body_widths_m']:
            radius=width/2;start,end=lo[0]+radius,hi[0]-radius
            if start>end:start=end=(lo[0]+hi[0])/2
            targets={}
            for side in p['objects'][label]['sides']:
                y=hi[1]+radius+p['clearance_m'] if side=='+Y' else lo[1]-radius-p['clearance_m']
                for i,x in enumerate(np.linspace(start,end,p['line_samples'])):targets[side+'_'+str(i)]=(matrix@np.array([x,y,0,1]))[:2].tolist()
            scenario=measure(state,room,bundle,{**p['measurement'],'square_body_widths_m':[width],'approach_points_xy_m':targets})['scenarios'][0]
            strips.append(dict(width_m=width,side_reachable={side:sum(v['reachable_from_entry'] for k,v in scenario['targets'].items() if k.startswith(side+'_')) for side in p['objects'][label]['sides']},details=scenario))
        fixed=measure(state,room,bundle,p['measurement'])
        results[label]=dict(object_id=oid,width_depth_height_m=(hi-lo).tolist(),local_bbox_floor_area_m2=float(np.prod((hi-lo)[:2])),access_strips=strips,fixed_access=fixed,
            at_least_one_allowed_face_point_each_width=all(any(v>0 for v in x['side_reachable'].values()) for x in strips))
    old=states['open_rail'];new=states['closed_wardrobe'];keep=set(old['objects'])-{p['objects']['open_rail']['id']}
    assert keep<=set(new['objects']) and all(old['objects'][k]==new['objects'][k] for k in keep)
    assert old['room_geometry']==new['room_geometry']
    accepted=results['closed_wardrobe']['at_least_one_allowed_face_point_each_width'] and all(v['reachable_from_entry'] for s in results['closed_wardrobe']['fixed_access']['scenarios'] for v in s['targets'].values())
    result=dict(status='preflight_face_comparison_complete',evaluated_at=datetime.now(timezone.utc).isoformat(),plan_sha256=sha256(a.plan),results=results,
        native_candidate_admitted=accepted,other_object_records_unchanged=len(keep),native_edits=0,model_calls=0,training_updates=0,
        scope='Wardrobe state is proposed, not executed; old rail state already executed. Only geometry/reachability of declared body probes, not opening or grasping.',
        fairness_limits=['A same-anchor same-yaw failure cannot establish the wardrobe type is worse: door-side orientation and other feasible placements are uncontrolled.', 'An open rail may be approached from either side; a wardrobe rear is not a valid door-side access substitute.', 'Static closed door geometry does not establish interior capacity, articulation, access speed, privacy preference or cleanliness.', 'Source supports a short-stay rationale, not a historical rejection or long-stay preference label.'])
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(admitted=accepted,counts={k:[x['side_reachable'] for x in v['access_strips']] for k,v in results.items()})),flush=True)


if __name__=='__main__':main()
