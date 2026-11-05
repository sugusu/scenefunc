"""Object-relative checks of policy outputs; does not read reference candidate poses."""
import argparse
import copy
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import numpy as np
from native_project import sha256,validate_bundle
from inspect_native_mesh_contacts import world_transform
from measure_room_access import measure
from measure_family_choice import hit_segment
from preflight_sourced_trajectory import boxes


def evaluate(state, initial, room, root, spec, condition):
    missing=set(initial['objects'])-set(state['objects']);extra=set(state['objects'])-set(initial['objects'])
    changed=[]
    for oid in set(initial['objects']) & set(state['objects']):
        a,b=copy.deepcopy(initial['objects'][oid]),copy.deepcopy(state['objects'][oid])
        if not a['immutable']:a.pop('transform');b.pop('transform')
        if a!=b:changed.append(oid)
    result=dict(missing_ids=sorted(missing),extra_ids=sorted(extra),changed_protected_records=changed,
        identity_and_dimensions_preserved=not(missing or extra or changed),condition=condition,
        conditional_proxy_pass=False,whole_room_success='unknown_not_professionally_accepted')
    required=set(spec['seat_ids']+[spec['table_id'],spec['observer_object_id'],spec['target_object_id']]+spec['functional_front_ids'])
    if not required.issubset(state['objects']):
        result['measurement_error']='Required activity object deleted; no success from empty space.';return result
    furniture={k:v for k,v in state['objects'].items() if v['object_type']=='furniture'}
    envelopes={k:(lo,hi) for k,(lo,hi) in boxes(state).items() if k in furniture}
    geometry=state['room_geometry'];outside=[k for k,(lo,hi) in envelopes.items()
        if lo[0]<-geometry['length']/2 or hi[0]>geometry['length']/2 or lo[1]<-geometry['width']/2 or hi[1]>geometry['width']/2]
    overlaps=[dict(objects=[a,b],overlap_m=(np.minimum(ah,bh)-np.maximum(al,bl)).tolist())
        for (a,(al,ah)),(b,(bl,bh)) in itertools.combinations(envelopes.items(),2)
        if np.all(np.minimum(ah,bh)-np.maximum(al,bl)>1e-6)]
    transforms={k:world_transform(v) for k,v in state['objects'].items()}
    oi,ti=spec['observer_object_id'],spec['target_object_id']
    old_obs_inv=np.linalg.inv(world_transform(initial['objects'][oi]));old_target_inv=np.linalg.inv(world_transform(initial['objects'][ti]))
    rays=[]
    for height in spec['sight']['heights_m']:
        source_old=np.array([*spec['sight']['observer_xy'],height,1.])
        origin=(transforms[oi]@old_obs_inv@source_old)[:3]
        for xy in spec['sight']['target_xy']:
            target=(transforms[ti]@old_target_inv@np.array([*xy,height,1.]))[:3]
            blocked=[k for k,(lo,hi) in envelopes.items() if hit_segment(origin,target,lo,hi)]
            rays.append(dict(origin=origin.tolist(),target=target.tolist(),blockers=blocked,clear=not blocked))
    table=state['objects'][spec['table_id']];tm=transforms[spec['table_id']];inv=np.linalg.inv(tm)
    width,depth=spec['place_width_depth_m'];places=[];seat_data={}
    for oid in spec['seat_ids']:
        obj=state['objects'][oid];center=(inv@np.r_[transforms[oid][:3,3],1])[:3]
        corners=np.array(list(itertools.product(*zip(obj['bbox_min'],obj['bbox_max']))))
        corners=np.c_[corners,np.ones(len(corners))]@transforms[oid].T@inv.T
        side=1 if center[1]>=0 else -1
        edge=table['bbox_max'][1] if side>0 else table['bbox_min'][1]
        gap=float(side*((corners[:,1].min() if side>0 else corners[:,1].max())-edge))
        rect=[float(center[0]-width/2),float(min(edge,edge-side*depth)),float(center[0]+width/2),float(max(edge,edge-side*depth))]
        within=rect[0]>=table['bbox_min'][0] and rect[2]<=table['bbox_max'][0] and rect[1]>=table['bbox_min'][1] and rect[3]<=table['bbox_max'][1]
        valid=bool(within and spec['seat_table_gap_m'][0]<=gap<=spec['seat_table_gap_m'][1])
        places.append(dict(object_id=oid,side=side,gap_m=gap,rectangle_table_local=rect,valid=valid))
        direction=tm[:2,1]*side
        worldcorners=(np.c_[np.array(list(itertools.product(*zip(obj['bbox_min'],obj['bbox_max'])))),np.ones(8)]@transforms[oid].T)[:,:2]
        radius=float(max((worldcorners-transforms[oid][:2,3])@direction))
        seat_data[oid]=(transforms[oid][:2,3],direction,radius)
    place_overlaps=[]
    for a,b in itertools.combinations(places,2):
        x,y=a['rectangle_table_local'],b['rectangle_table_local']
        if min(x[2],y[2])-max(x[0],y[0])>1e-6 and min(x[3],y[3])-max(x[1],y[1])>1e-6:
            place_overlaps.append([a['object_id'],b['object_id']])
    scenarios=[]
    for body in spec['access_settings']['square_body_widths_m']:
        points=dict(spec['fixed_access'])
        for oid,(center,direction,radius) in seat_data.items():points[oid+':seat_back']=(center+direction*(radius+body/2+.05)).tolist()
        for oid in spec['functional_front_ids']:
            obj=state['objects'][oid];lo,hi=np.array(obj['bbox_min']),np.array(obj['bbox_max'])
            point=np.array([(lo[0]+hi[0])/2,hi[1]+spec['front_offset_m'],0,1])
            points[oid+':declared_front']=(transforms[oid]@point)[:2].tolist()
        cfg={**spec['access_settings'],'square_body_widths_m':[body],'approach_points_xy_m':points}
        scenarios.append(measure(state,room,root,cfg)['scenarios'][0])
    clear=sum(x['clear'] for x in rays);visual=clear==15 if condition=='shared_visual_contact' else clear==0
    all_access=all(x['reachable_from_entry'] for s in scenarios for x in s['targets'].values())
    four_places=all(x['valid'] for x in places) and not place_overlaps
    result.update(clear_rays=clear,total_rays=len(rays),rays=rays,visual_target_met_under_declared_rule=visual,
        places=places,overlapping_place_rectangles=place_overlaps,four_declared_places=bool(four_places),
        access=scenarios,all_role_relative_access=all_access,outside_furniture=outside,positive_furniture_aabb_pairs=overlaps)
    result['conditional_proxy_pass']=bool(result['identity_and_dimensions_preserved'] and visual and four_places and all_access and not outside and not overlaps)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--condition',choices=['shared_visual_contact','reduced_visual_distraction'],required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();spec=json.loads(a.plan.read_text())
    for name,h in spec['code_sha256'].items():assert sha256(Path(__file__).parent/name)==h,name
    original=Path(spec['initial_bundle']);assert sha256(original/'manifest.json')==spec['initial_manifest_sha256']
    manifest,house,_,_=validate_bundle(a.bundle);rid=manifest['selected_room_id'];initial=validate_bundle(original)[1]['rooms'][rid]
    result=evaluate(house['rooms'][rid],initial,a.bundle/('room_'+rid),a.bundle,spec,a.condition)
    result.update(evaluated_at=datetime.now(timezone.utc).isoformat(),plan_sha256=sha256(a.plan),
        input_manifest_sha256=sha256(a.bundle/'manifest.json'),model_calls=0,scene_edits=0,
        limits=['Declared geometric rules, no human-use/professional acceptance.',
            'View targets follow actual objects; missing objects fail rather than making empty-space success.',
            'Seated-place rectangles are planar assumptions; real knee room, posture, comfort and storage operation unverified.',
            'AABB visibility and routes are conservative; no acoustic or perceived privacy claim.'])
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ['condition','clear_rays','four_declared_places','all_role_relative_access','conditional_proxy_pass']}))


if __name__=='__main__':main()
