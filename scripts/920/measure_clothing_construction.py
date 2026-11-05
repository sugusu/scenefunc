"""Measure construction endpoint and keep both accessible and inaccessible rail sides."""
import argparse,json
from pathlib import Path
import numpy as np
from native_project import validate_bundle,sha256
from inspect_native_mesh_contacts import world_transform
from measure_room_access import measure

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    plan=json.loads(a.plan.read_text())
    for n,h in plan['code_sha256'].items():assert sha256(Path(__file__).parent/n)==h,n
    root=Path(plan['bundle']);manifest,house,_,_=validate_bundle(root);rid=manifest['selected_room_id'];state=house['rooms'][rid];room=root/('room_'+rid)
    rail=state['objects'][plan['rail_id']];definition=plan['rail_approach_definition'];base=plan['measurement'];strips=[]
    for body in base['square_body_widths_m']:
        radius=body/2;lo,hi=np.array(rail['bbox_min']),np.array(rail['bbox_max']);start,end=lo[0]+radius,hi[0]-radius
        if start>end:start=end=(lo[0]+hi[0])/2
        points={}
        for side in definition['local_sides']:
            y=hi[1]+radius+definition['clearance_m'] if side=='+Y' else lo[1]-radius-definition['clearance_m']
            for i,x in enumerate(np.linspace(start,end,definition['line_samples'])):
                points[side+'_'+str(i)]=list(map(float,(world_transform(rail)@np.array([x,y,0,1]))[:2]))
        strips.append(measure(state,room,root,{**base,'square_body_widths_m':[body],'approach_points_xy_m':points})['scenarios'][0])
    fixed=measure(state,room,root,base)
    result={'status':'construction_endpoint_measured','source_plan_sha256':sha256(a.plan),'manifest_sha256':sha256(root/'manifest.json'),'rail_id':plan['rail_id'],'fixed_access':fixed,'rail_approaches':strips,'interpretation':'Endpoint conditions only. Empty initial space has no clothing provision; route metrics are not a quality gain over emptiness. Opposite rail side failures retained.','unknowns':plan['unknowns'],'model_calls':0,'training_updates':0}
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'fixed_reachable_counts':[sum(t['reachable_from_entry'] for t in x['targets'].values()) for x in fixed['scenarios']],'rail_reachable_counts':[sum(t['reachable_from_entry'] for t in x['targets'].values()) for x in strips]}))

if __name__=='__main__':main()
