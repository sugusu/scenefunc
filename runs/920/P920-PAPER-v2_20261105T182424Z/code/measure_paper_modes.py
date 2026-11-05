"""Declared occupancy/mode diagnostics; a clear reserve is not a working sofa bed."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
from preflight_sourced_trajectory import boxes
from measure_room_access import measure
from native_project import validate_bundle,sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--states',type=Path,required=True);ap.add_argument('--phase',choices=['preflight','native'],required=True);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();plan=json.loads(a.plan.read_text());manifest,_,_,_=validate_bundle(a.bundle);room=a.bundle/('room_'+manifest['selected_room_id']);rows=[];states={}
    for i in range(1,11):
        name=f'proposed_step_{i:02d}.json' if a.phase=='preflight' else f'step_{i:02d}_after.json';state=json.loads((a.states/name).read_text());states[i]=state;bounds={k:v for k,v in boxes(state).items() if state['objects'][k]['object_type']=='furniture'}
        contacts=[[k,l] for (k,(lo,hi)),(l,(ol,oh)) in itertools.combinations(bounds.items(),2) if np.all(np.minimum(hi,oh)-np.maximum(lo,ol)>1e-6)]
        rect=plan['guest_reserve_xy_m'];lo=np.array(rect[:2]);hi=np.array(rect[2:]);occupancy={k:float(np.prod(np.maximum(0,np.minimum(v[1][:2],hi)-np.maximum(v[0][:2],lo)))) for k,v in bounds.items()};occupancy={k:v for k,v in occupancy.items() if v>1e-8}
        access=measure(state,room,a.bundle,plan['reference_measurement']);sweep=[]
        if i>6:
            oid=plan['moves'][i-7]['object_id'];old=boxes(states[i-1])[oid];new=bounds[oid];slo=np.minimum(old[0],new[0]);shi=np.maximum(old[1],new[1]);sweep=[k for k,(x,y) in bounds.items() if k!=oid and np.all(np.minimum(shi,y)-np.maximum(slo,x)>1e-6)]
        places=[]
        if i>=6:
            table=bounds['mobile_table_0'];tc=(table[0]+table[1])/2
            for oid,side in [('chair_left_0',-1),('chair_right_0',1)]:
                chair=bounds[oid];cc=(chair[0]+chair[1])/2;gap=side*(cc[0]-(table[1][0] if side>0 else table[0][0]));aligned=abs(cc[1]-tc[1])+(chair[1][1]-chair[0][1])/2 <= (table[1][1]-table[0][1])/2+1e-6
                places.append({'chair':oid,'center_to_table_edge_m':float(gap),'within_declared_alignment_and_distance':bool(aligned and .2<=gap<=.75),'human_use':'unknown'})
        rows.append({'step':i,'furniture':len(bounds),'positive_envelope_pairs':contacts,'reserve_occupants_overlap_m2':occupancy,'reserve_clear':not occupancy,'fixed_access':access,'swept_union_furniture_conflicts':sweep,'declared_dining_places':places})
    before,after=states[6],states[10];same_ids=set(before['objects'])==set(after['objects']);errors={k:float(np.linalg.norm(np.array(v['transform']['translation'])-after['objects'][k]['transform']['translation'])) for k,v in before['objects'].items()};maxerr=max(errors.values())
    summary={'steps':10,'all_endpoint_envelopes_clear':all(not x['positive_envelope_pairs'] for x in rows),'fixed_access_counts_by_step':[[sum(v['reachable_from_entry'] for v in q['targets'].values()) for q in x['fixed_access']['scenarios']] for x in rows],'guest_reserve_clear_at_step8':rows[7]['reserve_clear'],'everyday_places_at_step6':sum(x['within_declared_alignment_and_distance'] for x in rows[5]['declared_dining_places']),'restored_places_at_step10':sum(x['within_declared_alignment_and_distance'] for x in rows[9]['declared_dining_places']),'same_ids_after_return':same_ids,'max_return_translation_error_m':maxerr,'swept_union_furniture_conflicts':[x['swept_union_furniture_conflicts'] for x in rows[6:]],'sleeping_function_verified':False,'model_calls':0,'training_updates':0}
    result={'status':'declared_mode_diagnostics_complete','phase':a.phase,'mode_plan_sha256':sha256(a.plan),'manifest_sha256':sha256(a.bundle/'manifest.json'),'summary':summary,'rows':rows,'limits':['No sofa articulation, actual guest sleep, physical pushing, collision forces, chair comfort or professional recommendation established.','Swept union is a conservative fixed-yaw translation envelope against furniture; it is not a robot or human motion model and does not validate door operation.','Declared chair alignment/distance is not ergonomic or human-use validation.','Research dimensions and reserve are hypothetical; nominal pass cannot establish validity for the original45m²home.']}
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
