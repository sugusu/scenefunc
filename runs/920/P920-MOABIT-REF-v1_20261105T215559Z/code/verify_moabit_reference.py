"""Independent native-bounds and route checks for a proposed geometry reference."""
import argparse,copy,itertools,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from native_project import sha256,validate_bundle
from native_checkpoint import canonical_hash
from inspect_native_mesh_contacts import world_transform
from measure_room_access import measure,obstacles
import trimesh


def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    p=json.loads(a.plan.read_text());a.out.mkdir(exist_ok=False,parents=True)
    for f,h in p['code_sha256'].items():assert sha256(Path(__file__).parent/f)==h,f
    assert sha256(Path(p['solution']))==p['solution_sha256'];solution=json.loads(Path(p['solution']).read_text())
    bundle=Path(p['bundle']);assert sha256(bundle/'manifest.json')==p['manifest_sha256']
    manifest,house,_,task=validate_bundle(bundle);rid=manifest['selected_room_id'];initial=house['rooms'][rid];state=copy.deepcopy(initial)
    templates={e['template']['object_id']:e['template'] for e in json.loads((bundle/'catalog.json').read_text())['entries']}
    print(datetime.now(timezone.utc).isoformat(),'begin independent full-scene geometry and routes',flush=True)
    for row in solution['commands']:
        args=row['command']['arguments'];obj=copy.deepcopy(templates[args['asset_id']]);angle=math.radians(args['yaw']);oid=row['planned_id']
        assert oid not in state['objects']
        obj.update(object_id=oid,transform=dict(translation=[args['x'],args['y'],0],rotation_wxyz=[math.cos(angle/2),0,0,math.sin(angle/2)]))
        state['objects'][oid]=obj
    write(a.out/'PROPOSED_STATE.json',state)
    bodies=[]
    for oid,obj in state['objects'].items():
        if obj['object_type']!='furniture':continue
        corners=np.array(list(itertools.product(*zip(obj['bbox_min'],obj['bbox_max']))));pts=trimesh.transform_points(corners,world_transform(obj))
        bodies.append(dict(id=oid,lo=pts.min(0).tolist(),hi=pts.max(0).tolist()))
    structures=[x for x in obstacles(initial,bundle/('room_'+rid),bundle,1.8) if x['id'].startswith('structure:')]
    overlaps=[]
    for l,r in itertools.chain(itertools.combinations(bodies,2),itertools.product(bodies,structures)):
        depth=np.minimum(l['hi'],r['hi'])-np.maximum(l['lo'],r['lo'])
        if np.all(depth>1e-7):overlaps.append(dict(objects=[l['id'],r['id']],extent_m=depth.tolist()))
    geom=state['room_geometry'];lower=np.array([-geom['length']/2,-geom['width']/2,0]);upper=np.array([geom['length']/2,geom['width']/2,geom['wall_height']])
    outside=[x['id'] for x in bodies if np.any(np.array(x['lo'])<lower-1e-7) or np.any(np.array(x['hi'])>upper+1e-7)]
    access=measure(state,bundle/('room_'+rid),bundle,p['access_settings']);write(a.out/'CONDITIONAL_ACCESS.json',access)
    rows=[dict(body_width_m=s['square_body_width_m'],clear_points=sum(t['target_cell_clear'] for t in s['targets'].values()),reachable_points=sum(t['reachable_from_entry'] for t in s['targets'].values()),planned_points=len(s['targets']),failed_points=[k for k,t in s['targets'].items() if not t['reachable_from_entry']]) for s in access['scenarios']]
    result=dict(status='candidate_checked',proposed_state_sha256=canonical_hash(state),furniture=len(bodies),furniture_ids=[x['id'] for x in bodies],envelope_overlaps=overlaps,outside=outside,access_counts=rows,all_immutable_records_preserved=all(state['objects'][k]==v for k,v in initial['objects'].items()),native_execution_gate_passed=not overlaps and not outside and rows[0]['reachable_points']==rows[0]['planned_points'],gate='Post-search declared minimum: no bounds/structure conflicts and every declared approach reachable with smallest frozen 0.4m square. All three widths reported; not human usability.',professional_labels=0,model_calls=0,training_updates=0,native_actions=0,whole_room_success='unknown',plan_sha256=sha256(a.plan))
    write(a.out/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
