"""Check a single prospectively fixed input, with no future-state evaluation."""
import argparse,copy,hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
from preflight_sourced_trajectory import inspect
from measure_room_access import measure

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();plan=json.loads(a.plan.read_text());path=Path(plan['source_state']['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['source_state']['sha256'];initial=json.loads(path.read_text());state=copy.deepcopy(initial)
 for name,c in plan['changes'].items():
  oid='recon_'+name+'_0';angle=math.radians(c['yaw']);state['objects'][oid]['transform']['translation'][:2]=[c['x'],c['y']];state['objects'][oid]['transform']['rotation_wxyz']=[math.cos(angle/2),0,0,math.sin(angle/2)]
 assert len(state['objects'])==plan['expected_objects']
 geometry=inspect(state);m=measure(state,Path(plan['source_room']),Path(plan['source_root']),plan['settings']);coverage=[{k:v['reachable_from_entry'] for k,v in s['targets'].items()} for s in m['scenarios']];passed=not geometry['outside_declared_wall_inner_faces'] and not geometry['positive_3d_AABB_pairs'] and all(all(row[k] for k in plan['preserved_targets']) for row in coverage)
 result={'status':'input_qualified' if passed else 'input_rejected','checked_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'geometry':geometry,'measurement':m,'coverage':coverage,'preserved_targets':plan['preserved_targets'],'candidate_count':1,'future_state_read':False,'model_calls':0,'native_tool_calls':0,'training_updates':0,'scope':'Input-only conservative geometric qualification, not professional/physical usability.'}
 with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':result['status'],'geometry':geometry,'coverage':coverage},ensure_ascii=False));raise SystemExit(0 if passed else 1)
if __name__=='__main__':main()
