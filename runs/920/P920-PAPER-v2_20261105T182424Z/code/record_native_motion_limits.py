"""Record why fixed-orientation preflight sweeps do not prove native motion."""
import argparse,json,math
from pathlib import Path
from native_project import sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--native',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();plan=json.loads(a.plan.read_text());rows=[]
    for i,move in enumerate(plan['moves'],7):
        files=[a.native/f'step_{i-1:02d}_after.json',a.native/f'step_{i:02d}_after.json'];states=[json.loads(p.read_text()) for p in files];poses=[d['objects'][move['object_id']]['transform'] for d in states];qs=[p['rotation_wxyz'] for p in poses];angle=2*math.degrees(math.acos(min(1.,abs(sum(x*y for x,y in zip(*qs))))))
        rows.append({'step':i,'object':move['object_id'],'state_files':[{'path':str(p),'sha256':sha256(p)} for p in files],'poses':poses,'yaw_rotation_difference_degrees':angle,'fixed_orientation_assumption_exact':qs[0]==qs[1],'native_motion_trajectory_recorded':False})
    result={'status':'actual_native_motion_unverified','mode_plan_sha256':sha256(a.plan),'rows':rows,'interpretation':'Native tools set poses; actual intermediate human/physical motion is not recorded. Small endpoint rotations violate the exact fixed-orientation preflight assumption. Preflight continuous-AABB results are prospective diagnostics only, not actual pushing or articulated-sweep acceptance.','all_actual_sweeps_verified':False,'model_calls':0,'training_updates':0}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'rotation_differences_degrees':[x['yaw_rotation_difference_degrees'] for x in rows]}))
if __name__=='__main__':main()
