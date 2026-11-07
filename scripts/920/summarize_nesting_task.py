"""Archive actual two-condition reference evidence; no learned-policy claim."""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from native_project import validate_bundle
from preflight_sourced_trajectory import boxes, inspect


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--runtime',type=Path,required=True)
    ap.add_argument('--clear-floor',type=Path,required=True)
    ap.add_argument('--desk-work',type=Path,required=True)
    a=ap.parse_args();p=a.run.resolve();t=a.runtime.resolve()
    def read(f):return json.loads(f.read_text())
    def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
    task=read(p/'input_v2/TASK_INPUTS.json')
    public=p/'evidence';public.mkdir(exist_ok=True)
    images=p/'views';images.mkdir(exist_ok=True)
    rows=[];records={}
    for condition,directory in [('clear_floor',a.clear_floor),('desk_work',a.desk_work)]:
        result=read(directory/'RESULT.json')
        assert result['status']=='native_sequence_completed',condition
        assert all(c['passed'] for c in result['checks'])
        steps=read(directory/'STEPS.json')
        checkpoint=directory/('checkpoint_'+str(len(steps)))
        manifest,_,_,_=validate_bundle(checkpoint)
        for label in ['initial','after']:
            state=read(directory/(label+'_state.json'))
            measurement=read(directory/(label+'_measurement.json'))
            geometry=inspect(state)
            assert not geometry['outside_declared_wall_inner_faces'] and not geometry['positive_3d_AABB_pairs']
            knee=task['tasks']['desk_work']['knee_volume_lo_hi_m']
            hits=[oid for oid,(lo,hi) in boxes(state).items() if np.all(np.minimum(hi,knee[1])-np.maximum(lo,knee[0])>1e-6)]
            seat=state['objects']['recon_work_stool_0']['transform']['translation']
            region=task['tasks']['desk_work']['seat_target_region_xyxy_m']
            seat_ok=region[0]<=seat[0]<=region[2] and region[1]<=seat[1]<=region[3]
            row=dict(condition=condition,state=label,clear_floor=measurement['reserved_floor']['clear_under_declared_envelope'],
                desk_knee_clear=not hits,desk_seat_ready=seat_ok,desk_activity_proxy=not hits and seat_ok,
                occupied_activity_floor_m2=measurement['reserved_floor']['occupied_AABB_m2'])
            for scenario in measurement['scenarios']:
                row['failed_access_'+str(scenario['square_body_width_m'])]=','.join(k for k,v in scenario['targets'].items() if not v['reachable_from_entry'])
            rows.append(row)
            # Archive the actual bound image list, never infer ordering names.
            obs=read(directory/(label+'_observation.json'))
            view_records=[]
            for index,filename in enumerate(obs['image_paths']):
                path=Path(filename);dest=images/(condition+'_'+label+'_'+str(index)+path.suffix)
                shutil.copyfile(path,dest)
                view_records.append(dict(source=str(path),public_path=str(dest),sha256=sha(dest)))
            records[condition+'_'+label]=dict(actual_state_sha256=sha(directory/(label+'_state.json')),
                measurement=measurement,geometry=geometry,knee_colliders=hits,views=view_records)
        proposed=read(t/'feasibility_v2'/(condition+'_proposed_state.json'))
        actual=read(directory/'after_state.json')
        assert set(actual['objects'])==set(proposed['objects'])
        deviations={}
        for oid,x in actual['objects'].items():
            y=proposed['objects'][oid]
            for key in ('bbox_min','bbox_max'):
                assert np.allclose(x[key],y[key],atol=1e-8,rtol=0), (oid,key)
            # Native placement can perturb requested transforms. Report actual
            # deviations; activity acceptance above uses the saved actual state.
            deviations[oid]={key:float(np.max(np.abs(np.asarray(x['transform'][key])-np.asarray(y['transform'][key]))))
                for key in ('translation','rotation_wxyz')}
            assert x['object_type']==y['object_type'] and x['immutable']==y['immutable']
        for filename in ['RESULT.json','STEPS.json','initial_state.json','after_state.json','initial_measurement.json','after_measurement.json','HTTP_REQUESTS.json']:
            shutil.copyfile(directory/filename,public/(condition+'_'+filename))
        records[condition]=dict(native_tool_calls=result['tool_calls'],checkpoint=str(checkpoint),
            manifest_sha256=sha(checkpoint/'manifest.json'),planned_and_actual_inventory_and_local_bounds_match=True,
            actual_vs_requested_transform_max_component_difference=deviations)
    with (p/'RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    summary=dict(recorded_at=datetime.now(timezone.utc).isoformat(),status='two_native_references_completed',
        family_id=task['family_id'],source_families=1,conditions=2,native_tool_calls=sum(records[c]['native_tool_calls'] for c in ('clear_floor','desk_work')),
        observations=4,view_images=20,model_calls=0,training_updates=0,records=records,rows=rows,
        scientific_conclusion='Source-grounded conditional task has two actually executed reference outcomes under declared geometry. Stowing the table clears floor but obstructs desk knees; working configuration reverses the tradeoff. This is reference feasibility, not learned design or generalization.',
        limits=task['unknowns']+['0.6m conservative body proxy fails bath-entry approach in every state; do not average it away.',
            'Raw replay reserved_floor.scope retains a legacy bed-space label. Here its numeric bounds describe the central activity rectangle, not sleeping capacity. Raw evidence is preserved unchanged.',
            'V1 rejected input and motion were repaired before native execution; evaluation task is developer-exposed, not sealed.',
            'There is no model comparison on this task. Only one project family, with two conditions.'])
    (p/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],native_tool_calls=summary['native_tool_calls'],rows=rows)))


if __name__=='__main__':main()
