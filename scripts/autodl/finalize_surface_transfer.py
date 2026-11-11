"""CPU-only recomputation/packaging of a native tool regression, not an eval set."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

R=Path(__file__).resolve().parents[2]


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def move_function(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManipulandTools')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_move_manipuland_impl')
    fn.decorator_list=[]
    return fn


def matrix_near(a,b):return all(abs(x-y)<=1e-12 for ra,rb in zip(a,b) for x,y in zip(ra,rb))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    root=ap.parse_args().root.resolve();start=time.monotonic()
    result=read(root/'corrected_execution/results.json');assert result['status']=='completed'
    assert result['native_move_invocations']==6 and result['model_calls']==0
    assert read(root/'failed_attempt_v1.json')['actual_process_exit_code']==1
    assert read(root/'execution_state.json')['corrected_attempt_exit_code']==0
    first_attempt=read(root/'results.json')
    assert first_attempt['status']=='running'  # Preserved stale partial output, not a live job.
    assert len(first_attempt['rows'])==2
    source=R/'runs/autodl/20261110_live_manager_request/source/scenesmith/manipuland_agents/tools/manipuland_tools.py'
    original=source.read_text();head=read(root/'upstream_head_check.json');assert sha(source)==head['sha256']
    fn=move_function(original)
    target=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=='position_unchanged and rotation_unchanged')
    target.test=ast.BoolOp(op=ast.And(),values=[target.test,ast.parse('str(scene_obj.placement_info.parent_surface_id) == surface_id',mode='eval').body])
    proposed=original.replace('if position_unchanged and rotation_unchanged:',
      'if (\n                (position_unchanged and rotation_unchanged)\n                and str(scene_obj.placement_info.parent_surface_id) == surface_id\n            ):')
    assert ast.dump(fn,include_attributes=False)==ast.dump(move_function(proposed),include_attributes=False)
    assert len([n for n in ast.walk(fn) if isinstance(n,ast.If) and 'parent_surface_id' in ast.unparse(n.test)])==1
    rows=[]
    for item in result['rows']:
        req=item['request'];z={'lower_shelf':.4,'upper_shelf':.9}[req['surface_id']]
        angle=math.radians(req['rotation_degrees']);c=math.cos(angle);s=math.sin(angle)
        expected=[[c,-s,0,req['position_x']],[s,c,0,req['position_z']],[0,0,1,z],[0,0,0,1]]
        assert matrix_near(expected,item['expected_pose'])
        post=(item['final_parent']==req['surface_id'] and matrix_near(expected,item['final_pose']))
        assert post==item['binding_and_world_pose_postcondition']
        response=json.loads(item['raw_response'])
        assert response['success']==item['tool_reported_success']
        assert response['object_id']=='project_piece_0'
        if response['success']:
            p=response['world_position'];assert all(abs(p[k]-expected[j][3])<1e-12 for j,k in enumerate(['x','y','z']))
            assert response['parent_surface_id']==item['final_parent']
        dx=[item['final_pose'][j][3]-item['initial_pose'][j][3] for j in range(3)]
        distance=math.sqrt(sum(x*x for x in dx));assert abs(distance-item['world_displacement_m'])<1e-12
        assert item['object_inventory_preserved'] and item['object_and_geometry_identity_preserved']
        rows.append(dict(arm=item['arm'],case=item['case'],tool_success=response['success'],
                         error_type=response.get('error_type'),target_surface=req['surface_id'],
                         final_surface=item['final_parent'],postcondition=post,world_displacement_m=distance,
                         geometry_and_identity='Asserted during native execution; synthetic mesh'))
    assert len(rows)==6
    defects=[r for r in rows if not r['postcondition']]
    assert len(defects)==1 and defects[0]['arm']=='original' and defects[0]['case']=='other_frame_same_local'
    with (root/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');wr.writeheader();wr.writerows(rows)
    for row in read(root/'corrected_execution/source_before_after.json')['files']:
        assert sha(row['path'])==row['before']==row['after']
    protected=read(R/'runs/autodl/20261111_feedback_information_control/source_before_after.json')
    assert len(protected['original_inputs'])==579
    for row in protected['original_inputs']:assert sha(row['path'])==row['before'],row['path']
    dump(root/'independent_recompute.json',dict(status='verified',rows=rows,protected_sources_unchanged=579,
        unmodified_source_matches_checked_upstream_file=True,integration_patch_matches_executed_AST_guard=True,
        native_replay_kernel_seconds=result['wall_seconds'],
        corrected_native_method_invocations=6,initial_adapter_failed_method_invocations=2,
        total_recorded_method_invocations=8,
        excluded_costs='First failed adapter attempt and reading/fetching not measured; no efficiency comparison',
        initial_adapter_failure_not_a_native_semantic_result=True,model_calls=0,training_updates=0,
        new_room_generations=0,source_task_occurrence_unknown=True,
        replay_executed_by_this_finalizer=False,finalizer_seconds=time.monotonic()-start))
    required=['reading_ledger.json','algorithm_and_training.txt','stage_report.html','reproduce.txt',
              'executed_driver_v2.py','failed_driver_v1.py','candidate_integration.diff']
    for name in required:assert (root/name).is_file(),name
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name not in
                 {'evidence.zip','delivery_manifest.json','delivery_verification.json'})
    files += [R/'scripts/autodl/probe_native_surface_transfer.py',Path(__file__).resolve()]
    manifest=[dict(path=str(p.relative_to(R)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    dump(root/'delivery_manifest.json',dict(files=manifest,first_failed_attempt_kept=True,
         original_sources_or_licensed_assets_modified=False))
    with zipfile.ZipFile(root/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[root/'delivery_manifest.json']:z.write(p,str(p.relative_to(R)))
    with zipfile.ZipFile(root/'evidence.zip') as z:
        assert z.testzip() is None
        for row in manifest:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    dump(root/'delivery_verification.json',dict(status='verified',files=len(manifest),
         zip_sha256=sha(root/'evidence.zip'),zip_crc_and_hashes_checked=True,
         old_sources_unchanged=579,model_calls=0,training_updates=0,
         scope='Native synthetic regression with ordinary in-memory fix; no model or full scene-generation benefit claimed'))
    print(json.dumps(dict(status='verified',corrected_native_invocations=6,total_recorded_invocations=8,defect_cases=1,
                         ordinary_patch_resolves_bounded_defect=True,model_calls=0)))


if __name__=='__main__':main()
