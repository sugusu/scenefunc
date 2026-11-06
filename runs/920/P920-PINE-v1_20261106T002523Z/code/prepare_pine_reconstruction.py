"""Freeze one source-conditioned, explicitly reconstructed Pine transfer episode."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone

def main():
    r=Path(__file__).resolve().parents[2];d=r.parent
    p=r/'runs/920/P920-PINE-v1_20261106T002523Z';t=d/'runtime'/p.name
    now=datetime.now(timezone.utc).isoformat()
    def h(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
    def ref(f):return dict(path=str(f),sha256=h(f))
    def write(name,v):
        with (p/name).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
    assert not (p/'RECONSTRUCTION_PROTOCOL.json').exists()
    measurement=json.loads((p/'ASSET_MEASUREMENT.json').read_text())
    assert measurement['assets'][0]['asset_id']=='03f226dd0e012925a8674564c4de19cb786c9a88'
    # Both previews were inspected by the executing analyst before this freeze.
    rotation=[[-1,0,0],[0,0,1],[0,1,0]]
    write('ASSET_REVIEW.json',dict(reviewed_at=now,preview_record=ref(t/'previews/PREVIEWS.json'),
        images_seen=['03f226dd0e012925a8674564c4de19cb786c9a88_front.png','03f226dd0e012925a8674564c4de19cb786c9a88_back.png'],
        finding='A single static desk with top, drawer/cabinet and knee opening. Metadata +Y side shows rear crossbar; opposite side shows drawer faces and knee opening.',
        frame_correction='Rotate metadata native frame 180 degrees about Z; +Y denotes inspected working side.',
        source_to_native_rotation=rotation,rescaled=False,accepted_scope='Static surface/pose substitute only; not source furniture or human ergonomic acceptance.',
        unknown=['Knee clearance','load','drawer articulation','chair comfort','ease of moving','original IKEA model']))
    task=dict(task_id='pine_transfer_study',researcher_authored_brief=
        '为两位住户布置一间书房兼访客房。两人需要同时做各自的桌面工作；保留旧桌与原椅实例，可以移动它们，允许增加家具。保留书架、沙发和从入口到它们的使用通路。访客睡眠和创作活动也是用途，但现有资产并未证明可展开客床或具体创作工序，须如实标未知。窗、门和房壳固定。',
        original_resident_request=None,provenance='Research transfer based on reported activities; not an original brief or historic initial room.',
        geometry_assumptions=dict(interior_rectangle=[-2.4,-2.,4.8,4.],height=2.7,
            origin='All room dimensions, apertures and initial positions are researcher assumptions. 55m2 source area is whole apartment, not this room.',
            window='North x=-0.7..1.1, sill0.9m; no operation/daylight simulation',door='South x=-0.1..0.9; no leaf sweep'),
        required_activities=['Two concurrent desktop workstations','Retain old desk/chair identity','Book and collection storage access','Sofa access','Guest sleeping and creative use: report unknown mechanisms'],
        uncertainties=['No surveyed source-room dimensions','No original IKEA or bespoke asset','No historical initial state or operation order','Guest sleeping mechanism and clearances','Storage capacity','Work equipment, knee/ergonomic comfort','Creative process/material requirements','Moving path/effort'])
    write('TASK_INPUT.json',task)
    initial=[dict(role=role,asset_name=asset,object_id=oid,x=x,y=y,yaw=yaw,immutable=imm) for role,asset,oid,x,y,yaw,imm in [
        ('retained_old_desk','work_desk','old_desk_0',0.,1.65,180,False),
        ('retained_old_chair','dining_chair_1','old_chair_0',0.,.8,0,False),
        ('retained_sofa','sofa','sofa_0',-1.85,0.,270,True),
        ('retained_bookshelf','bookshelf','bookshelf_0',-1.1,-1.75,0,True)]]
    build=dict(project=dict(project_id=p.name,family_id='sydney_pine_nicholas_gurney_2022',provenance='source_conditioned_research_transfer',independent_test=False),
        scenesmith=str(d/'resources/full_runtime_20261104/scenesmith'),
        inputs=dict(task=ref(p/'TASK_INPUT.json'),catalog_manifest=ref(d/'runtime/P920-WHOLE-v1_20261104T191014Z/build/bundle/manifest.json'),download=ref(t/'download/DOWNLOAD.json')),
        code_hashes={str(r/'scripts/920'/n):h(r/'scripts/920'/n) for n in ['build_pine_input.py','build_reference_room.py','native_project.py']},
        source_to_native_rotation=rotation,initial_furniture=initial,budget=dict(bundle_max_bytes=120000000))
    write('BUILD_PLAN.json',build)
    commands=[dict(role=role,planned_id=oid,command=dict(name=tool,arguments=args)) for role,oid,tool,args in [
        ('retain_and_reposition_source_reported_old_desk','old_desk_0','move_furniture_tool',dict(object_id='old_desk_0',x=2.09,y=.25,yaw=90)),
        ('retain_and_reposition_research_initial_chair','old_chair_0','move_furniture_tool',dict(object_id='old_chair_0',x=1.18,y=.25,yaw=270)),
        ('provide_window_workstation_substitute','work_desk_0','add_furniture_to_scene_tool',dict(asset_id='catalog_03f226dd0e012925a8674564c4de19cb786c9a88',x=.3,y=1.65,yaw=180)),
        ('provide_second_seat','dining_chair_1_0','add_furniture_to_scene_tool',dict(asset_id='catalog_6dba8ae88c9979be6b022dbf4d80cd88dd8e5b02',x=.3,y=.8,yaw=0))]]
    points=dict(sofa_front=[-.9,0.],bookshelf_front=[-1.1,-1.05],central_activity=[-.4,.25])
    access=dict(grid_m=.02,assumed_body_height_m=1.8,square_body_widths_m=[.4,.5,.6],entry_center_xy_m=[.4,-2.],entry_inward_normal_xy=[0,1])
    write('RECONSTRUCTION_PROTOCOL.json',dict(frozen_at=now,source_record=ref(p/'SOURCE_RECORD.json'),task=ref(p/'TASK_INPUT.json'),
        source_boundary='Gurney designed window desk/bookshelves; residents reused an existing IKEA desk along an adjacent wall for concurrent work. Co-produced reported outcome, not all professional-authored operations.',
        reconstruction_boundary='One declared 19.2m2 transfer room. Two unscaled substitutes for distinct original desk types. Adjacent-wall arrangement has a corner gap; not a continuous bespoke L desktop. One shelf is a book-access proxy, not surrounding-window cabinetry. Sofa does not prove guest sleeping.',
        commands=commands,reference_measurement=access,fixed_approach_points=points,
        initial_role_points=dict(old_seat_side=[-.65,.8]),final_role_points=dict(old_seat_side=[1.18,-.4],new_seat_side=[-.35,.8]),
        work_pairs_before=[['old_desk_0','old_chair_0']],work_pairs_after=[['old_desk_0','old_chair_0'],['work_desk_0','dining_chair_1_0']],
        surface_probe=dict(width_m=.4,depth_m=.3,near_top_band_m=.03,min_up_normal=.99,not_normative=True),
        simultaneous_seat_proxies=dict(square_widths_m=[.4,.5,.6],centred_on_chairs=True,not_human_models=True),
        interpretation='Measure desk-surface patches, chair facing/separation, actual static mesh contacts, retained IDs/bindings and common access. No claim that one table cannot serve two people, that this layout is unique/optimal, or that all room uses pass.',
        sensitivity='Three declared square-body widths only; not source-dimension validation. No tuned retries or empirical generalization from one assumed room.',
        budget=dict(tool_calls=4,observations=5,native_seconds=600,build_seconds=120,preflight_seconds=120,analysis_seconds=120,model_calls=0,training_updates=0,candidate_sequences=1,retries=0,first_infrastructure_error_stop=True)))
    status=json.loads((p/'STATUS.json').read_text());status.update(status='reconstruction_frozen',updated_at=now,next_step='Build, preflight, then at most four native edits with five observations.',configuration_hashes={**status['configuration_hashes'],'build':h(p/'BUILD_PLAN.json'),'protocol':h(p/'RECONSTRUCTION_PROTOCOL.json')})
    status['completed_planned']['mesh_downloads']=[1,1];status['completed_planned']['native_edits']=[0,4]
    (p/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    (r/'runs/920/latest.json').write_text(json.dumps(dict(run_id=p.name,stage='pine_reconstruction_frozen',updated_at=now,run_directory=str(t)),indent=2)+'\n')
    print(json.dumps(dict(run_id=p.name,protocol_sha256=h(p/'RECONSTRUCTION_PROTOCOL.json'),native_edits=4,training_updates=0)))

if __name__=='__main__':main()
