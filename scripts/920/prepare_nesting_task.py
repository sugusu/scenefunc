"""Prepare one held-out-family transfer of a sourced nesting-table decision.

All geometry and requests below are researcher-authored. This does not recover
the original 16 m² apartment, platform, bed mechanism, or original furniture.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    a = ap.parse_args()
    r = Path(__file__).resolve().parents[2]
    p, t = a.run.resolve(), a.runtime.resolve()
    now = datetime.now(timezone.utc).isoformat()
    source = json.loads((p/'SOURCE_RECORD.json').read_text())
    registry = json.loads((r/'research/920/data_readiness_v1/PROJECT_REGISTRY.json').read_text())
    allocation = next(x for x in registry['projects'] if x['family_id'] == source['family_id'])
    assert not allocation['training_eligible'] and not allocation['retrieval_eligible_for_other_projects']

    def write(name, obj):
        with (p/name).open('x') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')

    def record(name, x, y, z, w, h, d, facing=0, color='#CCC5B8'):
        return dict(name=name, location=[x,z,-y], dimensions=[w,h,d], facing=facing, color=color)

    structure = [record('floor',0,0,-.02,4,.04,4),record('ceiling',0,0,2.6,4,.04,4)]
    for name,x,y,w,facing in [('north_wall',0,2,4,180),('south_wall',0,-2,4,0),
                             ('west_wall',-2,0,4,90),('east_wall',2,0,4,270),
                             ('bath_south_wall',-1.25,.7,1.5,0),('bath_east_wall',-.5,1.35,1.3,270)]:
        structure.append(record(name,x,y,1.3,w,2.6,.02,facing,'#ECE7DF'))
    structure += [record('entry_door',.35,-2,1.05,.9,2.1,.02),
                  record('bath_door',-.5,1.05,1.05,.6,2.1,.02,270),
                  record('south_window',-1.15,-2,1.55,1.2,.9,.02)]
    furniture, specs = [], {}

    def obj(name,x,y,w,d,h,base=0,movable=False,color='#C6BBA5'):
        furniture.append(record(name,x,y,base+h/2,w,h,d,color=color))
        specs[name] = dict(kind='procedural_proxy',object_type='furniture',base_z=base,
                           immutable=not movable, note='Generic researcher box proxy; dimensions and pose assumed, not original furniture.')

    # Separate fixed tabletop and side supports preserve the genuine empty
    # cavity. A single solid desk box would invalidate the nesting decision.
    obj('desktop',-1.64,0,.62,1.2,.04,base=.72)
    obj('desk_north_support',-1.64,.575,.62,.05,.72)
    obj('desk_south_support',-1.64,-.575,.62,.05,.72)
    obj('mobile_table',-.9,-.85,.54,.40,.40,movable=True,color='#E1B34B')
    obj('work_stool',-1.65,-.85,.40,.40,.44,movable=True,color='#739BA3')
    obj('sofa',-.95,-1.6,1.6,.70,.70)
    obj('bed',1.30,-.65,1.2,2.0,.48)
    obj('kitchen_counter',1.25,1.65,1.3,.60,.90)
    obj('wardrobe',.0,1.72,.55,.46,2.0)
    obj('bath_shower',-1.55,1.58,.70,.70,.10)
    obj('bath_toilet',-1.1,1.00,.40,.50,.43)
    obj('bath_vanity',-.78,1.70,.40,.40,.80)
    unknowns = ['All dimensions, initial poses, apertures and body thresholds are research assumptions.',
        'Fixed bed and separate sofa replace original retractable bed; platform kitchen and rail mechanism are not reproduced.',
        'Mobile table and stool are boxes: no real casters, pushing force, comfort or load validation.',
        'Bathroom/kitchen/storage are static proxies: capacity, plumbing, operation and ventilation unknown.',
        'Source read by task designer; family excluded from training/retrieval, not sealed or pretraining-unseen.',
        'Only geometric activity proxies may pass. Independent professional or whole-room human-use acceptance is absent.']
    common = '房壳、门窗及固定家具不能动。保留所有对象实例，仅可移动小桌和凳子。保留床侧、厨房、衣柜和卫浴的接近条件；未知功能如实报告。'
    tasks = {
        'clear_floor': {'request':'现在暂不使用书桌，需要在标明的中央地面区域进行活动，请清空该区域。'+common,
                       'reserved_floor_xyxy_m':[-1.25,-1.10,.25,.50], 'desk_active':False},
        'desk_work': {'request':'现在需要坐在原书桌前学习，让凳子、桌面和桌下腿部空间可用。无需保留中央地面活动区。'+common,
                      'desk_active':True, 'desk_top_work_patch_xyxy_m':[-1.90,-.25,-1.40,.25],
                      'knee_volume_lo_hi_m':[[-1.85,-.25,.05],[-1.15,.25,.68]],
                      'seat_target_region_xyxy_m':[-1.00,-.10,-.80,.10]}}
    write('TASK_INPUTS.json',dict(created_at=now,family_id=source['family_id'],tasks=tasks,
        original_resident_request=None,request_provenance='researcher transfer, not original project brief',unknowns=unknowns,
        scoring_rule='Use declared activity geometry and full-room side effects; never score distance to reference actions or poses.',
        distinct_projects=1,conditions=2,independent_professional_labels=0))
    reference = dict(room_structure=structure,furniture=furniture,
        _convention='Explicit generic 4×4×2.6m rectangular transfer. x/up/south coordinates. No original plan traced.')
    write('INITIAL_REFERENCE.json',reference)
    build = dict(created_at=now,room_id='nesting_table_transfer',project=dict(project_id=p.name,
        family_id=source['family_id'],provenance='source_conditioned_research_transfer_not_original_apartment',sealed_test=False),
        room_description='Generic compact studio; only the nesting-table strategy is sourced.',
        initial_reference=dict(path=str(p/'INITIAL_REFERENCE.json'),sha256=digest(p/'INITIAL_REFERENCE.json')),
        assets={},objects=specs,task=dict(design_request=tasks['clear_floor']['request'],original_request=None,
        request_provenance='adapted',unknowns=unknowns))
    write('BUILD_PLAN.json',build)
    def move(oid,x,y):
        return dict(name='move_furniture_tool',arguments=dict(object_id='recon_'+oid+'_0',x=x,y=y,yaw=0))
    # No selection based on model results. These are withheld feasibility
    # references, never supplied as policy inputs or assumed to be optimal.
    references = {
        'clear_floor':[move('mobile_table',-.9,0),move('mobile_table',-1.64,0)],
        'desk_work':[move('work_stool',-.9,-.30),move('work_stool',-.9,0)]}
    write('FEASIBILITY_PLAN.json',dict(created_at=now,references=references,historical=False,model_generated=False,
        rules='One fixed candidate per condition; retain rejection, no automatic geometry search or reference tuning.',
        access_settings=dict(grid_m=.02,assumed_body_height_m=1.8,square_body_widths_m=[.4,.5,.6],
        entry_center_xy_m=[.35,-2],entry_inward_normal_xy=[0,1],
        approach_points_xy_m=dict(bed_side=[.35,-.5],kitchen_front=[1.25,1.0],wardrobe_front=[0,1.15],bath_entry=[-.25,1.05]))))
    py = str(r.parent/'envs/scene_designer311_native/bin/python')
    command = [py,str(r/'scripts/920/build_reference_room.py'),'--plan',str(p/'BUILD_PLAN.json'),
               '--scenesmith',str(r.parent/'resources/full_runtime_20261104/scenesmith'),'--out',str(t/'build')]
    write('EXECUTION_PLAN.json',dict(frozen_at=now,build_command=command,
        budget=dict(native_builds=1,build_seconds=240,feasibility_candidates=2,model_calls=0,training_updates=0),
        hashes={str(f):digest(f) for f in [p/'BUILD_PLAN.json',p/'INITIAL_REFERENCE.json',p/'TASK_INPUTS.json',p/'FEASIBILITY_PLAN.json',
            r/'scripts/920/build_reference_room.py',Path(__file__)]},
        native_edit_permission='Not started by this plan; feasibility must be measured before reference replay.',
        stop_rule='No retry or new layout candidate after a geometric rejection; report exact task limitations.'))
    print(json.dumps(dict(prepared=True,furniture=len(furniture),conditions=2,model_calls=0,training_updates=0)))


if __name__ == '__main__':
    main()
