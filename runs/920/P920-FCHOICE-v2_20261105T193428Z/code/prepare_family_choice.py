"""Freeze an attributed shared-living transfer task and two researcher candidates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_table_modes import furniture_mesh


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--public', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    a = ap.parse_args()
    p, t = a.public.resolve(), a.runtime.resolve()
    r = Path(__file__).resolve().parents[2]
    now = datetime.now(timezone.utc).isoformat()
    sources = json.loads((r / 'research/920/decision_packets_v1/SOURCED_DECISIONS.json').read_text())
    source = next(x for x in sources['records'] if x['decision_id'] == 'fhouse_existing_furniture_shared_living')
    registry = json.loads((r / 'research/920/data_readiness_v1/PROJECT_REGISTRY.json').read_text())
    assert next(x for x in registry['projects'] if x['family_id'] == source['family_id'])['allocation'] == 'train'
    write(p / 'SOURCE_RECORD.json', source)
    write(p / 'SOURCE_RECHECK.json', dict(checked_at=now,
        url='https://www.nevertoosmall.com/post/f-house-osaka', author='Joel Beath', published='2022-10-20',
        architect='Kazuteru Matsumura / Coilkma', locator='Opening project description, line56 of web extraction',
        supported_fact='Architect used a relatively open plan to accommodate existing furniture and prioritise common living space.',
        reported_home_area_m2=57, reported_storeys=3, reported_residents=4,
        original_dimensions_or_furniture_inventory_available=False,
        reconstruction_reviewed_by_source_designer=False, raw_page_cached=False, raw_page_sha256=None,
        reuse='Short attributed factual paraphrase only; no source photographs, plans or article copied.'))
    def row(name, x, y, z, dims, facing, color):
        return dict(name=name, location=[x,z,-y], dimensions=dims, facing=facing, color=color, holes=[])
    structure = [row('floor',0,0,0,[7,.04,4.8],0,'#C7BCA8'),
                 row('ceiling',0,0,2.7,[7,.04,4.8],0,'#FFFFFF')]
    for name,x,y,width,facing in [('north_wall',0,2.4,7,180),('south_wall',0,-2.4,7,0),
                                 ('west_wall',-3.5,0,4.8,90),('east_wall',3.5,0,4.8,270)]:
        structure.append(row(name,x,y,1.35,[width,2.7,.02],facing,'#E4E0D8'))
    structure += [row('entry_door',-3.5,-1.6,1.05,[1,2.1,.02],90,'#B9A18B'),
                  row('east_window',3.5,0,1.5,[1.6,1.4,.01],270,'#A9D8E5')]
    assets = t / 'generated_assets'; assets.mkdir()
    table, stool = assets / 'table.glb', assets / 'stool.glb'
    furniture_mesh(table,1.6,.85,.74,False); furniture_mesh(stool,.44,.44,.45,False)
    old = json.loads((r/'runs/920/P920-AIR-v1_20261105T042212Z/BUILD_PLAN.json').read_text())
    sofa = old['assets'][old['objects']['sofa']['asset_id']]
    asset_specs = {sofa['asset_id']:sofa}
    for key,path in [('family_research_table',table),('family_research_stool',stool)]:
        asset_specs[key] = dict(asset_id=key,name=key,source_path=str(path),sha256=sha(path),
            source_to_native_rotation=np.eye(3).tolist(),
            provenance='Research-generated substitute; not an original F-House item.',
            generator_sha256=sha(Path(__file__).parent/'prepare_table_modes.py'))
    layout = [('sofa',2.5,0,[1.802,.907,.958],90),('dining_table',-1.6,0,[1.6,.74,.85],0),
              ('seat_nw',-2.05,.8,[.44,.45,.44],0),('seat_ne',-1.15,.8,[.44,.45,.44],0),
              ('seat_sw',-2.05,-.8,[.44,.45,.44],0),('seat_se',-1.15,-.8,[.44,.45,.44],0),
              ('sideboard',2.6,-1.9,[1.4,.7,.4],0),('tall_storage',1.3,1.95,[1.8,1.8,.35],0)]
    ref = dict(_convention=['7x4.8x2.7m research shared living/dining room, not surveyed F-House.',
        'Every item and initial pose is a research assumption; object retention applies to these declared instances only.'],
        room_structure=structure, furniture=[])
    specs = {}
    for name,x,y,dims,facing in layout:
        ref['furniture'].append(row(name,x,y,dims[1]/2,dims,facing,'#D7D0C2'))
        spec = dict(kind='procedural_proxy',object_type='furniture',immutable=False,
                    note='Declared existing research item; not an original historical instance.')
        if name == 'sofa': spec.update(kind='mesh',asset_id=sofa['asset_id'])
        if name == 'dining_table': spec.update(kind='mesh',asset_id='family_research_table')
        if name.startswith('seat_'): spec.update(kind='mesh',asset_id='family_research_stool')
        specs[name] = spec
    write(p/'INITIAL_REFERENCE.json',ref)
    unknowns = ['Only a shared living/dining room is represented, not the original three-storey57m²home; cooking, sleeping and bathroom spaces are outside this room.',
        'All dimensions, existing items, openings, observer points and task variants are researcher assumptions.',
        'Tall storage is a solid opaque proxy; actual doors, shelves, capacity and source furniture are unknown.',
        'Sight rays are geometric indicators, not perceived openness, privacy, acoustic isolation or professional preference.',
        'Square-body approach checks do not establish real seating, reach, comfort or moving feasibility.']
    task = dict(original_request=None,request_provenance='adapted',
        design_request='研究者迁移任务：四人家庭的共同起居和用餐房间需保留已有八件家具的身份与尺寸，支持四个餐位、沙发起居和收纳取放，保持入口与窗前通行。家具位置可调整，具体活动需要由单独的条件卡给出。房间、已有家具和尺寸均为研究假设；本任务不指定采用哪种分隔或摆法。',unknowns=unknowns)
    build = dict(run_id=p.name,frozen_at=now,room_id='fhouse_shared_living_transfer',
        room_description='Declared shared living/dining transfer room with retained furniture.',
        project=dict(project_id=p.name,family_id=source['family_id'],independent_test=False,
            provenance='source_principle_transfer_reconstruction',geometry_template_family='rectangular_7x4p8_retained_furniture'),
        initial_reference=dict(path=str(p/'INITIAL_REFERENCE.json'),sha256=sha(p/'INITIAL_REFERENCE.json')),
        objects=specs,assets=asset_specs,task=task,code_sha256=sha(Path(__file__).parent/'build_reference_room.py'))
    write(p/'REFERENCE_BUILD_PLAN.json',build)
    variants = [dict(id='shared_visual_contact',
        request='家人在餐桌处理日常事务时，希望仍能看见沙发上的家人，便于共同交流；保留所有家具以及用餐、起居、收纳和通行。',
        provenance='Researcher operationalisation of the sourced shared-living preference, not a resident quotation.',
        indicator='More unobstructed declared dining-to-sofa sight rays is preferred, conditional on retained access.'),
        dict(id='reduced_visual_distraction',
        request='一人在餐桌处理事务，另一人在沙发活动时，希望两处之间减少直接视线干扰；保留所有家具以及用餐、起居、收纳和通行。不要求隔音。',
        provenance='Researcher-constructed contrast; not an F-House resident requirement or historical rejected alternative.',
        indicator='Fewer unobstructed declared dining-to-sofa sight rays is preferred, conditional on retained access.')]
    write(p/'TASK_VARIANTS.json',dict(shared_task=task,variants=variants,
        no_prescribed_action_in_requests=True,professional_preference_labels=False,
        actual_human_activities_validated=False))
    choice = dict(run_id=p.name,frozen_at=now,source_decision_id=source['decision_id'],
        family_id=source['family_id'],source_record_sha256=sha(p/'SOURCE_RECORD.json'),
        task_variants_sha256=sha(p/'TASK_VARIANTS.json'),
        source_reason='为容纳既有家具并优先共同生活，F-House采用了较开放的空间组织。',
        candidates={'perimeter':dict(x=-1.6,y=1.95,yaw=0),'divider':dict(x=.4,y=0,yaw=90)},
        candidates_origin='Researcher-generated alternatives; neither is a historical architect trajectory.',
        measurement=dict(grid_m=.02,assumed_body_height_m=1.8,square_body_widths_m=[.4,.5,.6],
            entry_center_xy_m=[-3.5,-1.6],entry_inward_normal_xy=[1,0],
            approach_points_xy_m={'entry_inside':[-2.95,-1.6],'seat_nw_back':[-2.05,1.4],
                'seat_ne_back':[-1.15,1.4],'seat_sw_back':[-2.05,-1.4],'seat_se_back':[-1.15,-1.4],
                'sofa_front':[1.45,0],'sideboard_front':[2.6,-1.3],'window_front':[3.05,1.3]}),
        storage_approach_by_candidate={'perimeter':[-1.6,1.4],'divider':[-.25,0]},
        sight=dict(observer_xy=[-1.15,.8],target_xy=[[2.5,y] for y in [-.6,-.3,0,.3,.6]],
            heights_m=[1.05,1.15,1.25],definition='Straight segments through opaque furniture AABBs at equal declared seated eye heights; no perceptual or acoustic claim.'),
        budget=dict(native_candidates=2,tool_calls_per_candidate=1,observations_per_candidate=2,
            native_seconds_per_candidate=360,model_calls=0,training_updates=0,
            retries=0,first_infrastructure_error_stop=True),unknowns=unknowns)
    write(p/'CHOICE_PLAN.json',choice)
    print(json.dumps(dict(family=source['family_id'],furniture=8,candidates=2,task_conditions=2,training_updates=0)))


if __name__ == '__main__':
    main()
