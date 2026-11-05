"""Declare a distinct living/kitchen reconstruction for an attributed table decision."""
import argparse,copy,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from prepare_table_modes import furniture_mesh

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args();p=a.public.resolve();t=a.runtime.resolve();r=Path(__file__).resolve().parents[2];now=datetime.now(timezone.utc).isoformat()
    source=next(x for x in json.loads((r/'research/920/decision_packets_v1/SOURCED_DECISIONS.json').read_text())['records'] if x['decision_id']=='paper_table_mobility_for_overnight_guest')
    registry=json.loads((r/'research/920/data_readiness_v1/PROJECT_REGISTRY.json').read_text());assert next(x for x in registry['projects'] if x['family_id']==source['family_id'])['allocation']=='train'
    write(p/'SOURCE_RECORD.json',source)
    prior=r/'runs/920/P920-AIR-v1_20261105T042212Z';b=json.loads((prior/'BUILD_PLAN.json').read_text());ref=json.loads((prior/'INITIAL_REFERENCE.json').read_text())
    structures=[]
    def row(name,x,y,z,dims,facing,color):return {'name':name,'location':[x,z,-y],'dimensions':dims,'facing':facing,'color':color,'holes':[]}
    structures += [row('floor',0,0,0,[6.4,.04,4.4],0,'#C7BCA8'),row('ceiling',0,0,2.6,[6.4,.04,4.4],0,'#FFFFFF')]
    for name,x,y,w,facing in [('north_wall',0,2.2,6.4,180),('south_wall',0,-2.2,6.4,0),('east_wall',3.2,0,4.4,270),('west_wall',-3.2,0,4.4,90)]:structures.append(row(name,x,y,1.3,[w,2.6,.02],facing,'#E4E0D8'))
    structures += [row('entry_door',-3.2,-1.5,1.05,[.9,2.1,.02],90,'#B9A18B'),row('east_window',3.2,-1.45,1.5,[1.4,1.4,.01],270,'#A9D8E5')]
    reference={'_convention':['Native x/y floor; source x/up/south input.','6.4x4.4x2.6m living/kitchen research area; not the full45m²home or a surveyed plan.','West entrance/east window, fixtures, dimensions and furniture positions are researcher assumptions.'], 'room_structure':structures,'furniture':[]}
    layout=[('kitchen_counter',1.6,1.85,[2.4,.9,.6],180),('cooktop',1.0,1.85,[.5,.04,.5],180),('kitchen_sink',2.2,1.85,[.6,.05,.45],180),('sofa',-1.8,1.3,[1.802,.907,.958],0),('low_fridge',2.8,.85,[.6,.85,.6],270),('pantry',2.8,-.55,[.6,1.8,.55],270),('chair_left',-2.85,0,[.44,.45,.44],0),('chair_right',-.75,0,[.44,.45,.44],0),('mobile_table',-1.8,0,[1.2,.74,.7],0)]
    assets=t/'generated_assets';assets.mkdir();table=assets/'table.glb';stool=assets/'stool.glb';furniture_mesh(table,1.2,.7,.74,True);furniture_mesh(stool,.44,.44,.45,False)
    sofa=b['assets'][b['objects']['sofa']['asset_id']];build_assets={sofa['asset_id']:sofa}
    for key,path in [('paper_research_table',table),('paper_research_stool',stool)]:build_assets[key]={'asset_id':key,'name':key,'source_path':str(path),'sha256':sha(path),'source_to_native_rotation':np.eye(3).tolist(),'provenance':'Research-generated static mesh; no original furniture replication or simulated wheels.','generator_sha256':sha(Path(__file__).parent/'prepare_table_modes.py')}
    specs={}
    for name,x,y,dims,facing in layout:
        reference['furniture'].append(row(name,x,y,dims[1]/2,dims,facing,'#D7D0C2'))
        if name in ['kitchen_counter','cooktop','kitchen_sink','sofa','low_fridge']:spec=copy.deepcopy(b['objects'][name])
        else:spec={'kind':'procedural_proxy','object_type':'furniture','note':'Declared substitute, not historical furniture or physical validation.'}
        spec['immutable']=name in ['kitchen_counter','cooktop','kitchen_sink']
        if name=='mobile_table':spec.update(kind='mesh',asset_id='paper_research_table')
        if name.startswith('chair_'):spec.update(kind='mesh',asset_id='paper_research_stool')
        specs[name]=spec
    write(p/'INITIAL_REFERENCE.json',reference)
    unknowns=['Only the reported shared living/dining/kitchen activity area is represented; permanent bedroom, bathroom and corridor are outside this selected room. The full45m²home is not reconstructed.','The source supports a movable co-designed table and a convertible sofa. This static sofa has no conversion; a declared guest floor reserve is not evidence that anyone can sleep there.','All room/furniture dimensions, entrance/window positions and candidate movements are researcher assumptions. Source geometry/coordinates are unavailable.','Generated table wheels are visual meshes only; moving tools teleport poses. No physical pushing, swept articulation, braking or load claim.','Capacity, comfort, seating, kitchen services, privacy and professional quality remain unknown.']
    task={'design_request':'研究者重建任务：为两位住户布置共享餐厨起居区，满足日常做饭、用餐、起居和食品收纳；偶尔来客留宿时需要移开桌子，保留其他家具、入口、厨房、储物和窗前的接近，之后恢复日常用餐。房间6.4米×4.4米、位置和家具均为声明的替代条件。研究只检查家具与声明来客预留区的占用，静态沙发不具有已验证的展开或睡眠功能。卧室与卫浴在所选活动区域之外，不作为本房间生成目标。','original_request':None,'request_provenance':'adapted','unknowns':unknowns}
    build={'run_id':p.name,'frozen_at':now,'room_id':'paper_shared_living_reconstruction','room_description':'Declared shared living/dining/kitchen research room, not the original45m²apartment.','project':{'project_id':p.name,'family_id':source['family_id'],'provenance':'source_conditioned_research_reconstruction','geometry_template_family':'rectangular_6p4x4p4_west_entry_east_window_no_bath','independent_test':False},'initial_reference':{'path':str(p/'INITIAL_REFERENCE.json'),'sha256':sha(p/'INITIAL_REFERENCE.json')},'objects':specs,'assets':build_assets,'task':task,'code_sha256':sha(Path(__file__).parent/'build_reference_room.py')}
    write(p/'REFERENCE_BUILD_PLAN.json',build)
    mode={'run_id':p.name,'frozen_at':now,'source_decision_id':source['decision_id'],'source_reason':'同一餐厨起居区需要在日常用餐、待客和偶尔留宿之间转换时，可移动的桌子便于按活动腾出空间。','researcher_actions_are_not_source_reason':True,'guest_reserve_xy_m':[-2.6,-.55,-1.0,.75],'reserve_is_actual_bed':False,'movable_source_ids':['recon_sofa_0','recon_low_fridge_0','recon_pantry_0','recon_chair_left_0','recon_chair_right_0','recon_mobile_table_0'],'mode_checkpoints':{'step_06':'everyday_dining','step_08':'guest_reserve_clear','step_10':'everyday_dining_restored'},'moves':[{'object_id':'chair_right_0','x':1.75,'y':-1.55,'yaw':0},{'object_id':'mobile_table_0','x':.75,'y':-1.55,'yaw':0},{'object_id':'mobile_table_0','x':-1.8,'y':0,'yaw':0},{'object_id':'chair_right_0','x':-.75,'y':0,'yaw':0}], 'reference_measurement':{'grid_m':.02,'assumed_body_height_m':1.8,'square_body_widths_m':[.4,.5,.6],'entry_center_xy_m':[-3.2,-1.5],'entry_inward_normal_xy':[1,0],'approach_points_xy_m':{'kitchen_front':[1.6,1.0],'fridge_side':[2.15,.85],'pantry_side':[2.1,-.55],'living_east':[0,1.3],'window_front':[2.65,-1.5],'entrance_inside':[-2.65,-1.5]}},'budget':{'native_calls':10,'observations':11,'native_seconds':1200,'candidate_sequences':1,'model_calls':0,'training_updates':0,'first_error_stop':True},'unknowns':unknowns}
    write(p/'MODE_PLAN.json',mode)
    print(json.dumps({'family':source['family_id'],'reference_furniture':9,'construction_additions':6,'mode_moves':4,'training_updates':0}))
if __name__=='__main__':main()
