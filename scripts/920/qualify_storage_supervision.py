"""Preserve weak-rule outcome but reject its vacuous functional interpretation."""
import argparse,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from shapely.geometry import box
from native_checkpoint import canonical_hash
from preflight_sourced_trajectory import boxes
from audit_native_review import export

def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[2];base=root.parent;pub=root/'runs/920'/a.run_id;runtime=base/'runtime'/a.run_id;replay=runtime/'replay';summary=read(pub/'SUMMARY.json');initial=read(replay/'initial_state.json');after=read(replay/'after_state.json');before_metrics=read(replay/'initial_measurement.json');after_metrics=read(replay/'after_measurement.json')
 assert summary['conditional_rule_passed'] and summary['living_free_gain_m2']>0
 for m in [before_metrics,after_metrics]:
  assert all(s['entry_cell_clear'] and not any(v['reachable_from_entry'] for v in s['targets'].values()) for s in m['fixed_access']['scenarios'])
  assert all(not any(v['reachable_from_entry'] for v in s['targets'].values()) for s in m['storage_front_approaches'])
 ids=['recon_chair_west_0','recon_table_0','recon_kitchen_counter_0'];env=boxes(initial);chain=[('recon_chair_west_0','recon_table_0'),('recon_table_0','recon_kitchen_counter_0')];barriers=[]
 for w in [.4,.5,.6]:
  pad=w/2+.01;hits=[]
  for a_id,b_id in chain:
   al,ah=env[a_id];bl,bh=env[b_id];x=box(al[0]-pad,al[1]-pad,ah[0]+pad,ah[1]+pad);y=box(bl[0]-pad,bl[1]-pad,bh[0]+pad,bh[1]+pad);hits.append({'objects':[a_id,b_id],'inflated_envelopes_intersect':x.intersects(y),'unexpanded_planar_distance_m':box(*al[:2],*ah[:2]).distance(box(*bl[:2],*bh[:2]))})
  assert all(x['inflated_envelopes_intersect'] for x in hits);barriers.append({'body_width_m':w,'connections':hits})
 old=pub/'pre_interpretation_export';old.mkdir()
 for name in ['OBJECTIVE_REVIEW.json','QA_EPISODE_CANDIDATE.json','EXPORT_CHECKS.json']:shutil.copy2(pub/name,old/name)
 qualification={'audited_at':datetime.now(timezone.utc).isoformat(),'run_id':pub.name,'classification':'local_floor_gain_with_preexisting_proxy_access_failure','frozen_rule_passed':True,'frozen_rule_limitation':'No-loss comparison is vacuous when all initial and final approach counts are zero. Keep original rule/summary, reject inference that activities are preserved in usable form.','source_rationale_still_supported':True,'actual_legal_actions':5,'source_guided_transfer_executed':True,'local_living_floor_gain_m2':summary['living_free_gain_m2'],'fixed_points_reachable_before':[0,0,0],'fixed_points_reachable_after':[0,0,0],'storage_front_points_reachable_before':[0,0,0],'storage_front_points_reachable_after':[0,0,0],'entry_cells_clear':[True,True,True],'entry_component_area_m2_before':[s['reachable_floor_center_area_m2'] for s in before_metrics['fixed_access']['scenarios']],'entry_component_area_m2_after':[s['reachable_floor_center_area_m2'] for s in after_metrics['fixed_access']['scenarios']],'unchanged_neck_evidence':barriers,'interpretation':'In the fixed conservative proxy, unchanged west dining chair/table/kitchen envelopes form part of the entrance bottleneck. This is an input-layout deficit, not proof the storage principle is bad, and not proof a real person cannot pass.','supervision':{'knowledge_usable':True,'source_action_correspondence_available':True,'objective_diagnostic_QA_available':True,'mechanical_replay_candidates':5,'positive_design_OTA_admitted':0,'independent_professional_preference':False,'training_authorized':False},'corrective_policy':'Do not retune this frozen trial. Future positive examples must verify meaningful baseline activity coverage and adequate final access, not only non-degradation. Input repair must be separately frozen and retain this record.','summary_sha256':sha(pub/'SUMMARY.json'),'old_export_sha256':{f.name:sha(f) for f in old.iterdir()},'scene_edits':0,'model_calls':0}
 save(pub/'SUPERVISION_QUALIFICATION.json',qualification)
 record=read(pub/'OBJECTIVE_REVIEW.json');record['judgment']['claims'][0]['assessment']=f'声明起居区空地增加{summary["living_free_gain_m2"]:.6f}平方米，但0.4/0.5/0.6米代理下前后固定活动点均为0/7，柜前点均为0/15可达。入口格本身为空，连通范围很小，不能用零到零的无损失宣称用途保留。'
 record['judgment']['claims'].append({'candidate_id':'after','object_ids':ids,'activity':'从入口进入用餐与其他活动区域','assessment':'未移动的西餐椅、餐桌、厨房台包络在三种代理宽度膨胀后连接，构成入口瓶颈的一部分。它属于既有重建初态缺陷；不能把局部空地收益当成房间可用，也不能据此否定源项目的设计原则。','evidence_ids':['qualification']});record['judgment']['evidence'].append({'evidence_id':'qualification','artifact':{'path':str((pub/'SUPERVISION_QUALIFICATION.json').relative_to(base)),'sha256':sha(pub/'SUPERVISION_QUALIFICATION.json')}});record['provenance']['qualification']='Objective diagnostic only. All baseline/final approaches fail; no positive design target admission.'
 for name,value in [('OBJECTIVE_REVIEW.json',record),('QA_EPISODE_CANDIDATE.json',export(record,base,['initial','after']))]:
  path=pub/name;path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 checks=read(pub/'EXPORT_CHECKS.json');checks.update(qualification_sha256=sha(pub/'SUPERVISION_QUALIFICATION.json'),positive_design_OTA_admitted=0,objective_qa_updated_with_zero_coverage=True,old_export_preserved=True,qualifier_code_sha256=sha(__file__));(pub/'EXPORT_CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n');shutil.copy2(__file__,pub/'code'/Path(__file__).name)
 print(json.dumps({'classification':qualification['classification'],'fixed_before_after':[[0,0,0],[0,0,0]],'positive_design_OTA_admitted':0,'five_actual_moves_preserved':True}))
if __name__=='__main__':main()
