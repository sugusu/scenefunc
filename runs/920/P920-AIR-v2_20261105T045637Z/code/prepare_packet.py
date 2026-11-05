"""Export one conditional corrective example and objective comparison, never human preference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
P=Path(__file__).resolve().parent.parent; R=P.parents[2]; D=R.parent
T=D/'runtime'/P.name; replay=T/'replay_attempt_2'; OLD=R/'runs/920/P920-AIR-v1_20261105T042212Z'
sys.path.insert(0,str(R/'scripts/920'))
from audit_native_review import export
from decision_state import canonical
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref=lambda p:{'path':str(p.resolve().relative_to(D)),'sha256':sha(p)}
def save(f,v):
 with f.open('x') as out:json.dump(v,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')
s=read(replay/'SUMMARY.json');frozen=read(P/'FROZEN_PROGRAM.json');result=read(replay/'RESULT.json')
assert s['conditional_objective_pass'] and [x['after'] for x in s['fixed_point_access']]==[7,7,7]
assert result['tool_calls']==result['observations']==1
candidates=[];measurement={};evidence=[]
for label in ('initial','after'):
 obs=read(replay/(label+'_observation.json'));binding=read(Path(obs['state_binding']))
 c={'candidate_id':label,'state':ref(replay/(label+'_state.json')),'context':ref(replay/(label+'_context.json')),
    'observation':ref(replay/(label+'_observation.json')),'binding':ref(Path(obs['state_binding']))}
 candidates.append(c);measurement[label]=read(replay/(label+'_measurement.json'))
 evidence.append({'evidence_id':label+'_access','artifact':ref(replay/(label+'_measurement.json'))})
 save(P/(label.upper()+'_VIEWS.json'),{'state':c['state'],'context':c['context'],'binding':c['binding'],'images':binding['images'],
  'newly_rendered':label=='after','reuse_scope':None if label=='after' else 'AIR-v1 bound endpoint, numerically equivalent to actual cold-restored state; exact difference in RESTORATION.json.'})
for key,path in [('rail_access',replay/'SUMMARY.json'),('move',replay/'STEP.json'),('restore',replay/'RESTORATION.json'),('sweep',replay/'ACTUAL_SWEEP.json')]:
 evidence.append({'evidence_id':key,'artifact':ref(path)})
measurement['rail_access']={'before':s['before_rail_approaches'],'after':s['rail_approaches']}
claims=[
 {'candidate_id':'after','object_ids':['open_clothes_rail_0','recon_chair_east_0','recon_table_0'],'activity':'从入口接近东侧餐位',
  'assessment':'0.5和0.6米方形代理恢复dining_east路径，三宽度均为7/7固定点可达。没有移动餐桌、坐椅或其他23个对象；只移动原衣架。','evidence_ids':['initial_access','after_access','move']},
 {'candidate_id':'after','object_ids':['open_clothes_rail_0','recon_low_fridge_0'],'activity':'接近开放衣架',
  'assessment':'南侧五个点在三宽度下均可达，北侧均不可达。0.4米由两侧10/10降为单侧5/10；0.5和0.6米总数仍5/10但可达侧由北改南。不能把餐位恢复写成每项使用条件均改善。','evidence_ids':['rail_access']},
 {'candidate_id':'after','object_ids':['open_clothes_rail_0'],'activity':'保留开放挂衣供给并调整房间',
  'assessment':'对象和未缩放网格保留，约0.396平方米外包络不变。保守直线扫掠无所列障碍相交，但不证明人能搬动、挂衣容量或舒适度；全房专业偏好仍未知。','evidence_ids':['move','sweep','rail_access']}
]
# Verify object names against saved native state; do not infer a semantic ID from a display label.
ids=set(read(replay/'after_state.json')['objects'])
fridge=[oid for oid in ids if 'fridge' in oid];assert len(fridge)==1
claims[1]['object_ids'][1]=fridge[0]
record={'review_id':P.name+'_objective_revision','created_at':datetime.now(timezone.utc).isoformat(),
 'provenance':{'kind':'objective_geometric_review_of_source_guided_reconstruction','project_id':'budapest_airbnp_position_collective_2015',
  'family_id':'budapest_airbnp_position_collective_2015','allocation':'train_source_candidate','unique_review_unit':1,
  'professional_review':False,'training_eligible':False,'training_authorized':False},
 'brief':read(replay/'initial_context.json')['design_request'],
 'source_context':{'url':'https://position-concept.com/portfolio/airbnp-2/','supported':'短住条件下开放挂杆便于取用衣物；原文不指定本重建摆位、搬移动作或通行尺寸。'},
 'question':'在保留开放挂衣供给的修订前后，哪些固定通路恢复，哪些衣架接近条件损失？区分几何条件验收与整屋专业偏好。',
 'assumptions':frozen['assumptions']+['修订动作根据已知v1失败选择；不是随机对照、独立项目或训练收益。','初态视图沿用v1终态；冷恢复仅一处四元数5.42e-20差异，详见恢复记录。'],
 'candidates':candidates,'measurement_input':measurement,
 'judgment':{'verdict':'insufficient_evidence','ranking_groups':[],'claims':claims,'evidence':evidence,
  'unknowns':['整体专业偏好、真实人体使用、衣物容量、承重和搬运能力未知。','源挂杆与落地衣架安装不同；房间比例及设施代理假设未作尺寸稳健性检验。']}}
save(P/'OBJECTIVE_REVIEW.json',record);qa=export(record,D,['initial','after']);save(P/'QA_EPISODE_CANDIDATE.json',qa)
ctx=read(replay/'initial_context.json');step=read(replay/'STEP.json');obs=read(P/'INITIAL_VIEWS.json')
reason='来源在短住条件下采用开放挂杆以便取用衣物。本重建保留开放挂衣供给，并根据已测的东侧餐位通路损失移动衣架；位置由研究者选择，需在执行后检查是否恢复通路及是否损害另一侧接近。'
objects=[{k:v[k] for k in ('object_id','object_type','name','transform','bbox_min','bbox_max','immutable')} for v in ctx['objects'].values()]
task={'request':ctx['design_request'],'request_provenance':ctx['request_provenance'],
 'revision_request':'保持全部对象与衣架实例，恢复东侧餐位入口路径，同时保留至少一侧开放衣架接近；只能移动衣架。',
 'revision_request_origin':'Researcher task after v1 failure, not original resident quotation.',
 'objects':objects,'available_measurement':measurement['initial'],'prior_rail_approaches':s['before_rail_approaches'],
 'unknowns':frozen['assumptions'],'observation_scope':'Views and object table show original v1 endpoint. The execution input differs only by recorded quaternion serialization roundoff.'}
ota={'stage':'furniture','system':'Revise this reconstructed room using only move_furniture_tool(object_id, x, y, yaw). Units are metres/degrees. Only open_clothes_rail_0 may move. Preserve every object. Propose one edit and do not assert its unmeasured outcome.',
 'task_context':{'scene_description':canonical(task)},
 'provenance':{'family_id':'budapest_airbnp_position_collective_2015','decision_id':frozen['source_decision'],'role':'source_conditioned_corrective_reconstruction_candidate',
  'allocation':'train_source_candidate','training_eligible':False,'training_authorized':False,'historical_designer_trace':False,'model_generated':False,
  'professional_reason_scope':'Only short-stay/open-clothing strategy is attributed. Need to retain the rail and this coordinate/action are researcher choices using objective feedback.',
  'outcome_scope':'Conditional geometric acceptance with opposite-side access loss; no professional preference.'},
 'evaluation_only':{'outcome':s['acceptance'],'transition':ref(replay/'RESTORATION.json'),'after_state':ref(replay/'after_state.json')},
 'turns':[{'role':'assistant','content':reason,'images':[str(Path(x['path']).relative_to(D)) for x in obs['images']],'tool_calls':[step['command']]}]}
save(P/'OTA_CORRECTION_CANDIDATE.json',ota)
save(P/'SUPERVISION_BOUNDARIES.json',{'source_supported_fragment':'短住条件下采用开放挂杆以便取用衣物。','researcher_reason_fragment':'保留此供给，根据实测餐位路径损失选择移动及坐标；不预告执行结果。',
 'source_only_knowledge_record':ref(P/'SOURCE_RECORD.json'),'inference_input_excludes_target_action_and_final_metrics':True,'target_action_executed':ref(replay/'STEP.json'),
 'candidate_data_not_admitted':True,'main_comparison_rule':'Both arms must receive the same truthful source fragment, researcher explanation and action; mismatch remains auxiliary. No optimizer authorized.'})
packet={'decision_id':frozen['source_decision'],'run_id':P.name,'prior_run':OLD.name,'family_id':'budapest_airbnp_position_collective_2015',
 'source':ref(P/'SOURCE_RECORD.json'),'source_scope':record['source_context'],'applicable_condition':'Two short-stay visitors; open-clothing provision is retained while a measured dining access conflict is corrected.',
 'reconstruction_assumptions':frozen['assumptions'],'researcher_choice':'Translate the same freestanding rail about 0.35m toward +Y, informed by v1 loss; not designer historical operation.',
 'initial':candidates[0],'action':ref(replay/'STEP.json'),'actual_restored_input':ref(replay/'actual_restored_state.json'),'restoration':ref(replay/'RESTORATION.json'),'after':candidates[1],
 'effects':{'fixed_points_before':[7,6,6],'fixed_points_after':[7,7,7],'rail_points_before':[10,5,5],'rail_points_after':[5,5,5],
  'each_rail_side_points':5,'body_widths_m':[.4,.5,.6],'gain':'Dining east entry paths restored at .5/.6m.','cost':'North-side access lost; .4m two-sided access becomes single-sided.',
  'unknowns':record['judgment']['unknowns'],'summary':ref(replay/'SUMMARY.json')},
 'outcome':{'classification':'conditional_objective_repair_pass_with_tradeoff','positive_professional_demonstration':False,'training_admitted':False,'model_learning_evidence':False},
 'supervision':{'knowledge_valid_independently':True,'objective_review_candidate':ref(P/'OBJECTIVE_REVIEW.json'),'corrective_action_candidate':ref(P/'OTA_CORRECTION_CANDIDATE.json')},
 'stopping_rule':'End AIR pose revision after this single candidate. Keep v1 and failed restoration attempt. Next work concerns corpus qualification and independent evaluation, not another AIR tuning round.'}
save(P/'DECISION_PACKET.json',packet)
save(P/'REVIEW_EXPORT_CHECKS.json',{'native_state_context_image_bindings_checked':True,'named_objects_and_evidence_exist':True,'qa_images':10,
 'ota_pre_action_images':5,'ota_executed_calls':1,'new_loader_or_tokenizer_run':False,'training_admitted':False,
 'qa_sha256':sha(P/'QA_EPISODE_CANDIDATE.json'),'ota_sha256':sha(P/'OTA_CORRECTION_CANDIDATE.json')})
print('Exported objective QA plus one conditional corrective OTA candidate. Source and researcher reasons separated. No loader/training invocation.')
