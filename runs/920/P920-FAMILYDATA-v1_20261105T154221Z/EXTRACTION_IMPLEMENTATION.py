from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
from collections import Counter
r=Path('/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc');now=datetime.now(timezone.utc);run='P920-FAMILYDATA-v1_'+now.strftime('%Y%m%dT%H%M%SZ');p=r/'runs/920'/run;p.mkdir();stamp=now.isoformat()
def read(f):return json.loads(f.read_text())
def write(f,v):f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
registry_path=r/'research/920/data_readiness_v1/PROJECT_REGISTRY.json';registry=read(registry_path);byfamily={x['family_id']:x for x in registry['projects']}
sources={
 'fhouse':{'url':'https://www.nevertoosmall.com/post/f-house-osaka','title':'F-House, Osaka','publication_date':'2022-10-20','author':'Joel Beath','project_designer':'Kazuteru Matsumura / Coilkma','family_id':'osaka_f_house_coilkma','reported_area_m2':57,'reported_occupants':4,'reported_storeys':3,'short_locator':'To accommodate existing furniture','evidence_kind':'original_project_editorial_attributing_architect_decisions'},
 'michelet':{'url':'https://www.nevertoosmall.com/post/michelet-paris','title':'Michelet, Paris','publication_date':'2022-09-01','author':'Joel Beath','project_designer':'L’atelier - Nomadic Architecture Studio','family_id':'paris_michelet_nomadic','reported_area_m2':50,'reported_occupants':5,'short_locator':'allowing both rooms to be used at the same time','evidence_kind':'original_project_editorial_attributing_architect_decisions'},
 'families':{'url':'https://www.nevertoosmall.com/post/designing-small-spaces-for-families','title':'Designing Small Spaces for Families','publication_date':'2024-06-25','author':'Kate Kolberg','project_designer':'Pierre Escobar / L’atelier - Nomadic Architecture Studio','family_id':'paris_michelet_nomadic','short_locator':'the ability to change the layout as the family grows','evidence_kind':'editorial_with_named_architect_interview'},
 'paper':{'url':'https://www.nevertoosmall.com/post/how-two-graphic-designers-live-and-work-in-45sqm-484sqft-home','title':'How Two Graphic Designers Live and Work in a 45sqm Home','publication_date':'2026-01-22','author':'Camilla Janse van Vuuren','project_designer':'Associates Architecture with Giovanni Cavalleri and Cecilia Bianchini','family_id':'milan_paper_paper_associates','reported_area_m2':45,'short_locator':'a custom table they designed with the architects','evidence_kind':'resident_interview_with_explicit_architect_co_design_attribution'}
}
for x in sources.values():x.update(checked_at=stamp,full_page_cached=False,raw_page_sha256=None,source_correlation_group='never_too_small_editorial',reuse_scope='Attributed short factual paraphrases and short locators only; no images/full article copied. Public accessibility is not training or image reuse permission.')
write(p/'SOURCE_NOTES.json',sources)
# Conditions and benefits below are source paraphrases. Tool proposals and limits are our analysis, not source facts.
specs=[
 ('fhouse_existing_furniture_shared_living','fhouse','storage_and_existing_assets','四口之家已有家具，希望主要空间供共同生活。','以较开放的空间组织容纳既有家具并优先共同起居。','Main paragraph on existing furniture/open plan, web line56','partial_candidate','若重建任务另行给出既有对象表，可检验保留实例并安排共同活动；不能推断原项目具体家具ID。',['原具体家具清单、尺寸及历史初态未取得。','三层建筑组织未由当前单层家具工具复现。']),
 ('fhouse_curtains_flexible_zones','fhouse','changeable_activities','希望紧凑家庭空间保持可分可合，并控制费用。','以魔术贴固定的帘代替常规门，灵活分区并遮蔽收纳。','Paragraph on velcro curtains, web line60','knowledge_only','当前静态家具移动不能验证帘的闭合、私密性、固定方式或真实成本。',['布料、帘轨/魔术贴、开合与视线遮挡未测。','费用下降为报道的设计理由，非本研究成本测算。']),
 ('fhouse_standard_furniture_vertical_fit','fhouse','cost_construction_maintenance','选型涉及定制成本，儿童区需兼顾睡眠、学习与收纳。','减少定制家具，将桌与收纳配置在上下床下方。','Paragraph beginning Avoiding custom furniture, web line71','knowledge_only','多层支撑、床下净高与上下床安全未实现；不能将物体穿插计成功。',['采购价格、荷载、净高及儿童安全未验。']),
 ('michelet_adapt_family_lifecycle','families','changeable_activities','家庭成员长大、居住安排会变化。','空间可随家庭演变调整；原长子卧室后来改为办公室。','Michelet section and Pierre Escobar statement, web lines118–128','knowledge_only','可记录用途转换原则，但原改造过程、家具变更与完整尺寸不足，不能构造历史动作。',['不知完整改造时间表、搬迁代价与实际家具处置。']),
 ('michelet_separate_hygiene_concurrency','michelet','zoning','一家五口需要同时使用卫生相关功能。','把洗手/储物/洗衣与更私密的如厕/淋浴分成两间，支持并行使用。','Final project paragraph on original bathroom, web line88','knowledge_only','当前工具不能新建两间房和可操作门；摆两个设施不证明并行使用或隐私。',['两间房准确尺寸、门扇和管线未知。','程序尚无两人同时操作的使用证据。']),
 ('michelet_guest_table_capacity','michelet','changeable_activities','五口之家有亲友来访时需更多餐位。','厨房容纳可延展餐桌，报道最大可坐八人。','Kitchen paragraph line65; family feature line122 confirms guest context','knowledge_only','当前刚体目录无伸缩机构，不将缩放桌面当作真实扩展动作。',['八人是来源报告容量，非本研究验证。','伸缩过程、净空、餐位尺寸与收整代价未知。']),
 ('paper_table_mobility_for_overnight_guest','paper','changeable_activities','同一生活区需吃饭、聚会并偶尔容纳留宿者。','住户与建筑师共同设计带轮桌，可移开以支持来客和过夜活动。','Dining and Living section, web lines74–85','partial_candidate','可在声明重建中移动同一桌实例并测几何后果；现有静态沙发不能冒充展开后的沙发床。',['原桌尺寸、轮子和搬运阻力未知。','沙发床展开、停放与真实睡眠/餐位未验证。'])
]
records=[]
for did,key,category,condition,reason,locator,level,expression,unknowns in specs:
 source=sources[key];family=source['family_id'];project=byfamily[family];assert project['allocation']=='train'
 record={'decision_id':did,'family_id':family,'allocation':'train_source_candidate','knowledge_category':category,
  'professional_evidence':{'evidence_kind':source['evidence_kind'],'primary_crosscheck':{'url':source['url'],'attribution':source['project_designer']+'; report by '+source['author'],'locator':locator,'checked_at':stamp,'full_page_cached':False,'raw_page_sha256':None},'raw_resident_brief_available':False,'reuse_scope':source['reuse_scope'],'project_facts':{k:v for k,v in source.items() if k.startswith('reported_')},'professional_attribution_limit':'Named co-design with residents; not a verbatim architect explanation.' if key=='paper' else 'Reported architect decision, not an independent assessment of our reconstruction.'},
  'condition_summary':condition,'condition_origin':'Retrospective project/editorial account; not raw commissioning brief.',
  'rationale_summary':reason,'rationale_origin':'Researcher Chinese factual paraphrase; source supports the stated relationship, not our future action coordinates.',
  'decision_scope':{'source_decision':reason,'unsupported_part':'No native episode, measured improvement, designer historical operation or expert ranking derived in this batch.'},
  'unknowns':unknowns,'tool_expression':{'level':level,'full_source_decision_supported':False,'executable_subdecision':expression if level=='partial_candidate' else None,'qualification':expression},
  'source_to_scene':{'historical_action_sequence':None,'reconstructed_action_sequence':None,'initial_proxy_ids':[],'decision_packet':None},
  'maturity':{'source_supported_knowledge':True,'source_to_native_objects_bound':False,'source_grounded_action_replayed':False,'whole_room_consequences_reviewed':False,'independent_professional_preference_label':False,'paired_episode_ready':False},
  'readiness_by_layer':{'professional_knowledge':{'supported':True,'use':'Attributed knowledge/retrieval candidate; preserve reporter/resident/architect attribution distinctions.'},'professional_evaluation':{'available':False},'scene_action':{'executed':False,'tool_calls':0},'professional_decision_pair':{'executed':False,'positive_demonstration':False,'core_training_ready':False}},
  'split_fixed_at':project['split_fixed_at'],'derived_knowledge_at':stamp,'training_eligible':False,'training_authorized':False,'knowledge_record':str((p/'NEW_KNOWLEDGE.json').relative_to(r))}
 records.append(record)
write(p/'NEW_KNOWLEDGE.json',records)
source_path=r/'research/920/decision_packets_v1/SOURCED_DECISIONS.json';all_records=read(source_path);assert not(set(x['decision_id'] for x in records)&set(x['decision_id'] for x in all_records['records']));all_records['records']+=records;all_records['updated_at']=stamp
for key in ['records_reviewed_here','source_supported_knowledge']:all_records['counts'][key]+=7
all_records['counts']['partially_tool_expressible_candidates']+=2;all_records['counts']['prospectively_train_allocated_knowledge_records']+=7
write(source_path,all_records)
for family in set(x['family_id'] for x in records):
 project=byfamily[family];chosen=[x for x in records if x['family_id']==family]
 project.update(identity_status='Original project editorial verified by title, designer attribution, household and reported area. Not independently surveyed.',source_reviewed_at=stamp,derived_knowledge_records=len(chosen),knowledge_decision_ids=[x['decision_id'] for x in chosen],knowledge_record=str((p/'NEW_KNOWLEDGE.json').relative_to(r)),action_qualification='Knowledge only; two partial planar subdecisions across this batch remain unexecuted, mechanisms and original object identities unknown.')
 if family=='paris_michelet_nomadic':project['primary_verified_url']=sources['michelet']['url']
registry['updated_at']=stamp;write(registry_path,registry)
covp=r/'research/920/data_readiness_v1/KNOWLEDGE_COVERAGE.json';cov=read(covp);counts=Counter(x['knowledge_category'] for x in all_records['records'])
for row in cov:
 row['source_supported_knowledge_records_in_current_packet']=counts[row['category']]
 new=[x['decision_id'] for x in records if x['knowledge_category']==row['category']]
 if new:row['latest_family_source_additions']={'run_id':run,'decision_ids':new,'native_episodes':0,'professional_evaluations':0}
write(covp,cov)
old=read(r/'runs/920/P920-LOSS-v1_20261105T092125Z/SOURCE_ATTRIBUTION.json')
review={'status':'next_data_scope_review_not_applied_to_frozen_runs','reviewed_at':stamp,'pilot_reason_records':10,'source_principle_containing_records':[x['id'] for x in old if x['source_principle_in_reason']],'researcher_only_narration_records':[x['id'] for x in old if not x['source_principle_in_reason']],'future_supervision_rule':'Keep all actual actions. Researcher-only narration is provenance, not sourced professional reasoning. For the three principle-containing records explicitly separate source-supported sentence from reconstructed execution plan before future training; original pilot and live KBASE corpus remain unchanged.','all_reason_records_include_researcher_plan':True,'new_training_authorized':False}
write(p/'REASON_SCOPE_REVIEW.json',review)
write(p/'SUMMARY.json',{'run_id':run,'recorded_at':stamp,'status':'seven_attributed_knowledge_records_added','new_project_families':3,'new_source_supported_knowledge':7,'new_partially_expressible_candidates':2,'new_native_episodes':0,'new_professional_evaluations':0,'new_training_samples_admitted':0,'total_project_families':len(registry['projects']),'allocations':dict(Counter(x['allocation'] for x in registry['projects'])),'total_source_supported_records':all_records['counts']['source_supported_knowledge'],'model_calls':0,'training_updates':0,'live_inference_inputs_modified':False})
(p/'README.md').write_text('''# 家庭活动、既有家具与共同设计来源批次\n\n新增3项目家族、7条有出处知识；仅2条可考虑局部刚体重建，0新动作/训练/人评。项目先登记train来源再结构化提取，搜索片段已暴露，不能称未见测试。知识总数21，项目家族总数25。\n\n- F-House：容纳既有家具与共同起居；帘的灵活分区；非定制家具与床下组合。保留原家具的总体要求有来源，但没有原物清单或ID。\n- Michelet：家庭生命周期用途变化；卫生功能分间支持同时使用；伸缩桌支持来访。现有地面家具工具不能复现新隔墙、门扇、伸缩和多层床。\n- Paper Paper：带轮桌让生活区适应留宿来客。来源明确住户与建筑师共同设计，不能把住户采访改称建筑师原话；静态沙发不能计为可展开床。\n\n来源/作者/日期/定位见 SOURCE_NOTES.json，逐项适用条件、理由、工具边界和未知见 NEW_KNOWLEDGE.json。页面正文通过公开原始报道核读，没有复制文章或图片，未保存原HTML，不伪造网页哈希；使用权字段仍未核实。没有因为缺完整3D而抹去知识，也不把知识入库视作训练准入。\n\nREASON_SCOPE_REVIEW.json 把原10段理由中的3段来源原则与7段研究者操作叙述分开；它只约束未来数据，当前冻结KBASE仍使用原全部10段，不在观察结果后换材料。\n''')
Path('/tmp/familydata_current').write_text(run);print(run,len(records),'total knowledge',len(all_records['records']))
