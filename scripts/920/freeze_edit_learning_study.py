"""Build the finite six-family edit-study candidate package. No model or training."""
import copy,hashlib,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    r=Path(__file__).resolve().parents[2];d=r.parent;now=datetime.now(timezone.utc)
    run='P920-EDITSTUDY-v1_'+now.strftime('%Y%m%dT%H%M%SZ');p=r/'runs/920'/run;t=d/'runtime'/run
    p.mkdir();t.mkdir();sys.path.insert(0,str(r/'scripts/920'))
    from compact_catalog_context import OBJECT_FIELDS
    curated=read(r/'runs/920/P920-CURATE-v1_20261105T212407Z/DATASET.json')
    lifecycle=read(r/'runs/920/P920-LIFECYCLE-v1_20261105T221829Z/dataset/DATASET.json')
    ledger=read(r/'research/920/decision_packets_v1/SOURCED_DECISIONS.json')
    byid={x['decision_id']:x for x in ledger['records']}
    selected=[];excluded=[]
    for row in curated['rows']:
        keep=row['group'] in ['storage_edit','air_edit','waterloo_edit'] or (row['group']=='storage_build' and row['step']>=7) or (row['group']=='clothing_build' and row['step']==7) or (row['group']=='paper_table_modes_build' and row['step']>=7)
        if not keep:excluded.append(dict(id=row['id'],reason='Earlier construction scaffolding excluded from this furnished-room edit study.'));continue
        ref=row['documents'].get('paired',row['documents'].get('shared'));path=d/ref['path'];assert sha(path)==ref['sha256']
        selected.append(dict(id=row['id'],family=row['family_id'],decision_id=row['decision_id'],doc=read(path),origin=dict(path=str(path),sha256=sha(path)),reason=row['reason'],scope=row['reason_scope'],consequences=row['consequences']))
    life_reason='住户生活阶段变化后，可将原卧室调整为新的用途；Michelet报道原长子卧室后来改为办公室。'
    for row in lifecycle['rows']:
        ref=row['documents'].get('paired',row['documents'].get('shared'));path=d/ref['path'];assert sha(path)==ref['sha256']
        selected.append(dict(id=f'lifecycle_{row["step"]:02d}',family='paris_michelet_nomadic',decision_id='michelet_adapt_family_lifecycle',doc=read(path),origin=dict(path=str(path),sha256=sha(path)),reason=life_reason if row['step']==1 else '',scope='Source supports role adaptation; bed removal and all poses are reconstruction, not historical operations.',consequences=dict(path='runs/920/P920-LIFECYCLE-v1_20261105T221829Z/CONSEQUENCES.json',sha256=sha(r/'runs/920/P920-LIFECYCLE-v1_20261105T221829Z/CONSEQUENCES.json'))))
    f=r/'runs/920/P920-FCHOICE-v2_20261105T193428Z';obs=read(f/'perimeter/native/initial_observation.json');step=read(f/'perimeter/native/STEPS.json')[0]
    context=json.loads(obs['room_prompt'].split('\nCURRENT NATIVE STATE:\n',1)[1]);variants=read(f/'TASK_VARIANTS.json');brief=variants['shared_task']['design_request']+'\n'+variants['variants'][0]['request']
    ctx={k:context[k] for k in ['original_request','request_provenance','asset_bindings','asset_catalog']};ctx.update(design_request=brief,objects={oid:{k:obj[k] for k in OBJECT_FIELDS} for oid,obj in context['objects'].items()},recent_actual_tool_receipts=context['prior_tool_receipts'][-2:],physical_unknowns=variants['shared_task']['unknowns'])
    doc=dict(stage='furniture',system=obs['system'],task_context=dict(scene_description=json.dumps(ctx,ensure_ascii=False,sort_keys=True)),provenance=dict(family_id='osaka_f_house_coilkma',kind='research_source_principle_transfer'),evaluation_only=dict(sentinel='EVAL_ONLY_EDIT_STUDY',actual_reply=step['reply'],after_state=str(f/'perimeter/native/step_01_after.json')),turns=[dict(role='assistant',content='',images=obs['image_paths'],tool_calls=[step['command']])])
    selected.append(dict(id='fhouse_perimeter_01',family='osaka_f_house_coilkma',decision_id='fhouse_existing_furniture_shared_living',doc=doc,origin=dict(path=str(f/'perimeter/native/STEPS.json'),sha256=sha(f/'perimeter/native/STEPS.json')),reason=read(f/'CHOICE_PLAN.json')['source_reason'],scope='Source shared-living condition only. Divider and reduced-distraction counterfactual excluded from professional supervision.',consequences=dict(path=str((f/'SUMMARY.json').relative_to(r)),sha256=sha(f/'SUMMARY.json'))))
    assert len(selected)==22 and len({x['family'] for x in selected})==6
    heldout={'sydney_pine_nicholas_gurney_2022'}
    registry_path=r/'research/920/data_readiness_v1/PROJECT_REGISTRY.json';registry=read(registry_path)
    milan=[x for x in registry['projects'] if 'SET' in str(x.get('reported_name','')) or ('milan' in x['family_id'] and x.get('allocation')=='test')]
    assert len(milan)==1,[(x['family_id'],x.get('reported_name')) for x in milan]
    heldout.add(milan[0]['family_id']);assert not heldout&{x['family'] for x in selected}
    knowledge=[]
    for did in sorted({x['decision_id'] for x in selected if x['reason']}):
        src=byid[did];sample=next(x for x in selected if x['decision_id']==did and x['reason'])
        knowledge.append(dict(decision_id=did,family_id=sample['family'],condition=src['condition_summary'],principle=sample['reason'],source=src['professional_evidence'],scope=sample['scope']))
    write(p/'KNOWLEDGE_POOL.json',dict(records=knowledge,policy='Fixed six-source knowledge bundle, same in every arm and every sample. No retrieval ranking model; all six fit in prompt. Source metadata is provenance, concise condition/principle/scope enter prompt.',excluded_families=sorted(heldout)))
    ktext=json.dumps([{k:x[k] for k in ['decision_id','condition','principle','scope']} for x in knowledge],ensure_ascii=False)
    decisions={
        'clothing_build_07':('两位短住者已具有主要生活家具，还需常用衣物取放。','增加开放挂杆；不替换已有生活家具。','需检查挂杆两侧取用与通行；承重、容量和封闭储物偏好未知。'),
        'storage_build_07':('主要生活家具已放入紧凑房间，还需补充收纳并保留连续起居空间。','开始将新增柜体沿边界集中组织；后续柜体属于同一重建序列。','集中储物仍可能占用窗前或柜前空间；完整集成墙的技术功能未复现。'),
        'storage_edit_01':('已有柜体需要保留，当前要求连续起居、柜前及窗前接近。','移动本步指定柜体，作为集中收纳布局的首个重建动作。','须逐步检查其他柜体、餐位、窗前和厨卫通行；不预先承诺全屋改善。'),
        'paper_table_modes_build_08':('当前活动从日常用餐转为需要让出访客预留区，仍需保留餐桌。','将现有桌移到临时停放位置，保留同一实例。','日常用餐布置暂时改变；需要检查停放处与厨房、通路冲突，客床展开未验证。'),
        'waterloo_edit_02':('当前需要共享活动工作面；已有两张桌可以重新组合。','移动第二张桌，形成共享桌面的重建方案，保留原桌身份。','独立工作位模式暂时改变；须检查坐席和通路，移动阻力、拼缝及稳定性未知。'),
        'lifecycle_01':('研究任务已明确不再需要原儿童床，同时需要新的办公用途。','移除该床以开始用途转换，保留仍需的主床和其他房间功能。','损失该床实例和原睡眠位置；仅在该睡眠需求已解除时适用，搬运处置及办公舒适度未知。'),
        'fhouse_perimeter_01':('当前需求强调餐桌与沙发之间的共同交流，且已有家具必须保留。','保留高柜并将其移向房间边缘，减少两活动区间的分隔。','需要检查柜前、餐位、窗前及入口接近；实际隐私、声学和专业偏好未测。')}
    rows=[]
    for x in selected:
        doc=copy.deepcopy(x['doc']);turn=doc['turns'][0];assert len(doc['turns'])==1 and len(turn['images'])==5
        images=[]
        for path in turn['images']:
            f=Path(path) if Path(path).is_absolute() else d/path;assert f.is_file();images.append(dict(path=str(f.resolve()),sha256=sha(f)))
        turn['images']=[z['path'] for z in images]
        ctx=json.loads(doc['task_context']['scene_description']);ctx['available_source_knowledge']=json.loads(ktext)
        doc['task_context']['scene_description']=json.dumps(ctx,ensure_ascii=False,sort_keys=True)
        doc['system']+='\n对需要权衡的步骤，可先在<think>内用短决定记录说明当前条件、来源原则适用性、涉及对象及预期代价，再给出一个原生tool_call。未知项保持未知；不要在动作前写成已经获得实测改善。'
        target=None
        if x['reason']:
            condition,decision,cost=decisions[x['id']]
            target=dict(current_condition=condition,principle_applicability=dict(source_decision_id=x['decision_id'],sourced_principle=x['reason']),proposed_decision=dict(description=decision,object_or_asset_references=turn['tool_calls'][0]['arguments'].get('object_id',turn['tool_calls'][0]['arguments'].get('asset_id'))),expected_cost_and_unknowns=cost)
        docs={}
        for arm in ['action_only','decision_action']:
            new=copy.deepcopy(doc);new['turns'][0]['content']=json.dumps(target,ensure_ascii=False) if target and arm=='decision_action' else ''
            new.setdefault('provenance',{}).update(study=run,source_family=x['family'],historical_designer_trajectory=False,decision_text_authorship='Researcher state/object mapping of attributed principle; only sourced_principle is a source paraphrase',training_authorized=False)
            new.setdefault('evaluation_only',{})['study_origin']=x['origin'];new['evaluation_only']['decision_scope']=x['scope']
            path=p/'dataset'/arm/x['id']/'episode.json';write(path,new);docs[arm]=dict(path=str(path),sha256=sha(path))
        objects=ctx['objects'];values=objects.values() if isinstance(objects,dict) else objects
        count=sum(z['object_type']=='furniture' for z in values);assert count>0
        rows.append(dict(id=x['id'],family_id=x['family'],decision_id=x['decision_id'],existing_furniture_count=count,images=images,command=turn['tool_calls'],decision_target=target,decision_scope=x['scope'],documents=docs,source=x['origin'],consequences=x['consequences']))
    counts=dict(actions_per_arm=len(rows),families=6,decision_positions=sum(x['decision_target'] is not None for x in rows),unique_source_principles=len(knowledge),ordinary_action_only=sum(x['decision_target'] is None for x in rows),tools=dict(Counter(x['command'][0]['name'] for x in rows)))
    assert counts['decision_positions']==7
    write(p/'DATASET.json',dict(schema='edit-study-candidate-v1',created_at=now.isoformat(),counts=counts,rows=rows,knowledge_pool_sha256=sha(p/'KNOWLEDGE_POOL.json'),excluded_old_rows=excluded,training_authorized=False,training_eligible_pending_review=True,original_episodes_modified=False,new_native_actions=0,model_calls=0))
    # This user-directed reassignment is recorded before any new training.
    events=[]
    for project in registry['projects']:
        if project['family_id'] in heldout:
            events.append(dict(family_id=project['family_id'],previous_allocation=project['allocation'],new_allocation='development',reason='User study plan explicitly reserves Pine and Milan as exposed development holdouts; exclude training and retrieval.',recorded_at=now.isoformat(),source_already_exposed=True))
            project.update(allocation='development',training_eligible=False,training_authorized=False,sealed_test_eligible=False,study_holdout=run)
    registry.setdefault('allocation_change_events',[]).extend(events);registry['updated_at']=now.isoformat();write(registry_path,registry)
    for rec in ledger['records']:
        if rec['family_id'] in heldout:rec.update(allocation='development_source_review',training_eligible=False,training_authorized=False,study_holdout=run)
    ledger['counts']['prospectively_train_allocated_knowledge_records']=sum(x['allocation']=='train_source_candidate' for x in ledger['records']);ledger['updated_at']=now.isoformat();ledger['count_scope']='22 attributed knowledge records; current study allocation is in PROJECT_REGISTRY. Pine reassigned from prospective training source to exposed development before new optimization. Historical episode counts are unchanged.';write(r/'research/920/decision_packets_v1/SOURCED_DECISIONS.json',ledger)
    write(p/'SPLIT.json',dict(events=events,train_families=sorted({x['family_id'] for x in rows}),development_families=sorted(heldout),sealed_test_projects=0,excluded_from_training_and_retrieval=sorted(heldout),historical_exposure_preserved=True))
    tasks=[dict(task='Pine',family_id='sydney_pine_nicholas_gurney_2022',bundle=str(d/'runtime/P920-PINE-v1_20261106T002523Z/build/bundle'),task_input=str(r/'runs/920/P920-PINE-v1_20261106T002523Z/TASK_INPUT.json'),evaluation_scope='Complete selected study/guest room; not complete55m2 apartment. Guest sleeping remains unknown, never success.'),dict(task='Milan',family_id=milan[0]['family_id'],bundle=str(d/'runtime/P920-MILAN-TASK-v1_20261105T224015Z/build/bundle'),task_input=str(r/'runs/920/P920-MILAN-TASK-v1_20261105T224015Z/TASK_INPUT.json'),evaluation_scope='Complete declared apartment reconstruction; original dimensions and real comfort unknown.')]
    for task in tasks:
        path=Path(task['bundle'])/'manifest.json';task['manifest_sha256']=sha(path);task['task_sha256']=sha(task['task_input'])
    write(p/'TASKS.json',dict(tasks=tasks,model_input_excludes=['reference actions/coordinates','developer outcome metrics','heldout project source rationale','reference final scene'],reference_outputs_evaluation_only=True))
    study=dict(study_id=run,status='candidate_package_materialized_not_training_authorized',question='Does source-grounded decision supervision improve complete furnished-room adaptation beyond identical action training and same-knowledge strong prompting?',arms=['frozen_base_knowledge_feedback','action_only','decision_action'],mechanism_claim='This3-arm study tests added decision supervision, not pairing versus same-material separate learning. Add truthful separate control only if warranted.',train_family_count=6,development_tasks=2,evaluation_seeds=[17,29],final_rooms_planned=12,training=dict(adapters=2,max_full_dataset_updates_per_adapter=8,max_seconds_per_adapter=7200,max_total_seconds=14400,lora='language q/v r8 alpha16; frozen base and vision; inherited BF16/SDPA/CPU checkpoint-input offload, optimizer config pending actual-loader audit',equal_action_exposure=True,loss='equal family mean action CE; decision CE separately normalized and reported; exact coefficient must be frozen before approval',not_started=True),evaluation=dict(max_total_gpu_seconds=7200,full_batch_total_gpu_seconds_upper_bound=21600,common_knowledge_pool='KNOWLEDGE_POOL.json',common_tools_views_assets_feedback=True,model_generates_own_decision=True,max_tool_calls_per_room=12,max_total_generated_tokens_per_room=12288,seeds_are_not_independent_projects=True,stochastic_decode_must_be_frozen=True,stopping_rule='Do not extend budgets or cherry-pick a checkpoint after development results.'),serialization=dict(chosen='Official OTA think + tool_call format for this study; do not silently reuse STRUCTBASE raw-JSON decoder or old encoded caches.',action_only='Empty decision content, supervise tool action only; no gold decision prefix.',decision_action='At7supported positions supervise short D plus same A; ordinary15positions supervise A only.',deployment='Every arm receives same planning permission and generates its own D/action; no gold D supplied.'),primary_acceptance=['Verify requested activity proxies, including simultaneous occupancy and usable surfaces/seats.','Retained instance/asset IDs and newly introduced hard side effects.','Compare area/edit/tool failure/token/GPU cost conditional on use outcomes.','Unknown guest sleep/comfort/capacity remain unknown; do not erase a valid local demonstration or claim whole-room gold.'],decision_rules=['Strong retrieval achieves requirements and training adds no benefit: stop extending training on these tasks.','Both adapters similar: credit action/task adaptation, no decision-mechanism claim.','Decision supervision improves use with no additional hard violation: add same-material separate control, then new unexposed projects.','Retrieval/parser/tool errors dominate: resolve named engineering cause, do not relabel it as missing design knowledge.','Training tasks fitted but no development gain: report result and inspect condition coverage; no automatic expansion.'],remaining_before_approval=['Actual official-loader encoding/masks and CPU dry-run of new finite training entry.','Freeze stochastic inference, feedback evaluator and exact GPU/time accounting.','Validate full12-cell executable command matrix and recovery behavior.'],authorization='Latest user message proposed a bounded budget; no new optimizer execution is authorized by this preparation.')
    write(p/'STUDY.json',study)
    write(p/'EVALUATION_MATRIX.json',dict(cells=[dict(task=task['task'],family_id=task['family_id'],seed=seed,arm=arm,status='not_started') for task in tasks for seed in study['evaluation_seeds'] for arm in study['arms']],results_available=0,results_planned=12))
    write(p/'STATUS.json',dict(run_id=run,status=study['status'],question=study['question'],updated_at=now.isoformat(),run_directory=str(t),completed_planned=dict(source_families=[6,6],candidate_actions=[22,22],decision_positions=[7,7],official_encoding=[0,1],training_updates=[0,16],evaluation_rooms=[0,12]),configuration_hashes=dict(study=sha(p/'STUDY.json')),input_hashes=dict(dataset=sha(p/'DATASET.json'),knowledge=sha(p/'KNOWLEDGE_POOL.json'),tasks=sha(p/'TASKS.json')),latest_verifiable_artifact=str(p/'DATASET.json'),failure_reason=None,next_step=study['remaining_before_approval']))
    write(r/'runs/920/latest.json',dict(run_id=run,stage=study['status'],updated_at=now.isoformat(),run_directory=str(t)))
    print(json.dumps(dict(run_id=run,counts=counts,development=sorted(heldout),model_calls=0,training_updates=0),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
