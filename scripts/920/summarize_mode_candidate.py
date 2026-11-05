"""Publish bounded reconstruction facts without converting unknown use into success."""
import argparse,json,hashlib,shutil,shlex
from pathlib import Path
from datetime import datetime,timezone

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args();p=a.public.resolve();t=a.runtime.resolve();r=Path(__file__).resolve().parents[2];now=datetime.now(timezone.utc).isoformat()
    mode=read(p/'MODE_PLAN.json');old=r/'runs/920'/mode['prior_rejected_run'];oldroot=t.parent/old.name;receipt=read(t/'native_execution/receipt.json');measured=read(t/'MODE_MEASUREMENT.json');audit=read(t/'FINAL_AUDIT.json');restore=read(t/'cold_restore/RESULT.json');export=read(t/'export/RESULT.json');data=read(t/'dataset/DATASET.json')
    assert receipt['exit_code']==0 and restore['receipt_count']==10 and read(p/'ARCHIVE_CHECKS.json')['images_hash_and_decode_checked']==55
    earlier=read(old/'MODE_PLAN.json');assert earlier['reference_measurement']==mode['reference_measurement'] and earlier['guest_reserve_xy_m']==mode['guest_reserve_xy_m'] and earlier['moves'][1:]==mode['moves'][1:]
    assert read(old/'EXECUTION_PLAN.json')['bundle']==read(p/'EXECUTION_PLAN.json')['bundle']
    correction={'before':earlier['moves'][0],'after':mode['moves'][0],'same_room_input_bundle':True,'same_measurement_and_reserve':True,'same_other_actions':True,'first_version_native_calls':0,'first_version_effects':read(oldroot/'MODE_PREFLIGHT.json')['summary'],'second_version_actual_effects':measured['summary'],'interpretation':'Researcher correction informed by retained geometric failure, not an independent comparison or learned-model gain.'};write(p/'CORRECTION_COMPARISON.json',correction)
    for root,public in [(oldroot,old),(t,p)]:
        commands=['# 已执行命令','',f'工作目录：`{r}`。此处为回执，查看结果无需重跑。','']
        for f in sorted(root.glob('*execution/receipt.json')):
            x=read(f)
            if x['status']=='running':continue
            for name in ['receipt.json','stage.log']:
                dest=public/'execution'/f.parent.name/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f.parent/name,dest)
            commands += [f"## {f.parent.name}",'',f"{x['started_at']} → {x['ended_at']}；退出{x['exit_code']}，{x['elapsed_seconds']:.6f}秒。",'','```bash',shlex.join(x['command']),'```','']
        (public/'COMMANDS.md').write_text('\n'.join(commands))
        for name in ['MODE_PREFLIGHT.json','TRANSLATION_PREFLIGHT.json','TRANSLATION_ANALYTIC_CHECKS.json']:
            if (root/name).exists():shutil.copy2(root/name,public/name)
    (old/'README.md').write_text('初版只完成输入构建与计算预检，没有原生动作或模型调用。椅子停放(1.75,-1.55)使第7–9步0.6米宽假设下的window_front不可达。原冻结方案、计算结果与实际退出回执保留；后续另冻结v2，只改停椅点，不更改检查标准。\n')
    failed=[{'width_m':x['square_body_width_m'],'unreachable':[k for k,v in x['targets'].items() if not v['reachable_from_entry']]} for x in audit['access']['scenarios']]
    summary={'run_id':p.name,'created_at':now,'status':'source_linked_construction_and_modes_reviewed','family_id':'milan_paper_paper_associates','actual_start':receipt['started_at'],'actual_end':receipt['ended_at'],'actual_seconds':receipt['elapsed_seconds'],'actual_exit_code':0,'native_actions':10,'additions':6,'moves':4,'initial_supplied_fixtures':3,'final_furniture_and_fixtures':len(audit['inventory']),'observations':11,'images':55,'images_in_git':7,'model_calls':0,'training_updates':0,'new_source_knowledge_records':0,'new_source_decisions_with_native_evidence':1,'source_reason_step':8,'ordinary_action_only_steps':9,'supervised_action_examples_per_arm':10,'source_reason_examples_per_arm':1,'mode_results':measured['summary'],'terminal_triangle_contacts':audit['surface_intersection_pairs'],'terminal_outside':sum(x['outside_nominal_room_xy'] for x in audit['bounds']),'immutable_records_unchanged':all(audit['immutable_records_unchanged'].values()),'generic_approach_failures':failed,'actual_motion_sweeps':'unverified; endpoint rotation changes and no recorded intermediate physical motion','cold_restore_status':restore['status'],'restore_roundoff_count':len(restore['quaternion_roundtrip_differences']),'glb':export['glb'],'source_geometry':'Researcher6.4x4.4m selected living/kitchen; not original45m²full home or surveyed plan.','sleeping_function_verified':False,'professional_preference_verified':False,'training_admitted':False,'limits':['Furniture inventory and declared reserve/points do not establish full human use.','Kitchen/fridge/pantry proxies, source sofa articulation, load, wheels, actual overnight capacity and comfort unknown.','V1 failure guided V2 research correction; not model learning or an independent/generalization result.','Shared assets/building code remain; different assumed topology/dimensions is not proof of real-world spatial diversity.']};write(p/'SUMMARY.json',summary)
    packet={'decision_id':mode['source_decision_id'],'family_id':summary['family_id'],'run_id':p.name,'source':{'path':str((p/'SOURCE_RECORD.json').relative_to(r)),'sha256':sha(p/'SOURCE_RECORD.json')},'source_principle':mode['source_reason'],'reconstruction_assumptions':mode['unknowns'],'native_steps':{'path':str((p/'native/STEPS.json').relative_to(r)),'sha256':sha(p/'native/STEPS.json')},'source_linked_step':8,'ordinary_actions':9,'source_does_not_supply':['coordinates','construction order','stool parking','dimensions','candidate reserve rectangle'],'benefit':'Declared guest reserve cleared in step8; all six fixed access points remain reachable under three square-body widths; all object IDs retained and everyday layout restored.','cost':'Everyday dining placement is interrupted while guest reserve is active; chair staging changes use until return. Original staging candidate lost window access and was rejected pre-execution.','unknowns':summary['limits'],'outcome':'conditional_geometric_mode_demonstration_not_professional_sleeping_acceptance','training_admitted':False,'independent_professional_review':False};write(p/'DECISION_PACKET.json',packet)
    previous=r/'runs/920/P920-ABUILD-v1_20261105T165434Z/SOURCE_ONLY_DATASET_INDEX.json';index=read(previous);index.update(created_at=now,prior_index={'path':str(previous.relative_to(r)),'sha256':sha(previous)});index['cases'].append({'case':'paper_table_modes','family_id':summary['family_id'],'candidate_manifest':str((p/'dataset/DATASET.json').relative_to(r)),'candidate_manifest_sha256':sha(p/'dataset/DATASET.json'),'server_input_root':str(t),'actions_per_arm':10,'reasons_per_arm':1,'same_targets':True,'actual_token_budget_verified':False});index.update(total_actions_per_arm=26,total_reasons_per_arm=3,ordinary_action_only_steps=23,source_family_count=3,declared_geometry_templates=2,independent_geometry_distributions=False,actual_loader_mixed_budget_verified=False,training_admitted=False,limits_file=str((p/'SUMMARY.md').relative_to(r)));write(p/'SOURCE_ONLY_DATASET_INDEX.json',index)
    (p/'SUMMARY.md').write_text(f'''# Paper Paper 来源决定：建设与活动切换

实际完成6次添加、4次移动，11组五视图；{receipt['elapsed_seconds']:.6f}秒，退出0。九件家具/设施包括三件输入中的固定厨具。没有模型推理或训练。

本例重建共享餐厨起居活动区域，6.4×4.4米、开口和替代家具是研究假设，并非原45平方米住宅平面。来源支持带轮桌与活动转换的关系；不支持本例坐标、停椅位置或动作次序。

- 第6步日常状态：两处声明的凳桌对齐/距离条件满足。
- 第8步来客状态：声明预留区清空；日常餐位暂时不可用，并非同时满足全部活动。
- 第10步恢复：对象ID和原位保留，原两处对齐/距离条件恢复。
- 十个步骤中的六个固定活动点，在0.4/0.5/0.6米方形身体假设下全部可达。终态三角接触、越界为0；通用接近探针仍有失败，详见SUMMARY.json，不能据此宣称人体使用已验收。

初版只作预检，停椅导致0.6米下窗前点受阻，已保留。v2只改停椅位置；这属于研究者改进，不能算模型学习。保守扫掠包络提示沙发冲突，固定朝向直线平移区间核对在预检中排除此误报；实际引擎有约0.025/0.135度落位差，运动过程未记录，因此实际搬运与扫掠仍未知。

静态沙发未展开，预留区不等于可睡床。厨房、冰箱、食品储物仍有代理，容量/开门/人体取用、舒适及专业评价均未知。原生冷恢复保留10次回执和资产绑定；未做恢复后再次编辑。

第8步对应一条来源原则，其余9步仅监督动作。加入前两案后候选为每组26动作、3来源理由；知识总数仍21，并无新增知识。两组同目标字符串，新的混合token/Trainer预算尚未核定，更没有新增训练。三来源不等于三种独立真实空间分布。
''')
    (p/'README.md').write_text('# 来源决定与模式切换\n\n[结果及边界](SUMMARY.md)、[决定记录](DECISION_PACKET.json)、[三案候选索引](SOURCE_ONLY_DATASET_INDEX.json)、[原始命令](COMMANDS.md)、[恢复](RECOVERY.md)。\n\n[日常俯视](views/step_006.topdown.png) → [来客预留俯视](views/step_008.topdown.png) → [恢复俯视](views/step_010.topdown.png)。Git保存7张关键视图，全部55张原图在服务器，逐图哈希见bindings。\n')
    (p/'RECOVERY.md').write_text(f'''# 恢复与查看

原生checkpoint：`{t/'native/checkpoint'}`。manifest SHA256：`{sha(t/'native/checkpoint/manifest.json')}`。恢复计划RESTORE_PLAN.json，已执行命令及退出见COMMANDS.md，使用新输出目录可只读复核，不应覆盖原结果。

静态GLB：`{t/'export/room_geometry.glb'}`，{export['glb']['bytes']}字节，SHA256 `{export['glb']['sha256']}`。完整资产、检查点和55张原始五视图留在服务器；Git保存可审阅状态、小日志、绑定/哈希和7张关键图。

输入资产/房壳来自v1参考构建，版本与哈希见BUILD_PLAN、REFERENCE_BUILD_PLAN、input_metadata/manifest；v2复用同一bundle，仅冻结不同的停椅动作。恢复检查已保留对象、资产与回执，未验证后续编辑或实际家具机械运动。
''')
    st=read(p/'STATUS.json');st.update(status=summary['status'],updated_at=now,ended_at=receipt['ended_at'],actual_exit_code=0,completed_planned={'native_calls':[10,10],'observations':[11,11]},latest_verifiable_artifact=str(p/'SUMMARY.json'),failure_reason=None,next_step='Expand source/action coverage and independent tasks; aggregate candidate review before any separately authorized training.');write(p/'STATUS.json',st);write(r/'runs/920/latest.json',{'run_id':p.name,'stage':st['status'],'updated_at':now,'run_directory':str(t)})
    print(json.dumps({'status':st['status'],'actions':10,'candidate_total_actions_per_arm':26,'source_only_reasons_per_arm':3,'new_native_linked_source_decisions':1,'training_updates':0}))
if __name__=='__main__':main()
