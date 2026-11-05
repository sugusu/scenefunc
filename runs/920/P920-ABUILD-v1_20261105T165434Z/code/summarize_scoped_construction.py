"""Summarize actual AIR construction and same-target source-only candidates."""
import argparse,hashlib,json,shutil,shlex
from pathlib import Path
from datetime import datetime,timezone

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args();p=a.public.resolve();t=a.runtime.resolve();r=Path(__file__).resolve().parents[2];now=datetime.now(timezone.utc).isoformat()
    receipt=read(t/'native_execution/receipt.json');native=read(t/'native/RESULT.json');audit=read(t/'FINAL_AUDIT.json');endpoint=read(t/'SOURCE_ENDPOINT.json');restore=read(t/'cold_restore/RESULT.json');export=read(t/'export/RESULT.json');assert receipt['exit_code']==0 and native['tool_calls']==7 and native['observations']==8
    assert restore['receipt_count']==7 and restore['asset_bindings_exact'] and read(p/'VISUAL_REVIEW.json')['final_views_inspected']==5
    cases=[]
    for tag,folder,root in [('clothing','dataset',t),('storage','storage_dataset',t.parent)]:
        data=read(t/folder/'DATASET.json');targets={};full=[]
        for arm,entries in data['arms'].items():
            targets[arm]=[]
            for entry in entries:
                f=root/entry['path'];assert sha(f)==entry['sha256'];doc=read(f);turn=doc['turns'][0];role=entry['supervised_role'];targets[arm].append((role,turn['content'] if role=='reason' else turn['tool_calls']))
                for image in turn['images']:assert (root/image).is_file()
            full.append(len(entries))
        assert targets['paired']==targets['separate']
        if folder=='storage_dataset':
            for f in (t/folder).rglob('*.json'):
                dest=p/folder/f.relative_to(t/folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        cases.append({'case':tag,'family_id':read(root/data['arms']['paired'][0]['path'])['provenance']['family_id'],'candidate_manifest':str((p/folder/'DATASET.json').relative_to(r)),'candidate_manifest_sha256':sha(t/folder/'DATASET.json'),'server_input_root':str(root),'actions_per_arm':data['action_samples_per_arm'],'reasons_per_arm':data['reason_samples_per_arm'],'same_targets':True,'actual_token_budget_verified':False})
    index={'created_at':now,'cases':cases,'total_actions_per_arm':sum(x['actions_per_arm'] for x in cases),'total_reasons_per_arm':2,'ordinary_action_only_steps':14,'reason_target_scope':'Two source-only factual paraphrases; no researcher coordinate/order instructions in reason targets.','source_family_count':2,'independent_geometry_distributions':False,'new_model_calls':0,'training_updates':0,'actual_loader_mixed_budget_verified':False,'training_admitted':False,'limits_file':str((p/'DATASET_SCOPE.md').relative_to(r))}
    write(p/'SOURCE_ONLY_DATASET_INDEX.json',index)
    generic=[{'body_width_m':x['square_body_width_m'],'unreachable':[k for k,v in x['targets'].items() if not v['reachable_from_entry']]} for x in audit['access']['scenarios']]
    fixed=[sum(v['reachable_from_entry'] for v in x['targets'].values()) for x in endpoint['fixed_access']['scenarios']]
    rail=[{side:sum(v['reachable_from_entry'] for k,v in x['targets'].items() if k.startswith(side)) for side in ['+Y','-Y']} for x in endpoint['rail_approaches']]
    summary={'run_id':p.name,'recorded_at':now,'status':'construction_and_source_only_candidates_reviewed','new_native_actions':7,'initial_fixed_fixtures':6,'final_furniture':len(audit['inventory']),'observations':8,'images':40,'actual_start':receipt['started_at'],'actual_end':receipt['ended_at'],'elapsed_seconds':receipt['elapsed_seconds'],'actual_exit_code':receipt['exit_code'],'mesh_contacts':audit['surface_intersection_pairs'],'outside_room':sum(x['outside_nominal_room_xy'] for x in audit['bounds']),'immutable_records_unchanged':all(audit['immutable_records_unchanged'].values()),'fixed_activity_counts':fixed,'rail_side_counts':rail,'generic_failed_targets':generic,'cold_restore_status':restore['status'],'cold_restore_roundoff_count':len(restore['quaternion_roundtrip_differences']),'glb':export['glb'],'source_only_candidates':{'actions_per_arm':16,'reasons_per_arm':2,'source_families':2,'new_independent_source_decisions':0},'whole_room_professional_acceptance':False,'new_model_calls':0,'training_updates':0,'failed_postprocessing':['archive_execution: wrong Python environment lacked jsonschema; retained, native construction not repeated.'],'next_step':'Expand geometry and decision diversity around source-supported construction; prepare any mixed training budget only after candidate review. No new training authorized.'}
    write(p/'SUMMARY.json',summary)
    (p/'SUMMARY.md').write_text(f'''# AIR 来源决定建设记录

从六件固定厨卫设施开始，真实添加床、沙发、桌、两把椅子、小冰箱和开放挂杆，共七次工具调用、八次五图观察。实际耗时 {receipt['elapsed_seconds']:.6f} 秒，退出 0；没有模型推理或训练。

终态十三件家具与设施，检测三角接触和越界均为零，固定对象记录保留。0.4/0.5/0.6 米方形身体假设下七个固定活动点均可达；挂杆 -Y 侧五点可达，+Y 侧五点均不可达。全部通用探针失败明细见 SUMMARY.json；不得用挑选的活动点掩盖它们。容量、取衣高度、人体使用与完整厨卫功能未知。

来源理由只用于第七步“提供开放挂杆”的策略依据。坐标、顺序、房壳和替代资产是研究者重建。前六步只示范动作。终态五图与添加挂杆前俯视图已审阅；新进程恢复保留七次回执与资产身份。可查看 GLB 与完整 checkpoint 留在服务器，见 RECOVERY.md。

另对既有 SBUILD 九步重新包装，去除理由目标中的研究者操作说明，没有重跑旧场景。两案合计每组十六动作、两个来源理由，字符串目标相同；新混合集合的 tokenizer、训练预算与训练尚未运行。两案共享相关矩形壳模板和部分代理，不能当独立空间分布或泛化证据。

归档首次因 Python 缺 jsonschema 失败，已保留日志；切换现有原生环境后归档通过，未重跑构建。此记录是有来源的条件化建设候选，不是专业推荐、模型成果或已准入训练集。
''')
    manifest=t/'native/checkpoint/manifest.json'
    (p/'RECOVERY.md').write_text(f'''# 查看与恢复

服务器几何：`{(t/'export/room_geometry.glb')}`\n\nSHA256：`{export['glb']['sha256']}`，{export['glb']['bytes']} 字节。此 GLB 是静态展示；对象身份与编辑恢复使用完整原生检查点。\n\n原生检查点：`{manifest.parent}`\n\nmanifest SHA256：`{sha(manifest)}`。源资产目录来自 BUILD_PLAN.json 的 base_manifest；版本、输入哈希与新建身份映射见 input_metadata 和 checkpoint_metadata。恢复与导出实际命令见 COMMANDS.md；这些命令已经运行，不应覆盖其输出目录。将完整检查点复制到新位置并按 manifest 逐文件验哈希，再在 RESTORE_PLAN 中绑定新路径/manifest 并使用新的输出目录。\n\n本次新进程只读恢复已通过；未执行恢复后再次编辑测试，不宣称优化器恢复。原始渲染、来源重建、权重、完整资产与 checkpoint 不进入 Git。\n''')
    commands=['# 实际命令与退出','',f'工作目录：`{r}`。回执记录已执行命令，重阅结果无需重跑。','']
    for f in sorted(t.glob('*execution/receipt.json')):
        x=read(f)
        if x['status']=='running':continue
        for name in ['receipt.json','stage.log']:
            dest=p/'execution'/f.parent.name/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f.parent/name,dest)
        commands += [f"## {f.parent.name}",'',f"{x['started_at']} → {x['ended_at']}；退出 {x['exit_code']}，{x['elapsed_seconds']:.6f} 秒。",'','```bash',shlex.join(x['command']),'```','']
    (p/'COMMANDS.md').write_text('\n'.join(commands))
    status=read(p/'STATUS.json');status.update(status=summary['status'],updated_at=now,ended_at=receipt['ended_at'],actual_exit_code=0,completed_planned={'native_actions':[7,7],'observations':[8,8]},latest_verifiable_artifact=str(p/'SUMMARY.json'),failure_reason=None,postprocessing_failure_retained=summary['failed_postprocessing'],next_step=summary['next_step']);write(p/'STATUS.json',status)
    write(r/'runs/920/latest.json',{'run_id':p.name,'stage':status['status'],'updated_at':now,'run_directory':str(t)})
    print(json.dumps({'status':summary['status'],'actual_new_actions':7,'combined_actions_per_arm':16,'source_only_reasons_per_arm':2,'fixed_access':fixed,'rail_side_counts':rail,'generic_failed_targets':generic},ensure_ascii=False))
if __name__=='__main__':main()
