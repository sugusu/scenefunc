"""Finalize an archived development diagnostic without rerunning the model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from native_project import sha256


def read(p): return json.loads(p.read_text())
def write(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run', required=True); a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2]; p=repo/'runs/920'/a.run; t=repo.parent/'runtime'/a.run
    now=datetime.now(timezone.utc).isoformat(); summary=read(p/'SUMMARY.json'); plan=read(p/'PLAN.json')
    visual=read(p/'VISUAL_REVIEW.json'); assert len(visual['inspected_images'])==10
    archive_receipt=read(t/'archive_execution/receipt.json'); assert archive_receipt['exit_code']==0
    for name in ['receipt.json','stage.log']:
        dest=p/'execution/archive_execution'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(t/'archive_execution'/name,dest)
    checks=read(p/'ARCHIVE_CHECKS.json'); checks.update(excluded_running_self=[],completed_archive_receipt=archive_receipt)
    write(p/'ARCHIVE_CHECKS.json',checks)
    initial_bundle=Path(read(p/'EVALUATION_PLAN.json')['initial_bundle'])
    manifest=read(initial_bundle/'manifest.json'); house=read(initial_bundle/'house_state.json')
    initial=house['rooms'][manifest['selected_room_id']]
    rows=[]; recovery=[]
    for arm in plan['execution_order']:
        row=summary['arms'][arm]; final=read(p/arm/'rollout/final_state.json')
        row['changed_transform_ids']=[k for k,v in initial['objects'].items()
            if k in final['objects'] and v['transform']!=final['objects'][k]['transform']]
        row['room_geometry_unchanged']=initial['room_geometry']==final['room_geometry']
        assert row['room_geometry_unchanged']
        row['objective_improved_from_initial']=not row['initial_proxy_pass'] and row['final_proxy_pass']
        row['peak_allocated_bytes']=max(x['model']['peak_allocated_bytes'] for x in read(p/arm/'rollout/RESULT.json')['rounds'])
        rows.append(f"| {arm} | {row['initial_clear_rays']} → {row['final_clear_rays']} /15 | {row['initial_proxy_pass']} → {row['final_proxy_pass']} | {row['model_calls']}/{row['tool_calls']} | {row['parse_error_attempts']}/{row['schema_mismatch_calls']} |")
        recovery.append(f"- {arm}: checkpoint `{row['checkpoint']}`, manifest SHA256 `{row['checkpoint_manifest_sha256']}`; GLB `{row['export']['path']}`, SHA256 `{row['export']['sha256']}`。")
    summary.update(run_id=a.run,updated_at=now,final_visual_inspection=visual['status'])
    write(p/'SUMMARY.json',summary)
    batch=read(t/'batch_execution/receipt.json')
    text='''# 同一家具房间、两种用途条件的冻结底座诊断

在同一八家具初态和同一五图上，只改变“保持相互看见”与“减少直接视线干扰”的用途要求。模型未见研究候选坐标、示范操作、来源事后理由或评分答案。每组最多4模型调用/4工具尝试，拒绝及解析失败计预算；BF16/SDPA、贪心、GPU0，无适配器、0训练。任务及代理评价在推理前冻结。

| 条件 | 清晰视线：初态→终态 | 条件代理通过：初态→终态 | 模型/工具尝试 | 解析失败/其他schema不符 |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+f'''

整批实际 {batch['started_at']} → {batch['ended_at']}，{batch['elapsed_seconds']:.3f} 秒，退出 {batch['exit_code']}。实际动作、原始输出和回执在各组 `rollout`；`ACTION_SCHEMA_REVIEW.json`逐条区分原生接受和schema合规。schema外参数被接受不表示参数一定有效，也不能抹掉实际发生的编辑。

初态已满足共享视线条件。因此该组即便终态通过，也不能记作模型创造了收益。第二组要从同一初态改变视线关系；需按实际终态而非模型意图判断。两个不同需求不是两个独立项目；这是一例暴露开发房间的有限调用诊断，不是整屋专业设计验收、跨项目泛化或配对训练对照。

评价随对象位姿移动观察点和目标，检查四个声明餐位、角色相对接近点、保留ID/尺寸和包络相交，另保留三角接触及通用前侧探针。四方凳的语义正面未定义，facing提示不能替代坐姿或可见性判断。15条线段是预定不透明包围盒代理：共享要求15条通、干扰减少要求15条均挡，此严格阈值不是人评或住户标准。座姿、转头、动态遮挡、舒适性、柜门、真实容量、稳定性及搬运路径均未知。

两种研究参考布局已在FCHOICE-v2实际执行，说明这些声明条件有可行端点；模型本次是否发现端点是另一问题。调用预算、输出格式和工具回执也影响行为，不能仅凭失败断言模型缺专业知识或训练必需。禁止把这一开发任务反复调到通过后充作独立测试。

`INTERPRETATION.md`记录具体编辑及后果；`VISUAL_REVIEW.json`记录实际查看的终态十图。所有轮次观察均核状态绑定、图像哈希与解码；完整图、原生检查点和GLB留服务器，见 `RECOVERY.md`。研究知识/示范库不自动收录模型输出为正例，训练资格仍不变。
'''
    (p/'SUMMARY.md').write_text(text)
    (p/'RECOVERY.md').write_text('# 运行恢复与产物\n\n'+ '\n'.join(recovery)+
        '\n\n两终态均经独立原生进程加载，保留资产绑定、实例ID、目录和4条回执；没有恢复后编辑。各组 RESTORE_PLAN.json 与 execution/restore_execution/receipt.json 给出实际命令。GLB只含家具几何，不替代可编辑原生状态。模型固定版本/本地文件哈希在各组 PLAN.json，运行命令及小日志在 execution。源资产的来源、维度、生成/代理标记由输入清单和检查点元数据记录。不要因网络失败重跑本批。\n')
    status=read(p/'STATUS.json'); status.update(status=summary['status'],updated_at=now,completed=2,planned=2,
        started_at=batch['started_at'],ended_at=batch['ended_at'],actual_exit_code=batch['exit_code'],
        first_log=batch['first_log'],input_sha256={k:v['request_sha256'] for k,v in read(p/'PREPARATION_REVIEW.json')['arms'].items()},
        latest_verifiable_artifact=str(p/'SUMMARY.json'),failure_reason=None,
        task_outcomes={k:dict(initial_proxy_pass=v['initial_proxy_pass'],final_proxy_pass=v['final_proxy_pass']) for k,v in summary['arms'].items()},
        next_step='Publish preserved outputs; improve source-linked decision evidence and independent evaluation. No inference restart or new training authorization.')
    write(p/'STATUS.json',status)
    write(repo/'runs/920/latest.json',dict(run_id=a.run,stage=summary['status'],updated_at=now,run_directory=str(t)))
    shutil.copy2(Path(__file__),p/'code'/Path(__file__).name)
    print(json.dumps(dict(status=summary['status'],model_calls=summary['model_calls'],training_updates=0)))


if __name__=='__main__': main()
