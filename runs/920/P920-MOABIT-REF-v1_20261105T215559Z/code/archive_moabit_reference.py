"""Close a rejected reference search with traceable evidence, not a usable scene."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
from measure_room_access import obstacles


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args();p=a.public.resolve();t=a.runtime.resolve()
    plan=load(p/'SEARCH_PLAN.json');v=load(t/'verify/RESULT.json');assert not v['native_execution_gate_passed']
    print(datetime.now(timezone.utc).isoformat(),'archive failed conditional reference, no reoptimization',flush=True)
    for folder in ['search','search_stage','verify','verify_stage']:
        shutil.copytree(t/folder,p/folder)
    state=load(t/'verify/PROPOSED_STATE.json');bundle=Path(plan['bundle']);rid=state['room_geometry']['room_id'] if 'room_id' in state['room_geometry'] else load(bundle/'manifest.json')['selected_room_id']
    # Explanatory post-result cross-section. Not a new acceptance criterion.
    y=-1.5;blocked=[]
    for b in obstacles(state,bundle/('room_'+rid),bundle,1.8):
        if b['lo'][1]<y<b['hi'][1]:blocked.append(dict(object=b['id'],interval=[max(-1.75,b['lo'][0]),min(1.75,b['hi'][0])]))
    blocked=[b for b in blocked if b['interval'][1]>b['interval'][0]];merged=[]
    for lo,hi in sorted(b['interval'] for b in blocked):
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    cursor=-1.75;free=[]
    for lo,hi in merged:
        if lo>cursor:free.append([cursor,lo])
        cursor=max(cursor,hi)
    if cursor<1.75:free.append([cursor,1.75])
    cut=dict(y_m=y,obstacles=blocked,free_intervals=free,largest_free_width_m=max(hi-lo for lo,hi in free),
        selection='Post-result explanatory horizontal cut; not a prespecified metric.',limits='Conservative native AABB obstacles across body height. Does not prove human physical impossibility or original-task infeasibility.')
    write(p/'BOTTLENECK.json',cut)
    sr=load(t/'search_stage/receipt.json');vr=load(t/'verify_stage/receipt.json');assert sr['exit_code']==vr['exit_code']==0
    summary=dict(run_id=p.name,status='candidate_rejected_due_to_disconnected_access',search_sequences_completed=1,search_sequences_planned=1,
        planned_native_actions=14,actual_native_actions=0,model_calls=0,training_updates=0,
        geometry_candidate_roles=14,envelope_conflicts=len(v['envelope_overlaps']),outside=len(v['outside']),route_results=v['access_counts'],
        native_execution_gate_passed=False,solver_exit_code=sr['exit_code'],verification_exit_code=vr['exit_code'],
        solver_stage_seconds=sr['elapsed_seconds'],verification_stage_seconds=vr['elapsed_seconds'],
        original_task_feasible='unknown',original_task_infeasible='not_established',professional_positive_demonstration=False,
        training_admitted=False,sealed_test=False,task_evaluation_ready=False,
        decision='Do not execute candidate or launch policy evaluation. Preserve failed search; no retries or shape/asset edits in this run.',
        next_step='Return to professional-decision/data and fair-study design. Moabit remains a prepared, exposed input with feasibility unresolved; do not make its layout optimization a prerequisite for all research.')
    write(p/'SUMMARY.json',summary)
    text=f'''# Moabit 全功能角色候选未通过通路检查\n\n一次预定预算的几何搜索给出14件家具候选，保留床、沙发、冷藏/操作台/水槽/炉灶、书架/箱、桌和两椅、淋浴/洗面盆/厕所。实际目录尺寸不缩放，固定结构/开口不改。\n\n独立读取原生包并构造**待执行状态**后，包围盒重叠0、越界0；它不是已经执行/导出的原生场景。\n\n| 方形通行包络 | 接近点局部未占用 | 从入口可达 | 总点数 |\n|---|---:|---:|---:|\n'''
    for row in v['access_counts']:text+=f"| {row['body_width_m']}m | {row['clear_points']} | {row['reachable_points']} | {row['planned_points']} |\n"
    text+=f'''\n最小0.4m包络只到达卫浴的4点，起居/睡眠/厨房/两餐位均不可达。事后解释截面y=-1.5m处，完整横截面最大连续空档约{cut['largest_free_width_m']:.3f}m；详见BOTTLENECK.json。它说明本候选的路径断开，不能证明真实住户不能通行或原始任务无解。\n\n搜索只约束局部接近方块，没有优化全屋连通；独立检查揭示了这一区别。搜索一次后停止，没有追加解、换资产、缩家具、删除功能或修改房壳。仅求解和几何检查，**0原生家具动作、0模型调用、0新训练**；不渲染已被拒绝的候选，不生成正向专业标签。\n\n搜索阶段{sr['elapsed_seconds']:.6f}秒/exit0；复核{vr['elapsed_seconds']:.6f}秒/exit0。进程成功不等于功能验收成功。入口/卫浴门全开、AABB、2cm网格和方形人体都是研究代理，非人体工学标准。安装、门扇隐私、容量、新艺术风格和专业整体评价仍未知。\n\n本源初态保持evaluation_ready=false、sealed_test=false。三个预定形状变体仍只有一个项目家族；本次只检查名义壳。这个负结果不进入训练/检索，不拿它评价模型能力。停止把这一间房的排布求解扩成研究主线，后续以有来源决定数据及公平学习对照为主。\n\n## 证据与恢复\n\n- SEARCH_PLAN.json：搜索前冻结代码/输入哈希、单次180秒上限（实际提前结束）、角色、假设和目录。\n- search/RESULT.json：唯一候选及合法工具命令的**拟执行**参数。\n- VERIFY_PLAN.json：独立复核代码/输入；最小0.4m通路执行门槛在搜索后、复核前声明，未伪称完全预注册。\n- verify/PROPOSED_STATE.json、CONDITIONAL_ACCESS.json、RESULT.json：待执行状态、逐点测量、拒绝原因。\n- search_stage/、verify_stage/：真实命令、启动/结束/首日志/退出。\n- ../P920-MOABIT-INPUT-v1_20261105T213853Z/RECOVERY.md：原生输入及未缩放网格的服务器恢复。\n\n公开小文件用于审阅。源网格、目录原生包留服务器；没有新的checkpoint或模型权重。复核可写到新的独立输出目录；不重新运行搜索并覆盖已有结果。原两组各2优化更新的许可已经用完，本轮没有消耗或重新解释那项批准。\n'''
    (p/'README.md').write_text(text)
    status=load(p/'STATUS.json');now=datetime.now(timezone.utc).isoformat();status.update(status=summary['status'],updated_at=now,started_at=sr['started_at'],ended_at=vr['ended_at'],actual_exit_code=0,first_log=sr['first_log'],completed=1,planned=1,failure_reason='Functional candidate rejected: only 4/14, 1/14, 0/14 approach points reachable for 0.4/0.5/0.6m; not a process failure.',latest_verifiable_artifact=str(p/'SUMMARY.json'),next_step=summary['next_step'])
    write(p/'STATUS.json',status);write(p.parent/'latest.json',dict(run_id=p.name,stage=status['status'],updated_at=now,run_directory=str(t)))
    write(p/'ARCHIVE_CHECKS.json',dict(source_receipts_terminal=True,input_manifest_unchanged=sha(bundle/'manifest.json')==plan['manifest_sha256'],search_code_unchanged=sha(Path(__file__).parent/'search_moabit_reference.py')==plan['code_sha256'],copied_results_exact=sha(p/'verify/RESULT.json')==sha(t/'verify/RESULT.json'),no_native_checkpoint=not(t/'native/checkpoint').exists()))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
