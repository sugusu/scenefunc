"""Recompute published research progress from pinned Git receipts, not STATUS labels."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-ref', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    def blob(path):
        raw = subprocess.check_output(['git','show',a.source_ref+':'+path])
        hashes[path] = hashlib.sha256(raw).hexdigest()
        return raw.decode()
    train = json.loads(blob('runs/920/P920-STOPDATA-v1_20261107T102628Z/TRAINING_RESULT.json'))
    eval_rows = list(csv.DictReader(io.StringIO(blob('runs/920/P920-EVALFIX-v1_20261107T130916Z/LEARNING_RESULTS.csv'))))
    matched = json.loads(blob('runs/920/P920-MATCHEDROLL-v1_20261108T062940Z/SUMMARY.json'))
    latest = json.loads(blob('runs/920/P920-LEARN-v1_20261108T075646Z/START_RECEIPT.json'))
    latest_plan = json.loads(blob('runs/920/P920-LEARN-v1_20261108T075646Z/TRAIN_PLAN.json'))
    loader = blob('scripts/920/train_equal_material.py')
    assert "(arm != 'action_only' or r['component'] == 'action')" in loader
    aggregate = {}
    for arm in sorted({r['arm'] for r in eval_rows}):
        group = [r for r in eval_rows if r['arm']==arm]
        aggregate[arm] = dict(n=len(group), activity_pass=sum(r['activity_status']=='pass' for r in group),
            gain_without_new_harm=sum(r['scoped_improvement_without_new_loss']=='True' for r in group),
            whole_room_unknown=sum(r['whole_room_success']=='unknown' for r in group))
    matched_groups = {}
    for arm in sorted({r['arm'] for r in matched['rows']}):
        group = [r for r in matched['rows'] if r['arm']==arm]
        ready = [r for r in group if r['transition']['initial_status']=='pass']
        deficit = [r for r in group if r['transition']['initial_status']=='fail']
        matched_groups[arm] = dict(ready_n=len(ready), deficit_n=len(deficit),
            ready_preserved_and_stopped=sum(r['trajectory_gate_pass'] for r in ready),
            deficit_completed=sum(r['trajectory_gate_pass'] for r in deficit))
    transfer = json.loads(Path('runs/autodl/20261110_transfer_controls/summary.json').read_text())
    record = dict(observed_at=datetime.now(timezone.utc).isoformat(), source_ref=a.source_ref,
        current_autodl_parent=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes=hashes,
        completed_training={k:dict(exit_code=v['exit_code'], updates=v['training']['optimizer_updates'],
            changed_adapter_tensors=v['training']['changed_adapter_tensors']) for k,v in train['arms'].items()},
        eval_recomputed=aggregate, matched_recomputed=matched_groups,
        eval_cost=dict(trajectories=len(eval_rows),model_calls=sum(int(r['model_calls']) for r in eval_rows),
                      native_calls=sum(int(r['native_calls']) for r in eval_rows)),
        latest_published_start=dict(observed_at=latest['observed_at'],exit_state=latest['exit_state'],
            updates_at_snapshot=latest['micro_progress_at_observation']['optimizer_updates'],
            backward_at_snapshot=latest['micro_progress_at_observation']['backward_completed']),
        latest_actual_training_components=latest_plan['components'],
        latest_plan_is_equal_material_comparison=False,
        latest_loader_verified='action_only selects separate action components only, discarding its decision components; separate vs paired is supported but not the started arm_order.',
        actual_920_live_state='not inspected; published running snapshot is not evidence of current execution',
        material_counts=transfer['data_counts'], conditioning_counts=transfer['conditioning_counts'],
        recommendation='Continue one bounded equal-material conditioning study; do not expand training or claim CVPR-level evidence.',
        scope='Recomputed existing Git metrics. No new model/training/complete-scene experiment.')
    (a.out/'git_progress_review.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    tr=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["activity_pass"]}/{v["n"]}</td><td>{v["gain_without_new_harm"]}/{v["n"]}</td></tr>' for k,v in aggregate.items())
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Git 进展与投入判断</title>
<style>body{{max-width:1000px;margin:40px auto;padding:0 24px;font:17px/1.8 system-ui;color:#20344b}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #ccd4dc}}a{{color:#075d97}}</style>
<h1>值得继续一次机制验证，还不值得扩大训练</h1>
<p>实际核对 runs/920 固定提交 <code>{a.source_ref}</code>、训练终态、逐轨迹 CSV、MATCHEDROLL 和最新启动回执。研究已超过概念方案阶段：有真实适配器更新、实际模型输出和原生工具后果；但尚无稳定的完整场景收益。</p>
<p>STOPDATA 两适配器各完成 8 次更新、各 144 个张量改变，正常退出。它比较动作监督与额外理由监督，不能用它单独证明相同材料下的理由条件机制。</p>
<table><tr><th>EVALFIX 条件</th><th>活动检查通过</th><th>改善且不新增损害</th></tr>{tr}</table>
<p>共 12 条轨迹、72 次模型调用、56 次原生调用。4 次/组不是 4 个独立项目，整屋成功均未知。MATCHEDROLL 的合格起点保持并停止：底座 1/2、动作组 1/2、理由动作组 2/2；需要修复的起点各 0/2。该信号值得保留，但来自一个训练曝光项目，且有解析流程敏感性，不能当泛化收益。</p>
<p>最新 LEARN 在 Git 中仍只有 11 月 8 日启动快照：5 次反向、0 次更新、未记录最终退出。这里没有连接另一台 920 机器核验，不能说它现在仍在运行、已经成功或失败。重复训练前，应找回原运行的最终回执。</p>
<p><strong>名称与实际配置须分开：</strong>LEARN 的 purpose 写 equal_material，但 arm_order 是 action_only / paired。实际计划为前者 24 组件、1,643 动作 token、0 理由 token；后者 32 组件、相同 1,643 动作 token 加 1,332 理由 token。加载器明确把 action_only 的理由组件过滤掉。这轮即使有提升，也不能单独归因于“显式理由条件”。真正同材料的 separate / paired 已在新分阶段代码中准备，但没有新训练结果。</p>
<h2>当前最有价值的修正</h2>
<p>双方都学相同场景下的理由和动作，核心差别是动作预测时是否显式接收刚生成的理由。它检验“理由的条件作用”，不是把一组正确对应与另一组故意错误对应比较。保留同知识底座、充分反馈与可计算普通规则，才能排除信息和工程优势。</p>
<p>当前 24 单元来自 6 家族；10 个动作有理由前缀监督，14 个没有。8 个有来源应用对应 5 条原则，同一原则尚无跨家族监督应用。9/24 单元已恢复完整或证明等价的视觉输入，其余仍缺本地完整证据。当前所有家族都是开发材料，不能把新视角或同源模板算新项目。</p>
<h2>我支持继续做什么</h2>
<ol><li>先恢复已有 LEARN 的最终证据，明确本轮真正可用的输入和图像，不重复盲训。</li>
<li>以相同材料、工具、资产、观察和累计预算比较：同知识底座／普通规则、辅助理由监督但直接动作、显式理由条件动作。模型必须自行生成理由；后续观察必须来自它自己的实际场景。</li>
<li>先做固定的小批完整任务轨迹，按项目来源与结构模板隔离，记录要求完成、旧要求损坏、未知项及总成本。当前原生宿主接入只提供执行基础；家具增删、脚本 STOP 和渲染成功不算整屋生成成绩。</li></ol>
<p>若提升只在解释 loss、格式合法或已合格场景停止上出现，不扩大当前学习方案；若在独立完整任务中能比强对照更好地完成要求，或同质量明显省总成本，再扩大数据与训练。即使取得此信号，理由条件预测本身仍有近邻，必须把贡献落在三维条件与设计决策的迁移及实际后果。</p>
<p>当前不请求 GPU，也不新增付费 API。新执行接口的实际验证与两次渲染失败另记，不混入旧研究成绩。</p>
<p><a href="git_progress_review.json">本轮逐项重算</a> · <a href="stage_report.html">CPU 原生宿主执行报告</a> · <a href="../20261110_phase_runtime/reading_ledger.json">前轮最近邻阅读范围</a> · <a href="../20261110_transfer_controls/summary.json">材料与来源划分审计</a></p></html>'''
    (a.out/'git_progress_review.html').write_text(page)
    print(json.dumps(dict(eval=aggregate,matched=matched_groups),ensure_ascii=False))


if __name__=='__main__':main()
