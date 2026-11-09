"""Render the measured missing-decision objective study; no experiments run."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();root=a.root
    d=json.loads((root/'actual_v1/summary.json').read_text())
    r=json.loads((root/'reference_check.json').read_text())
    rows=''.join(f"<tr><td>{x['K']}</td><td>{x['enumerated_ordered_samples']}</td><td>{x['max_error_to_marginal']:.6f}</td></tr>" for x in d['rws_finite_K_bias'])
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>缺少理由标注时，训练目标到底是什么</title>
<style>body{{font:17px/1.8 system-ui,sans-serif;max-width:1060px;margin:38px auto;padding:0 22px;color:#17202b}}h1,h2{{line-height:1.4}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cad2dc;padding:8px;text-align:left}}code,pre{{background:#edf2f7}}pre{{white-space:pre-wrap;padding:15px}}a{{color:#1554a0}}.note{{border-left:4px solid #637f9f;padding:14px;background:#f2f5f9}}</style>
<h1>14 个没有理由标注的单元：保留数据，明确学习目标和估计器</h1>
<p class="note">本轮交付了可执行的两类普通潜变量损失、全部 24 个真实单元的目标路由、精确 CPU 梯度核验及 LaTRO 官方源码对照。没有生成新理由标签、调用语言模型、更新模型参数或生成新房间。它修正下一轮实验的算法定义，尚不证明三维设计收益，也不授权新训练。</p>
<h2>1. 为什么继续这一项</h2>
<p>上轮 Git 复核没有取得新模型结果。协调聊天最新用户消息说明 920 暂时不可用，因此没有把反复连接它当成继续工作的前提；Git 中 LEARN 只有启动快照，终态仍未确认。当前可直接处理的问题是：配对策略运行时要求先生成 D 再行动，但 24 个单元里只有 10 个有 D 标注。其余 14 个原来仍按不带 D 的动作预测训练。输入去重不能消除这个支持范围差异。</p>
<p>本次没有删除这 14 个单元，也没有从目标动作反写“正确理由”再冒充专业真值。读取上一轮规范化后的同一材料，逐条验证分开组与配对组的原始目标、系数、来源和可观察输入一致。完整原始/等价视图仍只有 9 个单元可用；缺失图像不在本轮被悄悄替换。</p>
<table><tr><th>实际材料</th><th>数量</th><th>本轮处理</th></tr>
<tr><td>有理由标注的单元</td><td>10</td><td>保留现有理由及动作监督；包括两个停止样本，不统一称为专业行动示范。</td></tr>
<tr><td>无理由标注的单元</td><td>14</td><td>建立潜变量动作学习的可选路由；没有新增标注。</td></tr>
<tr><td>原始目标 token</td><td>{d['action_target_tokens']} 动作，{d['reason_target_tokens']} 理由</td><td>保留阶段拆分后的原 token；数量不同于旧 LEARN 的拆分，不混用两版。</td></tr>
<tr><td>模型调用／参数更新／新场景</td><td>0／0／0</td><td>有限概率表和真实材料索引不算模型实验。</td></tr></table>
<h2>2. 两个已经实现、但不等价的目标</h2>
<p>X 是当前场景和要求，D 是模型提出的决策记录，A* 是已有参考动作。参考动作只是一个监督目标，不代表全部合法摆法，更不自动代表最佳设计。对无理由的单元，已有方法提供两种不同选择：</p>
<pre>边际动作目标： Lm = −cA log ΣD pθ(D|X) pθ(A*|X,D)
期望对数目标： Le = −cA E[D~pθ] [log pθ(A*|X,D) − β log(pθ(D|X)/p0(D|X))]</pre>
<p>前者可用自归一化重要性采样估计梯度；后者对应 LaTRO 的理由策略梯度加动作学习项。两者都属于普通方法，不是新损失。代码分别实现 exact_marginal_nll、rws_surrogate、exact_expected_log_nll、latro_rloo_surrogate，API 文档明确样本分布、概率归一化、停止梯度和系数要求。</p>
<p>二者的差异有实际含义：在固定二选一理由概率下，动作成功概率为 [0.99,0.01] 的策略平均概率是 0.5；[0.4,0.4] 的策略是 0.4。边际目标偏好前者，期望对数目标偏好后者。这个例子只说明目标不能混用，不是实测三维结果。</p>
<h2>3. 真正运行的数值检查</h2>
<p>在三个理由的有限分布中，逐一穷尽所有有序抽样结果，包含重复样本，不挑选“好理由”。理由先验为 [0.8,0.15,0.05]，对应动作概率为 [0.01,0.2,0.8]。自归一化边际梯度的最大绝对误差如下：</p>
<table><tr><th>K</th><th>穷尽抽样组合</th><th>相对精确边际梯度的误差</th></tr>{rows}</table>
<p>K=1 从当前先验采样时，理由分布参数的期望梯度为零，而真实边际目标的该部分梯度不为零。因此不能为了省算力，只采一个理由、把它当真值训练，便宣称实现了边际学习。有限 K 的偏差也不能由“公式能反向传播”排除。</p>
<p>LaTRO 核心估计器在 K=2、3、4 和 β=0、0.1 的穷尽计算中，与其自身期望对数目标的精确梯度一致；另外验证了理由与动作共享同一参数向量的情形，误差 {d['shared_parameter_gradient_error']:.3g}。这不是对边际目标的无偏性结论。边际梯度有限差分误差 {d['finite_model']['finite_difference_error']:.3g}。将全部 14 个真实动作的原系数用于有限模型求和，误差 {d['real_corpus_weighted_gradient_error']:.3g}；没有按 K 重复增加目标权重。</p>
<p>主计算耗时 {d['wall_seconds']:.3f} 秒，峰值 RSS {d['peak_rss_kib']/1024:.1f} MiB，CUDA 未初始化。没有优化器，也没有有限表“训练到成功”的过程。</p>
<h2>4. 官方实现核对改变了复用方式</h2>
<p>固定 LaTRO 源码提交 <code>{html.escape(r['commit'])}</code>。完整读训练主循环、配置和启动入口，定向读概率计分与 padding 处理，并核了示例命令。其损失为 <code>{html.escape(r['loss_expression'])}</code>；配置默认 <code>sft_penalty=0</code>，公开的活动示例命令没有覆盖它。仅抽取这个标量表达式，在 CPU 上验证默认设置确实移除了显式动作似然梯度；系数为 1 时该项恢复。</p>
<p>这意味着复用时必须显式指定目标及系数，不能把默认入口自动说成论文式 (4) 的两项联合更新。共享参数仍可能间接改变动作概率；本轮没有核实作者实际实验命令或权重，不能由这个默认设置推翻论文结果。第三方源码未修改，Apache-2.0 许可证随原文件保留。</p>
<h2>5. 接回三维生成时，哪些步骤已经具体，哪些仍缺</h2>
<ol><li>对有标注单元沿用普通监督；对无标注单元，先从当前观测采样理由，再在同一固定观测上计算已有动作的条件概率。目标动作不能出现在理由生成提示中。</li>
<li>理由生成阶段和动作阶段使用不同的显式任务前缀；概率必须按各自上下文计算。选择期望对数目标时，同时保留理由的 RLOO 项和动作梯度。</li>
<li>原始单位与项目系数保持不变。K=2 时，全体单元一遍需要 28 条新理由、38 个动作评分、38 个理由评分；β&gt;0 另需 28 个参考理由评分。这是逻辑序列数，未测得 GPU 时间或内存，不能称等计算量。</li>
<li>运行时用模型自己的理由和动作，经同一原生工具执行、刷新视图，再检查原始要求。教师前缀似然提高不能替代自主轨迹或完整房间质量。</li></ol>
<p>仍须补齐 15 套视图，并明确畸形、截断理由的概率与处罚处理。本轮正有限概率内核会拒绝零/非有限输入；不会丢弃失败抽样以制造漂亮估计。实际多模态采样、评分及 Trainer 集成尚未执行。当前数值代码不是已经可启动的 GPU 训练入口。</p>
<h2>6. 对研究方向的约束</h2>
<p>本轮增加的是可复用的普通对照与明确的训练定义。新比较不能再称为“仅多了一个理由前缀”：处理 14 个缺失理由单元也改变了目标。应保留旧的部分条件化组，同材料辅助理由组，以及同知识基础模型与普通搜索；最后比较独立项目中原始要求、保留功能、完整度和成本。</p>
<p>三维贡献仍需来自用途与真实空间条件下的设计取舍，而不是潜变量命名。源码和理论整理已避免两种可预见的训练混淆；没有证明该机制优于普通方法。当前不请求资源，不扩大训练，不以这份数值结果宣布 CVPR 成果。</p>
<h2>来源与可复查产物</h2>
<p><a href="https://arxiv.org/html/1406.2751v4">RWS §§2.2–2.3</a>；<a href="https://arxiv.org/html/2411.04282v2#S4">LaTRO §4</a>；<a href="https://arxiv.org/html/2312.02179v1">TRICE §2</a>。阅读范围和未完成项见 <a href="reading_ledger.json">阅读记录</a>。</p>
<p><a href="actual_v1/summary.json">实际数值结果</a> · <a href="actual_v1/unit_routes.json">24 单元路由</a> · <a href="reference_check.json">官方标量损失核查</a> · <a href="ALGORITHM.json">完整算法定义与边界</a> · <a href="source/manifest.json">固定源码清单</a> · <a href="COMMANDS.txt">复查命令</a></p></html>'''
    (root/'stage_report.html').write_text(text)
    summary={'previous_goal_turn':'no_progress_for_algorithm: completed Git status reply restated already-known research evidence',
       'this_goal_turn':'progress: executable missing-label objectives, exact finite estimator comparison and newly verified reference configuration difference',
       'units_retained':24,'invented_reason_targets':0,
       'new_loss_kernels':['exact_marginal_nll','rws_surrogate','exact_expected_log_nll','latro_rloo_surrogate'],
       'ordinary_methods_not_novel':True,'model_calls':0,'optimizer_updates':0,'new_scenes':0,
       'training_ready':False,'gpu_requested':False,'source_files_read_or_inspected':8,
       'legacy_material_sha256':d['source_sha256'],
       'meaningful_next_work':'Define bounded sampled-reason handling and native scoring without changing task/observations, then complete the missing views before any admitted model experiment.',
       'full_goal_complete':False}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    sources=[Path('scripts/autodl/missing_decision_objective.py'),
             Path('scripts/autodl/check_missing_decision_objective.py'),
             Path('scripts/autodl/check_latro_reference.py'),Path(__file__)]
    paths=sorted({p.resolve() for p in root.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
                 | {p.resolve() for p in sources})
    manifest=[{'path':str(p.relative_to(Path.cwd())), 'bytes':p.stat().st_size,
               'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    (root/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest)}))


if __name__=='__main__':main()
