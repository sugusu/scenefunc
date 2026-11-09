"""Package an actual CPU measurement and scoped Git/source review.

No inference, training, mesh mutation, or edits to historical experiments.
The geometric domains are read from the completed measurement, not generated
again by this reporting script. The old incomplete label table stays intact.
"""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args, cwd):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--sceneteract', type=Path, required=True)
    args = p.parse_args()
    repo = args.repo.resolve()
    out = repo / 'runs/autodl/20261110_joint_representation'
    current = read(out / 'analysis/result.json')
    summary = read(out / 'measured/summary.json')
    old = read(repo / 'runs/autodl/20261110_objective_source_review/calculation/target_scope.json')
    old_git = read(repo / 'runs/autodl/20261110_consequence_review/git_progress_review.json')
    s = current['summary']

    # Complete only previously missing primitive A labels for the fixed eligible
    # pool. Geometry-rejected candidates retain unknown primitive functionality.
    labels = {row['candidate']: row for row in current['rows']}
    merged, added = [], []
    for row in old['rows']:
        row = dict(row)
        candidate, condition = row['candidate'], row['condition']
        before = row['primitive_function_target']
        if candidate in labels:
            val = labels[candidate]['separate_witness' if condition == 'A' else 'joint_witness']
            if before is not None and before != val:
                raise ValueError('Measured functionality disagrees with old known label')
            row['primitive_function_target'] = val
            row['conjunction_target'] = val  # Fixed geometry-eligible pool only.
            row['new_measurement_available'] = before is None
            if before is None:
                added.append({'candidate': candidate, 'condition': condition, 'label': val})
        else:
            row['new_measurement_available'] = False
        merged.append(row)
    counts = {c: {key: sum(r['condition'] == c and r['conjunction_target'] is value for r in merged)
                  for key, value in [('positive', True), ('negative', False), ('unknown', None)]}
              for c in ('A', 'B')}
    if len(merged) != 98 or len(added) != 11:
        raise ValueError('Unexpected coverage change')
    transition = {
        'old_report_unchanged': True, 'additional_primitive_measurements': added,
        'all_fixed_pool_conjunction_labels': counts,
        'old_all_pool_identity_oracle': {'correct': 86, 'known': 87},
        'new_all_pool_identity_oracle': {'correct': 86, 'known': 98},
        'new_eligible_pool_identity_oracle': {'correct': s['condition_blind_candidate_id_oracle_correct'],
                                              'known': s['condition_blind_candidate_id_oracle_denominator']},
        'primitive_known': sum(r['primitive_function_target'] is not None for r in merged),
        'primitive_unknown': sum(r['primitive_function_target'] is None for r in merged),
        'warning': 'Identity oracle uses labels. Neither it nor the posthoc support rule is a measured learned policy.'}
    write(out / 'label_coverage_update.json', transition)
    with (out / 'merged_label_table.csv').open('w', newline='') as f:
        columns = ['candidate', 'condition', 'primitive_function_target', 'conjunction_target',
                   'new_measurement_available', 'whole_room_target']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in merged)

    source_root = args.sceneteract.resolve()
    source_pin = git('rev-parse', 'HEAD', cwd=source_root)
    if source_pin != '03922d997ab97d6dab68e4490d040204181c3cbf':
        raise ValueError('Unexpected official SceneTeract source version')
    source_rows = []
    coverage = {
        'src/sceneteract/grpo/rewards.py': 'Full file, including answer parsing, asymmetric correctness, format and keyword bonus; optional length function.',
        'src/sceneteract/grpo/dataset.py': 'Atomic record construction, first failed property, image availability, scene-level 80/20 split and preparation.',
        'src/sceneteract/grpo/train.py': 'TrainingConfig, _rebalance, rebalance condition and configured reward functions; not the full trainer stack.',
        'src/sceneteract/grounding/verifier.py': 'Plan execution, target resolution defaults and agent position/posture updates; not every property implementation.'}
    for relative, read_scope in coverage.items():
        path = source_root / relative
        committed = subprocess.check_output(['git', 'show', f'{source_pin}:{relative}'], cwd=source_root)
        match = hashlib.sha256(committed).hexdigest() == sha(path)
        if not match:
            raise ValueError('Official working source differs from pin')
        source_rows.append({'path': str(path), 'sha256': sha(path), 'matches_git_commit': match,
                            'read_scope': read_scope,
                            'url': f'https://github.com/leopoldmaillard/sceneteract/blob/{source_pin}/{relative}'})
    write(out / 'source_index.json', {'repository': 'https://github.com/leopoldmaillard/sceneteract',
                                      'pin': source_pin, 'files': source_rows,
                                      'code_execution': 'Not executed; no model, training, property engine or external API call.'})
    ledger = [
        {'title': 'SceneTeract', 'url': 'https://arxiv.org/html/2603.29798v1',
         'read': ['3.1–3.2 task-agent-scene and atomic verification', '4.3–4.4 VLM evaluation and training',
                  '5 conclusions/limitations', 'D.1–D.4 training, prompts, rewards and overoptimization',
                  'selected Appendix A execution description'],
         'source': 'Four local official source files pinned in source_index.json',
         'finding': 'Geometrically grounded functional labels and VLM post-training already exist. Static scene objects with agent posture updates are not full scene regeneration.',
         'not_done': ['No complete paper/figure audit', 'No dataset/model download', 'No independent training or verifier execution']},
        {'title': 'Human-Aware 3D Scene Generation with Spatially-constrained Diffusion Models (SHADE)',
         'url': 'https://arxiv.org/html/2406.18159v1',
         'read': ['Abstract/introduction/related work', 'Method: formulation, conditional diffusion and guidance, equations 1–6',
                  'Dataset calibration subsection text'],
         'finding': 'Conditions on all supplied contact humans, motion free-space and floor plan; jointly generates objects with collision guidance.',
         'not_done': ['No full experiment/appendix read', 'No code audit', 'No reproduction']},
        {'title': 'Functional 3D Scene Synthesis through Human-Scene Optimization',
         'url': 'https://arxiv.org/html/2502.06819v1',
         'read': ['Abstract/introduction/related work', '3.1–3.3 and Algorithm 1', 'Table 1 and experiment introduction'],
         'finding': 'Text-to-graph, retrieved human poses and functional-group-aware layout adjustment already exist; Algorithm 1 can discard intersecting objects outside the functional group.',
         'not_done': ['No full experiments/appendix or implementation audit', 'No reproduction', 'No estimate of how often removal affects required objects']},
        {'title': 'Prior source and objective review',
         'local_record': '../20261110_objective_source_review/reading_ledger.json',
         'read': 'Reuse the explicitly scoped Scene-Designer, Haisor, 3D-Layout-R1, Function2Scene and pinned 920 evidence; no new full read claimed.'}]
    write(out / 'reading_ledger.json', ledger)

    # Analytical decision boundary, NOT a neural training result. Format rewards
    # cancel only for equally valid formats; keyword rewards are intentionally
    # excluded. Resampling derivation assumes unchanged class-conditionals.
    q_threshold = Fraction(2, 5)
    prior_old, prior_train = Fraction(1, 4), Fraction(2, 5)
    odds_multiplier = (prior_train / (1-prior_train)) / (prior_old / (1-prior_old))
    p_threshold = q_threshold / (odds_multiplier * (1-q_threshold) + q_threshold)
    write(out / 'objective_theory.json', {
        'source_reward_correct_infeasible': 3, 'source_reward_correct_feasible': 2,
        'correctness_only_failure_posterior_threshold': str(q_threshold),
        'assumed_original_failure_prior': str(prior_old), 'assumed_rebalanced_failure_prior': str(prior_train),
        'class_prior_odds_multiplier': str(odds_multiplier),
        'implied_original_distribution_threshold': str(p_threshold),
        'equations': ['choose infeasible iff 3q > 2(1-q)', 'q=2p/(1+p)', 'equivalent p>1/4'],
        'assumptions': ['Equal format rewards; exclude keyword and optional length bonus',
                        'Only class prior changes; class-conditionals fixed',
                        'Ideal expected correctness reward maximization, not a claim about finite GRPO dynamics'],
        'source_default_target_false_ratio': 0.0,
        'interpretation': 'Failure-sensitive utility is not a calibrated probability. The paper sampling recipe requires an explicit configuration; it is not the default source setting.',
        'our_920_sft_objective_is_different': True, 'training_executed': False})

    heads = {}
    for line in git('ls-remote', 'origin', 'refs/heads/main', 'refs/heads/runs/920', 'refs/heads/runs/autodl', cwd=repo).splitlines():
        digest, ref = line.split()
        heads[ref] = digest
    if heads['refs/heads/runs/920'] != old_git['remote_heads']['runs/920']:
        raise ValueError('920 has new published evidence; review it before finalizing')
    snapshot = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'remote_heads_before_this_commit': heads,
                'local_parent_commit': git('rev-parse', 'HEAD', cwd=repo),
                'remote_920_live_process_accessed': False,
                'status_scope': 'Only published files. No inference about the current live process or unpublished results.'}
    write(out / 'git_snapshot.json', snapshot)
    for path, digest in summary['source_hashes'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Historical measurement source changed')

    table = ''.join(f'<tr><td>{html.escape(r[0])}</td><td>{r[1]}</td><td>{html.escape(r[2])}</td></tr>' for r in [
        ('各项活动分别存在可用位置', '21/21', '两个矩形分别能够放下，不要求同时成立'),
        ('两项活动同时存在兼容位置', '9/21', '固定离散位置与朝向下的水平不相交'),
        ('仅用“各自可用”预测同时可用', '9/21 正确', '12 个状态需要组合信息'),
        ('普通精确几何查询', '21/21 正确', '与完整笛卡尔积枚举及旧求解器一致'),
        ('事后“不同支撑面”规则', '21/21 正确', '看过数据后形成，只解释此例，不作为泛化成绩')])
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Git 进展与联合用途 CPU 实测</title>
<style>body{{font:17px/1.75 system-ui,sans-serif;max-width:1080px;margin:45px auto;padding:0 24px;color:#1c2630;background:#fafafa}}h1,h2{{line-height:1.35}}h2{{margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:10px;border:1px solid #ccc;text-align:left}}code,pre{{background:#eee}}pre{{padding:16px;white-space:pre-wrap}}.note{{border-left:4px solid #557b96;padding:12px 20px;background:#eef4f7}}a{{color:#245e89}}</style>
<h1>值得继续验证，但尚不值得扩大训练</h1>
<p>Git 核查时间（UTC）：{html.escape(snapshot['checked_at_utc'])}。本报告把已发布训练、当前 CPU 实测和待检验方法分开。没有新增付费 API、模型推理、训练或 GPU 使用；旧实验只读。</p>
<p class="note">当前项目已经有真实场景、可执行工具和可复核的训练对照，超过了只有想法与伪代码的阶段。但尚无稳定证据证明“加入推理／历史／用途判断”比合理普通方法更好。此次普通几何对照全部正确，进一步收紧了值得训练的范围。</p>
<h2>1. Git 到了哪一步</h2>
<p><code>main={heads['refs/heads/main'][:7]}</code>；<code>runs/920={heads['refs/heads/runs/920'][:7]}</code>；本报告提交前 <code>runs/autodl={heads['refs/heads/runs/autodl'][:7]}</code>。主分支并未包含所有实验，必须分别看实验分支。</p>
<table><tr><th>证据</th><th>已实现</th><th>目前能得出的结论</th></tr>
<tr><td>920 EVALFIX</td><td>4 个起点；原模型活动代理 3/4，动作训练 2/4，理由＋动作 2/4；无新损害的改善均 0/4</td><td>没有稳定正收益；代理并非完整使用与场景质量</td></tr>
<tr><td>920 MATCHEDROLL</td><td>一个训练已见家族，4 个起点×3 组，35 次模型调用；缺陷起点均 0/2 修复</td><td>原本合格的起点，理由组保持并合法结束 2/2，另两组各 1/2。样本少且存在解析／动作拒绝因素，不能当作泛化证明</td></tr>
<tr><td>同材料对照</td><td>separate 与 paired 各 32 个组件，24 个动作、8 个理由，6 个家族；内容已核对</td><td>最新计划仍是 action_only 与 paired，缺少真正隔离配对作用的 separate 实跑</td></tr>
<tr><td>最新 LEARN 发布记录</td><td>快照写明 5 次前向、5 次反向、0 次优化器更新</td><td>Git 中未见本轮最终结果；不能据此声称远端现在仍运行、已失败或已完成</td></tr>
<tr><td>AutoDL 当前增量</td><td>实际网格上的 21 个候选、42 条功能标签，普通精确强对照</td><td>是同一旧场景的诊断，不是 21 个房间、训练或新完整生成</td></tr></table>
<p>上述训练数据来源及限定见<a href="../20261110_consequence_review/git_progress_review.html">已发布训练审阅</a>和<a href="../20261110_objective_source_review/research_review.html">训练目标与源码审阅</a>。</p>
<h2>2. 新完成的 CPU 实测</h2>
<p>固定已有 Paris B 场景、20 个原先几何准入的沙发位移候选及保持原状。重新加载真实网格（9 件家具，35 个网格节点），核对实际桌面与存档，再逐状态构建可用区域。工作矩形 0.8×0.5 米、用餐矩形 0.6×0.4 米，沿用原来的 1 厘米网格和 0/90 度朝向；没有为了结果换样本或阈值。</p>
<table><tr><th>检查／方法</th><th>结果</th><th>含义</th></tr>{table}</table>
<p>全体状态的工作区域候选都是 400 个。12 个不兼容状态只能在办公桌上找到两种活动各自的位置；9 个兼容状态额外在餐桌上找到用餐区域。因此“不同活动需要联合分配位置”是实际测到的问题；但此池仍能由一个事后支撑面规则完全解释，不足以建立复杂学习方法的必要性。</p>
<p>重新测量补齐了过去 11 条未知的 A 条件标签。旧报告的 86/87 是旧覆盖下、使用答案的候选 ID 上限；补齐后，同一全池变为 86/98，几何准入子池是 30/42。分母和标签覆盖变了，旧结果没有被覆盖。两个条件下的差异不是人工偏好真值。</p>
<p>CPU 测量总计 {summary['total_seconds']:.3f} 秒，包含约 {summary['setup_seconds']:.3f} 秒网格准备；进程峰值 RSS {summary['peak_rss_kib']/1024:.1f} MiB。每状态构建空间候选中位数 {s['timings_seconds']['domain_construction']['median']:.3f} 秒；已有候选数组上的精确查询中位数 {s['timings_seconds']['new_extrema_query']['median']*1e6:.1f} 微秒。后者不包括网格准备、区域构建、渲染或模型成本，不能当作整个生成器加速。</p>
<p>固定最小移动代价时，分别使用条件可保持原状，同时使用条件可选 <code>preview_23</code>（沙发位移 0.2 米）。这是旧候选池上的普通约束选择，并未重新导出或执行该场景，也没有声称它比旧的几何优先目标更好。</p>
<h2>3. 普通强对照为何精确</h2>
<p>令 W、D 是已通过单项几何检查的两个矩形候选集合。分别可用只要求两者非空；同时可用还要求存在内部不相交的一对。对轴对齐矩形，内部不相交等价于四个方向至少一个分离。由于候选可独立选取，对应的存在性可以由端点极值判断：</p>
<pre>exists(w,d) with disjoint interiors  iff
min_d x_max(d) &lt;= max_w x_min(w), or
max_d x_min(d) &gt;= min_w x_max(w), or
min_d y_max(d) &lt;= max_w y_min(w), or
max_d y_min(d) &gt;= min_w y_max(w).</pre>
<p>必要性：任意可行对至少满足一条分离不等式，极值只会扩大该分离。充分性：取对应极值的两个实际候选就是证据。线性扫描 O(|W|+|D|) 后只需四次比较，并能返回具体候选。这是普通计算几何，不是新算法。本轮另用完整两两枚举核对全部候选，避免只让两个相似实现互相确认。</p>
<p>该结论限定为两个轴对齐、零厚度活动区域与当前水平互斥定义。三个以上任务可能两两都有可行分配却不能共同分配（例如三个任务争用两个互斥位置），应使用普通约束求解作对照；不能把本公式直接推广为完整人体活动、动态过程或人的设计偏好。</p>
<h2>4. 最近邻把创新边界限制在哪里</h2>
<p><a href="https://arxiv.org/html/2603.29798v1">SceneTeract</a>已有几何功能验证和 VLM 训练；论文限定静态场景，并提出闭环生成作为后续用途。故“几何反馈训练 Critic”本身已有直接近邻。</p>
<p><a href="https://arxiv.org/html/2406.18159v1">SHADE</a>同时接收接触人体、运动空闲区域与房间边界，联合生成对象并施加碰撞约束。“多项人体条件共同影响场景”不能作为我们的独有主张。</p>
<p><a href="https://arxiv.org/html/2502.06819v1">Functional 3D Scene Synthesis through Human-Scene Optimization</a>已有文本到场景图、人体姿态检索及功能组布局调整；其算法允许移除部分相交对象。我们必须保留必需实例分母，但这项评测原则本身也不构成新算法。</p>
<p>当前只做了这些原文的指定方法和训练章节阅读，未完成所有图、附录或实现复现；逐项范围见 <a href="reading_ledger.json">阅读记录</a>。Haisor 的学习价值与搜索、Scene-Designer 的分支执行与反馈等近邻仍沿用上轮核查，不宣布排重完成。</p>
<h2>5. 训练目标需要说清什么</h2>
<p>已核对 SceneTeract 官方固定提交的奖励、数据构造和训练配置。其正确性项对正确不可行给 3、正确可行给 2；若只考虑这一项，同等格式下应预测不可行当且仅当训练后验 q&gt;2/5。若仅将失败先验从 1/4 重采样为 2/5、类条件分布不变，则 q=2p/(1+p)，相当于原分布 p&gt;1/4。这个推导说明该奖励表达失败敏感效用，不是校准概率。</p>
<p>关键词奖励另外按字符串命中给分，不能单独证明推理在空间上正确。实际 GRPO 还受其他奖励、优化与有限样本影响；以上不是对其模型行为的测量。源码默认 <code>target_false_ratio=0.0</code>，论文采样设置需要显式启用。更不能把这套 GRPO 解释套给本项目 920 的 SFT 负结果。源文件与哈希见 <a href="source_index.json">固定源码</a>，推导假设见 <a href="objective_theory.json">目标分析</a>。</p>
<h2>6. 是否值得继续</h2>
<p><strong>值得保留完整三维生成研究和目前可复核的执行基础；不值得现在扩大当前样本上的 Critic 训练或宣称 CVPR 主创新成立。</strong>直接训练“同时可用”判定在本例缺少必要性：普通几何已准确，查询本身也很便宜。</p>
<p>下一次方法比较须定位到一个真实成本或错误来源：从场景证据构造活动可行域是否困难；多个要求的对象与空间分配是否让现有普通求解失效或成本过高；生成器是否提出了合适方案。先把这些因素与最后的布尔判断分开。同样精确候选给所有方法后，不能靠让普通对照只看两个布尔值来制造收益。</p>
<p>现有训练线最清楚的未决问题，是“同样的理由与动作材料，配对学习是否超过分开学习”。separate 已准备，paired 与 action_only 不足以单独回答它。先取得现有运行的完整结果、区分格式／执行失败与功能决策，再决定是否运行这个有边界的对照；不因 Git 暂缺结果就重复开新训练。</p>
<p>这些继续条件尚不是新方法成果。需要独立房间中的真实差异、同工具同信息的强对照，以及整场要求完整度与代价的改善，才能提升为论文主张。当前 CPU 工作已经完成；本轮没有提出新的 GPU 资源请求。</p>
<h2>7. 交付与边界</h2>
<p><a href="analysis/summary.csv">21 候选逐项结果</a> · <a href="merged_label_table.csv">98 条合并标签</a> · <a href="label_coverage_update.json">新增标签和分母说明</a> · <a href="measured/summary.json">实际测量与旧源哈希</a> · <a href="COMMANDS.txt">复跑命令</a> · <a href="verification_receipt.json">交付核验</a></p>
<p>网格碰撞准入沿用存档，未重跑 FCL；重新计算的是实际网格桌面平面区域及有限矩形候选。有限搜索失败不等于连续空间无解。未做人体实验、独立人工偏好、完整语义验收或新场景生成。旧源文件前后哈希相同。</p></html>'''
    (out / 'stage_report.html').write_text(doc)
    print(json.dumps({'report': str(out / 'stage_report.html'), 'added_labels': len(added),
                      'pinned_sources': len(source_rows), 'remote_heads': heads}, ensure_ascii=False))


if __name__ == '__main__':
    main()
