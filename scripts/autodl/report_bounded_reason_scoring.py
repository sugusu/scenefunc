"""Render the bounded-reason audit from actual saved numerical evidence."""
import argparse
import html
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();r=a.root/'actual_v2'
    def read(name):return json.loads((r/name).read_text())
    s=read('summary.json');f=read('finite_distribution.json');o=read('objective_selection.json')
    b=read('batch_dependent_penalty.json')
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>停止边界与失败样本：可执行训练目标检查</title>
<style>body{{max-width:1040px;margin:36px auto;padding:0 22px;font:17px/1.75 system-ui;color:#182332}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cad2db;padding:9px;text-align:left}}code,pre{{background:#eef1f4}}pre{{padding:14px;white-space:pre-wrap}}.scope{{background:#f1f5fa;padding:14px}}a{{color:#176a9c}}</style>
<h1>先保证失败样本的训练含义正确，再决定是否训练</h1>
<p>2026-11-10。继 f1b9124 的有限支持损失检查，本轮补上真实停止边界、无效理由和外部中断的处理。
已实现并执行 CPU 数值对照；没有语言模型推理、参数更新、新场景或旧运行修改。本轮发现使“可用于训练”的判断更严格，不能据此宣称三维方法新颖或完整场景质量提高。</p>
<p class="scope">新的辅助模块尚未接入生产采样器。现有 phase_policy.py、训练材料和模型条件均未修改。
当前同监督、不同动作条件化的实验仍是优先事项；这批潜变量函数仅是缺理由单元的普通备选对照，不自动替换原研究。</p>
<h2>这次新完成的内容</h2>
<table><tr><th>内容</th><th>实际证据</th><th>边界</th></tr>
<tr><td>停止概率</td><td>22 个完整停止结果，总概率 {f['full_probability']:.15f}；省去停止 token 后合计 {f['omit_actual_stop_probability_mass']:.6f}</td><td>有限自回归分布，非真实模型概率</td></tr>
<tr><td>失败保留的梯度</td><td>枚举全部 484 个 K=2 有序样本；期望梯度最大误差 {f['RLOO_expected_gradient_max_error']:.3g}</td><td>固定逐样本效用、同策略独立采样的条件下</td></tr>
<tr><td>删除失败样本</td><td>条件重归一化与原失败效用目标的梯度最大差 {f['drop_failed_samples_gradient_difference']:.6f}</td><td>说明目标改变，不是某个三维系统的实测偏差</td></tr>
<tr><td>已有自由输出</td><td>12 份全部保留，其中 {s['archived_text_without_closing_reason']} 份文本没有闭合理由标签</td><td>4 个开发单元×3 条件，缺原始停止 token 和 logits；没有伪造概率或重新定义旧执行成绩</td></tr>
<tr><td>资源</td><td>最后一轮 {s['wall_seconds']:.2f} 秒，RSS {s['peak_rss_kib']/1024:.1f} MiB</td><td>零模型调用、零更新、未初始化 CUDA；actual_v1 的 15.03 秒也保留</td></tr></table>
<h2>随机变量与停止规则</h2>
<p>令 X 为当前场景的五视图、状态、要求和工具定义；Z 是模型在最多 L 个 token 内实际生成的序列。
遇到第一个闭合理由 token 或原生 EOS 即停止；未遇到时在 L 截断。Z 包括实际产生的停止 token，截断时不追加 EOS。
所有叶子事件互斥且概率和为 1。有效性 v(Z) 由既有解析器决定，仅表示语法上可以进入动作阶段。</p>
<pre>pθ(Z | X) = ∏(t=1..|Z|) pθ(z_t | X, z_&lt;t)
v(Z) = 1: 闭合、无角色控制注入且符合理由格式
v(Z) = 0: 早停 EOS、未闭合截断、闭合但格式不合法
外部 SIGKILL / 时间中断 / 未知停止: 保留尝试与成本，不能当作固定 L 下的 IID 完整样本</pre>
<p>理由闭合后用于拼接的分隔换行属于宿主格式，不是新采样 token。动作评分必须在动作阶段的真实前缀下计算，不能沿用理由阶段 KV 缓存或把参考动作放入理由生成提示。只保存解码文本不足以还原原始生成 token，重新分词也不能补出真实停止原因。</p>
<h2>固定失败效用：梯度正确仍可能优化错目标</h2>
<pre>rθ(Z) = v(Z) log pθ(A* | X,D(Z)) + (1-v(Z)) c
J = E_Z[rθ(Z) - β log(pθ(Z|X)/p0(Z|X))]
∇J = E_Z[(rθ - β log(pθ/p0) - b) ∇log pθ(Z|X)
          + v(Z) ∇log pθ(A*|X,D(Z))]</pre>
<p>b 必须独立于自己的采样；K≥2 的逐样本固定效用可用 leave-one-out。参考模型冻结，停止分布完整，采样与评分参数及温度相同；裁剪、重采样、top-k、贪心或过期参数均不在这个证明内。KL 导数中的常数项因完整分布的期望 score 为零而消失。</p>
<p><strong>反例：</strong>有效动作 log p=-20，失败 c=-4，当前有效概率 q=0.5，则 J=-4-16q。
其负效用对 q 的 logit 梯度为 +4；梯度下降反而降低有效输出率。任何有限 c 都无法对所有可能极小的动作概率保证“失败总是更差”。
因此本轮 -4 仅是声明过的数值用例，不能变成生产训练默认值。平均每 token 的 log 概率也没有有限下界，不能自动消除这个问题。</p>
<h2>普通边缘似然对照与全失败批次</h2>
<pre>L = -log Σ_Z v(Z) pθ(Z|X) pθ(A*|X,D(Z))
采样权重 w_k ∝ v(Z_k) pθ(Z_k|X) pθ(A*|X,D(Z_k)) / μ(Z_k|X)</pre>
<p>这个目标拟合“有效理由且产生参考动作”的联合概率。无效输出保留自身概率质量，但成功贡献为零；没有把条件分布 p(Z|v=1) 当成原分布。
它在上述反例的梯度为 −0.5，会提高有效率。然而这仍只是参考动作似然，不能代替完整场景功能。</p>
<p>K=2 的有限例中，全失败批次概率为 {o['marginal_K2']['all_invalid_probability']:.6f}。新函数明确返回 None 与失败记录，不允许自动补采到成功。
若调用方将这种批次记为零更新，仍有有限 K 偏差，本例最大梯度偏差为 {o['marginal_K2']['finite_K_gradient_max_bias']:.6f}。
不能把“没有可用估计”写成正常训练完成；也不能偷偷换成另一种效用。本轮没有选定新的训练目标。</p>
<h2>官方源码核查的适用范围</h2>
<p>复核 <a href="https://arxiv.org/html/2411.04282v2#S4">LaTRO §4.1–4.3</a>，并实际执行固定提交
54f9356 的 <a href="https://github.com/SalesforceAIResearch/LaTRO/blob/54f9356974cc138cd39ef79ec93eb9a421d5976f/LaTRO/utils/trainer_utils.py#L178">truncate_response</a> 纯函数。
该函数把停止 token 本身及之后位置填成 padding；后续评分不包含它。这与本项目把闭合 token 作为随机停止事件的合同不同，直接移植不能保留同一个分布证明。原始源码没有修改。</p>
<p>另一个默认分支用同批原奖励均值计算未停止罚分。其他样本的惩罚因此可能依赖当前样本，普通 leave-one-out 的独立性条件不再自动成立。
两结果、K=2 的反例中，实际期望批效用梯度为 {b['expected_batch_utility_negative_gradient']:.6f}，按单样本 score 套用 RLOO 为 {b['own_score_RLOO_expected_gradient']:.6f}。
这只限制我们移植后的理论主张；未重跑上游模型，也不推出其公开经验结果错误。</p>
<h2>对完整三维生成研究的影响</h2>
<p>这一步补齐训练目标的失败边界，避免把格式失败删掉或给它错误激励。它没有证明理由会改善设计。
下一项实际模型比较仍须使用同样的知识、资产、原生工具、原始要求与累计预算，比较理由作为辅助监督与作为动作条件的差别，并保留普通规则/搜索。
最终判据是完整场景的要求、保留功能、几何、未知项与成本；不能以理由闭合率、动作模仿概率或本页数值误差代替。</p>
<p>15 个单元缺完整原视图，14 个没有理由标注，现有家族均已用于开发。完整多模态采样/重评分尚未执行，本模块也未处理模型版本、dropout、温度等采样来源验证。
这些阻止“已经可训练”的判断，但不要求现在开 GPU。保留同监督主对照，不因本次源代码差异再扩大潜变量训练分支。</p>
<h2>可复核交付</h2><ul>
<li><a href="actual_v2/summary.json">最后一轮实际记录</a> · <a href="actual_v1/summary.json">首轮记录</a></li>
<li><a href="actual_v2/finite_distribution.json">完整停止树与梯度枚举</a> · <a href="actual_v2/objective_selection.json">罚分反例与全失败批次</a></li>
<li><a href="actual_v2/upstream_stop_helper.json">官方纯函数输出</a> · <a href="actual_v2/batch_dependent_penalty.json">同批罚分反例</a></li>
<li><a href="actual_v2/archived_output_audit.json">12 份真实旧文本审计</a> · <a href="actual_v2/token_boundary_fixtures.json">真实 tokenizer 的声明式边界测试</a></li>
<li><a href="../../../scripts/autodl/bounded_reason_scoring.py">评分和普通损失代码</a> · <a href="../../../scripts/autodl/check_bounded_reason_scoring.py">可运行枚举与源码检查</a> · <a href="COMMANDS.txt">复核命令</a></li>
</ul></html>'''
    (a.root/'stage_report.html').write_text(body)


if __name__=='__main__':main()
