"""Render the source-backed objective review. No training or inference entrypoint."""
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile


REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / 'runs/autodl/20261110_execution_supervision_review'


def save(name, obj):
    (STAGE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    data = json.loads((STAGE / 'actual_v1/summary.json').read_text())
    source = json.loads((STAGE / 'mapo_source_manifest.json').read_text())
    existing = {row['path']: row for row in source['files'] if 'error' not in row}
    for name, row in existing.items():
        assert hashlib.sha256((STAGE / 'source_mapo' / name).read_bytes()).hexdigest() == row['sha256']
    assert {'LICENSE', 'nsm/agent_factory.py', 'nsm/model_factory.py', 'table/experiment.py'} <= set(existing)
    ledger = {
        'scope': 'Targeted objective/tool-source review; not complete field novelty clearance',
        'papers': [
            {'title': 'Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing',
             'url': 'https://arxiv.org/html/1807.02322v3',
             'conference_pdf': 'https://papers.neurips.cc/paper_files/paper/2018/file/f4e369c0a468d3aeeda0593ba90b5e55-Paper.pdf',
             'read': 'arXiv v3 sections2-4.4 (equations1-8, algorithms1-2), 5.1 and 5.2 methodological comparison; no complete appendix/figure review',
             'version_boundary': 'The acquired conference PDF and arXiv v3 have different wording/configuration numbers. No experimental number is mixed between them.',
             'source_pin': source['pin'],
             'code_read': {'nsm/agent_factory.py': '105-242,366-510: probability weights, train, good-trajectory buffer admission/replay',
                           'table/experiment.py': '195-234,850-929: flags, inside/outside weighting and rejection membership'},
             'not_done': 'No TensorFlow2/Python2 upstream runtime, benchmark or training reproduction. model_factory acquired but not fully reviewed.'},
            {'title': 'From Language to Programs: Bridging Reinforcement Learning and Maximum Marginal Likelihood',
             'url': 'https://aclanthology.org/P17-1097.pdf',
             'read': 'Abstract/introduction; section4 objectives/gradient/exploration; complete section5 including equation11; opening section6 setup. Not all tables or appendices.',
             'finding': 'Several successful programs and spurious programs are prior problems; randomized search and meritocratic weighting are established controls.'},
            {'title': 'SceneSmith source',
             'url': 'https://github.com/nepfaff/scenesmith',
             'pin': 'e888297a8b0a2f3acbb8634edf9def59cceab836',
             'read': {'asset_manager.py': '148-195,486-695,1009-1042',
                      'furniture_agents/tools/furniture_tools.py': '60-118,273-382,480-694',
                      'placement_noise.py': '1-93 including pose-derived seed',
                      'configurations/furniture_agent/base_furniture_agent.yaml': '345-370'},
             'finding': 'Native success is a subsystem receipt; placement noise is deterministic from pose. Do not infer stochasticity merely from default_rng.'}],
        'existing_review_reused_not_newly_read': [
            '../20261110_native_physics_qualification/algorithm_and_theory.json',
            '../20261110_missing_decision_objective/reading_ledger.json',
            '../20261110_complete_generation_bridge/theory_and_method_difference.json'],
        'access_failures_preserved': ['GitHub API tree403; README raw SSL error; guessed replay_buffer.py404; shell pdftotext unavailable'],
        'fallback': 'Resolved actual master pin through git ls-remote and fetched bounded raw files; read papers through primary HTML/PDF tool. Download does not count as full reading.'}
    save('reading_ledger.json', ledger)
    spec = {
        'status': 'Detailed falsifiable comparison; neither a novel loss claim nor authorization to train',
        'target': 'Complete editable indoor scene from the original description, with original required instances and relations; execution receipts alone never establish completion',
        'current_candidate': 'Compare same-source decision-conditioned action learning against decision-auxiliary direct-action learning and direct-action search. Keep native asset/tool execution common and frozen.',
        'notation': {'X': 'original requirements + actual available images/state/history',
                     'D': 'model-produced decision proposal, not a verified fact',
                     'A': 'serialized executable action including termination',
                     'omega': 'tool-side latent outputs/asset choice/execution outcomes',
                     'tau': 'complete action/observation trajectory',
                     'G': 'full-task verifier, tri-valued true/false/unknown',
                     'B': 'unique complete verified-success sequences for exactly the same task/environment identity'},
        'primary_objective': 'J(theta;psi0) = E_{tau,omega~p(theta,psi0)}[1{G(X,S_T)=true}] at equal total execution budget. Unknown is not success.',
        'supervised_controls': {
            'action_only': 'Original action-token weighted CE; original corpus/family mass preserved',
            'auxiliary': 'Same A and original D labels, separately conditioned on X; deployment emits A(X)',
            'conditioned': 'Same original A,D labels; deployment samples its own D then A(X,D)',
            'fairness': 'Same source units/images/base initialization/native tool version. Direct A may use saved reasoning compute for ordinary candidates. Missing D is not fabricated.'},
        'MML_baseline': {'formula': 'L_B = -log sum_{tau in B} p_theta(tau|X; frozen deterministic tools)',
                         'gradient': '-sum_{tau in B} softmax(log p_theta(tau)) * grad log p_theta(tau)',
                         'ordinary_prior': 'MML/RANDOMER/MAPO; not a proposed new algorithm',
                         'empty_buffer': 'Undefined positive likelihood: do not silently create a successful label or force an arbitrary log-epsilon loss',
                         'current_admission': False,
                         'why': 'Current72 development decisions are evaluation-only local proxies, not verified complete success sequences'},
        'shared_model_boundary': {
            'actual_fact': 'LocalVLMService tool RPC uses the same existing SceneBuilder8B base checkpoint as local action experiments; no updates performed',
            'recommended_identified_control': 'Freeze and hash tool-side base weights/adapter state; compare policy arms with the same tool service. A policy adapter must not silently become active for physics/routing.',
            'if_shared_mutable_weights': 'grad J = E[R * sum_t(grad log pi_theta(A_t|H_t) + grad log kappa_theta(omega_t|H_t,A_t))], when reward itself has no direct theta dependence',
            'meaning': 'Updating both changes the estimand. Omitting tool terms does not estimate this full shared-model score gradient; ordinary fixed-reward RL remains a different valid objective.'},
        'data_boundary': {'training_units': 24, 'units_with_original_D': 10, 'action_only_units': 14,
                          'units_with_recovered_original_views': 19, 'units_missing_original_views': 5,
                          'evaluation_decisions_not_admitted': data['recorded_decisions'],
                          'full_scene_positive_labels_created': 0},
        'theory_limits': [
            'With sound B subset of complete valid sequences, 0<=P(B)<=P(full success). A partial collision/support checker does not establish this premise.',
            'Weighted token CE preserves the original supervision experiment but exp(-weighted CE) is not in general normalized sequence probability. Do not feed it to probability-mass replay weights.',
            'Teacher-forced histories score recorded actions, not trajectories generated by the tested policy after its own edits.',
            'Exact action strings, tool argument equality, and geometric/task equivalence are distinct. Deduplicate only exact strings unless equivalence and measure are proved.',
            'One-step preservation is not the trajectory objective. Bounded joint edits can pass through intermediate degradation; check final constraints and retain rollback.',
            'Old fixed trajectories lack complete counterfactuals and logged proposal propensities; local comparisons do not identify unbiased whole-policy improvement.',
            'Fixed frozen backend may be stochastic without invalidating policy-gradient expectation; reusing a single old return as deterministic requires further justification. The reviewed placement noise alone is pose-seeded deterministic, not evidence of such stochasticity.'],
        'algorithm': [
            'Freeze original task instances/clauses and tool/corpus identity; unsupported checks remain explicit unknown.',
            'From each policy own actual state construct identical allowed observations; keep D as a proposal.',
            'Emit direct A or self-generated D then A according to assigned arm; ordinary search uses identical action space.',
            'Execute a bounded single/joint candidate on a copy through common native tools; charge every model/tool-side VLM/render/failure.',
            'Evaluate the final candidate, preserve identity and required clauses, keep rollback. Intermediate proxy decline alone does not prohibit a joint candidate.',
            'Commit the allowed candidate, obtain new actual observation, and continue within the same cumulative budget.',
            'Accept STOP as full success only if the complete frozen task is verified; otherwise log failure/unknown and keep denominator.',
            'Compare complete-scene task quality, damage, unknown coverage and all-in cost on held-out source rooms. Neither D fluency nor native success is the endpoint.'],
        'novelty_decision': 'D->A, valid-set marginalization, replay memory, outcome checking and rollback individually overlap prior work. No established 3D-specific algorithmic advantage is claimed. Require independent whole-room benefit over the ordinary controls before escalation.',
        'no_new_training_or_inference': True, 'goal_achieved': False}
    save('algorithm_specification.json', spec)
    with (STAGE / 'method_differences.csv').open('w') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['method', 'learned_target', 'deployment', 'current_status', 'what_it_controls'])
        w.writerows([
            ['frozen_base_search', 'none', 'common native tools + ordinary bounded search', 'no new run in this review', 'ordinary geometry/compute'],
            ['action_only_SFT', 'original action weighted CE', 'A(X)', 'objective implemented; no new training', 'extra decision labels'],
            ['auxiliary_D_direct_A', 'same original A and D', 'A(X)', 'objective implemented; no new training', 'label augmentation alone'],
            ['self_D_conditioned_A', 'same original A and D', 'D_hat(X),A(X,D_hat)', 'candidate; no independent whole-room benefit', 'conditioning versus auxiliary supervision'],
            ['valid_trajectory_MML', 'verified complete sequence set', 'same policy/tools', 'not admitted: verified buffer missing', 'multiple valid solutions; existing method']])
    report = '''<!doctype html><meta charset="utf-8"><title>三维执行监督与训练目标审查</title>
<style>body{max-width:1060px;margin:35px auto;font:17px/1.75 sans-serif;padding:0 18px}pre{white-space:pre-wrap;background:#f3f5f8;padding:15px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}aside{background:#fff3d5;padding:14px}</style>
<h1>把完整场景目标、动作监督和工具内模型分清</h1>
<p>这轮得到的是可以审查的训练与对照定义、对真实执行材料的准入判断，以及原生工具分支证据。没有进行新训练、没有新的模型调用，也没有把局部工具成功当作完整生成成果。当前主张仍待验证，不能承诺CVPR级贡献。</p>
<h2>1．现有数据究竟能支持什么</h2>
<pre>DATA</pre>
<p>上表来自重新读取已保存的12条开发轨迹，共72个决定、56次原生工具调用；不是新跑72次。56次工具返回都成功，其中25次损坏了至少一项已记录检查。5次有局部改善且没新增已测损坏，只能叫局部改善。它们仍来自同一房间家族，不是5个新完整场景。</p>
<p>这些数据原来明确标记evaluation_only、training_admission=false，本轮保持不变。41组上下文中14组有多个非空观测动作，但缺少完整合格轨迹、未知动作结局与采样概率。不能拿它们直接宣称获得了完整有效动作集合，或把没见过的动作当负例。阈值敏感性检查包含不同通道宽度，所有代理合取也不是唯一任务真值。</p>
<h2>2．为什么不把集合似然包装成新算法</h2>
<p><a href="https://aclanthology.org/P17-1097.pdf">RANDOMER（ACL2017）§4–5</a>已经比较程序执行的RL与边际似然，并处理偶然答对的程序；<a href="https://arxiv.org/html/1807.02322v3">MAPO（NeurIPS2018）§3–4</a>已经利用高奖励轨迹记忆及缓冲区内外采样。因而“正确三维动作有多个，所以求和学习”只能先作为成熟基线。普通搜索和奖励检查也必须足够强。</p>
<p>从MAPO固定源码读到：AllGoodReplayBuffer只保存正回报、终止且未重复的程序；回放计算模型概率和总概率，再用于内外权重。其概率裁剪会改变原目标，不能保留裁剪却仍笼统声称完全无偏。源码是Python2/TensorFlow时代实现，本轮未运行官方训练，也未把论文数字迁移成三维预期收益。</p>
<h2>3．训练目标必须与完整任务相连</h2>
<p>令X为原始要求和实际可用观察，D为决策提议，A为可执行动作，ω为工具内部输出，τ为完整轨迹。G(X,S<sub>T</sub>)取真、假或未知；未知不计成功。目标是相同总预算下的完整任务完成概率：</p>
<pre>J(θ;ψ₀) = E[1{G(X,S_T)=true}],  (τ,ω) ~ p(π_θ, κ_ψ₀).</pre>
<p>现有监督首先保持原始24单元和原token系数：10单元有原始D，14单元仅有A；19单元恢复了原观察，另5个仍缺失。这三个数不能互换。缺D不编造，缺图不拿新图替代再称同源对照。辅助D、直接A与自身生成D再A，必须使用同样原始材料和参数初始化；直接A获得节省下来的推理预算做普通候选搜索。</p>
<pre>L_aux = Σ l_A(X,A) + Σ_(有原始D) l_D(X,D)
L_cond = Σ_(有原始D) l_A(X,D,A)
       + Σ_(无原始D) l_A(X,A) + Σ_(有原始D) l_D(X,D)</pre>
<p>这属于标准监督目标的受控拆分，不是新损失。部署不能把老师D注入条件组，也不能从旧轨迹逐轮重置到正确状态。应让每组使用自己的真实编辑结果获得后续观察。</p>
<h2>4．集合边际似然成立的前提</h2>
<p>固定可复现后端，令B是同一任务/环境下的、不重复的、完整验证成功的终止动作序列集合。只有B真的包含于完整有效解时，才有：</p>
<pre>0 ≤ P_θ(B|X) ≤ P_θ(完整成功|X)
L_B = −log Σ_(τ∈B) P_θ(τ|X)
∇L_B = −Σ_(τ∈B) [P_θ(τ|X)/P_θ(B|X)] ∇log P_θ(τ|X).</pre>
<p>空B不能伪造正例；只检查支撑或碰撞不能建立这个前提。动作字符串不同、参数数值相同、三维结果等价是三件事；等价类概率必须把所有成员的概率相加，不能任意取一个代表或用候选频次代替。当前只有观测支持，不存在已准入的完整B，因此代码没有计算一个看似有效的MML成绩。</p>
<p>原来的加权token CE用于保持监督质量和家族权重，但exp(−加权CE)一般不是归一化的动作概率。若将来实现边际似然，轨迹概率须来自完整的原始自回归概率和终止事件，家族权重放在每个样本目标外面；不能偷偷改变原对照的权重后仍称同一个实验。</p>
<h2>5．同一模型做动作与工具判断时，必须界定谁在学习</h2>
<p>实际RPC从既有SceneBuilder8B检查点加载工具VLM；它不是免费的确定真值接口。若只训练动作adapter而冻结工具服务，研究的是在相同工具下的决策收益。若工具也启用新adapter或同步更新权重，就同时改变了观察、资产估计及环境转移，不能只归因于决策机制。</p>
<pre>若双方共用可更新θ，且奖励没有直接θ依赖：
∇J = E[R Σ_t (∇log π_θ(A_t|H_t)
                + ∇log κ_θ(ω_t|H_t,A_t))].
若κ=κ_ψ₀冻结：工具项不属于θ梯度，保留动作项即可。</pre>
<p>这是分布求导的标准结果，不是新定理，也不禁止固定工具的策略学习。建议第一轮明确冻结工具检查点、adapter关闭状态、提示与解析器，并保存哈希。若研究共享更新，就另设相同能力的共享更新对照，不能混在一起。</p>
<p>源码还纠正了一个容易出现的误判：placement_noise.py不是每次无种子抽样，而是由输入姿态构造种子，因此该分支不能用来证明重复执行随机。检索映射、渲染、模型解码或其他后端是否随机，需要实际证据；此轮没有重新运行以制造随机性结论。</p>
<h2>6．可以直接审查的完整执行算法</h2>
<ol><li>冻结原描述、必需实例和关系、候选资产与工具版本；无法验证的条目保留未知。</li>
<li>从当前实际场景构造两组相同可用观察。条件组先生成D，直接组输出A；D始终是提议。</li>
<li>所有组共享动作和有界联合编辑能力，在副本执行，完整累计动作模型、工具VLM、检索、渲染、失败和回滚成本。</li>
<li>检查完整候选的要求保留、身份、几何和新增目标；允许有界联合编辑中的临时退步，不能强迫每一步都立刻提升布尔指标。</li>
<li>提交通过共同规则的候选，继续读取它自己的真实场景。预算结束、拒绝、缺物体和未知均保留分母。</li>
<li>STOP只表示策略要求结束；只有完整冻结任务验证通过才算成功。最终比较完整场景质量与总预算曲线。</li></ol>
<p>这个执行循环本身仍有反馈、搜索和回滚的直接先例。它的作用是让候选机制受到公平检验，不能作为主要创新。只有在独立房间中，条件决策超过同源辅助监督、普通搜索和强通用反馈，才值得进一步研究三维特有的表示或训练机制。</p>
<h2>7．当前取舍与资源结论</h2>
<aside>保留原始同源训练对照，不在现有局部轨迹上启动结果奖励训练；先固定工具侧模型边界。集合似然纳入已知方法清单，待真正有完整验证轨迹后再决定是否需要。当前未确认新颖算法，未证明学习收益，也没有终止整个研究。</aside>
<p>整场入口仍缺任意新文本CLIP检索能力、索引映射溯源和完整语义覆盖；不能因为单资产管理器跑通就宣布资源和算法准备充分。先解决这些具体公共前提，下一轮应验证完整场景，而非继续制造局部修复样例。当前不需要为写报告或继续源码阅读新增资源。</p>
<p><a href="algorithm_specification.json">机器可读算法与理论</a> · <a href="method_differences.csv">方法差异</a> · <a href="reading_ledger.json">阅读范围</a> · <a href="actual_v1/observed_support.json">真实观测动作支持</a> · <a href="../20261110_live_manager_request/stage_report.html">原生管理器实际运行</a></p>'''
    (STAGE / 'algorithm_review.html').write_text(report.replace('DATA', html.escape(json.dumps(data, ensure_ascii=False, indent=2))))
    save('summary.json', {**data, 'source_pin': source['pin'],
                           'new_paper_method_reviews': 2, 'new_source_training_runs': 0,
                           'new_theorem_or_loss_claim': False, 'goal_achieved': False})
    save('handoff.json', {
        'status': 'Continue full-room research; this stage is not a new method result',
        'ordinary_controls_required': ['same-source auxiliary D', 'direct action with ordinary search', 'shared frozen native backend', 'MML only after verified complete buffer'],
        'no_training_authorization': True, 'no_automatic_model_calls': True,
        'next_concrete_gap': 'General text-to-asset retrieval and whole-room source/constraint coverage; do not repeat old coordinate diagnoses',
        'do_not_promote_evaluation_rows': True})
    # Acquired paper PDFs are local reading material, not redistributed here.
    files = sorted(p for p in STAGE.rglob('*') if p.is_file()
                   and 'papers' not in p.relative_to(STAGE).parts
                   and p.name not in ('artifact_manifest.json', 'evidence.zip'))
    manifest = [{'path': str(p.relative_to(STAGE)), 'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    save('artifact_manifest.json', {'files': manifest, 'excluded': 'Local paper PDFs; no full-paper redistribution'})
    with zipfile.ZipFile(STAGE / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(STAGE))
        z.write(STAGE / 'artifact_manifest.json', 'artifact_manifest.json')
    print(json.dumps({'report': str(STAGE / 'algorithm_review.html'),
                      'published_files': len(files), 'source_files_acquired': len(existing),
                      'new_training_updates': 0, 'new_model_calls': 0}))


if __name__ == '__main__':
    main()
