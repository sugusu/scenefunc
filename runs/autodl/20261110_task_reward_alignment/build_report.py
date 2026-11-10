"""Publish executed reward diagnostics and task-aligned control definitions."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    s = json.loads((HERE/'actual_v1/summary.json').read_text())
    rows = json.loads((HERE/'actual_v1/trajectory_reward_rows.json').read_text())
    shaped = json.loads((HERE/'actual_v1/finite_horizon_shaping.json').read_text())
    corpus_path = ROOT/'runs/autodl/20261110_native_phase_host/summary.json'
    corpus = json.loads(corpus_path.read_text())
    coverage = corpus['native_action_compatibility']['counts']
    reading = [
        {'title': 'Hylos: Operability Contracts for Model-Native Spatial Intelligence',
         'url': 'https://arxiv.org/html/2605.24728v1',
         'read': ['Abstract/introduction and scope 1.2', '3.3–3.5 positioning',
                  '4.1–4.4 formal state/dependency/transaction contract',
                  '5.2 agency gates', '6.1–6.4 evaluation', '7.1–7.3 results',
                  '8.1–8.3 staged roadmap', '9.5 metrics', '10–11 boundaries',
                  '12 opening and selected roadmap; references to prototype report'],
         'finding': 'Already proposes dependency-based upstream repair, admissible operations and validated commit/rollback. Public empirical anchor is one qualitative repair family, conceptual controls and a broader evaluation roadmap, not a broad quantitative scene-generation result.',
         'limits': ['Not full-paper/figure visual review', 'No implementation acquired or executed',
                    'Prototype implementation-specific details are omitted in 3.5; the referenced prototype report is internal'],
         'implication': 'Do not claim dependency tracing plus transactions as the main new algorithm. Its bounded evidence also does not establish that a learned complete-room method has been solved.'},
        {'title': 'SDGScenes: User-intent driven indoor scene generation via semantic dependency graph',
         'url': 'https://www.sciencedirect.com/science/article/pii/S0031320326006394',
         'doi': '10.1016/j.patcog.2026.113674',
         'read': ['Publisher-indexed abstract, highlights, introduction and conclusion snippets'],
         'finding': 'Semantic dependency graph for implicit intent, VLM-derived spatial constraints and optimization with physical/reachability constraints are direct prior capabilities.',
         'limits': ['Direct full-page requests returned403', 'No full method/loss/source reading',
                    'No inference about exact solver, datasets, training or numerical superiority'],
         'publication_boundary': 'Publisher lists Pattern Recognition179 PartB, November2026; no new claim about acceptance/online-first date'},
        {'title': 'Policy invariance under reward transformations: Theory and application to reward shaping',
         'url': 'https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf',
         'read': ['Abstract and introduction', '2.1 MDP assumptions',
                  '3 Theorem1 sufficient form and proof', 'Corollary2 and remarks on policy values',
                  'Terminal-potential condition in the undiscounted case'],
         'finding': 'Potential-based reward shaping is a classical control, not a new loss. Finite diagnostic paths are checked by the telescoping identity with the terminal potential handled explicitly.',
         'limits': ['No full necessity-appendix proof audit', 'No reproduction of original grid experiments',
                    'No guarantee of unchanged finite GRPO/Adam updates or neural convergence']},
        {'title': 'Scene-Designer pinned training and evaluator source',
         'url': 'https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src/train_grpo.py',
         'pin': s['source_pin'], 'sha256': s['source_sha256'],
         'read': ['train_grpo.py140–170 gate definition', '211–249 scalar rewards/count loader',
                  '272–312 Critic wrapper contract', '390–425 local reward assembly',
                  '808–862 trajectory global rewards/normalization', '1018–1038 CLI gate',
                  'src_critic/scene_io.py135–203 loader'],
         'finding': 'Global scalar sums length and object-count rewards; local rewards separately combine format, Critic rank and improvement, with the actual local legality gate ON by default.',
         'limits': ['Only the two original scalar function bodies executed as AST',
                    'Native type counts are diagnostic covariates, not an execution of count_objects',
                    'No Critic, model, full module, optimizer or author evaluation executed'],
         'comment_discrepancy': 'make_reward_critic docstring says pure Critic by default/global legality. Actual module and CLI use the local gate ON, while the inspected global loop sums length and object count. Record the discrepancy without claiming a paper-result defect.'},
    ]
    dump(HERE/'reading_ledger.json', reading)
    theory = {
        'status': 'Executable objective diagnostics and ordinary controls; no independent algorithmic benefit claimed',
        'endpoint': 'Complete editable indoor generation from original text, with original required instances, attributes and relations; not local proxy optimization alone',
        'new_observation': {
            'source_functions': s['source_definitions_executed'],
            'diagnostic_archive': '12 fixed trajectories, 72 decisions, one source-room family',
            'result': 'All length+count diagnostic sums2.0; all12 terminal states have failed recorded checks',
            'limits': s['scope'],
        },
        'task_definition': {
            'C_U': 'Fixed set of original user-required instances/attributes/relations, including unresolved clauses; model hypotheses do not replace it',
            'v_c': 'true/false/unknown per original clause, with evidence and verifier coverage',
            'verified_completion': 'G_V=1 iff every original clause is covered and verified true under the declared assessment; unknown is not a verified success and is not relabeled false',
            'success_objective': 'Maximize E[G_V(S_T,U)] under a shared total execution budget, while separately reporting false and unknown coverage, actual costs and collateral damage',
            'important_limit': 'This is success under the specified verifier. Unsound/incomplete semantic or physical checks cannot certify real-world full completion.',
        },
        'ordinary_shaping_control': {
            'formula': 'r_t^Phi = r_t + gamma*Phi_{t+1}(S_{t+1},U)-Phi_t(S_t,U)',
            'identity': 'sum_{t=0}^{T-1} gamma^t*r_t^Phi = sum_{t=0}^{T-1} gamma^t*r_t - Phi_0(S_0,U) + gamma^T*Phi_T(S_T,U)',
            'finite_horizon_condition': 'Set Phi_T=0 on every terminal status, including budget exhaustion/failure, if the original episodic objective ends there. If time-limit truncation is a continuing task, bootstrap instead; do not confuse truncation and termination.',
            'gamma_one': 'With gamma1 and Phi_T0 the entire shaped return differs only by the initial-state constant. This preserves the original undiscounted success objective.',
            'gamma_below_one': 'The identity preserves that discounted base objective, which already prefers earlier reward. It is not a guarantee of equivalence to undiscounted success/cost.',
            'scope_of_guarantee': 'Pathwise returns for fixed state/task and terminal rules, with bounded potential. It does not prove identical finite stochastic gradients, clipping/KL behavior, data coverage, exploration or convergence.',
            'not_valid_substitutes': ['Reward only positive progress and ignore undoing it',
                                     'Treat each intermediate improvement as independently cumulative success',
                                     'Leave a geometry/completeness potential at STOP and claim original-objective invariance',
                                     'Use a large but arbitrary weighted soft score and assert hard feasibility'],
            'implemented_diagnostic': 'fixed original proxy-check denominator, no new truth labels, gamma1/.99;24 finite-path identities, maximum residual recorded in summary',
            'not_a_novel_method': True,
        },
        'training_and_decision_algorithm': [
            'Preserve source tasks, original instance identities and unresolved clauses. Keep a model decision/proposal separate from observed execution and from original user requirements.',
            'Keep native base asset/tool models frozen across policy arms; policy updates must not silently alter physics, retrieval validation or routing services.',
            'Use the same original family/component-weighted supervision for action-only, D-auxiliary direct action and self-generated D-conditioned action. Missing D/views are retained, not synthesized from answers.',
            'If testing a geometric goal H, label it desired/proposed, not measured future. Give the direct-action auxiliary control the same eligible H targets. No privileged future or teacher H at deployment.',
            'All arms receive the same actual state and operation rights. Use ordinary known geometry previews/search as a common baseline. Allow bounded joint candidates when an intermediate move must temporarily worsen a proxy.',
            'Native asset generation/acquisition, placement, multi-stage continuation and original-task verification remain necessary. An add from a supplied registry is not proof of learning how to acquire missing assets.',
            'During any later authorized RL comparison, distinguish actual original source reward, explicit terminal task reward, and terminal-correct potential shaping. Changing the reward is a control and must be applied consistently across representation arms.',
            'Execute each policy on its own actual states, maintain candidate rollback, and score whole descriptions under fixed budget. Charge all model/tool-side VLM, renderer, failed and rejected candidates.',
        ],
        'strong_controls': [
            'Common native backend + ordinary state/geometry/search, no learned decision mediator',
            'Original D auxiliary supervision with direct actions, same labels as D-conditioned arm',
            'Self-generated D-conditioned action under exactly the same task reward',
            'If testing geometry: same-data geometry auxiliary direct-action control',
            'If testing rewards: terminal completion alone vs ordinary potential shaping; keep representation/data/operations fixed',
        ],
        'reused_corpus_coverage': {
            'source': str(corpus_path.relative_to(ROOT)), 'sha256': sha(corpus_path),
            'counts': coverage,
            'interpretation': '24 original single-step targets; six add-existing-registry,15 move,one remove,two stop. No target for acquisition/floor-plan/wall/ceiling/manipuland stages. This does not prove the base model cannot do them; it limits what this training corpus demonstrates.',
            'this_turn_inventory_rerun': False,
        },
        'novelty': {
            'not_claimed': ['Dependency graph plus transaction rollback', 'Intent graph to ordinary constraints',
                            'Reason then act alone', 'Potential shaping or changing count rewards alone'],
            'candidate_remains': 'Whether a learned, requirement-grounded decision representation improves full generation beyond same-source auxiliary labels and equal-budget ordinary search, under a shared native backend and properly aligned task assessment',
            'current_evidence': 'No new trained advantage established; scalar/source diagnostics narrow the valid comparison rather than establish CVPR novelty',
        },
        'resource_decision': 'No further GPU/RAM/large downloads needed for these source and CPU objective checks. Training readiness still requires complete observations, validated task outcomes and a separately measured adapter-training memory envelope.',
        'goal_achieved': False,
    }
    dump(HERE/'algorithm_and_theory.json', theory)
    differences = [
        ['Hylos', 'Dependency/provenance, upstream repair and transaction contract', 'One public qualitative repair family, not broad learned full-generation results', 'Engineering scaffold cannot carry a new learning claim'],
        ['SDGScenes', 'Intent/commonsense graph to VLM constraints and optimization', 'Publisher abstract/snippets only; methods/code not acquired', 'Intent parsing plus solver already has a direct scene-generation neighbor'],
        ['Scene-Designer scalar global terms', 'Shorter trajectories and more objects', 'Isolated exact functions; not local Critic or total GRPO reward', 'Task completion must be separately verified'],
        ['Ordinary potential shaping', 'Transition potential differences', 'Terminal conditions and original base reward must be correct', 'Required strong control, not a new loss'],
        ['Current learned decision question', 'Same-source D auxiliary vs D-conditioned action', 'No independent full-room improvement yet', 'Keep representation and reward effects separately identifiable'],
    ]
    with (HERE/'method_differences.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(['method','mechanism','evidence_boundary','implication']);w.writerows(differences)
    (HERE/'COMMANDS.txt').write_text('''From repository root, CPU only; no model/native tool/training calls:
python scripts/autodl/audit_task_reward_alignment.py --reward-source /root/autodl-tmp/agent_method_preparation_20261027/work/context_credit_source_review_20261103/sources/train_grpo.py --out runs/autodl/20261110_task_reward_alignment/reproduce_v2
python runs/autodl/20261110_task_reward_alignment/build_report.py
Use a fresh output directory; source SHA is checked against pinned author revision.
Do not interpret diagnostic source-function values as author GRPO results or local Critic scores.
''')

    h = ['<h1>完整生成的训练奖励，是否真的区分任务完成？</h1>']
    h.append('<p>本轮执行了固定源码中的两个奖励函数，复核训练分支，并补充三项原文阅读。新增结果是奖励目标的可区分性诊断，不是模型训练或新的整场生成。研究目标仍是完整可编辑室内场景。</p>')
    h.append('<h2>实际计算结果</h2><p>12条旧轨迹来自一个原始房间族，各有6次决策；按记录中的native furniture类型与真实缺失ID回执，终态保留10至12件家具。把这些诊断统计输入原函数后，长度奖励与物体数奖励均为1，总和全部为2.0，而12个终态都仍有原检查失败。该全局量在这组诊断输入上没有区分度。</p>')
    h.append('<table><tr><th>轨迹</th><th>决策数</th><th>家具类型数</th><th>诊断全局和</th><th>失败检查数</th></tr>')
    for r in rows:
        h.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['cell'],r['recorded_decisions'],r['terminal_native_furniture_type_count'],r['diagnostic_global_sum'],len(r['terminal_failed_scoped_checks'])])+'</tr>')
    h.append('</table><p><strong>适用范围：</strong>这是原length_reward与objcount_reward函数的CPU执行。作者的count_objects读取导出的placed_furniture，本轮没有调用该加载器；这里提供的是旧记录的家具类型数。旧决策数也不是声称取得了原GRPO buffer.traj_len，不过0至6都处在默认满分区间。未运行局部Critic、格式／改善奖励和完整优化，因此不能推导总训练奖励或梯度为零，更不能据此推翻论文成绩。</p>')
    h.append('<h2>源码分支核对</h2><p>固定版本a0565e50…的两个函数在211–234行；默认阈值是长度≤20满分、≥38为零，物体数≥8满分、≤3为零。808–862行将两项相加后做轨迹间标准化。局部奖励另外结合格式、Critic排序和父子改善；模块与CLI默认启用局部合法性门控。make_reward_critic的说明文字与实际门控默认值、全局合法性位置存在不一致，本报告以执行分支为准。远端固定文件哈希核验结果另存，不据此宣称作者使用了哪一组训练参数。</p>')
    h.append('<h2>让密集反馈不替换原任务</h2><p>原始要求集合必须固定，逐条记录真、假或未知。完整成功只计所有原要求在声明评测范围内均已验证；未知不算成功，也不能偷偷改成失败标签。模型自行提出的布置方案与参数不进入用户硬要求集合。</p><pre>rΦ_t = r_t + γ Φ_(t+1)(S_(t+1),U) − Φ_t(S_t,U)\nΣ γ^t rΦ_t = Σ γ^t r_t − Φ_0(S_0,U) + γ^T Φ_T(S_T,U)</pre>')
    h.append('<p>对于确实结束的有限任务，应对所有终止状态正确处理末端势函数；Φ_T=0时，γ=1的总回报只增加相同初态常数。若把继续任务的时间截断误当终止，则需要正确bootstrap。γ&lt;1保留的是该折扣任务，不能说仍等于不折扣的完成率目标。此为<a href="https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf">经典势函数塑形</a>，不能作为新损失主张。</p>')
    h.append(f'<p>在旧轨迹上按固定原检查分母计算势函数，分别检查γ=1与0.99，共{s["shaping_path_identities_checked"]}条路径恒等式，最大数值残差{s["shaping_max_abs_identity_residual"]:.3g}。不做末端处理会留下终态相关奖励；只奖励正向改善也可能忽略之后的破坏。这些计算没有创造专家标签或证明RL提升。原终态检查并非完整语义与功能真值；塑形只能保留所定义的基准目标，不能让错误验证器变正确。</p>')
    h.append('<h2>新近邻对方法主张的影响</h2><p><a href="https://arxiv.org/html/2605.24728v1">Hylos</a>已经描述沿依赖定位上游修改、约束允许动作、验证和回滚。它公开的证据是一个定性修复家族及概念对照，不能扩写成完成大规模场景学习的实证。<a href="https://www.sciencedirect.com/science/article/pii/S0031320326006394">SDGScenes</a>的出版商摘要已说明从用户意图构建语义依赖图、生成约束并求解布局；全文访问403，未声称读过精确优化器或代码。因此“依赖图＋约束工具”不能单独承担当前创新。</p>')
    h.append('<h2>当前算法比较及强对照</h2><p>保留同源动作监督、D作辅助目标后直接动作、自生成D后条件化动作三组；工具、标签与任务奖励一致。引入几何目标时，给直接动作组同标签的几何辅助监督；引入奖励修改时，固定表征和数据，单独比较原奖励、明确终态目标与普通势函数塑形。这样才能区分改善来自目标修正、额外标签、普通搜索，还是所研究的决策表示。</p>')
    h.append('<p>从原描述开始、获取资产、放置并完成各生成阶段仍是必须的端到端证据。复用已有语料覆盖记录：24个目标含6次添加已有注册资产、15次移动、1次移除、2次STOP，没有资产获取及其他原生阶段目标。这限制了该小语料能证明什么，不代表预训练模型不具备那些能力。不能将这些目标或脚本回放改称完整生成训练集。</p>')
    h.append('<p>候选研究问题仍是：在同样数据、工具和总预算下，学到的要求驱动决策表示，能否超过同源辅助监督与普通搜索，改善未参与调参的完整房间。当前没有这样的独立收益证据。更换奖励或加入交易式回滚属于合理基线；它们本身不构成CVPR级新方法。</p>')
    h.append('<h2>交付</h2><ul>')
    for name in ['actual_v1/summary.json','actual_v1/trajectory_reward_rows.csv','actual_v1/finite_horizon_shaping.json','actual_v1/input_hashes.json','upstream_hash_verification.json','reading_ledger.json','algorithm_and_theory.json','method_differences.csv','COMMANDS.txt']:
        h.append(f'<li><a href="{name}">{name}</a></li>')
    h.append('</ul><p>没有模型／付费API／原生动作／渲染／训练／新场景调用，不删除旧结果或活动权重。本阶段新增少量文本，不需要扩大GPU或内存。准备工作仍包括完整观察覆盖、可靠任务评价和实际训练显存测量；不能把这些审计称为全部实验准备已完成。</p>')
    (HERE/'stage_report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>任务奖励对齐审计</title><style>body{max-width:1150px;margin:40px auto;padding:0 20px;font:17px/1.75 system-ui;color:#172330}td,th{border:1px solid #ccd4dd;padding:8px}table{border-collapse:collapse;font-size:14px}th{background:#edf3f8}pre{white-space:pre-wrap;background:#edf3f8;padding:16px}a{color:#165ba8}</style><body>'+''.join(h)+'</body></html>')
    dump(HERE/'summary.json', {**s, 'new_reading_items': len(reading), 'reused_corpus_operator_counts': coverage,
         'new_method_effect_established': False, 'goal_achieved': False})
    (HERE/'scripts').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'scripts/autodl/audit_task_reward_alignment.py', HERE/'scripts/audit_task_reward_alignment.py')
    files = [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['artifact_manifest.json','evidence.zip']]
    dump(HERE/'artifact_manifest.json', {'files':[{'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)], 'raw_author_source_vendored':False})
    with zipfile.ZipFile(HERE/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[HERE/'artifact_manifest.json']:z.write(p,p.relative_to(HERE))
    print(json.dumps({'files':len(files),'report':str(HERE/'stage_report.html'),'zip_bytes':(HERE/'evidence.zip').stat().st_size}))


if __name__ == '__main__':
    main()
