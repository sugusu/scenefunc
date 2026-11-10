"""Package the executed target audit and bounded primary-source review."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    s = json.loads((HERE / 'actual_v2/summary.json').read_text())
    native = json.loads((HERE / 'native_branch_contract.json').read_text())
    acquisition = json.loads((HERE / 'acquisition_v1/manifest.json').read_text())
    # Raw third-party README is local-only because no license file was acquired.
    dump(HERE / 'source_acquisition.json', acquisition)
    reading = [
        {
            'title': 'GeomVLA: Unifying Scene, Motion, and Action in 3D',
            'url': 'https://arxiv.org/html/2609.13812v1', 'version': 'v1, 2026-09-12',
            'read_scope': ['Abstract/introduction/related work', '3.1–3.4 method and training',
                           '4.1 setup fairness and 4.2 ablations', '4.3–4.4 selected matched comparisons',
                           '5 limitations', 'Appendix lifting and scene-trajectory encoder opening'],
            'not_claimed': ['All appendix equations/tables read', 'All figures viewed',
                            'Author implementation read or run', 'Robot demonstrations downloaded'],
            'method_note': 'Calibrated 3D scene/motion/action coordinates. Motion-token conditioning uses one initial velocity evaluation, not execution of a fully predicted future. First train trajectory pseudo-labels with frozen VLM; joint stage uses action loss, not the trajectory loss.',
            'control_note': 'Its five-task RoboTwin comparison does not use identical training sets for all variants; the paper states this. Shared rigid-frame ablation does not establish a unique advantage of the robot origin.',
            'repository': 'https://github.com/ziyin-xiong/GeomVLA',
            'availability': 'Project links this URL; web and GitHub API returned 404. git ls-remote failed TLS. No source availability beyond this observation is inferred.',
        },
        {
            'title': 'From Predictive Geometry to Action: Learning Complementary Predictive Representations for Embodied Manipulation',
            'url': 'https://arxiv.org/html/2610.28164v1', 'version': 'v1, 2026-10-20',
            'read_scope': ['Abstract/introduction/related work', '3.1–3.2 all method equations',
                           'Experiment setup', 'A.3 representation/pretraining objective comparisons through Table 9',
                           'B.1 data mixture, B.2 curriculum opening', 'D limitations'],
            'not_claimed': ['Full appendix read', 'All figure pixels viewed', 'Model implementation read or run'],
            'method_note': 'Pretraining combines reconstruction, masked-slot and latent-transition objectives with action supervision for a separate motor representation. Post-training uses future-informed frozen targets; deployment uses current observations only. The written latent-transition predictor is conditioned on current latent geometry, not a supplied candidate action.',
            'code_pin': 'f2982a37ed685ecbde33a98b323216fe791aa0f2',
            'repository': 'https://github.com/xuxiaoxxxx/CPR',
            'availability': 'Exact recursive tree is not truncated: README.md, asset/, asset/main.png. No model/loss/training Python source at this pin. README fully read; no weights/image downloaded.',
        },
        {
            'title': 'SceneSmith qualified native implementation',
            'repository': 'https://github.com/nepfaff/scenesmith',
            'code_pin': native['base_pin'],
            'read_scope': ['RoomScene object add/remove/get/move and surrounding ID methods, lines 825–925',
                           'FurnitureTools add 405–511, move 551–708, remove 709–795, generate 883–956',
                           'AssetManager HSSD acquisition 487–690 and dispatch/router entry 1009–1100',
                           'placement_noise.py full file'],
            'not_claimed': ['Complete router branch reviewed this turn', 'New native tool runs this turn'],
            'source_contract_receipt': 'native_branch_contract.json',
        },
    ]
    dump(HERE / 'reading_ledger.json', reading)

    theory = {
        'scope': 'Applicability and identified controls; not a new architecture, trained model, or verified CVPR contribution',
        'variables': {
            'S': 'Actual exposed native object state, registry/config where available',
            'U': 'Original user requirements, retaining ambiguity and unsupported clauses',
            'A': 'A proposed executable action with exact identities and arguments',
            'Y': 'Measured post-action geometry or a supported check vector',
            'mu': 'Archived behavior policy, not an expert by default',
            'H': 'A proposed desired geometric/relational subgoal, not a measured future',
        },
        'claims': [
            {
                'name': 'Known transition is not a novel world model',
                'formula': 'For fixed supported deterministic T and checker F, Y=F(T(S,A),U). Given S,A,T,F, H(Y|S,A)=0 for discrete Y.',
                'assumptions': ['Exact state/config/identity', 'Correct supported transition and checker',
                                'No unobserved stage, model output or physical interaction'],
                'not_implied': ['A neural policy can choose a good A', 'Representation learning cannot help finite models',
                                'All native stages are deterministic', 'Proxy success equals complete-scene success'],
                'evidence': 'Previously published ordinary-control 823/823 comparisons, reused without new predictor calls; native source contracts separately read',
            },
            {
                'name': 'Descriptive future and desired future differ',
                'formula': 'p_mu(Y|S,U)=sum_A p(Y|S,U,A) mu(A|S,U). Training to this mixture does not identify an optimal-task future distribution.',
                'assumptions': ['Observed actions belong to the logged behavior distribution'],
                'not_implied': ['Bad outcomes cannot train an action-conditioned dynamics/negative-outcome model',
                                'Multi-modal future prediction is inherently invalid', 'MSE ambiguity proves CPR or GeomVLA fails'],
                'evidence': 'New 72-row role audit, 25/35 native moves with recorded damage, no independent desired-future labels',
            },
            {
                'name': 'Prediction loss can average behavior, not solve the task',
                'formula': 'For squared error and fixed target coordinates, argmin_g E||Y-g(X)||^2=E[Y|X]; conditional variance is the irreducible risk for that X. SmoothL1 has a different estimating equation, not generally this mean.',
                'assumptions': ['This is a mathematical population statement for a fixed X and fixed target, not an estimate of model generalization'],
                'not_implied': ['The extracted state/task X contains full model history', 'The empirical table is a learned model',
                                'Zero in-sample X,A risk identifies deterministic population dynamics',
                                'A target mean must correspond to a feasible scene or useful action',
                                'A flexible latent representation cannot avoid this fixed-coordinate loss geometry'],
                'evidence': '10/41 state-task groups have different measured check vectors; native-move-only restriction still has 2/25 groups',
            },
            {
                'name': 'Information and computation are separate',
                'formula': 'A deterministic feature of available X has zero conditional information I(A;f(X)|X), but may change approximation, optimization or computation of a bounded model.',
                'assumptions': ['Same actual X and deterministic feature, no hidden labels supplied'],
                'not_implied': ['An auxiliary geometry feature is useless', 'Any improvement is automatically a new method'],
                'control': 'Compare explicit native-state/ordinary-lookahead inputs, pose auxiliary supervision and equal-budget direct-action search before attributing benefit to predictive geometry.',
            },
        ],
        'target_roles': [
            {'role': 'current-state reconstruction', 'data': 'Original audited current object geometry/available views',
             'meaning': 'Perception/representation auxiliary control', 'known_gap': 'Views missing on 5/24 original units; no replacement view fabrication'},
            {'role': 'executed effect conditioned on S,A', 'data': 'Archived after-state/check receipts',
             'meaning': 'Descriptive transition or negative outcome, not desired arrangement',
             'known_gap': '72 rows remain evaluation-only; ordinary control covers supported checks; no independent room/operator coverage'},
            {'role': 'desired goal H conditioned on S,U', 'data': 'Must be anchored to original requirements and verified candidates or independent demonstrations',
             'meaning': 'Goal proposal/action selection',
             'known_gap': 'No such independently established labels added by this audit; do not copy all behavior futures'},
            {'role': 'complete trajectory value', 'data': 'Native complete-room outcomes plus full-task assessment',
             'meaning': 'Long-horizon task success/cost',
             'known_gap': 'Local changes, name binding and unknown checks cannot be promoted to whole-room success'},
        ],
        'next_algorithm_comparison': {
            'primary': 'Original same-source action-only, D-auxiliary direct-action, and self-generated D-conditioned action comparison remains the identified supervised question; geometric future prediction is a control, not a replacement claim.',
            'source_specification': '../20261110_execution_supervision_review/algorithm_specification.json',
            'complete_generation_contract': [
                'Begin from the original text and common native backend; preserve required instance identities and all unmet/unknown clauses.',
                'Let all policies query the same actual state and use the same asset, add, move, remove and supported relation operations.',
                'For known transition/checks, use the same ordinary preview in all compared arms rather than call one arm a world model.',
                'If testing H, explicitly tag it as desired subgoal. Shared original H labels must also train an auxiliary-H direct-action control; no teacher H/future observations at deployment.',
                'Decode the action, execute on a candidate copy, obtain actual results and continue from that actual state. Do not use frozen teacher trajectories as end-to-end policy execution.',
                'Score full original requirements, semantic unknowns, instance completeness, geometric/function limits, collateral edits and all model/tool/render costs.',
            ],
            'required_controls': [
                'Direct action + common native state/ordinary endpoint check/search',
                'Direct action + original D auxiliary labels (same data as D-conditioned arm)',
                'Self-generated D -> action (no supplied teacher D at inference)',
                'If geometry is introduced: same geometry labels as ordinary auxiliary head with direct action',
                'If future-goal H is introduced: H auxiliary vs H-conditioned action with identical valid labels, model budget and candidate operations',
            ],
            'novelty_boundary': 'HSM/SceneSmith already combine generation, structured state and feedback; NaLA has explicit layout geometry. D conditioning, future auxiliary prediction, ordinary search and rollback are not independently new contributions.',
            'escalation_rule': 'No new large teacher/robot data download. First show a full-room, held-out benefit beyond ordinary state/geometry/auxiliary controls, or revise the mechanism. This audit provides no such benefit yet.',
        },
        'no_new_training_or_paid_api': True,
        'goal_achieved': False,
    }
    dump(HERE / 'theory_and_algorithm.json', theory)

    differences = [
        ['GeomVLA', 'Calibrated scene/motion/action latent conditioning', 'Robot interaction; original point tracks; action-flow controller',
         'Our archived room move assigns an absolute pose; cannot infer the same learning need', 'Paper methods read; implementation unavailable at checked URL'],
        ['CPR', 'Future-informed geometry and motor targets during training', 'Robot multi-frame clips and action chunks; no future input at deployment',
         'Training-only future supervision is a legitimate control, not new scene-generation evidence', 'Pinned repository contains README and one diagram, no model source'],
        ['Ordinary native preview', 'Compute supported F(T(S,A),U)', 'Exposed state and candidate actions, bounded proxy geometry',
         '823 archived comparisons already correct; reuse as common strong control', 'Prior executed evidence, no re-run this turn'],
        ['Desired 3D subgoal', 'Propose which requirement/object relation should become true', 'Requires valid task-conditioned labels or verified candidate outcomes',
         'Different target from copying executed future; independent method advantage unestablished', 'No labels or success manufactured'],
    ]
    with (HERE / 'method_differences.csv').open('w', newline='') as stream:
        w = csv.writer(stream)
        w.writerow(['method', 'learned_or_computed_object', 'assumptions', 'implication_for_this_project', 'evidence_boundary'])
        w.writerows(differences)

    move = s['selection_sensitivity']['native_moves_only']
    def esc(value):
        return html.escape(str(value))
    sections = []
    sections.append('<h1>动作后的三维状态，什么时候值得学习？</h1>')
    sections.append('<p>本轮完成新的训练目标识别审计、两项近期机器人方法的定向阅读，以及七个原生方法的源码边界提取。结论是：保留空间表征作为强对照，暂不把当前布局移动包装成需要学习动力学的世界模型。目标仍是完整可编辑室内场景生成；本轮没有新增整场生成、训练或模型调用，也未取得新的方法效果。</p>')
    sections.append('<h2>1. 新的实测证据</h2><table><tr><th>项目</th><th>结果</th><th>解释边界</th></tr>')
    for a, b, c in [
        ('保存的开发决策', '72条／12轨迹／1个原始房间族', '不是72个独立场景；仍为evaluation-only'),
        ('公开状态＋任务分组', '41组，其中10组后续检查向量不同，涉及30行', '没有包括完整对话、图像和历史，不能宣称完整条件下不可预测'),
        ('只保留真实移动', f"35条／25组，其中{move['x_groups_with_multiple_check_vectors']}组后果不同", '排除无调用、只读、删除后，仍有多种行为后果'),
        ('真实移动造成记录损坏', '25/35', '是几何／要求代理损坏，不是独立人评失败'),
        ('真实调用中修复且无新增记录损坏', '2/56', '仅局部代理改善，不能提升为完整成功或专家轨迹'),
        ('当前目标标签', '理想后续布局0；完整成功标签0；训练准入0', '没有为凑训练量重写原标签'),
        ('复用的普通预测器', '823/823支持检查匹配', '旧证据，未重跑；不覆盖完整功能与语义'),
    ]:
        sections.append('<tr>' + ''.join('<td>'+esc(v)+'</td>' for v in [a,b,c]) + '</tr>')
    sections.append('</table><p>所有记录下，按X查表的检查均方误差为0.013728；按X和动作查表为0。但68个X—动作组只有4行超过每组一行；只看真实移动时34组中只有1组重复、共2行。因此零误差不能证明泛化、确定性或学习效果。它只是描述性样本内统计；源码与已有普通控制才解释了受支持部分为何可直接计算。</p>')
    sections.append('<h2>2. 训练目标必须分开</h2><p>①当前几何重建：帮助模型看清现在。②给定候选动作的后果预测：描述动作会发生什么，可以学习失败后果，但不自动表示好设计。③给定用途的理想子目标：表达应该实现什么，需要合格来源。④整场价值：评价完整轨迹能否完成原始描述，不能由一步碰撞或名称绑定代替。现在这批日志主要属于②；不得直接升级成③或④。</p>')
    sections.append('<p>数学上，行为日志的未来分布是 pμ(Y|S,U)=ΣA p(Y|S,U,A)μ(A|S,U)。这描述的是旧策略做过什么。平方误差预测的是固定条件下的目标均值；SmoothL1并不一般等于这个均值，生成式多模态预测也不必输出均值。不能用本次歧义统计反驳GeomVLA或CPR，更不能认为坏结果没有任何学习用途。关键是标签角色和部署时能获得的条件必须匹配。</p>')
    sections.append('<h2>3. 与最近邻方法的实际区别</h2><p><a href="https://arxiv.org/html/2609.13812v1">GeomVLA</a>使用共享三维坐标下的运动特征指导动作；<a href="https://arxiv.org/html/2610.28164v1">CPR</a>将训练时看到未来的表征转移给部署时只看当前输入的策略。这些提供了合理的表征对照，但它们处理的是机器人交互，不能直接说明设定家具绝对姿态也需要同类世界模型。阅读范围逐项记录，未声称通读附录、逐页看图或运行作者系统。</p>')
    sections.append('<p>CPR固定提交 f2982a37… 的递归树只有README和一张框架图，没有损失或模型源码；GeomVLA项目链接的GitHub/API返回404，另一次Git请求发生TLS错误。这些都是当前访问记录，不能伪称已完成实现核查，也不能由此断言作者从未发布代码。</p>')
    sections.append('<h2>4. 原生工具到底做什么</h2><p>七个方法已按源文件哈希和AST定位：move检查对象与范围、施加配置中的姿态噪声、直接替换transform；remove删除对象；add复制已有注册资产再摆放；generate先获取资产，随后还需要摆放。HSSD获取过程又依赖检索、VLM朝向／材质／质量估计、网格处理和SDF生成。这些阶段不能合并成“已知移动动力学”。噪声实现使用姿态派生种子，也不证明所有生成服务确定。</p>')
    sections.append('<p>此前真实资产请求“wooden nightstand”，模型估为plastic，是尚未独立核实的分歧；不能把该预测当材质真值。未知部分仍包括语义、风格、合格用途选择和完整任务成功。新审计没有调用资产服务，也没有重新执行旧工具。</p>')
    sections.append('<h2>5. 当前可识别的算法比较</h2><p>沿用原训练问题：直接动作；同源决策D作辅助目标、部署直接动作；模型自行生成D再条件化动作。三组共享原始样本、工具和费用口径。若加入几何，先加“同标签几何辅助＋直接动作”强对照；若进一步引入理想三维子目标H，必须明确它是提议，给直接动作组同源H辅助监督，并在部署时由模型自行产生。原24单位中缺少的D或图像不得制造，72条开发日志仍不纳入训练。</p>')
    sections.append('<p>完整生成比较从原始描述和相同原生后端开始，三组都能查询实际状态并使用相同获取、添加、移动与删除操作。受支持的几何后果由共同普通预览处理；每次执行后读取真实状态继续，记录必需实例、原始关系、未知语义、附带破坏、模型与工具总成本。不得把冻结轨迹上的下一动作准确率报告为整场生成成功。</p>')
    sections.append('<p>这是一组可反驳的比较，不是已经证明新颖的损失或CVPR贡献。近期方法已有几何辅助与未来表征，场景方法已有工具反馈和回滚。只有在未参与调参的完整房间上，超过普通搜索、同标签辅助和通用策略，并能解释额外收益来自哪里，才支持上升为研究贡献。目前尚无该证据。</p>')
    sections.append('<h2>6. 交付与资源</h2><ul>')
    for name in ['actual_v2/summary.json', 'actual_v2/target_roles.csv', 'actual_v2/state_task_groups.json',
                 'actual_v2/input_hashes.json', 'native_branch_contract.json', 'reading_ledger.json',
                 'source_acquisition.json', 'theory_and_algorithm.json', 'method_differences.csv', 'COMMANDS.txt']:
        sections.append('<li><a href="'+name+'">'+name+'</a></li>')
    sections.append('</ul><p>审计脚本墙钟约0.09秒（不含阅读、网络、开发与打包）；进程RSS统计包含旧记录的读取解析，不能当GPU训练需求。没有模型调用、优化更新、渲染或新增整场。原输入前后哈希一致。v1保留为早期统计，v2增加仅真实移动敏感性与条件范围限定；没有根据质量换样本。当前无须增加GPU或内存。此阶段只新增小型文本证据，不触碰旧资产和权重。</p>')
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>预测几何的训练目标与工具边界</title><style>body{max-width:1100px;margin:40px auto;font:17px/1.75 system-ui;padding:0 20px;color:#172330}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #ccd4dd;vertical-align:top}th{background:#edf3f8}a{color:#165ba8}p{margin:18px 0}</style><body>' + ''.join(sections) + '</body></html>'
    (HERE / 'stage_report.html').write_text(document)
    (HERE / 'COMMANDS.txt').write_text('''From repository root, without training, network models or tool execution:
python scripts/autodl/audit_future_target_identification.py --out runs/autodl/20261110_predictive_geometry_applicability/reproduce_v3
python scripts/autodl/audit_native_prediction_boundary.py --out runs/autodl/20261110_predictive_geometry_applicability/reproduce_native_contract.json
python runs/autodl/20261110_predictive_geometry_applicability/build_report.py

Use a NEW output directory for the row audit. actual_v1 and actual_v2 are retained evidence.
The report builder reuses already acquired tiny text sources. It does not download weights or open APIs.
''')
    summary = {**s, 'reading_items': len(reading), 'native_methods_reviewed': len(native['source_methods']),
               'actual_result': 'Supervision-role and operator identifiability established for the saved archive; no independent learned benefit established',
               'goal_achieved': False}
    dump(HERE / 'summary.json', summary)
    with (HERE / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'value', 'scope'])
        for key in ['decisions', 'trajectories', 'independent_project_families',
                    'current_state_and_task_groups', 'groups_with_distinct_after_check_vectors',
                    'native_moves', 'native_moves_with_primitive_or_check_damage',
                    'desired_future_labels_available', 'training_admitted_rows', 'model_calls', 'training_updates']:
            writer.writerow([key, s[key], 'Archived development data; no new model/method result'])
    # Keep exact runnable analysis snapshots in the evidence package.
    (HERE / 'scripts').mkdir(exist_ok=True)
    for name in ['audit_future_target_identification.py', 'audit_native_prediction_boundary.py']:
        shutil.copy2(ROOT / 'scripts/autodl' / name, HERE / 'scripts' / name)
    files = [p for p in HERE.rglob('*') if p.is_file() and 'acquisition_v1' not in p.parts
             and '__pycache__' not in p.parts and p.name not in ('artifact_manifest.json', 'evidence.zip')]
    manifest = [{'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(files)]
    dump(HERE / 'artifact_manifest.json', {'files': manifest,
         'raw_third_party_text_distributed': False,
         'old_inputs_unchanged': native['old_inputs_unchanged'] and json.loads((HERE / 'actual_v2/input_hashes.json').read_text())['unchanged']})
    with zipfile.ZipFile(HERE / 'evidence.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files + [HERE / 'artifact_manifest.json']:
            z.write(p, p.relative_to(HERE))
    print(json.dumps({'report': str(HERE / 'stage_report.html'), 'files': len(files),
                      'evidence_bytes': (HERE / 'evidence.zip').stat().st_size}))


if __name__ == '__main__':
    main()
