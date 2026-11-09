"""Assemble source evidence, explicit algorithms and measured CPU receipts."""
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path


ROOT = Path('runs/autodl/20261110_decision_phase_study')
BASE = Path('/root/autodl-tmp/agent_method_preparation_20261027/work/autodl_continuation_20261110')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    phase = json.loads((ROOT/'actual_v3/phase_audit.json').read_text())
    encoding = json.loads((ROOT/'phase_encoding_and_loss_v3.json').read_text())
    reference = json.loads((ROOT/'objective_and_reference_check_v3.json').read_text())
    gate = json.loads((ROOT/'ordinary_stop_gate.json').read_text())
    theory = json.loads((ROOT/'theory_checks.json').read_text())
    cached = BASE/'phase_source_reading_20261110'
    source_rows = json.loads((cached/'source_manifest.json').read_text())
    metadata = json.loads((cached/'commit_metadata.json').read_text())
    read_ranges = {
        'scott_data_helper.py':'Full file read in preceding source pass; factual/counterfactual task markers and label masks.',
        'scott_main.py':'Lines 100–200 in preceding pass; lines 200–310 this pass: data loaders and mixed CE training loop. Not the complete file.',
        'frodo_frodo.py':'Lines 1–170, 200–441 in preceding pass; 230–370, 444–612 this pass. Exact ReasoningModule AST executed only with controlled scalar NLLs.',
        'frodo_train_frodo.py':'Lines 1–160: illustrative data and training entry setup. Not full training entry or actual paper data.',
        'frodo_test_frodo.py':'Downloaded and hashed only; not read as implementation evidence.',
    }
    for row in source_rows:
        path = Path(row['local_path'])
        assert digest(path) == row['sha256']
        row['read_scope'] = read_ranges[path.name]
        row['commit_metadata'] = next(m for m in metadata if m['pin'] == row['pin'])
    upstream_index = json.loads(Path('runs/autodl/20261110_objective_source_review/source_index.json').read_text())
    target_scopes = {
        'src/rollout_client.py':'Lines 760–905: stop handling, candidate sampling, branch evaluation, Critic selection, branch commit and fallback.',
        'src/grpo_rollout.py':'Lines 384–498: snapshot copies, executed siblings, random continuation; preceding review covered surrounding generation.',
        'src/train_grpo.py':'Lines 510–548 and 810–860: local normalization, broadcast trajectory advantage, length/object rewards.',
        'src/grpo_objective.py':'Full file: clipped token policy objective, reference KL, per-completion averaging.',
    }
    for row in upstream_index['sources']:
        if row.get('repository') != 'ZJU-OmniAI/Scene-Designer' or row['path'] not in target_scopes:
            continue
        assert digest(Path(row['local_source'])) == row['sha256']
        source_rows.append(dict(row, read_scope=target_scopes[row['path']], executed=False))
    server = cached/'scene_designer_src_tool_server.py'
    assert digest(server) == 'b0327a7db14b52a66cca585f12642a89127c3004dba767ff5a4503b1ec74c977'
    source_rows.append({'repository':'ZJU-OmniAI/Scene-Designer',
        'commit':'a0565e50ed7d32c0e09132b9311c0b9b91165405','path':'src/tool_server.py',
        'url':'https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src/tool_server.py',
        'local_source':str(server),'sha256':digest(server),
        'read_scope':'Lines 145–195, 235–428: execution/memory helper, stage restrictions, branch snapshots, restore and choose. No native server or renderer run.',
        'executed':False})
    write(ROOT/'source_manifest.json', source_rows)

    ledger = [
        {'title':'SCOTT: Self-Consistent Chain-of-Thought Distillation',
         'url':'https://aclanthology.org/2023.acl-long.304v2.pdf',
         'read':'Prior pass: abstract, introduction, methods 3.1–3.2 (Eq.1–5), experimental setup 4.1–4.4. This pass: exact official factual/counterfactual CE training loop.',
         'finding':'Reason-conditioned answers and counterfactual coupling are established controls, not a new contribution here.',
         'not_read':'Complete results/appendix and visual figures; no reproduction.'},
        {'title':'Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning (FRODO)',
         'url':'https://aclanthology.org/2024.findings-emnlp.882.pdf',
         'read':'Abstract, introduction, §2 interventions, §3.1–3.2 and Eq.1–4; §4 datasets, metrics and baselines. Pinned author-linked January 2026 source sections in source_manifest.',
         'finding':'DPO rationales plus correct/counterfactual answer losses and ranking. Current illustrative source has an NLL sign issue and duplicate positive CE branches in the probed methods.',
         'not_read':'Complete experimental results/appendix or visual figures. No evidence this 2026 demonstration source produced 2024 paper results; no paper-performance conclusion.'},
        {'title':'When Must Faithful Chain-of-Thought Fail? An Information-Theoretic Bound and Its Empirical Limits',
         'url':'https://aclanthology.org/2026.emnlp-main.1823.pdf',
         'read':'Abstract, introduction, related work, §3 definitions and §4 main statements/limitations. No full appendix proofs or empirical reproduction.',
         'finding':'Capacity claims require explicit assumptions. Our finite checks distinguish deterministic encoding from sufficient encoding and use the full mutual-information identity. The basic capacity inequality survives these corrections.',
         'not_read':'Full experiments, appendix proofs, visual figures. No claim that 3D decisions exceed a proven capacity limit.'},
        {'title':'Scene-Designer official tool branches and objective',
         'url':'https://github.com/ZJU-OmniAI/Scene-Designer/tree/a0565e50ed7d32c0e09132b9311c0b9b91165405',
         'read':'Pinned client, server and training sections listed individually in source_manifest.',
         'finding':'Inference ranks executed branches; server commits cached scene/memory state. Training continues one random sibling and broadcasts its global trajectory advantage to the sibling group.',
         'not_read':'All external SceneSmith snapshot internals or side-effect semantics; no full native execution. Broadcast is not a measured counterfactual final return for each sibling.'},
    ]
    write(ROOT/'reading_ledger.json', ledger)
    algorithm = {
        'status':'prospective_fair_control_not_established_novel_method',
        'variables':{'I':'Original task, current object/asset state, available measurements, ordered five views, tool receipts.',
                     'D':'Public supervised proposed-decision text; not a verified receipt or access to private model reasoning.',
                     'A':'One allowed tool call or assistant EOS proposing STOP.',
                     'E':'Actual executor and independent declared-requirement checks.',
                     'Y':'Environment/task correctness, not the policy output itself.'},
        'training':{
          'same_material_separate':'L_D(D|I,phase=D)+L_A(A|I,phase=A)',
          'same_material_paired':'L_D(D|I,phase=D)+L_A(A|I,D,phase=A)',
          'objective':'Sum over original source tokens t: w_t * -log p_theta(token_t | causal prefix_t). w_t=1/(6*N_original_family,original_component).',
          'phase_repartition':'Keep each original token once, including the two terminal records now assigned to D and EOS/newline to A. Never renormalize by the new stage lengths.',
          'missing_annotations':'14 edit states have no D target. Do not fabricate D labels or exclude them from A training.',
          'implementation':'scripts/autodl/phase_objective.py; actual causal shift and coefficient-gradient oracle passed. Full Trainer accumulation integration remains unexecuted.',
          'not_claimed':'Equal targets/coefficients do not imply equal logits, gradients, training compute or deployment performance; prompts/prefixes differ.'},
        'deployment':[
          'Use identical observation and tool access for all conditions. Count every generated token and model invocation.',
          'For the two-stage matched compute comparison, generate D from I in both arms; stop at the closing delimiter. Keep failures in denominator, no teacher lookup or free repair.',
          'Separate arm calls A on I only; paired arm calls A on I plus its actual generated D. D is never treated as a tool receipt.',
          'Execute the same permitted action, or verify measured scope on EOS. EOS alone is not completion. Record unknowns and preservation checks.',
          'Refresh observation from each arm’s own resulting state and repeat within one predeclared budget. Do not splice archived downstream model outputs after feedback changes.',
          'Also report efficient direct-action separate deployment; do not conceal the wasted D pass of the compute-matched diagnostic.'
        ],
        'theory':[
          'For fixed theta and D_hat=g_theta(I,epsilon) with epsilon independent of Y conditional on I, P(Y,D_hat|I)=P(Y|I)P(D_hat|I), hence I(Y;D_hat|I)=0. This is a no-new-environment-information statement, not a no-computation-benefit claim.',
          'A teacher D constructed from (I,A*) can carry target information beyond I; reduced teacher-forced NLL is not deployment evidence. Source supervision is legitimate, but the test prefix must be generated without A*.',
          'H=f(X) gives I(Y;H)<=I(Y;X), not equality without sufficiency. General chain identity subtracts I(Y;C|H). Since it is nonnegative, I(Y;H)<=H(C)+I(Y;H|C) remains valid.',
          'Changing an intermediate record can demonstrate behavioral dependence; it does not establish physical correctness or the hidden internal causal computation.',
          'Changing actual task/geometry can require a different valid action; changing only a false draft should not authorize violations. Irrelevant paraphrases should not require preference reversal.',
          'Multiple valid layouts form a set. Exact tool-string imitation is a fitting check; post-execution requirements, object preservation, unsupported aspects and total cost determine scene utility.'
        ],
        'interventions':[
          {'change':'Task changes, scene fixed','claim_tested':'Conditional design adaptation','outcome':'Recompute valid requirements; allow same valid action, different action, tie or unknown. No automatic flip label.'},
          {'change':'Scene geometry changes, task fixed','claim_tested':'Spatial applicability of a decision','outcome':'Use independently checked actual geometry; no template-only label.'},
          {'change':'Only D is contradicted or paraphrased','claim_tested':'Draft dependence and robustness','outcome':'Fixed physical/task truth. Report violations separately from action-change rate; diagnostic, not human preference.'}
        ],
        'necessary_controls':['Frozen base with full task/tools and same knowledge retrieval',
                              'Action-only adaptation', 'Same-material separated tasks',
                              'Same-material paired tasks', 'Ordinary declared-rule check and bounded geometric/search tools'],
        'scientific_gate':'Multiple independent project families with full-room endpoint improvements beyond ordinary controls; current data are development reconstructions and do not meet this gate.',
        'training_ready':False,
        'remaining':['52/68 prospective components lack complete original views locally.',
                     'Full Trainer scaling/accumulation and actual two-stage model deployment not run.',
                     'No new independent project validation, professional blind evaluation or full-room improvement.',
                     'Latest 920 training final result not in current Git snapshot; do not relaunch based on old status.'],
        'resource_decision':'Continue source/material work. Do not request GPU or expand training on this evidence alone.'
    }
    write(ROOT/'algorithm_and_theory.json', algorithm)
    with (ROOT/'summary.csv').open('w', newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['item','value','scope'])
        for r in [('source_states',24,'6 development families'),('source_tokens_per_arm',2975,'unchanged original targets'),
                  ('new_D_tokens',1726,'includes 394 existing terminal-prose tokens'),('new_A_tokens',1249,'existing targets repartitioned'),
                  ('D_annotated_states',10,'8 edits plus 2 existing stop records'),('D_missing_edit_states',14,'retained in action training'),
                  ('encoded_components',16,'of 68 fixed components; original image hashes'),
                  ('missing_components',52,'local original views incomplete'),('ordinary_gate_agreement',6,'6 supported states, 18 unknown; development only'),
                  ('model_calls',0,'no policy inference'),('optimizer_updates',0,'scalar gradient tests only')]:w.writerow(r)

    html_text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>决策监督、工具分支与理论边界</title>
<style>body{{max-width:1050px;margin:38px auto;padding:0 24px;line-height:1.75;font-family:system-ui;color:#17212b}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4dc;padding:9px;text-align:left}}pre{{white-space:pre-wrap;background:#f2f5f8;padding:16px}}a{{color:#165b9d}}.note{{background:#fff2d8;padding:14px}}</style>
<h1>把“理由影响动作”变成可公平检验的三维问题</h1>
<p>{datetime.now(timezone.utc).isoformat()}｜本轮 CPU 实际材料处理、目标函数检查与固定源码阅读；0 模型推理、0 参数更新、0 新场景生成。</p>
<p><strong>当前能交付的是可执行的同材料对照及其理论边界，还不能交付已经成立的新算法优势。</strong>说明与动作耦合已有直接前作。三维研究必须证明：依据实际用途和空间条件作出的决定，能改善完整执行结果，并超过普通知识检索和几何工具。</p>
<h2>1. 找到并修正的具体问题</h2>
<p>旧 24 个状态含 22 个编辑与 2 个停止。8 个编辑有独立 D 目标，14 个编辑没有；两个停止的说明和终止符原先全部算在 A。若部署统一采用“D→A”，便没有独立的 D→STOP 示范。这是拟议新协议的覆盖问题，不能倒推为旧单次生成失败的原因。</p>
<p>本轮只重新划分既有 token：原 8 个 D 保持；把两个停止记录的说明划给 D，EOS 与换行留给 A。v2 的动作提示仍要求输出说明，与 EOS-only 目标不一致，因此保存 v2 后改为 v3：A 明确输出工具或终止；终止仍须由执行检查验收。没有新增理由、正确坐标或专业标签。</p>
<table><tr><th>项目</th><th>结果</th><th>解释边界</th></tr>
<tr><td>每组监督内容</td><td>2,975 token，34 个成分</td><td>两组共 68 成分；不是 68 个独立样本</td></tr>
<tr><td>D / A token</td><td>1,726 / 1,249</td><td>只是阶段重分配，原每个 token 的系数保留</td></tr>
<tr><td>原始目标权重质量</td><td>11/6</td><td>原家族／成分归一化；不是将新阶段分别重新归一化</td></tr>
<tr><td>原图实际编码</td><td>{encoding['counts']['encoded']}/68</td><td>5 个状态完整；52 成分缺原图，仍保留在清单</td></tr>
<tr><td>CPU 编码及损失检查</td><td>{encoding['wall_seconds']:.2f} 秒，峰值 {encoding['peak_rss_kib']/1024:.1f} MiB</td><td>无模型权重；不能据此估算训练显存</td></tr></table>
<p>实际检查了原图张量哈希、提示与答案边界、D 前缀不重复计损失、标签移位和梯度。构造概率的独立 CE 计算为 {encoding['loss_probe']['actual']:.12f}，与代码相同；故意不移位得到 {encoding['loss_probe']['unshifted_wrong']:.12f}，能被分辨。分微批求和一致，但尚未接入完整 Trainer 验证其累积缩放。</p>
<p class="note">旧材料与结果未改。首次划分尝试的换行分词冲突、一次源文件路径错误均留档。相同 token 与权重不等于相同模型损失或梯度：两组的动作上下文刻意不同。</p>
<h2>2. 已实现的控制算法与真正的自变量</h2>
<pre>输入 I：原始任务、当前对象／资产、可见量测、五视图、工具回执
D：公开的拟议决定说明；A：合法工具调用或 EOS 提议停止

分开学习：D ~ pθ(D | I, D任务)；A ~ pθ(A | I, A任务)
配对学习：D ~ pθ(D | I, D任务)；A ~ pθ(A | I, D, A任务)
L = Σ原始目标token w_t · [-log pθ(token_t | 前缀)]
w_t = 1 / (6 × 原家族、原成分的token总数)

推理：两组都从同一观察生成 D；配对 A 读取自己生成的 D，分开 A 不读取。
两组每步执行同样工具、刷新自己的真实场景；错误及所有调用都计入预算。
EOS 仅是停止提议，不能自动宣布整屋完成。</pre>
<p>这套相同调用次数的诊断之外，还应报告分开组直接生成 A 的实际低成本部署；不能隐藏额外 D 调用。14 个没有 D 标签的状态仍训练 A，缺失不能用新编理由补齐后称“同材料”。新材料尚无完整训练准入。</p>
<h2>3. 理论上能证明什么</h2>
<p>固定模型参数，若 D̂=gθ(I,ε)，且随机性 ε 在给定 I 后独立于真实任务结果 Y，则 P(Y,D̂|I)=P(Y|I)P(D̂|I)，所以 I(Y;D̂|I)=0。这个普通条件独立关系只说明自生成说明不凭空增加环境信息；有限模型仍可能从更多计算、辅助监督与表达中受益。它不是否定理由学习的定理。</p>
<p>教师说明若根据正确动作 A* 编写，就可能携带 I 中没有的目标信息。因此教师前缀下损失更低不能替代真实部署。有限概率核查中，教师说明的条件互信息为 1 bit，自生成说明为 0；这是数学例子，不是三维测量或新论文结果。</p>
<p>新读的容量论文涉及另一种“说明是否充分表达内部信息”的问题。需要保留前提：确定性编码 H=f(X) 只保证 I(Y;H)≤I(Y;X)，等号需要充分性；一般链式关系还含 −I(Y;C|H)。本轮枚举了两项反例，并确认 I(Y;H)≤H(C)+I(Y;H|C) 的基本容量上界仍成立。不能把这些理论包装成“三维复杂所以模型必然需要本方法”。</p>
<p>更直接的研究界限是：<strong>动作受错误说明影响，不等于空间推理正确。</strong>改变真实用途／几何，与仅插入一份错误说明，是两种干预。前者应重新判断可行方案；后者不能让错误说明覆盖实际要求。合理方案可有多个，不能要求条件一变就必须反转，更不能把复现同一句动作当作整屋质量。</p>
<h2>4. 最近邻及源码核查</h2>
<table><tr><th>来源</th><th>已有机制／本轮发现</th><th>对本研究的约束</th></tr>
<tr><td><a href="https://aclanthology.org/2023.acl-long.304/">SCOTT</a></td><td>事实与反事实任务，说明前缀与答案损失；官方训练混合两项 CE。</td><td>“让模型不能忽略说明”本身不新。</td></tr>
<tr><td><a href="https://aclanthology.org/2024.findings-emnlp.882/">FRODO</a></td><td>说明偏好与答案依赖目标；论文、当前源码必须分开核对。</td><td>依赖强度不是三维任务正确性的替代指标。</td></tr>
<tr><td><a href="https://github.com/ZJU-OmniAI/Scene-Designer">Scene-Designer</a></td><td>已有说明—动作数据与执行后分支评价。</td><td>更多说明、多候选、回滚均不能独立作为新增贡献。</td></tr></table>
<p>FRODO 当前作者链接源码的提交日期为 2026-01-14，训练入口包含演示样本；本轮没有证据认定它就是 2024 论文实验代码。在该固定版本的 ReasoningModule 中，positive NLL=0.2、negative NLL=0.8 时，margin 实际值为 {reference['reference_margin_probe']['actual_loss']}，梯度为 [-1,+1]；按其“正确答案应得更高对数概率”的注释，应为 0.4、[+1,−1]。前两个 CE 分支还收到相同正确说明与答案，不能当成论文式(3)的反事实目标。精确方法 AST 的 CPU 标量核查已保存；这不否定论文报告的实验结果，也未修改第三方代码。</p>
<h2>5. 工具分支到底执行了什么</h2>
<p>客户端先生成候选，服务端从同一场景快照分别执行，并为每支复制工作记忆。Critic 读取执行后分支，再由 choose 恢复选中的已执行场景与记忆；choose 不再次调用那些工具。本轮核查的分支服务仅接受 furniture 阶段。所有分支工具、渲染、模型与 Critic 成本都须计入，不能只统计最后提交的一个动作。</p>
<p>训练入口则随机选择一个已执行兄弟分支继续，将最后轨迹的全局长度／对象奖励优势广播到该组全部分支。这个广播不是“每个兄弟动作都有独立完整续建的反事实标签”。若以后研究行动后果预测，必须明确哪些后果实际执行过、哪些只是共用轨迹权重。</p>
<p>这里只读了服务端的 scene/memory 恢复接口；没有运行外部 SceneSmith 快照恢复、检索、渲染或检验所有文件／资产副作用。不能把源码存在当作整个分支系统已经在本机可靠复现。</p>
<h2>6. 普通方法已经解释了哪些行为</h2>
<p>规则直接读取模型本来就可见的用途要求、桌间距、座位对齐与接近测量，输出“需编辑／限定范围内停止／未知”。固定 24 状态中仅 6 个具有支持的测量合同，规则在这 6 个上的编辑／停止判断全部与参考一致；另外 18 个明确未知。它不产生坐标、不执行场景，也不是未见项目成绩。但这一结果要求下一轮保留普通规则对照，不能只用模型停止准确率证明专业知识学习。</p>
<h2>7. 当前可以执行的研究判断</h2>
<p>值得保留的是“何时采用哪项设计决定，并保住完整要求”的问题。当前具体实现仍是普通同材料监督的机制检验，不是已确认的新损失或新规划算法。先收齐已有训练结果，完成材料与实际部署接口，再比较同材料分开／配对、动作适配、充分提示与同知识检索，以及普通工具。所有方法在真实场景终态上评分，允许平局和未知，按独立项目划分。</p>
<p>继续投入所需证据是：在多个未参与调参的完整项目中，配对机制改善实际要求完成或同等质量成本，且收益不能由材料增加、格式修复、工具信息不等或普通规则解释。若只改善说明文字或局部停止，不扩大论文主张。现有 CPU 结果没有达到这个条件，本轮不申请 GPU，不启动训练。</p>
<p><a href="algorithm_and_theory.json">算法与理论合同</a> · <a href="reading_ledger.json">实际阅读范围</a> · <a href="source_manifest.json">固定源码与哈希</a> · <a href="summary.csv">汇总</a> · <a href="phase_encoding_and_loss_v3.json">编码和梯度证据</a> · <a href="objective_and_reference_check_v3.json">来源权重与参考源码检查</a> · <a href="ordinary_stop_gate.json">普通规则逐项结果</a> · <a href="theory_checks.json">概率反例</a></p>
</html>'''
    (ROOT/'stage_report.html').write_text(html_text)
    source_files = ['audit_decision_phase_support.py','audit_declared_stop_gate.py','check_phase_objective_and_reference.py',
                    'phase_objective.py','verify_phase_material.py','audit_information_claims.py','report_decision_phase_study.py','verify_decision_phase_study.py']
    manifest = {'schema':'decision-phase-source-and-theory-v1',
                'source_sha256':{str(Path('scripts/autodl')/p):digest(Path('scripts/autodl')/p) for p in source_files},
                'output_sha256':{str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*') if p.is_file() and p.name not in ('manifest.json','verification_receipt.json')},
                'all_model_calls':0,'all_optimizer_updates':0,'new_scene_executions':0,
                'full_goal_complete':False,'gpu_request_ready':False,
                'prior_turn_audit':'Git review confirmed existing evidence but did not implement new methods. This continuation made new material, objective and source-level progress.'}
    write(ROOT/'manifest.json', manifest)
    print(json.dumps({'outputs':len(manifest['output_sha256']),'sources':len(manifest['source_sha256']),'report':str(ROOT/'stage_report.html')}))


if __name__ == '__main__':
    main()
