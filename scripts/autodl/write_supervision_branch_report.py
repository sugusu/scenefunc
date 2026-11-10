"""Build the evidence-backed algorithm comparison and reading record for this stage."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main(repo,s):
    prior=repo/'runs/autodl/20261110_complete_native_runtime'
    sdk=json.loads((s/'native_sdk_v1/result.json').read_text())
    admission=json.loads((s/'training_admission_v2/result.json').read_text())
    replay=json.loads((s/'native_reference_replay_v1/result.json').read_text())
    boundary=json.loads((s/'upstream_target_boundary_v1/result.json').read_text())
    original=json.loads((prior/'source_before_after.json').read_text())
    verified=[]
    for group in ('original_source_inputs','pre_inference_code'):
        for x in original[group]:
            p=Path(x['path']);now=sha(p) if p.exists() else None
            verified.append(dict(group=group,path=str(p),
                baseline_sha256=x['expected_sha256'],current_sha256=now,
                same=now==x['expected_sha256']))
    assert all(x['same'] for x in verified)
    source_manifest=json.loads((s/'sceneorchestra_source/source_manifest.json').read_text())
    for x in source_manifest['files']:
        p=s/'sceneorchestra_source/source'/x['path']
        assert sha(p)==x['sha256']
    sdk_protocol=json.loads((s/'native_sdk_v1/SDK_PROTOCOL.json').read_text())
    for path,expected in sdk_protocol['code_sha256'].items():
        assert sha(Path(path))==expected
    dump(s/'source_before_after.json',dict(checks=verified,unchanged=True,
         downloaded_prior_source_files_verified=len(source_manifest['files']),
         native_probe_frozen_code_unchanged=True,
         cleanup_is_separate_old_installation_not_source_or_evidence=True))
    reading=[
      dict(title='SceneOrchestra',URL='https://arxiv.org/html/2604.19907v2',
           version='v2, 2026-08-22',read='HTML abstract, introduction, related work, complete method, experiment setup/results/ablations, supplementary sections6-13. References traversed as text, not every cited paper read.',
           figures='Captions read; figures/PDF not visually inspected this stage.',
           code_pin=source_manifest['pin'],
           code_read=['constants.py','prompts.py','records.py','trajectory.py','scoring.py','datasets.py','sceneweaver.py','interleaved.py','README.md','NOTICE','LICENSE'],
           code_executed='Original lightweight trajectory/SFT/DPO/score builders only, with clearly synthetic boundary records. No SceneWeaver scene generation or training.',
           difference='Already learns local and full tool trajectories; mandatory strong prior for any learned planning claim. Scalar score-selected prefixes are not independent certificates of every original task clause.',
           data='Public code expects private/user-provided S1/S2/S3 instruction files and externally acquired SceneWeaver. No original full training trajectories acquired.'),
      dict(title='ThinkBLOX',URL='https://arxiv.org/html/2607.13539v1',
           read='This stage: abstract and targeted methodology III through III-B, including reasoning-action consistency and normalization equations. Earlier reading is linked in prior stages; no new full-paper claim.',
           code_executed=False,figures='Not visually inspected.',
           difference='Already trains reason-then-act placement and reasoning-action consistency. Such components alone cannot identify a new contribution.'),
      dict(title='SceneReVis',URL='https://arxiv.org/html/2602.09432v1',
           read='This stage: abstract and method3.1-3.4; exact original training rows not read because dataset access remains unacquired.',
           code_executed=False,figures='Not visually inspected.',
           difference='Already has joint diagnostic reasoning/tool SFT and multi-turn GRPO. POMDP formulation is prior art, not a new claim.',
           data_card_URL='https://huggingface.co/datasets/runder1/SceneChain-12K',
           dataset_revision='3fdc902027659d256aae4dc507b40dcd7e5b601d',
           gate='Public metadata confirms auto-gate and identity/research terms; no form submitted or identity invented. No dataset rows acquired. Optional data branch only.'),
      dict(title='Scene-Designer native dispatch and SceneSmith tools',
           URL='https://github.com/ZJU-OmniAI/Scene-Designer',
           dispatch_pin='a0565e50ed7d32c0e09132b9311c0b9b91165405',
           native_pin='e888297a8b0a2f3acbb8634edf9def59cceab836',
           read='Actual ToolEnvironment dispatch, memory/reset, normal agent/manager construction; AssetRegistry serialization/loading/scaling and furniture add/move/remove/rescale/snap paths.',
           actual_executed='Seven original SDK rescale calls through normal ToolEnvironment, restored real room assets and registry, no model.',
           qualification='Scene and registry aliases retained; rescale snapshot verified only. No new full-tool/RNG/worker-state qualification.',
           dataset='Publisher API currently lists Scene-OTC-Bench evaluation dataset; no OTA35K training download acquired. Evaluation-only dataset never used as training.'),
      dict(title='AI-Agents-in-Depth-zh-CN.pdf',
           read='No additional pages read this stage. Earlier exact reading boundaries remain in prior ledgers; no claim of full312-page reading.',
           pages_previous='1-6,223-230,239-243,248-256; other earlier targeted reading only as already recorded.',
           consequence='Do not infer new3D novelty from SFT/RL, OPD, memory or checkpoint terminology.')]
    dump(s/'reading_ledger.json',reading)
    algorithm={
      'status':'Falsifiable conditional-supervision study; not a verified novel method or a trained result',
      'target':'Full editable indoor3D generation from unchanged original text, with actual assets and independently evaluated required objects, relations and usability; not box repair or next-token accuracy alone.',
      'research_question':'With the same source knowledge, action examples, observations and execution budget, does correctly paired decision supervision improve the student policy on new full-room tasks beyond auxiliary supervision and existing joint reasoning-action training?',
      'novelty_boundary':'Reason-then-act, whole-trajectory SFT/DPO, memory, rollback, ordinary ID references, tri-state label admission and source provenance all have existing precedents. They are common controls or research hygiene here. No new loss or CVPR-level contribution claimed.',
      'observations':'X_t=(original_task, current_scene_graph, actual_multi_views, historical_tool_receipts, available_tools_and_assets, remaining_budget). An authored or generated rationale is not measured scene evidence.',
      'state':'Full simulator state includes scene objects, asset files, registry, relevant caches/RNG/workers. Current byte/registry reset is empirically qualified only for rescale on existing registered furniture. SceneJSON alone is insufficient; do not claim general Markov sufficiency from the finite probe.',
      'actions':'Composite output (D_t,A_t); D is self-generated deliberation, A is an executable native call or explicit completion. Native runtime feedback is produced by a shared frozen tool kernel kappa_0.',
      'estimand':'J(theta;kappa_0,B)=E[1{normal task termination AND every frozen original requirement AND common geometry/usability validity has sound true evidence before budget B}]. Unknown/unsupported checks and runtime deadlines remain in the evaluation denominator, never fabricated positive or preferred labels.',
      'costs':'Shared B=(forward_calls, generated_tokens, native_tool/candidate_attempts, elapsed_time). Report all dimensions. A scalar time penalty defines a different estimand unless its exchange rate is prospectively fixed. Data acquisition/training and test-time costs are separate.',
      'data_actual':admission,
      'arms':{
        'A':'Same action targets, ordinary action-only CE.',
        'AUX':'Same action CE plus same supported D targets predicted from X independently. Actions do not consume D. This isolates extra source-text supervision.',
        'J':'Original same-source joint autoregressive D then A in one response. It is a strong existing reason-action baseline, not omitted.',
        'P':'Generate own D in a separate pass, then produce A conditioned on that D. Extra forward/prefix/token costs charged. If token conditioning is identical to J, this is an execution factorization, not an independent new learning principle.',
        'T':'Whole-trajectory SFT/DPO objective as a strong prior. Original SceneOrchestra uses SceneWeaver idea-level tools, while this host uses native typed calls and runtime IDs. Adaptation is an explicitly labeled mechanism control, not paper reproduction; all arms share the ordinary reference binder.',
        'F':'Frozen base + equal source retrieval, actual feedback and bounded ordinary search using the same tools. Shared snapshot qualification applies to all branches.'},
      'losses':{
        'A':'L_A=-sum_i w_i^A log pi_theta(A_i*|X_i)',
        'AUX':'L_AUX=L_A-lambda sum_{i:original D present} w_i^D log pi_theta(D_i*|X_i)',
        'J':'L_J=-sum_{i:D present}[w_i^A log pi_theta(A_i*|X_i,D_i*)+lambda w_i^D log pi_theta(D_i*|X_i)] plus original no-D action targets. Token/family masses retained; missing D never authored from scoring answers.',
        'P':'Same component losses as the existing phase contract; different real prefixes must be reported. Teacher forcing uses D*, deployment uses sampled D; oracle D at evaluation is privileged input.',
        'T':'Existing stepwise and whole-trajectory CE/DPO on the SAME admitted task set; scalar composition-score preferences retain that label identity and are never called independently verified full-task preferences.'},
      'actual_new_implementations':[
        'NativeEditSnapshot: normal scene/registry aliases plus mutable SDF/auto-save bytes and render-cache reset; actual SDK rescaleAB/BA probe.',
        'trajectory_supervision_contract: common original-clause denominator and terminal label admission with unknown evidence preserved.',
        'audit_actual_training_admission: real35call ledger,24local-unit provenance and original10decision-target availability.',
        'native_trajectory_references: literal references into only earlier successful output receipts, exercised on all35archived calls;17calls reference returnedIDs.',
        'exercise_sceneorchestra_targets: unmodified pinned lightweight builders executed on explicitly synthetic boundary witnesses.'
      ],
      'not_done':[
        'No learned policy comparison or optimizer step in this stage.',
        'No new model-predicted full plan, native new room or reference-based full simulator execution.',
        'No complete positive room admitted from the audited new native bedroom.',
        'No author training dataset rows downloaded; reconstructed24units are not OTA35K or real architect tool trajectories.',
        'No general asset-acquisition, move/add/snap/remove or worker/RNG reset qualification.',
        'No claim that the sampled references/label rule are new3D algorithms.'
      ],
      'decision_rule':'P/J must improve actual original-task completion or quality-cost frontier on held-out room/source families beyond A/AUX/F/T. Better teacher-forced CE, more fluent D, successful SDK replies or more audit checks alone do not suffice. If AUX or existing J/T explains the effect, revise the claim instead of relabeling it.',
      'next_action_without_paid_API':'Use the now-audited common host/data contract to prepare actual rollout qualification and training-data acquisition separately. Do not silently reopen closed inference budgets or start training. Data-gate failure is optional, not a whole-project termination.',
      'goal_achieved':False,'training_updates':0,'paid_API_calls':0}
    dump(s/'algorithm_and_theory.json',algorithm)
    theory='''算法、识别条件与可否证结论（本轮无训练）

1. 比较的到底是什么
X为当时可用的真实输入，D为模型自己生成的决策文本，A为可执行动作。
pi(D,A|X)=pi(D|X)pi(A|X,D)只是概率链式分解，不是新损失。
在容量不受限、动作策略类相同的理想情况下，任意q(A|X)都可以用常量D和pi(A|X,D)=q(A|X)表示。
因此，插入D本身没有普遍提高最优可达成功率的定理。有限数据、优化、表示与计算分配可能改变实际表现，必须实验。
如果D只从X生成、没有另查外部信息，那么在固定模型下，它不是新观测到的房间真相。
训练时给正确D、测试时用自己生成D是分布变化。必须用自己的D闭环执行，不能把teacher-forced准确率当成功率。
拆成两次调用若保持完全相同的条件分布，只是同一联合策略的采样分解；前缀、位置或预算不同时须披露。

2. 可识别的监督作用
A对AUX检验额外决策文本监督；AUX对J检验动作是否消费匹配D；J对P检验两阶段执行机制。
F控制普通检索/反馈/搜索；T控制已有完整轨迹学习。各对照不是可以任意省去的弱基线。
同一项目的不同视角、同一布局变换和同源动作不能跨训练/测试泄漏。应按来源项目、房间及资产关系分组。
当前24条只覆盖6个本地重建来源族；10条有本地原始D目标，其中8条标有公开资料依据，2条是局部终止控制文本。
没有D的14条不能凭评分答案补写。保留既有token与来源族权重，不能通过复制理由改变有效数据量。
这不构成足够的整场学习样本量，也不把重建动作说成原设计师操作记录。

3. 工具是否也是学习策略的一部分
固定工具kernel kappa_0时，轨迹score包括每个生成的D和A的log概率。
若同一可训练模型也改变检索/工具VLM，正确联合梯度还含工具输出log kappa_theta；只算policy项识别不了共享模型的总效果。
若研究policy适配，应在工具角色明确禁用adapter并验证；本轮未训练，未验证adapter切换。
状态包括JSON之外的可变资产。实际同一个JSON父状态，文件/注册表未恢复时，下一次.95缩放的SDFscale为.4809375；干净分支为.534375。
这是实际原生SDK反例。普通字节/注册表恢复能纠正此有限分支；不能称为新算法，也不能外推到未测试的生成、缓存与RNG。

4. 哪种终止标签可信
设原始任务有m个固定子句c_j，结果标签v_j属于true/false/unknown。
若验证器对其已验证结论可靠且覆盖全部原始子句，所有v_j=true与正常终止可作为整场正样本的充分证据。
unknown不等于真实false，不能用于偏好负标签；但评测时未证明完成的尝试不能被计作成功，截止与拒绝也留在分母。
如果有一个可靠false子句，完整任务指标区间为[0,0]；全部可靠true为[1,1]；其余为[0,1]。
只有第一个结果的下界大于另一个上界时，这些证据才足以决定完整任务指标的严格偏好。
这里的区间是逻辑可识别范围，不是统计置信区间，也不是完整审美排序。
当前原生卧室五条原始要求复核仍unknown；共同几何有效性与功能可用性也各保留为unknown，不能只查名字就算整场正例。
v2明确区分原文五项与共同有效性两项；v1保留为范围较窄的历史记录，没有把旧结果改好。
所以没有新整场正样本，也没有把未证实的质量强行标成偏好负例。

5. 代理分数和原始目标
SceneOrchestra的组合分数同时考虑物体数、物理检查、视觉评分和累计时间。其原版构造器按门槛保留截断前缀并补成功终止。
本轮实际运行原版代码，构造清楚标记的逻辑边界例：给定分数8可通过7.5门槛，即使另一个原始要求尚未证实。
这只证明门槛规则不逻辑蕴含逐项完成；没有测其真实失败率，也未复现其效果。
保留其原始标签作强对照，不能偷偷替它换规则后仍宣称论文原样复现。
对象数量或无碰撞率增加不等于原始要求完成。不能用额外摆件替代缺少的主要家具，不能用名字代替类别/功能证据。

6. 当前能下的结论与下一步
已有共同基线可用的rescale回滚、动态ID绑定和真实数据来源账本；现有J/T等直接前作使“思考再行动/训练工具规划”不能单独承担创新。
仍值得检验的是：相同资料与同等预算下，匹配的、依据当前状态作出的设计决策监督，是否在新房间的实际完成中产生AUX/J/T不能解释的收益。
目前没有这个效果证据，不声称已找到CVPR主创新。即使文字损失更好，也必须回到完整三维场景闭环验证。
旧截止预算保持关闭；本轮没有新推理、训练、付费API或大模型下载。数据访问支线不影响继续做现有源代码与数据准备。
'''
    (s/'theory_notes.txt').write_text(theory)
    commands=[
      'PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/audit_actual_training_admission.py --repo . --out /dev/shm/native_admission_reproduction_NEW',
      'PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/exercise_sceneorchestra_targets.py --stage runs/autodl/20261111_supervision_and_branch_state --out /dev/shm/sceneorchestra_builder_reproduction_NEW',
      'PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/compile_native_reference_replay.py --repo . --out /dev/shm/native_reference_reproduction_NEW',
      'Native SDK probe requires native_python and native_dependency_paths from the prior ENVIRONMENT.json; command below reproduces7tool calls, normal CPU native workers, no model.',
      'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(python -c \'import json; print(":".join(json.load(open("runs/autodl/20261110_complete_native_runtime/ENVIRONMENT.json"))["native_dependency_paths"]))\')" /root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python scripts/autodl/probe_native_registry_branches.py --stage runs/autodl/20261111_supervision_and_branch_state --out /dev/shm/native_sdk_reproduction_NEW']
    (s/'reproduce.txt').write_text('Run from '+str(repo)+'\nUse fresh output paths; existing results are never overwritten.\n\n'+'\n\n'.join(commands)+'\n')
    csv_rows=[['native_SDK_rescale_calls',sdk['actual_native_tool_calls'],'actual original SDK, not generated rooms'],
              ['native_SDK_seconds',sdk['seconds'],'includes normal CPU worker startup'],
              ['JSON_only_asset_leak',sdk['JSON_only_same_scene_but_different_files_and_registry'],'diagnostic'],
              ['full_restore_order_invariant',sdk['byte_and_registry_restore_order_invariant_for_A_and_B'],'rescale only'],
              ['actual_native_saved_calls',admission['native_tool_calls'],'old model rollout, audited this stage'],
              ['calls_success_flag',admission['native_call_outcomes']['native_success'],'not semantic task completion'],
              ['local_units',admission['local_units'],'project reconstructions, not author teacher trajectories'],
              ['local_decision_targets',admission['existing_local_decision_targets'],'8 sourced rationale +2 local terminal controls'],
              ['complete_positive_rooms_admitted',0,'audited native bedroom only'],
              ['archived_call_bindings_replayed',replay['total_calls'],'argument replay, no tool execution'],
              ['calls_using_returned_ID_references',replay['referenced_calls'],'same ordinary binder for all policy arms'],
              ['new_model_forwards',0,'previous closed budgets stay closed'],
              ['new_training_updates',0,'no optimizer'],['paid_API_calls',0,'no new paid API']]
    with (s/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','value','scope']);w.writerows(csv_rows)
    links=['algorithm_and_theory.json','theory_notes.txt','reading_ledger.json','summary.csv',
           'native_sdk_v1/result.json','native_sdk_v1/measured_branches.json',
           'training_admission_v2/result.json','training_admission_v2/native_calls.jsonl',
           'training_admission_v2/local_units_provenance.json','training_admission_v2/terminal_admission.json',
           'native_reference_replay_v1/result.json','native_reference_replay_v1/archived_symbolic_plans.json',
           'upstream_target_boundary_v1/result.json','source_before_after.json',
           'sceneorchestra_source/source_manifest.json','storage_cleanup_receipt.json','reproduce.txt','evidence.zip']
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in csv_rows)
    article='''<h1>完整场景决策学习：监督、训练目标与原生分支的证据</h1>
<p>结论：继续有明确反证的学习比较；尚未证明一种新方法有效。此次完成了原生SDK分支反例、共同回滚、真实监督来源审计、公开原版数据构造器验证和完整轨迹对照所需的动态ID接口。没有训练、新模型推理或新生成房间。</p>
<h2>直接前作怎样改变对照</h2>
<p>SceneOrchestra已经做局部及整条工具轨迹学习；ThinkBLOX已有reason-then-act和一致性奖励；SceneReVis已有诊断推理、工具动作和多轮学习。因此，仅添加理由、回滚或规划不构成区别。逐项阅读范围与未做复现见阅读记录；没有把源码阅读写成模型复现。</p>
<h2>实际原生分支</h2>
<p>在新RAM副本中按正常原生构造创建工具与注册表，经ToolEnvironment执行7次缩放，保持场景与注册表对象别名。JSON恢复后逻辑scale正确，但资产SDF和注册表仍受前一分支污染；字节和注册表恢复能复现本例A/B。仅资格化已有资产的rescale，不外推到所有工具、随机数或工作进程状态。</p>
<h2>实际监督可用范围</h2>
<p>35条真实工具调用中30条有成功标志，4条是没有布尔成功证书的检查文本，1条是失败的资产生成批次（其中仍有部分资产生成）。19次决策文本不是19次成功设计。24条本地单元来自6个来源族的项目重建；10条有D目标，其中8条有资料支持，2条为局部终止控制。14条缺失D不补写。SceneChain只读到公开元数据，没有拿到训练行；评价基准未混入训练。</p>
<h2>共同接口和强基线</h2>
<p>整条计划不能预知原生工具稍后才返回的ID。普通literal引用适配器在35条真实存档调用上复原了原参数，其中17条引用先前返回的ID。它拒绝未来/失败结果引用，不执行任意代码；尚未让模型预测计划或重新执行整场。原始SceneOrchestra使用另一套工具，此处不能称其论文配置复现。</p>
<h2>理论与可否证算法</h2>
<p>主要比较A、AUX、J、P，加入普通反馈/检索/搜索F与完整轨迹学习T。资料、动作目标、工具权限、观察与预算一致；P必须消费自己生成的D。链式分解不提供新颖性或成功率保证。只有在未参与开发的新房间上改善原始要求完成或质量成本前沿，且AUX/J/T解释不了时，才支持继续扩大候选机制。</p>
<h2>运行证据</h2>'''
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>监督与分支状态研究记录</title><style>body{max-width:1000px;margin:40px auto;padding:20px;font:16px/1.65 system-ui}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:7px}pre{white-space:pre-wrap;background:#f5f5f5;padding:20px}a{overflow-wrap:anywhere}</style>'+article+'<table><tr><th>指标</th><th>值</th><th>范围</th></tr>'+rows+'</table><h2>推导与限制</h2><pre>'+html.escape(theory)+'</pre><h2>交付文件</h2><ul>'+''.join('<li><a href="'+x+'">'+x+'</a></li>' for x in links)+'</ul><p>来源：<a href="https://arxiv.org/html/2604.19907v2">SceneOrchestra</a>；<a href="https://arxiv.org/html/2607.13539v1">ThinkBLOX</a>；<a href="https://arxiv.org/html/2602.09432v1">SceneReVis</a>。未改变旧结果、未删除唯一原始证据。删除的仅是未在当前环境使用的旧NumPy安装，已保留逐文件哈希和安装元信息。</p></html>'
    (s/'stage_report.html').write_text(page)
    code_names=['native_edit_snapshot.py','probe_native_registry_branches.py',
                'trajectory_supervision_contract.py','audit_actual_training_admission.py',
                'exercise_sceneorchestra_targets.py','native_trajectory_references.py',
                'compile_native_reference_replay.py','write_supervision_branch_report.py']
    bundle=s/'evidence.zip'
    included=[p for p in sorted(s.rglob('*')) if p.is_file() and p.name not in ('evidence.zip','delivery_verification.json')]
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in included:z.write(p,str(p.relative_to(s)))
        for n in code_names:z.write(repo/'scripts/autodl'/n,'code/'+n)
    delivery=dict(files={str(p.relative_to(s)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in included},
                  evidence_zip=dict(bytes=bundle.stat().st_size,sha256=sha(bundle)),
                  scripts={n:sha(repo/'scripts/autodl'/n) for n in code_names},
                  source_hashes_unchanged=len(verified),
                  actual_qualified_SDK_calls=sdk['actual_native_tool_calls'],
                  delivered_files_are_not_generation_quality_tests=True,
                  paid_API_calls=0,model_forwards=0,training_updates=0,goal_achieved=False,
                  disk_free_GiB=shutil.disk_usage(s).free/2**30)
    dump(s/'delivery_verification.json',delivery)
    print(json.dumps(dict(stage=str(s),source_hashes_unchanged=len(verified),
                        evidence_zip_bytes=bundle.stat().st_size,
                        disk_free_GiB=delivery['disk_free_GiB'],goal_achieved=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve())
