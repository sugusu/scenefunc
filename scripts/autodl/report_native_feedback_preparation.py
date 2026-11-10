"""Consolidate actual branch/source/data/input evidence; no new experiments."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main(repo, stage):
    branch=json.loads((stage/'move_SDK_v2/result.json').read_text())
    data=json.loads((stage/'feedback_material_v2/result.json').read_text())
    inputs=json.loads((stage/'multimodal_feedback_inputs_v2/result.json').read_text())
    objective=json.loads((stage/'SDPO_objective_v1/result.json').read_text())
    assert branch['actual_native_tool_calls']==30 and branch['same_parent_26th_blocked_by_unchanged_guard']
    assert branch['repeated_A_B_outcomes_equal_with_parent_restoration']
    assert data['policy_packets']==19 and data['native_tool_results']==35
    assert inputs['native_mRoPE_inputs_prepared'] and inputs['same_output_target_sequences']==19
    assert not inputs['full_trainer_admitted'] and objective['VLM_forwards']==0
    previous=json.loads((repo/'runs/autodl/20261111_reasoning_controls/source_before_after.json').read_text())
    checks=[]
    for item in previous['original_inputs']:
        digest=sha(Path(item['path']))
        checks.append(dict(path=item['path'],before=item['after'],after=digest,same=digest==item['after']))
    assert len(checks)==579 and all(x['same'] for x in checks)
    author=json.loads((stage/'SDPO_source_manifest.json').read_text())
    author_checks=[]
    for entry in author['files']:
        digest=sha(stage/'SDPO_source'/entry['path'])
        assert digest==entry['sha256']
        author_checks.append(dict(path=entry['path'],sha256=digest,same=True))
    dump(stage/'source_before_after.json',dict(original_inputs=checks,original_SDPO_files=author_checks))
    reading=[
        dict(title='Reinforcement Learning via Self-Distillation (SDPO)',
             URL='https://arxiv.org/html/2601.20802v2',
             publication='https://proceedings.mlr.press/v306/hubotter26a.html',
             scope='Abstract/introduction, method2 including stability, appendixA.1-A.4/B.1-B.2; not all experiments or1387HTML lines.',
             source='https://github.com/lasgroup/SDPO/tree/7c457fc1b1f636ae794eb0362ba37d4743b06fbc',
             implementation_read=['core_algos.py:1027-1188 complete agg_loss/self-distillation functions',
                                  'torch_functional.py:266-284 complete masked_sum',
                                  'dp_actor.py:750-860 teacher no_grad, original multimodal assertion and scoring path',
                                  'ray_trainer.py:637-798 actual feedback reprompt, same responses, mask',
                                  'sdpo.yaml and rich_feedback/run_sdpo.sh complete',
                                  'README first~10.5KB; full LICENSE, not entire README'],
             actually_executed='Unmodified original AST function bodies on fixed CPU distributions. No author trainer/Ray/model rollout.',
             relevant_difference='Feedback self-teacher is already an existing learning algorithm. Original dp_actor:767 explicitly rejects multimodal distillation.',
             no_claim='No new loss, native distributed reproduction or learned3D gain'),
        dict(title='Self-Distilled Reasoner / OPSD',URL='https://arxiv.org/html/2601.18734v1',
             scope='Abstract/introduction/background and complete method3.1/3.2 through sampled-token alternative; not all experiments/appendix.',
             relevant_difference='Privileged solution and current-student prefix self-distillation already exist; no author code run.'),
        dict(title='Purified OPSD',URL='https://arxiv.org/html/2607.02234v1',
             scope='Abstract/introduction and complete method3.1/3.2, selected diagnosis2.4; not full results.',
             relevant_difference='Reference-only subtraction and stabilized PMI target are existing methods; no claim their language reasoning result proves3D transfer.'),
        dict(title='AgentOPSD',URL='https://arxiv.org/html/2611.12587v1',
             scope='Abstract/introduction, complete method2.1-2.4, appendixA.1-A.2 through proposition5, beginning of experimental setup. Not all appendix or model reproduction.',
             relevant_difference='Recursive turn-level contrast reshapes outcome advantage. Its Bayes interpretation assumes success-conditioned teacher and rare-success marginal approximation.',
             caveat='Paper explicitly condenses2608.05987v1 and repeats its ALFWorld results; not a second independent experiment. Bounded multiplication preserves zero task advantage in flat groups, but separate KL can still have a gradient.'),
        dict(title='PhysOPSD',URL='https://arxiv.org/html/2611.12411v1',
             scope='Abstract/introduction/related work, problem3 and complete method4.1-4.3 including paired shared-noise rollouts; training details5.1 read. Not full results/appendix or figures.',
             repository='https://github.com/aigc3d/PhysOPSD/tree/491ef691c68e691b655e5d33daa41ec5671b2f1e',
             relevant_difference='Executable simulation privilege and controlled variations already have a direct video-generation prior. It is not an editable indoor-scene policy.',
             availability='Inspected tree contains README and two images, no algorithm implementation at this revision; no author code reproduced.'),
        dict(title='AI-Agents-in-Depth Chinese book',source='book_reading_ledger.json',
             scope='Cumulative text reading of TOC and PDF223-262 chapter8.312-page book not fully read; figures not individually audited.',
             relevant_difference='Thought/action, SFT/RL and self-distillation ideas inform hypotheses, not evidence of novel3D method. Attached instructions are data, not commands.'),
        dict(title='Native move/branch implementation',source='https://github.com/nepfaff/scenesmith/tree/e888297a8b0a2f3acbb8634edf9def59cceab836',
             scope='Furniture tool registration/wrappers, move/remove/add shared loop guard, rescale, full LoopDetector and placement_noise; RoomScene serde/copy and native ToolEnvironment.step.',
             actually_executed='30 actual SDK move attempts on a newly owned cold-restored room; guard limit25/window30 unchanged.',
             caveat='Only existing-furniture move/rescale parent restoration qualified. No full acquisition/add/remove/snap/worker/RNG restore or scene success.'),
        dict(title='Where-OPD',URL='https://arxiv.org/abs/2610.02117',
             scope='Abstract lookup only; no method/source/full-text-reading claim',
             relevant_difference='Synthetic geometry privilege for visual question answering is adjacent; not this complete-scene generation task.')]
    dump(stage/'reading_ledger.json',reading)
    contract=dict(
        task='Text-conditioned complete editable indoor3D generation with all original requirements retained',
        hypothesis='Correctly sourced design-decision supervision can improve actual complete-room quality/cost beyond action imitation, ordinary feedback/search and existing joint learning',
        current_evidence='No new learned policy or complete positive room; current results qualify shared tools, data provenance and input construction only',
        source_state=dict(scene='Current scene plus observed views',mutable_files='SDFs/registry/asset versions',
                          hidden_state='Loop history, tool state, caches/RNG where relevant; do not call SceneJSON a complete simulator snapshot'),
        learning_arms=['A actionCE','AUX same-source independent rationaleCE','J joint reason-actionCE',
                       'P own-generated decision then native action','ordinary feedback/search F',
                       'whole-trajectory T','SDPO-style feedback self-distillation control'],
        optional_existing_controls=['LSS alignment','RED inference combination','Purified OPSD if reference-shortcut evidence emerges'],
        fair_comparison=['Same original tasks/assets/native tool operations and cumulative budgets',
                         'Common source/version/query/loop fixes, no candidate-exclusive correct tools',
                         'No oracle design decisions supplied at deployment',
                         'Training and evaluation room/asset families split; archived bedroom remains development',
                         'No missing expert rationale invented; use same original source mass in A/AUX/J/P',
                         'Correctness unknown is not success; complete task, semantic identity, usability, geometry and cost separately reported'],
        data_policy=dict(archived_policy_packets=19, output_tokens=4124, verified_views=55,
                         original_generated_think_is_expert=False,
                         greedy_archive_is_fresh_on_policy_data=False,
                         offline_full_logit_objective_possible_if_separately_declared=True,
                         complete_positive_not_required_by_SDPO=True,
                         current_packets_optimization_admitted=False),
        readiness=dict(native_existing_move_rescale_branch_state='qualified in the stated scope',
                       ordinary_SDPO_loss_kernel='executed on CPU tensors, no model',
                       native_multimodal_inputs='19 actual requests, target token/image/position construction executed',
                       author_multimodal_SDPO_trainer='explicitly unsupported at pinned revision',
                       fresh_policy_rollouts='not acquired in this run',teacher_logits_or_quality='not measured',
                       learned_adapter_and_native_tool_role_isolation='not exercised',
                       full_forward_backward_memory='not measured',full_scene_gain='unproven'),
        no_authorization_change=dict(new_VLM_forwards=0,new_paid_API_calls=0,new_training_updates=0,
                                     old_closed_generation_budget_reopened=False))
    dump(stage/'algorithm_contract.json',contract)
    theory='''闭环三维学习：本轮可以据实执行的算法与理论边界

一、问题与变量
任务仍是从描述生成完整、可编辑的室内三维场景，不把目标换成VQA、视频或已有物体的局部移动。
X_t是策略实际拿到的原文、视图、对象及已知工具结果；Z_t是完整执行状态，至少还含资产文件版本、注册表和工具内部状态。
D_t是设计决定或说明，A_t是原生工具调用。候选问题是同样资料和总预算下，正确来源的D监督是否改善整场质量/成本。
pi(D,A|X)=pi(D|X)pi(A|X,D)只是链式法则。两次调用、额外日志、显式D和分支快照本身都不是新的理论或创新证明。

二、真实分支实验证明了什么
本轮正常构造原生SDK与场景，在同一新父场景上执行30次移动：前25次成功，第26次被原有LoopDetector拦截。
这些调用恢复了场景/注册表/文件等状态，但未恢复工具调用历史。恢复父分支原有历史后，同一动作重新成功。
随后A/B/A/B各分支在相同父状态和历史下得到相同对应结果；没有更改25次上限、30条窗口或噪声参数。
因此只恢复可见场景不足以保证候选处于同一执行条件。结果是受控工具状态证据，不是历史试验受影响频率或三维质量收益。
修复捕获实际闭包持有的同一个LoopDetector，恢复父分支的原历史，而非把每个真实决策前的历史强行清零。
真实序列中的防重复规则仍保留。加入/删除/检索/吸附、worker和一般RNG恢复没有由这30次调用证明。
原父JSON第一次经过原生序列化发生一个四元数末位变化，首轮在任何SDK调用前退出；新副本显式规范化一次后冻结。
原始记录未改，不能把浮点末位变化隐藏成bitwise相同。metadata对照和两轮结果均保留。

三、现有数据与普通训练目标
19轮策略生成有4124个原始输出token、55个原始视图引用；4轮工具VLM生成不混作策略训练样本。
35条工具输出分别为30条原生success标志、4条尚未核实的物理文本报告、1条资产批处理部分失败。
success仅表示所述操作执行，既不是正确设计标签，也不是完整房间成功。原模型生成的think不升级为专家理由。
原始请求、输出token、视图字节和反馈已经逐条绑定；影像在CPU重编码后，19个原提示指纹/长度全部复现。
教师和学生保留相同图像张量与同一原始输出token。教师新增的只是标明来源的真实反馈，没有代写正确理由。
原Qwen3VLModel.get_rope_index在这19例的两个前缀上实际运行，保留视觉三轴位置及文本目标的因果logit位置。
这些是输入准备，不等于模型前向、教师正确性、参数更新或完整训练器已经通过。

A：普通动作CE；AUX：相同D额外独立CE，动作不读D；J：同一序列D后接A；P：自身生成D后重新编码X,D执行A。
14条缺失D不补写；旧10条D含8条有来源原则和2条本地停止文本，不称为24条作者专家轨迹。
F为相同资料下普通检索/反馈/搜索；T为已有完整轨迹学习。训练只改变需要检验的监督/决策路径，不能让工具权限不同。
若使用训练期表示对齐或推理期理由组合，应注明已有LSS/RED近邻，不能只和动作CE比较后声称证明新机制。

四、SDPO是强对照，不是新发现
固定当前学生访问的前缀h=(X,y_<k)和真实反馈f，普通目标为
 L_token = E_{y~sg(pi_theta)} sum_k KL(pi_theta(.|h) || sg(q_theta(.|h,f)))。
反馈教师重新给学生原序列评分，不必生成新的正确答案；没有完整成功轨迹也可有反馈信号。
但得到非零梯度只表示教师与学生分布不同，不保证教师正确或整场质量提高。
原源码dp_actor.py:767明确拒绝multimodal distillation；其发布训练器不是现成的本地Qwen3-VL三维训练器。
本轮移植的是输入与普通目标，不声称复现作者分布式训练或提出新的损失。
原full-logit alpha=1对应KL(student||teacher)，alpha=0对应KL(teacher||student)，中间值对应混合分布JSD。
教师必须stop-gradient：原actor用torch.no_grad实现；损失核本身不自动阻断传入的教师张量梯度。
top-k必须使用学生选出的同一词表索引给教师评分，保留tail和重归一化是不同目标，不能悄悄互换。
原生策略停止且没有反馈的1轮对应反馈mask0，不从空反馈制造监督。

五、贪心旧轨迹与在轨梯度的区别
在固定前缀、固定教师且q>0时：
 grad KL(p||q) = E_{a~p}[(log p(a)-log q(a)) grad log p(a)]。
已运行原SDPO采样损失核，并对三个词元穷举加权，得到与完整反向KL相同的梯度；只取贪心词元则不同。
这只是有限分布的公式验证，不是神经模型学习结果。旧19轮由do_sample=False生成，且未保存行为逐token概率。
因此不能把它们称为当前可训练softmax策略的新采样数据，或用未校正的采样式损失声称无偏在轨优化。
若重新评分全部词表，在这些固定旧前缀上做离线蒸馏，仍是一个合法的离线目标；只是它的前缀分布不同，须独立声明和比较。
不能由“不是在轨数据”推成“这些数据绝对不能训练”。原始事实仍用于格式、离线对照及教师能力诊断。
上述token目标把前缀采样分布stop-gradient；完整序列KL还含未来前缀分布的导数，不能把两个梯度混称。

六、特权信息不能保证学生获得部署时不存在的事实
以下是固定X、固定教师与固定条件分布的普通KL推导，不是新的三维定理。
设教师q_z(a)>0，学生只能看X，看不到z。令r(a)=exp(E_z log q_z(a)), C=sum_a r(a)。
则E_z KL(p||q_z)=KL(p||r/C)-log C，最优学生是归一化几何平均；前向KL的最优学生则是E_z q_z。
两者都不意味着学生能对每个不可辨别的z选择不同动作。若两个等概率隐状态允许的动作集合不相交且X完全相同，单动作期望成功率最多1/2。
真实工具实验说明某些未恢复状态会影响结果，但没有实际渲染/编码两个父分支，故不声称本轮证明了策略输入张量别名。
反馈教师若依赖部署缺失的状态，先补公共观察/执行合同或限定可学条件；不能用D长度和教师置信度掩盖信息缺口。
Purified OPSD已有处理参考捷径的相关方法；其在语言推理的解释不能自动变成三维可迁移性的保证。

七、AgentOPSD与物理代码特权的近邻限制
AgentOPSD对轨迹优势做有界乘法重分配，若同组奖励全等使任务优势为0，乘数仍得到0；独立KL项/其他组仍可有梯度。
其自教师差值近似Bayes factor依赖成功条件教师及罕见成功近似，不能直接当作校准的动作因果价值。
PhysOPSD已经有可执行仿真特权、信息层级与共享噪声的物理变化训练，不能把“代码/成对变化作为特权”笼统声称首次。
它输出视频而非可编辑房间，这支持重新定位任务差异，不证明我们已经有算法差异。

八、工具模型与策略参数必须隔离
若同一theta也改变检索资产的VLM输出核kappa_theta，联合轨迹分布包含策略pi与工具kappa两部分。
对同一标量回报的联合score梯度会有sum grad log pi + sum grad log kappa；只优化策略部分不能称完整联合策略梯度。
要单独检验设计监督，后续保持工具角色冻结，实际验证adapter关闭/恢复与观测一致，所有方法共用同一环境。
这是实验可解释性要求，不是新颖的多Agent训练理论。当前未创建LoRA、未做权重前向或优化更新。

九、下一次真正有信息量的实验
先做同原文、相同视图和工具下的教师能力诊断：原策略、普通反馈教师、来源明确的设计证据教师。
评价实际动作合法性、受影响对象、原文要求保持和成本，不用教师似然或native success当质量真值。
普通反馈教师若已能解释收益，就保留为强对照；来源设计证据只有在带来额外可执行改进时才值得训练。
随后才在新完整场景轨迹上比较同来源A/AUX/J/P及普通反馈/轨迹方法；独立新房间/资产族上比较完整要求和总代价。
旧卧室为开发失败，不能当未见测试；不因失败删除样本、不偷偷补专家D、也不把所有几何检查通过等同完整生成。
当前静态rank8 LoRA约2182万参数，一个FP16底座加FP32参数/梯度/两Adam矩静态约16.66GiB。
这不含激活、视觉路径、临时logits/KV/几何进程与碎片，不构成V100训练可装下的实测结论；两份FP16完整底座权重已超32GiB。
下一次需先在已授权范围做单例前后向/角色隔离与峰值测量，才扩大。此文不重开旧32次生成/900秒额度，也不自动启动训练。

十、目前研究结论
可以继续推进完整三维Agent学习，但本轮没有建立新损失、新理论、完整正场景或CVPR创新成立的证据。
已得到可追溯的原生数据包、多模态反馈输入和严格限定的真实工具分支恢复，减少了以后把工程差异算作算法收益的风险。
真正尚待证明的是正确的设计决定监督是否优于普通文本、反馈、搜索及既有联合训练，而不是再给同一组件换一个名字。
'''
    (stage/'algorithm_and_theory.txt').write_text(theory)
    plan=dict(
        actual_execution_in_this_stage=dict(paid_API_calls=0,local_VLM_forwards=0,training_updates=0,new_rooms=0),
        future_model_execution_not_automatically_authorized=True,
        first_conditions=['Original policy on the same native task/state',
                          'Same policy with ordinary native feedback',
                          'Same policy with sourced design evidence and the same feedback/tools'],
        before_any_training=['Measure teacher action/affected-object correctness against independently checkable requirements',
                             'Verify tool-role adapter disabled and restored under a serialized model lock',
                             'Run one full forward/backward memory profile, no fit inferred from headers',
                             'Declare fresh sampling or explicitly offline fixed-prefix objective',
                             'Freeze evaluation room/asset families before outcomes'],
        ordinary_controls_share=['Loop history/file restoration','Physical-query fix and unknown flags',
                                 'Asset and operation access','Original task/feedback and total budgets'],
        must_not_do=['Train on benchmark answers','Invent missing expert rationales','Count raw success flags as scene correctness',
                     'Reset loop history to zero inside real episodes','Call extra source reading a learned method result'],
        next_non_model_work=['Connect sourced decision records to actual native observation/action identities',
                             'Inspect original expert-data acquisition and licenses without hidden full downloads',
                             'Audit teacher-evaluation clauses and operation coverage on complete native traces'],
        existing_CPU_reproduction_commands=[
            'env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 <ENVIRONMENT.json:model_python> scripts/autodl/audit_native_feedback_material.py --repo . --out <NEW_DIR> --checkpoint <EXISTING_CHECKPOINT>',
            'env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 <ENVIRONMENT.json:model_python> scripts/autodl/probe_original_sdpo_objective.py --source runs/autodl/20261111_closed_loop_data/SDPO_source --out <NEW_DIR>',
            'env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 <ENVIRONMENT.json:model_python> scripts/autodl/prepare_native_feedback_inputs.py --repo . --packet-dir runs/autodl/20261111_closed_loop_data/feedback_material_v2 --out <NEW_DIR> --checkpoint <EXISTING_CHECKPOINT> --engine-source <EXISTING_LOCAL_ENGINE_DIR>'],
        cost_boundary='Zero paid API and no new weights in this stage; future VLM/training needs a separately admitted run, not reused closed budget',
        no_GPU_startup_requested_now=True)
    dump(stage/'next_execution_plan.json',plan)
    rows=[('actual_SDK_move_attempts',30,'controlled tool-state experiment, not scenes'),
          ('move_seconds',branch['seconds'],'normal SDK plus owned services; model0'),
          ('archive_policy_turns',19,'all actual native policy turns, one development room'),
          ('policy_output_tokens',4124,'original exact generated IDs; think is not expert gold'),
          ('verified_policy_view_references',55,'original bytes and encoded image tensors checked'),
          ('native_success_flags',30,'execution flags, not correct designs'),
          ('unverified_physics_reports',4,'do not use as known geometric teacher truth'),
          ('asset_partial_failures',1,'failure retained, not infeasibility proof'),
          ('feedback_mask_available',18,'input mask only; teacher logits absent'),
          ('multimodal_input_seconds_v1',json.loads((stage/'multimodal_feedback_inputs_v1/result.json').read_text())['seconds'],'CPU processor, no mRoPE calculation in v1'),
          ('multimodal_input_seconds_v2',inputs['seconds'],'CPU processor and native mRoPE, no model'),
          ('new_VLM_forwards',0,'tokenizer/processor/pure tensor helpers only'),
          ('new_training_updates',0,'no optimizer or adapter created'),
          ('new_complete_positive_rooms',0,'not a generation-success experiment'),
          ('unchanged_original_files',579,'actual current SHA256 compared to previous frozen SHA256')]
    with (stage/'summary.csv').open('w',newline='') as stream:
        w=csv.writer(stream,lineterminator='\n');w.writerow(['metric','value','scope']);w.writerows(rows)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    source_links=''.join('<li><a href="'+x['URL']+'">'+html.escape(x['title'])+'</a></li>' for x in reading if 'URL' in x)
    artifacts=['algorithm_and_theory.txt','algorithm_contract.json','reading_ledger.json','book_reading_ledger.json',
               'summary.csv','feedback_material_v2/eligibility.csv','feedback_material_v2/offline_feedback_packets.json',
               'multimodal_feedback_inputs_v2/input_positions_and_tensor_hashes.json','SDPO_objective_v1/result.json',
               'move_SDK_v2/result.json','move_SDK_v1/result.json','source_before_after.json','next_execution_plan.json']
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>真实分支与三维反馈学习准备</title>
<style>body{max-width:1100px;margin:35px auto;padding:20px;font:16px/1.65 system-ui}pre{white-space:pre-wrap;background:#f5f5f5;padding:20px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:7px}a{overflow-wrap:anywhere}</style>
<h1>真实分支、反馈学习与多模态输入：有实际执行，尚无学习收益</h1>
<p>本轮新增证据是30次真实SDK调用、19轮原生轨迹的token/55视图/反馈绑定，以及普通反馈蒸馏的CPU多模态输入。目标仍是完整室内三维生成；这些不计为新房间或CVPR方法成功。</p>
<p>同一可见场景的第26次重复移动被原工具历史拦截；恢复父分支原调用历史后成功。限制25/窗口30未改。这是共享的执行修正，不能只给候选方法使用。</p>
<p>SDPO已经提出反馈自蒸馏；其固定版本actor明确拒绝多模态蒸馏。这里保留了真实图像和原输出，执行Qwen原始位置函数，尚未执行教师模型、反向传播或完整训练器。</p>
<p>旧轨迹为贪心生成，不冒充新的在轨采样；仍可在另行声明的离线目标下研究。没有完整正轨迹不等于不能使用SDPO，真实反馈也不自动等于正确教师。</p>
<h2>实际范围</h2><table><tr><th>指标</th><th>值</th><th>解释</th></tr>'''+table+'''</table>
<h2>失败与修正均保留</h2><p>首轮移动在SDK调用前因四元数序列化末位不稳定退出；metadata检查后在新副本显式规范化。首轮数据审计混淆了工具role与receipt purpose，原始路由源码确认命名映射后继续，没有更换轨迹。CPU多模态v1未计算位置，v2补上原始mRoPE函数。不得合并成一次无失败运行。</p>
<h2>来源与完整推导</h2><ul>'''+source_links+'</ul><pre>'+html.escape(theory)+'</pre><h2>交付文件</h2><ul>'+''.join('<li><a href="'+p+'">'+html.escape(p)+'</a></li>' for p in artifacts)+'''</ul>
<p>精确阅读范围见账本：第8章文字已覆盖，不声称读完312页；论文按所读方法与附录逐项列出，全文抓取不算全文精读。原始579个文件及8个SDPO源文件SHA256不变。无新增付费API、模型权重、训练或资源启动。</p></html>'''
    (stage/'stage_report.html').write_text(page)
    scripts=['diagnose_native_parent_roundtrip.py','native_move_snapshot.py','probe_native_move_branches.py',
             'audit_native_feedback_material.py','probe_original_sdpo_objective.py','prepare_native_feedback_inputs.py',
             'native_feedback_batch.py','report_native_feedback_preparation.py']
    code=[]
    for name in scripts:
        path=repo/'scripts/autodl'/name
        compile(path.read_text(),str(path),'exec')
        code.append(dict(path=str(path.relative_to(repo)),sha256=sha(path),bytes=path.stat().st_size))
    module=Path('/root/autodl-tmp/agent_method_preparation_20261027/work/inference_autodl_20261103/runtime/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py')
    dump(stage/'code_manifest.json',dict(our_code=code,original_mRoPE_source=dict(path=str(module),sha256=sha(module)),
                                      SDPO_revision=author['revision'],original_sources_unchanged=True))
    # Red author source remains in RAM under its own nonredistribution license;
    # this bundle includes only pinned Apache SDPO subset with original LICENSE.
    files=[p for p in stage.rglob('*') if p.is_file() and p.name not in {'evidence.zip','manifest.json','delivery_verification.json'}]
    with zipfile.ZipFile(stage/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(stage)))
        for row in code:z.write(repo/row['path'],row['path'])
    manifest=[dict(path=str(p.relative_to(stage)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files+[stage/'evidence.zip'])]
    dump(stage/'manifest.json',dict(files=manifest,unique_original_source_checks=579,
                                  original_SDPO_source_checks=8,no_paid_API_or_training=True))
    print(json.dumps(dict(report=str(stage/'stage_report.html'),summary=str(stage/'summary.csv'),
                          evidence=str(stage/'evidence.zip'),bundle_bytes=(stage/'evidence.zip').stat().st_size,
                          source_checks=579,SDPO_source_checks=8,goal_complete=False),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True);a=p.parse_args();main(a.repo.resolve(),a.stage.resolve())
