"""Summarize measured native-phase work, ordinary objectives and open gates."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile


THEORY='''原生整场策略、条件化设计决定与训练目标：本轮算法合同

一、保留的任务和可反驳假设
目标仍是文字到完整、可编辑的室内三维场景。不是只修一个坐标，也不是把解释文字写得像设计师。
待检验：在相同任务、真实多视图、来源知识、工具和总预算下，让动作读取条件化设计决定，能否超过普通操作计划、直接动作和普通反馈/搜索。
现有8条来源原则应用是本地重建监督；不是建筑师的历史工具轨迹。14个动作没有D；另2个D是停止记录。不能删除它们或补造专家理由。
本轮是普通接口与监督边界修正，没有新优化器、学习效果或新颖性证明。

二、共同执行算法已实现到哪里
X_t包含原始任务、当前stage、真实native memory及按原顺序的视图；floor为0图，家具/墙/顶声明为5图。顶面没有实际保存轨迹可验。
J：一次原生连续输出；A：直接生成动作；I：生成D但动作不读D；P_m：先生成D，再把g_m(D)加入动作前缀。
g_m是完整记录、去原则字段、只保留操作计划；只读生成D，不接场景ID、动作答案或语料查表。非对应schema共同透传并记录。
新D提示只在确有来源原则时请求四字段JSON；缺少来源不得编造。schema覆盖率必须报告，不能把透传当成成功消融。
每次提议共享D+A输出token和时间限制；两阶段的两次真实模型请求全部记账。直接动作基线不执行无用的D。
动作阶段允许一个有序批次。所有条件共用严格JSON/EOS预检，并在派发前检查整批工具预算。
一次调用原生env.step(batch)：工具按原顺序执行，每个结果经过原生memory handler；原生step计数在整批后只加一次。
中断时分别记录预留工具数和已确认尝试数；未知不能记0。停止只表示策略结束意图，full_scene_success保持未知。
下一轮观察必须来自该条件自己的真实执行状态。不能在分支发生变化后继续拼接旧下游输出。
本轮仅将保存结果代入原生step。没有派发真实SDK、执行finalizer或跑新房间。地面小物体另有原生workflow，未被四阶段接口冒充覆盖。

三、结束符对齐后的普通SFT
设i为原始开发单元，w^D_ik、w^A_ij为原监督token系数。新匹配目标：
 L_m(theta)=-sum_ik w^D_ik log pi_theta(D*_ik|X_i,D*_<k,phase=D)
            -sum_ij w^A_ij log pi_theta(A*_ij|X_i,g_m(D*_i),A*_<j,phase=A).
I条件第二项不读D；所有匹配条件保留相同D项。缺D单元只提供动作监督，部署仍会遇到没有D监督覆盖的状态。
旧序列2975个token中，8个D结束符后换行、24个动作EOS后换行，在新目标中统一排除，共32个。
保留2943个监督token及各自原系数，不重新归一化：匹配总质量1.812419347110903，排除0.020913986222430014。
这明确改变了旧目标，不能声称仍逐token完全等同旧v3。原记录、排除位置与权重都保留。
连续J基线包含同样的2943个正权重token；8个D/A间隔换行仍在因果输入内，loss mask为0。不能删输入换行来冒充同一连续序列。
直接A基线只包含1225个动作监督token、质量0.8761436017492825；它不是等D监督的对照，必须单独解释。
文本构造不等于完整训练：缺失的原始图像、真实模型前后向、adapter/teacher隔离、优化器、峰值显存与多阶段训练覆盖仍未证明。
现有源材料没有新增floor/多工具专家训练样本。19条保存的策略输出也不能自动变成正确示范。

四、为什么这些对照对研究主张有判别力
完整D比只计划好，不自动证明原则有用，因为条件与未知字段也同时变化。去原则组用于进一步缩小差异；共同X仍含相同来源知识。
训练时D*=q(D|X,A*)可能包含目标对象选择；部署D_gen=f(X,U)看不到A*。金标准D下似然变好不能当成闭环效果。
若外部任务真值Y在给定X后与随机种子U独立，则I(Y;D_gen|X)=0；显式D仍可能帮助有限模型计算与使用已有X。
这不意味着I(A_model;D_gen|X)=0，更不是解释无用的定理。动作本来可以依赖D。
固定同一自身生成D做字段干预，是信息通路诊断；各训练模型自由生成自己的D并执行整场，才是部署比较。两者结果必须分开。
本轮旧19条D均为普通文本，字段路由0/19适用；三路相同是构造结果，完全不是原则无效的实验。

五、反馈蒸馏必须区分的分布
h是交互历史，u是当前输出token前缀。历史来源与u来源是两个独立设计轴；已有层级Agent蒸馏研究，不能将二者拆开作为新贡献。
对固定采集分布d_beta、r_beta和冻结teacher phi，可写普通目标：
 L_dist(theta;beta,phi)=E_(h,z)~d_beta E_u~r_beta [ KL(pi_theta(.|h,u) || stopgrad pi_phi(.|h,u,z)) ].
z可以是原工具反馈，但反馈存在、工具success、独立几何正确、整场满足要求是四件不同的事。未知与失败不删样本。
当前19条是旧冻结模型的greedy缓存；本轮只是构造输入，没有teacher logits、没有采样刷新，也没有形成新的on-policy训练。
采集上下文在本次梯度中固定。这不是对包含d_theta/r_theta分布导数的整个在线期望求无偏梯度；普通token级OPD通常不包含这些项。
详细反馈至少要与相同位置的普通提示、无反馈比较。Fin-SD已有通用提示对照，但仍共享依赖正确答案的错误定位/截断；不能推导成所有特权信息都无用。
不能将语言任务的一组结果数值迁移成三维预期收益。也不能在student错误前缀上假设teacher总能提供正确修正。
如果来源条件在相同h下存在不可见差异，反向KL的最优公共分布可能是几何平均，正向KL可能是算术平均；这不保证每个房间的可执行动作正确。

六、下一项可执行比较及结论标准
先用同一父状态、同一自身生成D，比较full/noR/plan；与直接A、原生J、D不被A读取的I、普通工具检查/搜索并列，累计所有角色成本。
同一模型取得的schema覆盖若不足，则先报告路由不可识别，不将格式失败解释成知识无效。不给某一组额外修复或重试。
通过后，各条件按新房间、资产与设计来源族隔离，独立自由执行完整任务。不得用同一项目换视角冒充泛化。
主结果同时保留原文必需实例、真实类别/材质、明确关系、原正确要求保持、几何与未知覆盖；候选工具数不能替代总成本，STOP不能替代完成。
若只计划即可解释收益，贡献应收窄；若来源原则应用超出计划、反馈和搜索并在新完整房间稳定改善，才有研究贡献证据。
现阶段没有训练后优胜、完整正场景或CVPR级方法证明。下一步还要接入统一全角色额度和原生stage handover；旧关闭生成额度不重启。
不为了继续这一CPU工作要求用户启动付费实例。新训练/模型生成须另按明确实验规模与实际内存证据安排。
'''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main(repo,out):
    read=lambda name:json.loads((out/name).read_text())
    replay=read('replay_v3/result.json');material=read('material_v4_2/result.json');inputs=read('multimodal_inputs_v1/result.json')
    sources=read('source_before_after.json')
    assert replay['source_policy_turns']==19 and replay['source_tool_results']==35
    assert inputs['phase_input_records']==190 and material['arms'][0]['target_tokens']==2943
    assert sources['all_originals_unchanged'] and sources['count']==579
    (out/'algorithm_and_theory.txt').write_text(THEORY)
    contract=dict(task='Complete editable indoor scenes from original text',
        implemented='Common native batch/phase contract and prospective SFT material; no validated learned method',
        shared_stages_declared=['floor_plan','furniture','wall','ceiling'],
        actual_saved_stages=['floor_plan','furniture','wall'],
        surface_workflow='Separate original path; not ported in this result',
        controls=['native_joint','action_direct','decision_independent','decision_conditioned:full_record',
                  'decision_conditioned:without_principle_field','decision_conditioned:plan_field_only',
                  'Ordinary feedback/search','Generic-feedback versus detailed-feedback distillation'],
        full_room_quality_improvement_measured=False,new_training_updates=0,new_model_generations=0,
        old_generation_budget_reopened=False,goal_complete=False,
        readiness=dict(actual_batch_memory_replay=True,actual_CPU_processor_mRoPE=True,
                       common_training_targets_text_material=True,full_source_images=False,
                       all_role_unified_inference_budget=False,model_forward_numerics=False,
                       trained_adapter_fit=False,whole_room_handover_test=False,
                       complete_room_performance_evidence=False))
    dump(out/'algorithm_contract.json',contract)
    rows=[('actual_source_policy_turns',19,'one existing partial development run'),
          ('actual_source_tool_results',35,'saved outputs, no SDK replay'),
          ('replay_conditions',6,'ordered calls/arguments preserved by construction'),
          ('pre_action_memory_equal',114,'original memory handler text compared'),
          ('actual_CPU_processor_encodes',inputs['unique_actual_processor_encodes'],'shared live-backend encoder helper'),
          ('phase_input_records',190,'includes reused identical prefixes; no model forwards'),
          ('original_image_references',55,'same source byte hashes and pixel tensors'),
          ('max_combined_input_tokens',inputs['maximum_prompt_plus_response_tokens'],'no truncation'),
          ('positive_targets_per_matched_arm',2943,'source-index and weight matched'),
          ('excluded_post_stop_whitespace_tokens',32,'new v4; old labels unchanged'),
          ('native_joint_positive_targets',2943,'8 internal separator tokens retained with zero weight'),
          ('direct_action_positive_targets',1225,'intentionally does not match D supervision'),
          ('old_plain_D_field_route_coverage',0,'of19; no evidence of a field effect'),
          ('protected_old_files_unchanged',579,'actual rehash'),
          ('new_model_generations',0,'no model weight load'),('new_training_updates',0,'no optimizer'),
          ('new_complete_scene_successes',0,'not inferred from transport results')]
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['measure','value','scope']);w.writerows(rows)
    commands='''CPU reproduction only; all output directories must be new. Do not launch an old inference budget.
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <existing-model-python> scripts/autodl/replay_native_phase_batches.py --repo <repository> --out <new-replay-dir> --checkpoint <existing-checkpoint>
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <existing-model-python> scripts/autodl/prepare_native_phase_material.py --repo <repository> --out <new-material-dir> --checkpoint <existing-checkpoint>
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <existing-model-python> scripts/autodl/prepare_native_phase_inputs.py --repo <repository> --out <new-input-dir> --checkpoint <existing-checkpoint>
'''.replace('<existing-model-python>','/root/autodl-tmp/agent_method_preparation_20261027/work/v100_runtime_20261104/runtime/bin/python').replace('<repository>',str(repo)).replace('<existing-checkpoint>','/root/autodl-tmp/agent_method_preparation_20261027/work/critic_public_runtime_20261104/checkpoint')
    (out/'COMMANDS.txt').write_text(commands)
    paragraphs=[
      '本轮补上原生有序批次、分阶段输入与监督结束符的共同合同。19轮旧输出、35个旧工具结果的回放保持顺序及动作前记忆；没有生成新房间或比较学习效果。',
      '新SFT材料保留同样的2943个监督token及原系数；统一排除32个停止符后的序列化换行。原生连续基线保留8个内部换行作因果输入，要求loss mask为零。旧文件不改，也不宣称新旧目标完全一致。',
      '实际CPU执行了共享编码器与Qwen3-VL位置函数：190份阶段输入对应76个不同前缀。19个原生连续输入复现旧prompt哈希/长度；55个源图引用在条件间保持字节和像素张量一致。',
      '重要限制：旧19条D全部是普通文字，不符合来源原则四字段schema，三个路由共同透传。因此本轮既没有原则效果，也没有原则无效的模型证据。',
      '方法仍是待检验假设：条件化设计决定是否超过同信息的普通操作计划、直接动作、完整连续生成、反馈和搜索。新字段路由和结束符修正本身不是主创新。',
      '已补读层级Agent蒸馏方法与Fin-SD的方法、通用提示对照和限制。它们分别要求区分历史/当前前缀来源，以及详细反馈内容与错误定位位置。不能把这两项现成机制重新命名成三维新算法。',
      '原始SDK没有重新运行，未加载模型权重、未训练、未调用付费API。顶面只有接口声明，桌面物体workflow未接入此桥；全角色预算、真实模型前后向、训练显存和整场handover仍未验证。579个受保护旧文件重新计算哈希一致。']
    links=[('算法与理论','algorithm_and_theory.txt'),('算法合同与未完成项','algorithm_contract.json'),
           ('实际阅读范围','reading_ledger.json'),('训练边界与系数','material_v4_2/result.json'),
           ('排除的原始token位置','material_v4_2/boundary_exclusions.json'),
           ('原生批次回放','replay_v3/result.json'),('多模态输入执行','multimodal_inputs_v1/result.json'),
           ('旧文件哈希核验','source_before_after.json'),('运行命令','COMMANDS.txt'),('汇总表','summary.csv')]
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in rows)
    body=''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
    body+='<h2>实际范围</h2><table><tr><th>指标</th><th>值</th><th>边界</th></tr>'+table+'</table>'
    body+='<h2>证据与复现</h2><ul>'+''.join('<li><a href="'+p+'">'+html.escape(label)+'</a></li>' for label,p in links)+'</ul>'
    body+='<p>直接近邻：<a href="https://arxiv.org/html/2610.31918v1">层级Agent蒸馏</a>；<a href="https://arxiv.org/html/2610.36082v1">Fin-SD</a>。具体已读章节和未读范围见reading_ledger。</p>'
    (out/'stage_report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原生多阶段合同与训练边界</title><style>body{max-width:1040px;margin:32px auto;font:17px/1.7 system-ui;padding:0 20px;color:#17202a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccc;padding:8px;text-align:left}a{color:#1456a0}</style><h1>原生多阶段合同与训练边界：实测结果与算法依据</h1>'+body+'</html>')
    scripts=['native_multistage_phase.py','native_phase_torch_backend.py','replay_native_phase_batches.py',
             'prepare_native_phase_material.py','prepare_native_phase_inputs.py','report_native_phase_bridge.py']
    manifest=[dict(path=str(repo/'scripts/autodl'/s),sha256=sha(repo/'scripts/autodl'/s)) for s in scripts]
    dump(out/'code_manifest.json',manifest)
    bundle=out/'evidence.zip'
    files=[p for p in out.rglob('*') if p.is_file() and p.name not in ('evidence.zip','manifest.json','delivery_verification.json')]
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(out).as_posix())
        for s in scripts:z.write(repo/'scripts/autodl'/s,'code/'+s)
    dump(out/'manifest.json',dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files],
                                 evidence_zip=dict(bytes=bundle.stat().st_size,sha256=sha(bundle))))
    print(json.dumps(dict(report=str(out/'stage_report.html'),bundle_bytes=bundle.stat().st_size,goal_complete=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.out.resolve())
