"""Write a source-grounded learning contract and executed-control evidence."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def main(repo,stage):
    result=json.loads((stage/'arithmetic_v2/result.json').read_text())
    assert result['status']=='original_RED_arithmetic_and_independent_objectives_checked'
    assert not result['cuda_initialized'] and result['VLM_forwards']==0
    memory=json.loads((stage/'adapter_static_memory.json').read_text())
    costs=json.loads((stage/'actual_cost_semantics.json').read_text())
    protected=json.loads((repo/'runs/autodl/20261111_contact_query_validity/source_before_after.json').read_text())
    checks=[]
    for row in protected['original_inputs']:
        actual=sha(Path(row['path']))
        checks.append(dict(path=row['path'],before=row['after'],after=actual,same=actual==row['after']))
    assert len(checks)==579 and all(x['same'] for x in checks)
    dump(stage/'source_before_after.json',dict(original_inputs=checks))
    reading=[
        dict(title='Reasoning Without Inference Cost: Latent Semantic Scaffolding for Robot VLA Policies',
             URL='https://arxiv.org/html/2609.04893v1',
             read='Complete HTML main textI–VI and references; figures read through captions only, not image inspection. No original code or encoder/model executed.',
             relevance='Training-only rationale-to-action representation alignment is already a direct prior. Text-CE AUX alone does not cover this control.',
             limitation='Paper evaluates bottle-manipulation human pretraining and robot transfer; it does not establish indoor scene generation gains. Phase boundaries and unverified generated rationales limit the result.',
             availability='No author implementation link found in inspected paper/search; this does not prove none exists elsewhere.'),
        dict(title='Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought',
             URL='https://arxiv.org/html/2507.07685v2',
             project='https://yshinya6.github.io/projects/red-cvpr2026/',
             source='https://github.com/yshinya6/red/tree/5c3a49780059a6a58e6f1b5c186b28aa6d6e8e38',
             read='HTML3–5(method, proof, experimental tables, interventions and efficiency) and conclusion. Official project, full README/license, full multipath.py/utils.py/make_input.py/output.py. No claim of all supplemental experiments or numerical model reproduction.',
             executed='Unmodified original module imported internally; _compile_logits called on8fixed CPU score examples. No model sampler, weights or VLM forward.',
             consequence='Inference-level rationale grounding is an existing control. Original next-token optimization is not a guarantee of geometric or full-task success.',
             implementation_boundary='Author README flags batched-lambda0 discrepancy. Fixed supplied-logit arithmetic passes; no model/cache cause diagnosed.',
             distribution='Custom internal evaluation license; no original source in Git or bundle. URLs, revision and hashes preserved.'),
        dict(title='Native local sequence metering',
             source='scripts/autodl/complete_native_model_host.py:67,78,102',
             read='Full host source; metered method binding and current actual23-entry ledger inspected.',
             consequence='Historical forwards field counts sequence generate requests. Internal model calls were not instrumented; preserve old records and refine future accounting.'),
        dict(title='Public original-data availability check',
             source='HF dataset search API:Scene-OTA,Scene-OTC,ThinkBLOX,SceneChain',
             read='Bounded metadata search only, no dataset contents or account forms.',
             result='No Scene-OTA/ThinkBLOX search hits at access time; Scene-OTC-Bench is evaluation data, excluded from training; SceneChain-12K is gated.',
             consequence='No new original reasoning trajectories acquired. This is not proof of universal unavailability.')]
    dump(stage/'reading_ledger.json',reading)
    arms=[
        dict(arm='A',learning='Original weighted action CE',deployment='Actual state -> native action',isolates='Base action imitation',implementation='Existing encoder/trainer; no new trained result'),
        dict(arm='AUX',learning='Same D CE as J/P, independently predicted from X; action sees noD',deployment='Actual state -> native action',isolates='Additional textual source supervision',implementation='Existing separate-phase material; Dgeneration unnecessary at deployment unless explicitly charged as a separate control'),
        dict(arm='J',learning='Same-source joint reason/action weighted CE',deployment='Single autoregressive reason then action',isolates='Existing reason-action supervision',implementation='Existing joint serializer/deployment adapter; no new trained result'),
        dict(arm='P',learning='Same component CE; real prefixes differ and must be recorded',deployment='Own D then fresh action pass over actual X,D',isolates='Two-pass execution, not a new probability factorization',implementation='Existing phase policy; no strict textual bottleneck or new learned comparison'),
        dict(arm='LSS-control',learning='Frozen rationale embedding aligned to action-state representations, plus same action CE',deployment='Drop alignment head/teacher; same direct action graph',isolates='Representation shaping without generated reasoning',implementation='Independent cosine arithmetic checked only; no original T5, semantic embeddings or scene-phase annotation. Not an author reproduction and not training-ready'),
        dict(arm='RED-control',learning='No additional training',deployment='Combine image/current-input and rationale-only next-token distributions',isolates='Inference-time rationale utilization',implementation='Original probability kernel actually exercised; dual-context model/cache/native decoder not integrated or measured'),
        dict(arm='F',learning='Frozen base',deployment='Equal source retrieval and ordinary feedback/search',isolates='Benefits of source access, feedback and search',implementation='Common tools/thresholds; physical-query and saved-file-version corrections shared, not candidate-exclusive'),
        dict(arm='T',learning='Existing step/whole-trajectory SFT/DPO',deployment='Whole-plan baseline with ordinary returned-ID references',isolates='Existing full-trajectory learning',implementation='Author builders and native references checked earlier; idea-level SceneWeaver vs native typed-call adaptation explicitly distinguished')]
    dump(stage/'strong_controls.json',arms)
    theory='''可以继续检验的算法，以及它不能声称的理论

1. 核心变量与研究主张
X_t是原文、当时真实场景/视图、已执行工具结果、资产及剩余预算。D_t是模型提出的设计决策，A_t是原生工具调用。
问题仍是：相同资料与总预算下，正确对应的设计决策监督能否改善新房间完整生成，超过额外文本、已有联合训练和普通工具方法？
这是一项经验假设。没有“增加D一定提升最优成功率”的定理；链式分解pi(D,A|X)=pi(D|X)pi(A|X,D)也不是新算法。
本轮不把LSS、RED、文件版本或AABB修正改名为贡献。

2. 实际准备的学习目标
A使用固定来源/族/目标token权重的动作CE。AUX保留同样D目标，但动作不读D；J让后续A消费同一响应中的D；P生成自己的D后以新上下文执行动作。
缺少D的14条原始动作继续保留，不能补写理由来凑监督。10条已有D中只有8条标有来源依据，另2条是本地终止控制文字。
LSS式对照把D当冻结向量监督，不当动作输入；它不是AUX的另一种叫法。当前只实现并核验公式核，尚无可用的场景语义向量及phase标签。
若采用不同文本编码器、不同动作特征位置或新的场景阶段映射，必须标为迁移对照，不能称原论文复现。
尤其对自回归工具调用，必须记录对齐的是预测当前目标之前的状态，还是已消费金标准动作后的表示；二者信息不同。
相同辅助损失变小不能证明动作学会了理由，投影头本身拟合也不能代替闭环行为证据。

3. 为什么还需要RED类强对照
对于相同输出前缀a_<t，RED分布为q(a_t)正比于p_visual(a_t)*p_reason(a_t)^lambda。
原版函数把原始logits加权后softmax；与分别log_softmax再相加，在有限固定logits下只差一个与词元无关的常数，最终分布一致。
已运行8个固定向量例子，包含lambda=0；误差约为浮点精度。不能据此声称双上下文模型的padding、position或KV-cache路径也正确。
设beta=1/lambda，F(q)=E_q[log p_reason]-beta KL(q||p_visual)。q*为上述分布时，F(q*)-F(q)=beta KL(q||q*)。
这是已有KL正则化最优解的核验，不是新理论；其奖励是理由条件似然，未等同于原文完成或几何合法。
明确的抽象反例：合法/非法两动作的视觉概率为(0.9,0.1)，理由概率为(0.01,0.99)，lambda=1时组合概率约为(0.0833,0.9167)。
组合更服从错误理由，反而选非法动作。这是构造的数学例子，不是观察到某个模型或真实房间失败。
因此，评测必须保留真实几何与原任务，不能把D敏感性本身作为要最大化的三维质量标签。

4. 共同工具与实际计量
所有条件共享同一修正后的原生反馈。原阈值、原始要求和失败分母保留；不让候选独享正确工具。
旧23条forwards实为23次model.generate：19次policy、4次tool，共4553输出token、46377输入token。
它们不是已测量的逐token神经forward数。RED每个输出位置计算两个上下文，P额外做一次前缀编码，都不能用旧字段隐去。
以后并列记录generate请求、顶层模型prefill/decode调用及各上下文token、原生工具/候选、总耗时、RAM/VRAM。只比较“调用次数”不足以称等计算。
同一可训练模型若也改变工具VLM，则目标梯度还含工具输出概率的score；欲只检验policy监督，必须冻结工具角色或验证adapter隔离。
当前没有训练adapter，也没有把基线共享文件快照宣称完整worker/RNG恢复。

5. 能运行之前还缺的证据
现有24条动作来自6个本地重建来源族，不是作者OTA原轨迹。原生卧室仍是未完成开发尝试；没有新增可核验完整正房间。
IL3D固定8项是静态合成活动室、69实例，没有原始全任务或理由；不能拿来冒充专家轨迹。
无需等待整场正例才能做所有动作级学习，但“原生调用success”只说明该工具报告成功，不是正确设计决策或偏好标签。
若用部分监督训练，只能按原来已支持的动作/理由身份解释；是否改善完整房间仍由独立原文、几何、功能评价决定。
不因当前缺完整标签而宣称方向失败；也不为了开训练，把unknown改成正例。当前阶段不启动训练或新增本地推理。

6. 下一项实际实验的顺序
先将共同反馈修正和前瞻文件版本记录接到新的真实生成中，原参数重放/回滚只在已资格化范围内使用；绝不重算未知旧SDF历史。
然后在有相同原任务、状态和资产证据的新房间上运行A/AUX/J/P及F/T的闭环对照。LSS/RED只有各自真实接口与资料可用才进入，不能用公式检查冒充实跑。
先报告可计算的原始要求覆盖、已证实满足/失败/未知、误改、运行截止和总成本；成功必须满足预先冻结的整场标准。
若AUX/J/T或普通反馈解释全部收益，就修订学习主张；若P仅因多算而变好，需报告质量—成本曲线而非归因于“决策机制”。
工程修正和原码算术验证现已形成可复核成果，但完整三维生成上的学习收益和CVPR贡献仍未被证明，研究目标继续保持。
'''
    (stage/'algorithm_theory.txt').write_text(theory)
    contract=dict(status='Prepared strong-control arithmetic and theory; no learned claim',
        target='Complete editable indoor3D generation, not rationale fluency or local geometry alone',
        actual_new_work=['Read complete LSS main text and targeted RED method/source',
            'Executed unmodified original RED combination arithmetic on CPU',
            'Implemented independent explicit-mask cosine loss, checked mathematical gradient only',
            'Audited historical23generate requests and existing8.767B checkpoint headers'],
        deployment_actions_must_see_actual_geometry=True,
        known_missing=['No new original expert reasoning trajectories','No learned full-room comparison',
            'No LSS scene embedding/phase dataset','No actual RED dual-context native model decoder',
            'No proven frozen-tool LoRA role switch','No measured LoRA training activation peak'],
        original_positive_room_labels_added=0,new_VLM_forwards=0,optimizer_updates=0,paid_API_calls=0,
        prospective_resource_request=False,goal_achieved=False)
    dump(stage/'algorithm_contract.json',contract)
    rows=[['RED_original_arithmetic_cases',len(result['probability_cases']),'fixed mathematical scores, not model outputs'],
          ['RED_arithmetic_seconds',result['seconds'],'includes original module import; not generation latency'],
          ['historical_generate_requests',costs['sequence_generation_requests'],'not internal neural-forward count'],
          ['historical_output_tokens',costs['completed_sequence_output_tokens'],'old model run, not new cost'],
          ['checkpoint_parameters',memory['parameter_count'],'existing weights, headers only'],
          ['rank8_language_LoRA_parameters',memory['plans'][0]['trainable_parameters'],'252target modules, proposed ordinaryLoRA'],
          ['rank8_static_base_adapter_GiB',memory['plans'][0]['one_FP16_base_plus_adapter_static_bytes']/2**30,'excludes activations/workspaces; not training-fit measurement'],
          ['new_VLM_forwards',0,'none'],['optimizer_updates',0,'none'],['paid_API_calls',0,'none'],
          ['new_verified_full_scenes',0,'not inferred from arithmetic checks']]
    with (stage/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','value','scope']);w.writerows(rows)
    (stage/'reproduce.txt').write_text('''Run from REPOSITORY. Every result output must be fresh.

Original RED source is NOT redistributed. The pinned file URLs and SHA256 are in RED_source_manifest.json. Internal evaluation must respect the author license; use unmodified source and its original dependencies, not copied/relicensed snippets.
The completed original function probe used the existing transformers4.57.1 environment, no model load:
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /root/autodl-tmp/agent_method_preparation_20261027/work/v100_runtime_20261104/runtime/bin/python scripts/autodl/probe_reasoning_control_arithmetic.py --stage NEW_STAGE_WITH_MANIFEST --out-name NEW_RESULT
First default-native-Python import failure is preserved in arithmetic_v1/result.json. One environment-only continuation completed arithmetic_v2; no source editing or model retry.

Header-only resource calculation:
python scripts/autodl/audit_adapter_static_memory.py --checkpoint EXISTING_CHECKPOINT --output NEW_RESULT.json

Reports consolidate saved evidence only:
python scripts/autodl/report_reasoning_controls.py --repo REPOSITORY --stage STAGE
None of these commands runs a scene-generating VLM, optimization/training, a paidAPI or a new GPU instance.
''')
    links=['algorithm_theory.txt','algorithm_contract.json','strong_controls.json','reading_ledger.json',
           'arithmetic_v1/result.json','ENVIRONMENT_CORRECTION.json','arithmetic_v2/result.json',
           'reason_correctness_counterexample.json','actual_cost_semantics.json',
           'adapter_static_memory.json','RED_source_manifest.json','source_before_after.json',
           'summary.csv','reproduce.txt','evidence.zip']
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    (stage/'stage_report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>三维决策学习强对照与理论</title><style>body{max-width:1050px;margin:40px auto;padding:20px;font:16px/1.7 system-ui}pre{white-space:pre-wrap}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style><h1>从“理由会被读取”到完整任务效果</h1><p>补齐LSS训练期表示对齐和RED解码对照，实际执行原版概率合并，并纠正历史成本字段含义。不是新损失或完整生成成绩；没有启动训练或新模型推理。</p><table><tr><th>指标</th><th>值</th><th>范围</th></tr>'+table+'</table><h2>算法与推导</h2><pre>'+html.escape(theory)+'</pre><h2>证据</h2><ul>'+''.join('<li><a href="'+x+'">'+x+'</a></li>' for x in links)+'</ul><p>一手来源：<a href="https://arxiv.org/html/2609.04893v1">LSS</a>；<a href="https://arxiv.org/html/2507.07685v2">RED</a>；<a href="https://github.com/yshinya6/red">RED作者仓库</a>。源码许可、阅读范围和未实现项均单独保留。</p></html>')
    scripts=['reasoning_control_objectives.py','probe_reasoning_control_arithmetic.py',
             'audit_adapter_static_memory.py','report_reasoning_controls.py']
    files=[p for p in sorted(stage.rglob('*')) if p.is_file() and p.name not in ('evidence.zip','delivery_verification.json')]
    assert not (stage/'original_source').exists()
    bundle=stage/'evidence.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(stage)))
        for name in scripts:z.write(repo/'scripts/autodl'/name,'code/'+name)
    dump(stage/'delivery_verification.json',dict(
        files={str(p.relative_to(stage)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
        scripts={name:sha(repo/'scripts/autodl'/name) for name in scripts},
        evidence_zip=dict(bytes=bundle.stat().st_size,sha256=sha(bundle)),
        protected_inputs=len(checks),data_free_GiB=shutil.disk_usage(stage).free/2**30,
        original_RED_source_distributed=False,VLM_forwards=0,optimizer_updates=0,paid_API_calls=0,
        goal_achieved=False))
    print(json.dumps(dict(report=str(stage/'stage_report.html'),evidence_bytes=bundle.stat().st_size,
                         protected_inputs=len(checks),original_source_distributed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve())
