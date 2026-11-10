"""Package completed CPU checks and the precise prospective research contrast."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/autodl/20261110_local_tool_model'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    enc=read('actual_encoding_v3/summary.json');peft=read('peft_probe_v3/summary.json')
    ctrl=read('controller_regression_v1/summary.json');cost=read('fixture_costs.json')
    assert enc['status']==peft['status']==ctrl['status']=='passed'
    assert peft['actual_merged_peft_adapter_rejected'] and enc['language_model_inference_calls']==0
    source=read('actual_encoding_v3/source_hashes.json')
    for p,h in source.items():assert digest(Path(p))==h
    for p,h in peft['source_hashes'].items():assert digest(Path(p))==h
    before=OUT/'full_generation_bridge_before.py'
    parent_source=subprocess.check_output(['git','show','4c88bcf3a756193a127a828b3319c8b362f6b098:scripts/autodl/full_generation_bridge.py'],cwd=ROOT)
    assert before.read_bytes()==parent_source
    changes=['full_generation_bridge.py','local_tool_vlm.py','check_local_tool_vlm.py',
             'check_peft_base_isolation.py','summarize_generation_cost.py','report_local_tool_model.py']
    write('source_check.json',dict(old_input_files=len(source),old_inputs_unchanged=True,
        previous_bridge_equals_committed_parent=True,
        source_hashes=source,new_code={p:digest(ROOT/'scripts/autodl'/p) for p in changes},
        peft_sources_unchanged=peft['source_hashes']))
    write('summary.json',dict(
        previous_goal_turn='Progress: authoritative Git branches and raw trajectory results checked; outcome/implementation gaps distinguished.',
        this_turn='Progress: shared local tool backend and cumulative budgets implemented; real PEFT behavior checked; cost and research contrasts made explicit.',
        actual_native_view_encoding=True,native_images=enc['image_count'],encoding_seconds=enc['wall_seconds'],
        encoding_peak_rss_kib=enc['peak_rss_kib'],actual_peft_linear_probe=True,peft_seconds=peft['wall_seconds'],
        peft_peak_rss_kib=peft['peak_rss_kib'],controller_cases=len(ctrl['cases']),controller_seconds=ctrl['seconds'],
        scripted_generation_calls=enc['fixture_generate_calls'],language_model_inference_calls=0,paid_api_calls=0,
        training_updates=0,new_complete_scenes=0,full_runtime_ready=False,training_ready=False,
        novel_algorithm_established=False,gpu_requested=False,
        remaining=['Actual local 8B tool inference and native physical metadata quality',
                   'All native services/assets and no-paid-client path qualification',
                   'Incomplete training observation/decision coverage and full-generation ordinary search',
                   'Independent full-scene benefit of equal-material training/deployment contrasts']))
    write('experiment_estimands.json',{
        'status':'Prospective mechanism comparison, not a new learning objective or authorization to run',
        'variables':{'X':'original task, actual scene observation, same permitted evidence and tool results',
                     'D':'explicit generated decision proposal, not hidden reasoning or an executed fact',
                     'A':'native action or proposed stop','M':'source units with recorded D'},
        'objectives':{
            'auxiliary':'L_aux = sum_i CE_w(A_i|X_i) + sum_{i in M} CE_v(D_i|X_i)',
            'conditioned':'L_cond = sum_{i in M} CE_w(A_i|X_i,D_i) + sum_{i not in M} CE_w(A_i|X_i) + sum_{i in M} CE_v(D_i|X_i)',
            'weights':'Same immutable source token coefficients in both objectives; split components must not be renormalized to equal weight.',
            'not_claimed':'No new loss, no universal improvement theorem, no causal interpretation of plausible prose.'},
        'arms':[
            {'name':'separate_direct_v3','training':'auxiliary','deployment':'A(X)','role':'Primary efficient learned control'},
            {'name':'separate_phase_v3','training':'auxiliary','deployment':'generate D for audit, then A(X) without D','role':'Diagnostic only; intentionally unused D is not a strong efficiency baseline'},
            {'name':'paired_phase_v3','training':'conditioned','deployment':'generate D, then A(X,D_hat)','role':'Candidate end-to-end mechanism'},
            {'name':'base_plus_shared_search','training':'none','deployment':'base policy, same knowledge/evaluation/tools, ordinary multi-operation search','role':'Strong control, full-generation search not yet implemented'}],
        'identification':[
            'Action-only versus rationale+action changes supervision quantity and conditioning; cannot identify the conditioning effect.',
            'Conditioned versus auxiliary with same materials estimates a training-and-deployment treatment, not pure semantic causality of D.',
            'A self-generated D may help finite computation even without new external information; factorization alone guarantees no gain.',
            'Removing/replacing D is an off-distribution intervention and is supplementary evidence, not an alternative performance score.',
            'If D is correct but A fails, distinguish grounding/action limitation; if A executes but room fails, distinguish metric/constraint coverage.',
            'Do not silently replace generated D with teacher D or reset actual states to saved correct rollouts.'
        ],
        'material_limit':{'units':24,'families':6,'recorded_decision_units':10,'missing_decision_units':14,
                          'all_targets_furniture':True,'consequence':'Retain missing-D action units. No invented decisions or claims of full-stage training; conditioning exposure is incomplete.'},
        'cost':{
            'model_tokens':'policy D+A plus tool-side analyses, invalid responses and retries; input and output separately',
            'local_time':'outer trial wall plus stage breakdown, never add nested stage time to outer wall',
            'assets':'same catalogue/metadata and cache policy; report one-time preprocessing separately and amortization assumption',
            'comparison':'quality against actual total cost and fixed limits; identical token caps do not equal identical latency or FLOPs',
            'saved_compute':'ordinary search may spend saved reasoning budget through the same tools, not a single-action weakened control'},
        'quality':{'primary':'all frozen mandatory instances and supported requirements satisfied simultaneously',
                   'also_report':['unknown/unsupported requirements','newly broken requirements','collisions and bounds','edit magnitude','whole-stage completion'],
                   'unit':'independent room/description family; seeds and views are nested, not independent samples'},
        'decision_rule':'Only expand after actual full-scene gains over auxiliary direct-action and common search, at matched meaningful cost, on independent descriptions. A positive pilot is not proof of CVPR novelty.'
    })
    write('reading_ledger.json',[
        {'source':'SceneSmith VLMService and mesh physics analysis','pin':'e888297a8b0a2f3acbb8634edf9def59cceab836',
         'local':'../20261110_native_asset_contract/tool_source/manifest.json',
         'scope':'Re-read native completion conversion and mesh analysis request/response path; previous turn read all three preserved tool files.',
         'implication':'Changing the tool model can change physical metadata even if policy is fixed. Same tool backend must be shared; numeric mass/material output is still an estimate.'},
        {'source':'PEFT 0.17.1','url':'https://github.com/huggingface/peft/blob/v0.17.1/src/peft/peft_model.py',
         'scope':'disable_adapter and get_model_status path; LoRA layer.forward and enable/disable branch, not whole package.',
         'implication':'Reject merged/irregular adapters; actual tiny forward confirms context restoration, not 8B equivalence.'},
        {'source':'Distilling step-by-step','url':'https://aclanthology.org/2023.findings-acl.507/',
         'code':'https://github.com/google-research/distilling-step-by-step/blob/ef944263c9ddfd40a9888557f3e7d17513fd1a7e/model_utils.py',
         'scope':'Re-read complete model_utils.py including separate prediction/explanation forwards. run.py fully read in preceding turn; no new full-paper reading claim.',
         'implication':'Auxiliary rationale supervision is an existing baseline; causal LM adaptation is not the exact T5 reproduction.'},
        {'source':'Measuring Faithfulness in Chain-of-Thought Reasoning','url':'https://www-cdn.anthropic.com/827afa7dd36e4afbb1a49c735bfbb2c69749756e/measuring-faithfulness-in-chain-of-thought-reasoning.pdf',
         'scope':'Abstract, introduction, methods 2.1–2.4 and table 2 in extracted text, PDF pages 1–5; no complete appendix/figure-image review.',
         'implication':'Changes after intervention and performance gains are different quantities; neither textual plausibility nor sensitivity alone establishes useful scene design.'}
    ])
    write('commands.json',{
        'python':'/root/autodl-tmp/agent_method_preparation_20261027/work/inference_autodl_20261103/runtime/bin/python',
        'working_directory':str(ROOT),
        'commands':[
            ['scripts/autodl/check_peft_base_isolation.py','--package',str(OUT/'peft_runtime/deps'),'--out','NEW_EMPTY_PROBE_DIRECTORY'],
            ['scripts/autodl/check_local_tool_vlm.py','--processor','/root/autodl-tmp/agent_method_preparation_20261027/work/inference_autodl_20261103/processor_only',
             '--observation-source','runs/autodl/20261110_native_observation_export/checkpoint_renderer/src/scene_observation.py',
             '--bindings','runs/autodl/20261110_stage_alignment_review/encoding_v2/observation_bindings.json',
             '--vlm-source','runs/autodl/20261110_native_asset_contract/tool_source/vlm_service.py','--out','NEW_EMPTY_ENCODING_DIRECTORY'],
            ['scripts/autodl/summarize_generation_cost.py','--policy',str(OUT/'actual_encoding_v3/fixture_policy_calls.json'),
             '--tool',str(OUT/'actual_encoding_v3/fixture_tool_calls.json'),'--fixture','--out','NEW_COST_JSON']
        ],'notes':'Set PYTHONDONTWRITEBYTECODE=1; reruns use new paths. Acquisition receipts pin two local probe packages; they are ignored, not committed. Images require existing licensed local resources.'})
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>工具模型隔离、真实库验证与研究对照</title>
<style>body{font:17px/1.7 system-ui;max-width:1060px;margin:40px auto;padding:0 20px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:9px}code{background:#eee}a{color:#165eb0}</style>
<h1>把工具模型保持一致，才能检验设计决定的作用</h1>
<p>本轮接通共享本地模型的工具服务绑定，完成真实图像编码和真实 PEFT 库的小型数值检查。没有语言模型推理、训练、完整场景生成或付费 API。前一轮 Git 审阅已确认研究未建立方法优势；本轮解决的是一项真实的实验混杂因素，不将工程修正称为创新。</p>
<h2>实际执行与边界</h2>
<table><tr><th>执行</th><th>结果</th><th>不能推出</th></tr>
<tr><td>五张原生场景视图，真实 Qwen processor 编码</td><td>文件输入与内嵌图像输入的 token、像素张量和网格完全一致；17.98 秒，峰值约804 MiB（v2记录）</td><td>历史像素完全复原、模型看懂图像、真实物理分析成功</td></tr>
<tr><td>PEFT 0.17.1 实际 LoRA，4×4 线性网络</td><td>关闭后精确恢复基座输出，嵌套与异常退出恢复状态；已合并适配器被接口拒绝</td><td>8B 权重/GPU配置和原生整场流程已验证</td></tr>
<tr><td>原生控制函数的七个替身场景</td><td>控制流程回归通过；中途结束不冒充整场完成</td><td>七个真实生成成功</td></tr>
<tr><td>调用成本汇总</td><td>三次脚本生成（含一次错误 JSON）均计入；缺失工具账本记未知</td><td>全系统成本已齐全或三次真实模型推理</td></tr></table>
<p>首次 PEFT 导入缺 accelerate 的失败仍保留；补充固定小型依赖后才通过。未安装到旧环境。各版本结果均保留，最终数值见<a href="summary.json">汇总</a>与<a href="peft_probe_v3/summary.json">PEFT 原始记录</a>。</p>
<h2>为什么这项接口影响研究结论</h2>
<p>原生资产处理会用 VLM 判断朝向、材质和质量。若训练后连这个工具也换成带实验适配器的模型，比较同时改变了规划与执行环境。新绑定在原生服务构造前安装，在原生主模型载入后绑定同一实例；工具分析暂时关闭实验适配器。未知模型映射、外部图片、截断输出和预算耗尽均显式失败。模型替换必须记录，不能称为原作者模型的等价复现。</p>
<p>覆盖仅限新构造的 VLMService；既有实例、Agents SDK、图像生成和其他客户端未被统一接管。图像编码是真实的，三次输出是固定脚本。全系统零付费路径、真实资产物理分析及8B共享实例仍未通过实际运行验证，因此 full_runtime_ready 继续为 false。</p>
<h2>保留的算法问题与正确强对照</h2>
<p>设 X 为任务与当前场景，D 为明确输出的设计决定，A 为实际动作。两组都使用同一来源的 D、A 与原 token 权重。辅助组学习 D|X 和 A|X；条件组学习 D|X 和 A|X,D。有 D 的来源才监督后一条件，缺少 D 的14个动作单元仍保留，不能补造理由或当作全部已配对。</p>
<p>实际生成时条件组只能使用自己生成的 D；执行后依据真实新场景继续。主要效率对照是辅助组直接行动，并允许普通搜索使用省下的计算。辅助组先生成但不读取 D，只能作为机制诊断，不能当作被故意浪费预算的主要竞争方法。同材料条件对照识别的是“训练＋部署”的组合变化，不能单独证明理由的语义因果作用。</p>
<p><a href="https://aclanthology.org/2023.findings-acl.507/">Distilling step-by-step</a> 的官方实现已分开计算预测与解释损失；这里的辅助目标不是新损失。<a href="https://www-cdn.anthropic.com/827afa7dd36e4afbb1a49c735bfbb2c69749756e/measuring-faithfulness-in-chain-of-thought-reasoning.pdf">Measuring Faithfulness</a> 用推理文本干预研究输出依赖，并区分依赖与性能。对我们而言，理由看似合理、动作对理由敏感、整场质量改善，是三项不同的证据。删除理由导致变化也可能来自输入分布改变，不能直接认定理由忠实或有用。</p>
<h2>完整场景与成本必须同时落地</h2>
<p>共同资产、工具模型、允许操作、未知项规则与评价保持一致。主要结果是必需实例和支持的要求同时完成，另报未验证项、碰撞、越界、损坏原正确关系和改动规模。停止或阶段保存成功不等于设计成功；房间家族才是独立统计单位。</p>
<p>成本包括策略 D+A、工具侧模型、错误输出及重试、资产准备与渲染求解。输入/输出 token 分开，阶段时间不重复叠加到整场时钟。相同输出上限不等于相同 FLOPs 或耗时。本轮汇总器明确拒绝从局部账本推出全系统成本，见<a href="fixture_costs.json">脚本账本汇总</a>。</p>
<p>这仍是值得做有限验证的机制，而非已确认的三维新方法。下一项有价值的证据应当是共同后端上的完整生成与强对照；当前还缺原始观察覆盖、可用仿真资产、真实工具推理与整场普通搜索。没有因本轮小检查通过就要求开启GPU，也没有开始扩充训练。</p>
<p><a href="experiment_estimands.json">训练与部署对照的具体定义</a> · <a href="reading_ledger.json">阅读范围</a> · <a href="commands.json">复核命令</a> · <a href="source_check.json">源文件核对</a> · <a href="actual_encoding_v3/summary.json">最终编码与绑定检查</a></p></html>'''
    # Substitute the actual final check timing, not an earlier execution's time.
    html=html.replace('17.98 秒，峰值约804 MiB（v2记录）',f"{enc['wall_seconds']:.2f} 秒，峰值约{enc['peak_rss_kib']/1024:.1f} MiB")
    (OUT/'stage_report.html').write_text(html)
    paths=[p for p in OUT.rglob('*') if p.is_file() and 'deps' not in p.parts and p.name!='artifact_manifest.json']
    write('artifact_manifest.json',{'files':[{'path':str(p.relative_to(OUT)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(paths)],
        'excluded':'Local downloaded probe dependencies; acquisition URLs/hashes retained. No model weights or licensed meshes copied.'})
    print(json.dumps({'status':'packaged','files':len(paths),'summary':str(OUT/'summary.json')}))


if __name__=='__main__':main()
