"""Summarize source review and controller evidence; never run a scene/model.

The small algebra examples are counterexamples to overbroad claims, not
experimental results. Source call sites indicate possible dependencies, not
observed network traffic or a complete transitive call graph.
"""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import subprocess


def write(path, value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def source_record(path, selected):
    raw=path.read_bytes();tree=ast.parse(raw)
    records=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in selected:
            calls=[]
            for child in ast.walk(node):
                if isinstance(child,ast.Call):
                    calls.append({'line':child.lineno,'function':ast.unparse(child.func)})
            records.append({'function':node.name,'start':node.lineno,'end':node.end_lineno,'calls':calls})
    missing=set(selected)-{r['function'] for r in records}
    if missing:raise ValueError('Missing source functions: '+str(missing))
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),functions=records)


def main(a):
    out=a.out
    summary=json.loads((out/'controller_v2/summary.json').read_text())
    assert summary['status']=='passed' and summary['actual_generated_scenes']==0
    source=a.base/'native_execution_inventory_20261104/source'
    manager=a.base/'native_room_bridge_20261104/source/scenesmith/agent_utils/asset_manager.py'
    audit={'scope':'Static source reading and call-site inventory; not a transitive dependency proof or actual provider billing.',
      'sources':[
        source_record(source/'src/agent_rollout_decoupled.py',
            ['main_async','driver_loop','finalize_stage','handover_after_floor_plan','make_roomscene_env']),
        source_record(source/'Designer/replay_chained.py',
            ['compose_full_cfg','start_all','_start_geometry','_start_hssd','_start_articulated']),
        source_record(manager,['generate_assets','_retrieve_hssd_assets','_generate_assets_with_router',
                               '_generate_images','_convert_mesh_to_simulation_asset']),
        source_record(out/'primary_source/model_utils.py',['__call__','compute_loss','prediction_step']),
        source_record(out/'primary_source/run.py',['run'])],
      'findings':[
        {'finding':'Native floor geometry errors can be caught and followed by a normal return; zero exit is not four-stage completion.',
         'source':'agent_rollout_decoupled.main_async'},
        {'finding':'Native main constructs its own policy once, after starting services. A separate already-loaded furniture model would duplicate model ownership.',
         'source':'agent_rollout_decoupled.main_async'},
        {'finding':'Router-enabled asset acquisition invokes request analysis before strategy execution.',
         'source':'AssetManager._generate_assets_with_router'},
        {'finding':'Router-disabled HSSD retrieval still invokes VLM mesh orientation/material analysis.',
         'source':'AssetManager._retrieve_hssd_assets'},
        {'finding':'Generated asset paths invoke image generation and subsequent VLM mesh analysis.',
         'source':'AssetManager._generate_images and _convert_mesh_to_simulation_asset'},
        {'finding':'The auxiliary rationale baseline has an official implementation with separate prediction and explanation inputs and weighted task losses.',
         'source':'Distilling step-by-step model_utils.py and run.py'}],
      'qualification':{
        'real_shared_adapter_runtime':'NOT VERIFIED: local reviewed Python has torch 2.7.0+cu128 / transformers 4.57.1 but no PEFT installation; controller uses explicit doubles.',
        'tool_side_no_paid_api':'NOT VERIFIED: model-capable paths remain; configuration alone does not establish provider use/cost.',
        'complete_scene_quality':'NOT EVALUATED: no new scene generated.',
        'next_concrete_work':'Resolve a common asset/service path with attributable metadata and zero paid requests; then run the actual shared-adapter runtime. Do not weaken all-stage or quality criteria to pass.'}}
    cfg=source/'integration/scene_designer.yaml'
    audit['config']={'path':str(cfg),'sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),
        'coverage':'Entire integration overlay read; effective composed upstream defaults not fully audited.',
        'meaning':'Local top-level policy does not establish that asset/tool-side model calls are local. No API call was made.'}
    write(out/'source_audit.json',audit)

    # Finite, explicit numerical counterexamples. No model or geometry run.
    # Two equally likely standardized values in each tier; unit weights are
    # illustrative, NOT asserted to be the authors' experimental hyperparameters.
    raw={'hard':[-.001,0.0],'soft':[1.0,0.0],'consistency':[1.0,0.0]}
    z={}
    for k,values in raw.items():
        mean=sum(values)/2;sd=(sum((v-mean)**2 for v in values)/2)**.5
        z[k]=[(v-mean)/sd for v in values]
    score=[sum(z[k][i] for k in z) for i in range(2)]
    assert score[0]>score[1]
    marginal_a0=.6*.51+.4*.01
    theory={'status':'Elementary counterexamples and experiment implications, not new theorems or scene results.',
      'soft_tiers_are_not_feasibility':{
        'raw_rewards':raw,'illustrative_weights':dict.fromkeys(raw,1.0),
        'standardized_rewards':z,'combined_advantage':score,
        'result':'The violating candidate gets the higher weighted score in this example. Positive affine final normalization preserves ordering.',
        'boundary':'Not a reproduction of ThinkBLOX results or its selected weights; normalization and tier names alone give no strict feasibility guarantee.',
        'consequence':'Keep actual required-object/geometry checks separate from learned or weighted scores; unknown checks remain unknown.'},
      'greedy_reasoning_is_not_marginal_decision':{
        'decision_probabilities':[.6,.4],
        'action_0_given_decision':[.51,.01],
        'marginal_action_0':marginal_a0,'marginal_action_1':1-marginal_a0,
        'result':'Greedy decision then greedy action selects action 0; the marginal action mode is action 1.',
        'boundary':'Does not show that more reasoning harms our model. It blocks a universal guarantee from the factorization alone.'},
      'source_to_method_difference':{
        'auxiliary_control':'Official Distilling step-by-step separates label and rationale tasks. Our action/decision weighted causal objective is an adaptation, not an exact T5 reproduction or a new loss.',
        'reason_then_act':'ThinkBLOX already trains progressive 3D reason-and-place sequences. Furniture D-to-A conditioning is not itself new.',
        'candidate_estimand':'Final full-room quality at fixed total cost, or cost to matched quality, for self-generated D conditioning versus auxiliary-D/direct-A learning with the same original supervision.',
        'important_control':'Direct-A plus additional ordinary candidate search must receive the saved inference budget; identical output caps alone do not imply equal cost.',
        'data_limit':'24 units / 6 development families / 10 original decisions; 14 action-only units do not train the conditioned pathway. No new labels created or units deleted.',
        'not_yet_identified':'No established three-dimensional inductive bias beyond ordinary rationale supervision/conditioning; a positive tiny pilot would justify investigation, not a CVPR novelty claim.'}}
    write(out/'theory_and_method_difference.json',theory)
    reading={'scope':'Targeted methods/source review; not exhaustive nearest-neighbor clearance.',
      'papers':[
        {'title':'ThinkBLOX','url':'https://arxiv.org/html/2607.13539v1',
         'read_this_turn':'Method III-A/III-B and reward equations 1–3 rechecked; earlier reading ledger covers other main-text sections. No new full-text/figure/official-code reproduction.',
         'finding':'Progressive 3D reasoning-action supervision is already present. Tier-wise normalized rewards remain a weighted objective; that formulation alone is not a hard feasibility constraint.'},
        {'title':'Distilling step-by-step','url':'https://aclanthology.org/2023.findings-acl.507.pdf',
         'read_this_turn':'Abstract/introduction rechecked; §3 method read previously. This turn fully read official model_utils.py and run.py at ef944263c9ddfd40a9888557f3e7d17513fd1a7e.',
         'finding':'Independent rationale and prediction tasks are a necessary established control. Did not run the official training.'}],
      'code_scope':{'native_controller':'main_async/driver_loop and selected finalize/handover paths',
        'replay_chained':'configuration and server startup branches, not full file',
        'asset_manager':'selected acquisition/router/analysis methods, not all 2190+ lines',
        'new_bridge':'complete implementation and controller tests',
        'peft':'not installed; actual adapter runtime not tested'}}
    write(out/'reading_ledger.json',reading)
    result={'previous_goal_turn':'progress: remote commits and existing actual results checked; comparison eligibility was reassessed without starting an experiment.',
      'this_turn':'progress: native shared-model bridge implemented, silent early-return failure tracked, selected tool dependencies and official auxiliary-supervision implementation audited.',
      'controller_cases':len(summary['cases']),'controller_seconds':summary['seconds'],
      'actual_language_model_calls':0,'training_updates':0,'actual_native_scene_actions':0,
      'actual_new_complete_scenes':0,'paid_api_calls':0,
      'full_runtime_ready':False,'gpu_requested':False,'novel_algorithm_established':False,
      'limitations':['Service/model doubles verify orchestration only.',
        'Ledger meters policy generation; tool-side calls are outside its scope.',
        'All four finalizers returning is execution completion, not scene success.',
        'Dataset observation and original-decision coverage remain incomplete.',
        'Full-generation common ordinary-search control is still incomplete.']}
    write(out/'summary.json',result)
    repro={'cwd':str(Path.cwd()),'python':str(a.base/'inference_autodl_20261103/runtime/bin/python'),
      'argv':['scripts/autodl/check_full_generation_bridge.py','--native-source',str(source/'src/agent_rollout_decoupled.py'),
              '--tokenizer',str(a.base/'inference_autodl_20261103/processor_only/tokenizer.json'),
              '--out','NEW_NONEXISTENT_OUTPUT_DIRECTORY'],
      'environment':{'PYTHONDONTWRITEBYTECODE':'1'},
      'scope':'Reproduces controller tests only. Production callable is full_generation_bridge.run_native_trial; current production dependencies are not qualified.',
      'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write(out/'commands.json',repro)
    rows=''.join('<tr><td>'+html.escape(c['case'])+'</td><td>'+html.escape(c['trial']['status'])+'</td><td>'+str(c['fixture_generate_calls'])+'</td></tr>' for c in summary['cases'])
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>整场生成接入与方法边界</title>
<style>body{font:17px/1.65 system-ui;max-width:1050px;margin:40px auto;padding:0 20px;color:#17212b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}code{background:#eee}a{color:#145eb0}</style>
<h1>整场流程有了接入，但仍未证明方法收益</h1>
<p>本轮实现共享模型的原生四阶段接入：只有家具阶段启用实验适配器，其他阶段恢复共同基座；两种方法均记录实际策略生成调用。完整生成质量仍未得到验证，没有新模型推理、训练、场景生成或付费调用。</p>
<h2>这次具体修正</h2><p>原生 main_async 在地面几何失败时捕获异常并正常返回。新接入逐阶段记录最终保存与交接，不能再把正常退出当作整场完成。最终保存返回也只代表执行经过该步骤，物体、关系和几何是否正确仍须独立验收。</p>
<table><tr><th>控制流程例</th><th>记录状态</th><th>替身生成调用</th></tr>'''+rows+'''</table>
<p>这些例子运行原生控制函数，但模型、服务、场景均为显式替身。它们不进入研究效果表。PEFT 未安装，真实适配器开关与模型运行仍未验证。</p>
<h2>零付费并非只替换主 Agent</h2><p>源码中资产路由会分析请求；关闭路由、直接使用 HSSD 仍会做 VLM 朝向和材质分析；生成资产还有图像生成路径。配置只能说明调用可能性，不能证明实际收费。这些工具侧调用不在当前策略计时器内。下一步须落实共同资产与服务路径，不能仅关闭分析后声称复现相同后端。</p>
<h2>方法应当怎样区别于已有工作</h2><p><a href="https://arxiv.org/html/2607.13539v1">ThinkBLOX</a> 已有三维渐进推理与动作监督；<a href="https://aclanthology.org/2023.findings-acl.507/">Distilling step-by-step</a> 已有解释作为辅助任务的方案。本轮固定并完整读取后者的两份官方实现文件。因此，分两个阶段生成，或增加解释交叉熵，都不能单独作为新算法。</p>
<p>保留的实验问题是：相同监督与工具下，自生成决策是否改善最终整场结果，且超过直接动作与普通搜索使用同等计算的结果？它目前是机制对照，尚未形成有证据的三维特有方法。不能把现有 24 条家具单元包装成完整生成训练集。</p>
<p>数值反例另外说明：多层奖励标准化不自动保证硬约束；先贪心选决策、再选动作也不保证获得边缘分布中的最佳动作。这些只是基础数学边界，不是新定理，也不是场景实验。</p>
<h2>接下来需要的实际证据</h2><p>先接通可追踪的共同资产/服务和真实共享模型，再完成固定描述的四阶段生成。随后比较基础模型、辅助监督直接动作、自生成决策后动作，以及等预算普通搜索。主要记录完整要求、缺项、破坏原正确关系、未知项和全流程成本。当前不据此要求开 GPU；尚有不依赖 GPU 的运行依赖工作。</p>
<p><a href="summary.json">结果范围</a> · <a href="source_audit.json">源码与依赖</a> · <a href="theory_and_method_difference.json">方法与数学边界</a> · <a href="reading_ledger.json">阅读记录</a> · <a href="commands.json">复核命令</a> · <a href="controller_v2/summary.json">七项原始结果</a></p></html>'''
    (out/'stage_report.html').write_text(report)
    files=[p for p in out.rglob('*') if p.is_file()]
    files += [Path('scripts/autodl')/name for name in ('full_generation_bridge.py','check_full_generation_bridge.py','report_full_generation_bridge.py')]
    write(out/'artifact_manifest.json',{'files':[{'path':str(p),'bytes':p.stat().st_size,
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]})
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    main(ap.parse_args())
