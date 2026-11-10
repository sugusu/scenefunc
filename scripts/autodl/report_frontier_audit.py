"""Publish a bounded source review and its executed CPU evidence."""
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/autodl/20261110_frontier_scene_evaluation'

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    summary=json.loads((OUT/'audit_v1/summary.json').read_text())
    reading=[
        {'source':'https://arxiv.org/html/2611.09017v1','sections':'Main text and A/B text reviewed across acquisition and continuation; A/B rechecked this turn',
         'not_done':'No full-resolution figure inspection, rankings reproduced, or official full evaluation executed.',
         'finding':'Benchmark inputs and whole-task score are useful references, not a ready indoor furniture-only evaluation. Human agreement study is reference-fidelity scoped; no claim of validating indoor functional preferences.'},
        {'source':'https://arxiv.org/html/2605.19587v1','sections':'Abstract, introduction, related work and method 3.1–3.3 through articulation compilation',
         'not_done':'Not full paper/appendices; no repository source or execution audited.',
         'finding':'Explicit plans, rule/LLM verification, execution repair and persistent scene records already appear; these components alone do not distinguish our candidate.'},
        {'source':'https://arxiv.org/html/2610.37326v1','sections':'Abstract, introduction, related work, 3.1–3.4; C.2–C.7 and D.1–D.3',
         'not_done':'No complete experiment/appendix audit, repository or simulation run. Project page failed access in this turn.',
         'finding':'Physical capability absence and executed interaction reliability are separated. Missing native capabilities are not evaluator-repaired; post-interaction measures need coverage.'},
        {'source':'https://github.com/wangpuyi/SceneCode','sections':'Project page and repository README only',
         'not_done':'No installed alternative backend; availability is not readiness.'},
        {'source':'https://github.com/Einsia/Frontier3D/tree/1462133985213af9bae14344d2b938e18868350b',
         'sections':'All 20 Brief2Scene instructions and attribution notices; all 20 task metadata parsed. README, scoring/experiment/licenses documentation. Full pure numeric aggregator, detail subset arithmetic and replacement; extension dispatch/finalizer and measurement branch inspected.',
         'not_done':'Other downloaded code is source inventory, not a full line-by-line review. Native image/Blender execution, judge calibration and reference assets not reproduced.'}
    ]
    dump('reading_ledger.json',reading)
    trace={
        'pin':'1462133985213af9bae14344d2b938e18868350b',
        'extension_path':['benchmark/engine.py:evaluate -> manifest harness_status',
             'tools/experiment_additions.py:evaluate -> numeric, baseline, detail, finalize',
             'tools/finalize_effort_addition.py:finalize -> level 2 dispatch',
             'tools/finalize_effort_addition_l2.py:finalize -> Brief2Scene dispatch',
             'tools/finalize_effort_addition_brief.py:finalize -> numeric_data_score and 0.6/0.3/0.1 blend'],
        'numeric_evidence':'tools/measure_effort_addition_numeric.py invokes a judge-backed run_codex using native catalogue and registered measurements. It is not wholly deterministic geometry scoring.',
        'legacy_caution':'src/evaluator/l2.py and l2_quality_scoring.py alone do not define the final release score.',
        'real_execution':'Only pure numeric_data_score and detail_from_criteria executed; full finalizer not executed.',
        'missing_policy':'Official numeric not_measurable blocks score, partial/observed retain zero credit in fixed denominator; invalid input population rejected.',
        'assets':'No meshes, LFS assets, model weights or external image references acquired.'}
    dump('scoring_path.json',trace)
    theory={
        'status':'Candidate hypothesis and identification limits; no new theorem of improvement and no new loss claimed',
        'state':'X_t contains original request, actual current scene observations, previous executed outcomes and same permitted evidence. D_t is a predicted design decision, not an executed fact. A_t is a tool action/stop.',
        'candidate':'D_t ~ pi_phi(D|X_t); A_t ~ pi_theta(A|X_t,D_t); actual native tool execution produces X_{t+1}. Never inject teacher D at evaluation or reset to a saved correct state.',
        'primary_control':'Same-source auxiliary training of D(X) and A(X), with direct A(X) deployment. Retain identical source token weights and missing-D action units. The separate-unused-D deployment is only a diagnostic.',
        'search_control':'Base policy plus ordinary multi-operation search has the same tools, assets, observations, edits and cumulative cost limits. Full-generation version still incomplete.',
        'supervision_limit':'24 units / 6 families, 10 original D labels, 14 missing; all furniture stage. Other stages stay common base. This does not establish general interior-design learning.',
        'information_argument':{
            'assumptions':'Condition on trained parameters; D generated using X and randomness independent of target Y conditional on X, without fresh external observations.',
            'statement':'Y -> X -> D is Markov, so I(Y;D|X)=0. D cannot add external target information under these assumptions.',
            'scope':'It can change finite-compute search/representation and thus empirical action quality. The information statement neither proves futility nor guarantees benefit. If D queries external tools, equal access and full costs must be controlled.'},
        'objective_identification':'Equal-material conditioned-vs-auxiliary comparison estimates a training/deployment intervention. Rationale removal/substitution may be distribution shift; sensitivity alone is not utility or faithful causal explanation.',
        'complete_task_metric':{
            'definition':'F(S,u)=AND_j c_j(S,u) over a prospectively frozen mandatory requirement set; false if a known violation, unknown if no known violation but unresolved checks; only all verified true yields pass.',
            'official_score':'Preserve published Q and all components separately. Our conjunction is a companion, not a replacement Frontier3DBench score.',
            'counterexample':'For actual task 013 with 32 numeric checks, one violated check gives R=31/32. If V=D=1, Q=0.990625 but the numeric conjunction is false. This is a synthetic algebra check, not an observed visual judgment or benchmark defect.',
            'general_bound':'For weighted mean R=sum_j w_j c_j, a single failed j yields 1-w_j; near-one average does not imply conjunction. This elementary fact is not a novel algorithm.',
            'trajectory_risk':'For properly defined failure events E_t, P(any E_t)<=sum_t P(E_t) by union bound, no independence needed. We do not have calibrated event probabilities, so this is not a safety guarantee or a learned constraint.'},
        'testable_claim':'At matched meaningful total cost, conditioned decisions improve full-room mandatory satisfaction and preservation over direct-action auxiliary learning and base+search, across independent description families.',
        'disconfirmation':'Stop scaling this conditioning mechanism if common search explains gains, gains only occur in trained room families, or rationale quality improves without executed scene gains.',
        'scope_retained':'Complete indoor scene generation remains the endpoint; no switch to outdoor/robot benchmark engineering merely to obtain a new score.',
        'next_actual_gate':'One full native generation with common local services and physical metadata, then the shared ordinary search on that same output. Only after this is executable and observations are complete can the equal-material learning comparison be qualified.'}
    dump('algorithm_and_theory.json',theory)
    dump('method_differences.json',[
        {'method':'SceneCode','known_overlap':'Object planning, validation, execution repair, editable records','candidate_difference':'Study learned decision conditioning at matched evidence, not claim to introduce plans or execution loops. No measured advantage yet.'},
        {'method':'ScIntBench','known_overlap':'Task-specific functional/physical evaluation','candidate_difference':'An evaluation resource, not our competing learning objective. Missing capabilities must remain missing; do not turn this into new physics-backend work.'},
        {'method':'Frontier3DBench','known_overlap':'Full executable scene delivery and multimodal quality checks','candidate_difference':'External evaluation reference; public Brief tasks do not directly fit current room generator. Whole-task feasibility and all-in trial cost still need separate reporting.'},
        {'method':'Ordinary search','known_overlap':'Existing native edits already resolve two local activity goals','candidate_difference':'Any learned gain must exceed this. Common full-generation search still absent.'}])
    dump('next_execution_boundary.json',{
        'do_not_launch_frontier_suite':True,'reason':'Unmodified tasks demand capabilities outside the audited room-generation contract; paid judge path also not authorized.',
        'do_not_train_yet':True,'do_not_request_gpu_for_this_audit':True,
        'remaining_before_resource_request':['Qualify all required native services without hidden paid calls','Complete actual observations and shared multi-operation search needed by the chosen comparison','Recover final state of already-started remote LEARN rather than silently retrain'],
        'concrete_cpu_deliverable':'Public input exporter and official score-semantics replay completed. No task requirements removed.'})
    # Compact only this newly acquired, uncommitted index, not old evidence.
    index_path=OUT/'repository_index.json';raw=index_path.read_bytes();index=json.loads(raw)
    manifests=[json.loads((OUT/n).read_text()) for n in ('source_manifest.json','source_manifest_supplement.json')]
    selected={r['path'] for m in manifests for r in m['files']}
    if 'compaction' not in index:
        index['compaction']={'original_index_sha256':hashlib.sha256(raw).hexdigest(),'original_bytes':len(raw),
            'original_entries':len(index['files']),'reason':'Keep exact selected-file git identities and upstream full-tree response hash; no assets downloaded.'}
        index['files']=[r for r in index['files'] if r['path'] in selected]
        index_path.write_text(json.dumps(index,indent=2)+'\n')
    summary.update(previous_goal_turn='progress: remote evidence review changed next execution priorities',
        this_goal_turn='progress: new primary-source/task review, official scoring replay and public-task exporter implemented and executed',
        full_runtime_ready=False,training_ready=False,novel_algorithm_established=False,
        source_files_acquired=len(selected),goal_complete=False,disk_available_bytes=shutil.disk_usage(OUT).free)
    dump('summary.json',summary)
    report='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>完整场景评价与机制边界复核</title>
<style>body{font-family:system-ui;max-width:1050px;margin:36px auto;line-height:1.75;color:#19242e}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd3d8;padding:10px;text-align:left}code{background:#eef1f5}a{color:#125b91}</style>
<h1>完整场景评价与机制边界复核</h1><p>2026-11-10。本轮完成公开任务接入、官方评分计算复核与近邻方法阅读；没有新增场景、模型调用、训练或付费服务。目标仍是完整室内三维生成。</p>
<h2>这轮实际增加了什么</h2><p>固定 Frontier3D 提交 1462133，获取全部20条 Brief2Scene 原文及元数据，保留任务和评分资料隔离。输入导出器逐字保留原文和哈希，不将评价规则或答案注入生成器。20份输入已构建校验，013导出命令已实际执行。</p>
<table><tr><th>证据</th><th>结果与含义</th></tr><tr><td>公开任务审阅</td><td>001–010是室外场景；011–020要求静态机器人几何与接触，不能删去这些要求来适配家具模型。</td></tr><tr><td>当前接入资格</td><td>没有任务已证明可由现有完整入口执行；这不是证明这些任务不可能完成。当前不启动新机器人或室外生成支线。</td></tr><tr><td>官方数值聚合函数</td><td>真实013评分计划有32项检查，5种状态及缺项/重复/错身份/非法状态已执行；缺测保留空值，未伪造场景测量。</td></tr><tr><td>官方细节聚合函数</td><td>构造四个变体的测量值，核实取最小变体；缺少一个变体则无总值。未运行渲染或视觉评审。</td></tr></table>
<h2>评分事实改变了什么</h2><p>官方最终文字场景分数采用视觉、要求、细节加权；extension数值检查也需要模型辅助读取已登记的原生测量，不是零API的纯几何流程。已从公开入口追踪到最终分支，不能用旧 l2.py 的默认值代替最终发布规则。</p>
<p>构造反例：013的通路项N24失败，其余31项满足，则数值分31/32。若视觉与细节各为1，总分为0.990625，但必需数值要求没有全部完成。这是公式层面的反例，不是对实际模型场景的评分，更不是指控视觉评审会忽略通路。我们保留官方分数，同时报告固定要求的合取结果、未知项和成本。</p>
<h2>算法与理论：保留候选，收紧可证主张</h2><p>候选机制仍是从相同当前证据X生成决定D，再生成动作A，并接受真实工具反馈。强学习对照使用相同来源、相同目标token权重，分别学习D(X)和A(X)，部署直接执行A(X)；普通搜索具有相同操作、观察和累计预算。推理时必须使用自生成D，不能给教师答案。</p>
<p>给定训练后参数，若D只来自X及独立随机性，它不增加X以外的目标信息；可能的收益是有限计算下的表示与搜索。这个事实不意味着D无用，也不给出改善保证。损失下降、理由更像教师或打乱理由后性能下降，都不能代替整场收益。</p>
<p>当前24个单元全部在家具阶段，14个缺原始决定；完整普通搜索仍未接通。因此不能把新输入/评分接入称为已具备开训条件。完整公式、假设、控制条件与否定条件见 <a href="algorithm_and_theory.json">算法与理论记录</a>。</p>
<h2>近邻限制</h2><p><a href="https://arxiv.org/html/2605.19587v1">SceneCode</a> 已有显式对象计划与执行修复；我们不能以“有计划、有状态记录”作为贡献。<a href="https://arxiv.org/html/2610.37326v1">ScIntBench</a> 区分交互能力缺失与执行可靠性，不能给生成结果补物理属性后再称原生成成功。本轮均为指定章节阅读，不是全文或代码复现。</p>
<p><a href="https://arxiv.org/html/2611.09017v1">Frontier3DBench</a> 的人工核查范围是参考保真度，不能移作室内功能偏好认证。公开任务也是新接触过的开发材料，不自动成为我们的未见封存测试集。</p>
<h2>可运行交付</h2><p><code>PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/frontier_task_audit.py --task 13 --out /tmp/frontier_public_013.json</code> 导出纯公开输入，拒绝覆盖。<code>--audit</code> 运行本轮算术审计，固定输出已存在会拒绝覆盖。重建需先运行 acquire_frontier_review.py 及其 --supplement。源码、原文按源清单核验；源任务本地保留，发布包不重复分发全部任务正文。</p>
<p><a href="audit_v1/task_coverage.csv">20条完整覆盖表</a> · <a href="audit_v1/official_score_semantics.json">官方函数实际运行结果</a> · <a href="reading_ledger.json">阅读边界</a> · <a href="scoring_path.json">最终评分调用路径</a> · <a href="method_differences.json">方法区别</a> · <a href="summary.json">本轮摘要</a></p>
<p>下一实质门槛仍是原有室内生成器的一次完整真实执行及其共同普通搜索，不是继续增加本页检查数，也不是启动20场机器人任务。论文贡献尚未成立，目标保持进行中。</p></html>'''
    (OUT/'stage_report.html').write_text(report)
    print(json.dumps({'report':str(OUT/'stage_report.html'),'disk_available_bytes':shutil.disk_usage(OUT).free}))

if __name__=='__main__':main()
