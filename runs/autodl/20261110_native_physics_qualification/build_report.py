"""Assemble source, model and exported-asset evidence; no service is rerun."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    process=read(ROOT/'actual_v1/process_receipt.json')
    model=read(next((ROOT/'actual_v1/model_call').glob('*/receipt.json')))
    first=read(ROOT/'asset_v1/result.json')
    second=read(ROOT/'asset_v2/result.json')
    consumed=read(ROOT/'drake_probe.json')
    physics=read(ROOT/'asset_v2/parsed_physics.json')
    assert process['returncode']==0 and not process['timeout']
    assert model['status']=='completed' and model['ended_with_eos']
    assert first['canonicalization_seconds']>0 and first['blender_terminal']
    assert second['collision_pieces']==128 and second['collision_server_terminal']
    assert consumed['status']=='existing_sdf_parsed_by_Drake'
    assert consumed['models']==1 and consumed['collision_geometries']==128
    assert consumed['inertia_physically_valid'] and consumed['old_asset_files_unchanged']
    old=read(REPO/'runs/autodl/20261110_native_asset_conversion/actual_v2/input_hashes.json')
    assert all(sha(p)==h for p,h in old.items())
    dump(ROOT/'input_hashes_rechecked.json',old)
    coll_events=[]
    for stage in ('asset_v1','asset_v2'):
        for line in (ROOT/stage/'collision/http_events.jsonl').read_text().splitlines():
            coll_events.append(dict(stage=stage,**json.loads(line)))
    assert sum(r['path']=='/generate_collision' for r in coll_events)==1
    dump(ROOT/'component_execution.json',{
        'model_subprocess_attempts':1,'completed_model_outputs':1,
        'canonicalization_RPCs':1,'new_canonical_mesh_collision_POSTs':1,
        'native_SDF_writes':1,'registered_assets':1,
        'SDF_parse_attempts':2,'SDF_parse_attempt_scope':'Initial parse succeeded before wrapper metadata error; one read-only verification parsed the same saved SDF. No asset-generation retry.',
        'native_registry_roundtrip_assertion':'Passed before the later num_visual_geometries metadata call in asset_v2.',
        'original_raw_frame_collision_request_in_previous_delivery_not_counted_here':True,
        'wrapper_failures_preserved':[first['error'],second['error']],
        'new_model_retries':0,'training_updates':0,'paid_api_calls':0,
        'whole_scene_generations':0})
    rows=[
        ['local_native_physics_VLM','returned_JSON',process['seconds'],'1585 input /98 output tokens; model estimate, not measured mass'],
        ['native_response_parser','returned','','+Z up /-Y front, wood10kg, range8-12kg'],
        ['native_canonicalization_RPC','exported',first['canonicalization_seconds'],'real same-asset geometry; author integration patch'],
        ['canonical_mesh_native_collision','returned128',second['collision_seconds'],'unchanged configured V-HACD; one request on transformed geometry'],
        ['native_SDF_writer','exported',second['sdf_seconds'],'native friction0.4 and inertia processing'],
        ['native_registry','registered1','','actual helper and class; no full AssetManager constructor'],
        ['Drake_saved_SDF_consumer','parsed',consumed['seconds'],'1model/128collision/1visual; no dynamics'],
        ['full_indoor_generation','not_executed','','all-stage gate remains unqualified'],
    ]
    with (ROOT/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['component','status','seconds','scope']);w.writerows(rows)
    summary={
        'status':'one_real_estimated_asset_registered_and_SDF_consumed',
        'previous_goal_turn':'progress: actual native conversion/four views and processor-only request published9aab586',
        'this_goal_turn':'progress: one bounded local VLM qualification, actual native parser/canonical/convex/SDF/registry and Drake consumption; targeted objective/nearest-work/theory review',
        'new_model_subprocess_attempts':1,'completed_model_outputs':1,
        'model_input_tokens':model['input_tokens'],'model_output_tokens':model['output_tokens'],
        'model_subprocess_seconds':process['seconds'],'engine_seconds_including_load':model['elapsed_seconds'],
        'generation_peak_cuda_allocated_bytes':model['peak_cuda_allocated_bytes'],
        'cuda_peak_scope':'Engine resets peak counters after model load; not full cold-load peak.',
        'downstream_first_wrapper_seconds':first['seconds'],
        'downstream_continuation_seconds':second['seconds'],
        'read_only_Drake_probe_seconds':consumed['seconds'],
        'recorded_execution_body_seconds_sum':process['seconds']+first['seconds']+second['seconds']+consumed['seconds'],
        'cost_scope':'Includes both failed downstream wrapper bodies; excludes code/source/report time. Not an account bill or full generation latency.',
        'estimated_physics':physics,'mass_measured':False,
        'collision_pieces':128,'registered_assets':1,'parsed_SDF_models':1,
        'SDF_inertia_physically_valid':True,'dynamic_simulation_executed':False,
        'wrapper_statuses':[first['status'],second['status']],
        'correction':'Do not rewrite failed wrapper records as successful. Final artifact consumption verified separately without regenerating it.',
        'old_source_asset_files_unchanged':len(old),
        'new_paid_api_calls':0,'new_training_updates':0,'new_complete_scene_generations':0,
        'full_runtime_ready':False,'whole_scene_success':None,'learned_benefit_established':False,
        'free_data_directory_bytes':shutil.disk_usage(ROOT).free,
        'source_prefix_review':'canonical_prefix_review.json; actual -Y asset agrees with documented prefix; +/-X matrix cases do not. No all-front qualification or patch.'}
    dump(ROOT/'summary.json',summary)
    dump(ROOT/'reading_ledger.json',{
        'scope':'Targeted primary-source rechecks and native asset code; no exhaustive newest-work guarantee or full-paper reread.',
        'papers':[
            {'title':'Scene-Designer','url':'https://arxiv.org/html/2610.23121v1','read':'Sections3.2-4.1 on actual branch execution, critic/local/global rewards and training/evaluation; no all-appendix/figure audit this turn.','boundary':'Reasoning-before-action and critic-guided alternatives already exist.'},
            {'title':'ThinkBLOX','url':'https://arxiv.org/html/2607.13539v1','read':'Targeted repeat III-A.1-2, III-B, IV rationale construction/refinement. Earlier source ledgers already identified it, so not newly discovered.','boundary':'State-conditioned reasoning/action and consistency reward are direct overlaps; not run or independently reproduced.'},
            {'title':'LaTRO','url':'https://arxiv.org/html/2411.04282v2','read':'Sections3-4.2, equations1-4 and frequency/majority distinction rechecked; no new training or whole-paper audit.','boundary':'Unannotated-rationale objectives have prior art; expected log likelihood is distinct from log marginal likelihood.'}],
        'source_read':[
            {'path':'scenesmith/agent_utils/mesh_physics_analyzer.py','scope':'Exact dataclass, direction helpers and final parsing try-block executed from original AST; full original function/SDK service not called as one unit.'},
            {'path':'scenesmith/agent_utils/mesh_canonicalization.py','scope':'Complete wrapper read/imported/executed via real Blender RPC; existing author patched canonicalization implementation used unchanged.'},
            {'path':'scenesmith/agent_utils/blender/canonicalization.py','scope':'Full function125-273 re-read; source matrix prefix125-181 executed for four basis fronts. No fake VLM/asset labels or full transform claim for counterexample.'},
            {'path':'scenesmith/agent_utils/materials.py and data/materials.yaml','scope':'Complete library read and loaded; woodfriction0.4 used, no missing-file fallback.'},
            {'path':'scenesmith/utils/inertia_utils.py','scope':'Complete validity/projection and XML processing read/imported; native SDF writer invokes actual helper.'},
            {'path':'scenesmith/agent_utils/sdf_generator.py','scope':'Full generate_drake_sdf body read/executed normally; rescale helpers not executed.'},
            {'path':'scenesmith/agent_utils/asset_registry.py','scope':'Native register/save/load functions read and executed normally.'},
            {'path':'AssetManager._create_scene_object/_generate_collision_geometry','scope':'Unchanged AST helper bodies executed with real registry/client; full constructor/import lifecycle remains unqualified.'}],
        'new_original_training_labels':0,'new_method_gain':False})
    dump(ROOT/'full_runtime_admission.json',{
        'full_stage_assets':False,'tool_side_no_paid_api':False,'common_evaluation':False,'common_action_scope':False,
        'one_retrieved_asset_physics_export_path_executed':True,
        'unknown':['all required asset categories/floor materials','all-stage constructors and local provider coverage',
                   'all front-axis root transformations','dynamic stability/true physical properties','independent whole-room results'],
        'meaning':'This observed call had zero paid API, but that is not a proof about every reachable native service branch.'})
    dump(ROOT/'handoff.json',{
        'new_real_asset':'asset_v2/asset/native_nightstand.sdf (local-only geometry); registry.json records original identity and paths.',
        'model_budget_closed':'Exactly1local call completed, no inference retry. No training or next model batch is launched.',
        'common_backend_next':'Compose a real retrieval-only shared AssetManager lifecycle using this now-executed path; retain image/texture guard until actual unavailable branches and task asset requirements are handled. Address the documented front-axis contract on common backend only, without attributing it to policy reasoning.',
        'research_next':'Use algorithm_and_theory.json to distinguish ordinary same-supervision conditioning from an independent3D mechanism. One asset does not qualify a larger training batch or establish CVPR novelty.',
        'resources':'All launched model/Blender/collision processes terminal. Continue remaining source/config work without keeping the billed bundle open for reports.',
        'goal_complete':False})
    local=[]
    for dirname in ('asset_v1/asset','asset_v2/asset'):
        for p in sorted((ROOT/dirname).rglob('*')):
            if p.is_file():local.append(p)
    for p in ROOT.glob('asset_*/collision/raw_response.json'):local.append(p)
    for p in ROOT.glob('actual_v1/model_call/*/request.json'):local.append(p)
    dump(ROOT/'local_only_resources.json',{'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in local],
        'scope':'Derived licensed mesh/texture data and inline-image requests remain local; metadata/output estimates and sources can be published.'})
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实物理分析到SDF与算法边界</title>
<style>body{max-width:1080px;margin:38px auto;font:17px/1.7 system-ui;color:#213044}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd5df;padding:8px}pre{white-space:pre-wrap;background:#f1f4f7;padding:16px;font-size:14px}</style>
<h1>一个真实资产已接到SDF和Drake；完整场景与新方法仍待验证</h1>
<p>本轮固定上一轮的同一HSSD床头柜及四图消息，实际完成一次本地8B物理分析，继续使用原生解析、坐标标准化、V-HACD、SDF和注册代码。没有换资产、人工填质量、修改阈值或重复推理。该资产现在有可读入Drake的SDF，并非只写了准备报告。</p>
<table><tr><th>组件</th><th>实际状态</th><th>秒</th><th>范围</th></tr>'''+table+'''</table>
<h2>实测结果及失败记录</h2>
<p>本地模型进程42.291秒，1585输入token、98输出token，正常EOS。既有适配器在上一轮1566token原生消息后增加了JSON提醒，故不能沿用旧输入计数。模型估计木材、10kg、范围8–12kg和第3视图为正面；原生解析得到+Z朝上、-Y正面。质量范围不是校准置信区间，也未进行称重。请求模型标签仍是原配置gpt-5.2，实际后端是已下载Scene-Builder8B；未保留原模型推理/verbosity/detail语义，不能称原论文配置复现。</p>
<p>标准化RPC0.369秒；变换后网格重新进行一次原配置分解，4.465秒返回128片；原生SDF写入0.273秒，使用实际源材料表木材摩擦0.4和惯量处理。新鲜进程读取已保存SDF，得到1模型、128碰撞几何、1视觉几何，惯量有效。未执行动态仿真，不证明物体稳定、接触准确或真实质量正确。</p>
<p>包装脚本有两处错误，均保留：首次在标准化完成后，新建AST节点缺少位置字段，尚未请求分解；补位置字段后从已保存标准化网格继续，没有再运行模型或Blender。随后SDF和注册均已完成、Drake已解析，但收集可视几何数量时调用了不存在的num_visual_geometries。修正后的只读探针使用GetVisualGeometriesForBody，未重写SDF或重新生成资产。不能把两份失败结果改写成原脚本顺利退出。</p>
<p>模型和两个下游包装进程、只读读取探针的已记录执行体合计约60.626秒，包含失败包装时间；不含源码读取、实现与报告时间。生成期记录显存分配峰值约16.75GiB，计数在权重载入后重置，不能当全进程峰值。所有启动进程已结束，显存已释放。</p>
<h2>源码发现：一个资产通过不能外推全部朝向</h2>
<p>实际标准化代码最终输出Y-up GLTF，再由SDF/Drake处理到Z-up；早先转换模块的流程注释与当前作者补丁实现存在差别，应以实际调用和代码为准。没有因此偷偷更改既有结果。</p>
<p>按标准化函数文档“把输入正面轴变到+Y”的契约，只执行原始矩阵前缀，±X两种输入都得到-Y，±Y得到+Y。本资产恰为-Y，因此此反例不推翻当前资产的消费结果；完整根节点层级效应尚未审计，也没有修复或生成新资产来追求通过。它说明通用后端仍需检查，不能把错误朝向全归因于策略缺乏推理。详见<a href="canonical_prefix_review.json">原始前缀证据</a>。</p>
<h2>当前算法对照的准确版本</h2>
<p>X包含原始要求、真实当前场景/视图和已有执行证据；D是未验证的公开决策记录，A是实际工具动作。D不是工具执行事实。相同监督的两种使用方式为：</p>
<pre>L_aux  = Σ_i l_A(X_i,A_i;原w_Ai) + Σ_i∈LD l_D(X_i,D_i;原w_Di)
L_cond = Σ_i∈LD l_A(X_i,D_i,A_i;原w_Ai)
       + Σ_i∉LD l_A(X_i,A_i;原w_Ai) + Σ_i∈LD l_D(X_i,D_i;原w_Di)
部署aux：A~p(A|X)
部署cond：D_hat~p(D|X)，A~p(A|X,D_hat)，然后执行并重新观察。</pre>
<p>保持原token权重、来源分组和目标，不能删除14条缺D动作样本或编造解释。当前24条开发单元仅19条完成观察编码；局部损失函数通过不代表完整Trainer累积尺度已验证。两组前缀长度不同，不能称“同token目标即同计算量”。普通直接动作/搜索应可以用节省的推理时间提出更多候选，所有策略共同承担本轮真实测到的工具VLM、分解、渲染及失败开销。</p>
<h2>理论能说明什么</h2>
<p>固定参数θ，如果D只来自X和独立随机数，则I(Y;D|X,θ)=0；这不排除有限计算下的策略改善，也不能保证改善。训练时使用参考D得到高动作似然，不代表部署时自产D有益。固定下游成功函数v_X(d)∈[0,1]时，参考分布q与实际分布p的成功差至多TV(q,p)；但单个参考记录不能估计q或TV，所以这里没有数值保证。</p>
<p>三维有效动作是集合V(X)，成功概率应统计Σ_a∈V(X)p(a|X)，不是唯一参考工具字符串的概率。对潜在D边缘化的标准混合模型，log似然梯度同时包含决策分布和条件动作分布的导数；丢掉一项便改变目标。期望log似然、log边缘似然和有限样本重要性估计也不是同一目标。经验频率收敛于边缘分布，并不意味着多数票胜者服从该分布。完整推导假设与限制已写入<a href="algorithm_and_theory.json">算法与理论记录</a>，均不宣称新定理。</p>
<h2>最近邻约束与研究判断</h2>
<p><a href="https://arxiv.org/html/2610.23121v1#S3.SS2">Scene-Designer</a>已有思考—动作训练、执行分支和评价改进；<a href="https://arxiv.org/html/2607.13539v1#S3.SS1">ThinkBLOX</a>已有渐进推理及推理动作一致性；<a href="https://arxiv.org/html/2411.04282v2#S4">LaTRO</a>已有无解释标注时的潜在推理优化。因此当前D→A配对应作为需要识别效果的对照，不是已确认主创新。本文仅定向复读相应方法，不宣称这些论文与全部补充材料重新全文精读。</p>
<p>下一步仍须把共同完整生成和普通搜索接上，再在独立任务中证明额外收益。不能让候选方法拥有额外资产或上游操作，也不能通过删除表面物体、跳过不可执行要求、固定几何答案或把人造解释当专业偏好证明获胜。一个资产成功不支持扩大训练；它只减少了一项真实后端缺口。目标仍为完整可编辑室内场景，当前没有新整场、学习收益或CVPR贡献成立的证据。</p>
<h2>证据与未完成事项</h2><p><a href="summary.json">汇总</a> · <a href="summary.csv">逐组件结果</a> · <a href="component_execution.json">实际次数与错误</a> · <a href="drake_probe.json">SDF只读消费</a> · <a href="reading_ledger.json">阅读范围</a> · <a href="full_runtime_admission.json">仍未通过的全流程准入</a> · <a href="evidence.zip">证据包</a></p>
</html>'''
    (ROOT/'stage_report.html').write_text(text)
    excluded={str(p) for p in local}
    skip={'artifact_manifest.json','evidence.zip','delivery_check.json'}
    paths=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and str(p) not in excluded and p.name not in skip and '__pycache__' not in p.parts]
    paths.extend([REPO/'scripts/autodl/qualify_native_physics_once.py',REPO/'scripts/autodl/finish_native_physics_asset.py'])
    manifest={'files':[{'path':str(p.relative_to(REPO)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths],
              'local_payloads':'local_only_resources.json'}
    dump(ROOT/'artifact_manifest.json',manifest)
    with zipfile.ZipFile(ROOT/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths+[ROOT/'artifact_manifest.json']:z.write(p,str(p.relative_to(REPO)))
    print(json.dumps({'status':summary['status'],'metadata_files':len(paths),'archive_bytes':(ROOT/'evidence.zip').stat().st_size}))


if __name__=='__main__':main()
