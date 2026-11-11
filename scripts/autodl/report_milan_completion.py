"""Report actual full-inventory ordinary completion without a method claim."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
    result=read(r/'completion_result.json');ev=read(r/'original_evaluator_result.json');exact=read(r/'exact_resource_checks.json')
    qualified=[x for x in ev['workspace_pairs'] if x['conditional_dedicated_work_geometry']]
    assert result['status']=='ordinary_conditional_workspace_completed' and len(qualified)==1
    w=qualified[0]
    costs=dict(resource_wrong_runtime_seconds=read(r/'asset_recovery_v1_wrong_runtime.json')['wall_seconds'],
        resource_matched_runtime_seconds=read(r/'asset_recovery_v2_fixture_roundoff.json')['wall_seconds'],
        resource_fixture_roundoff_resume_seconds=read(r/'asset_recovery.json')['wall_seconds'],
        exact_xml_native_restore_seconds=exact['wall_seconds'],
        main_completion_seconds=result['wall_seconds'],
        posthoc_candidate_diagnostic_seconds=read(r/'candidate_requirement_evidence.json')['wall_seconds'],
        cpu_topdown_render_seconds=read(r/'completed_topdown.json')['wall_seconds'],
        cpu_oblique_render_seconds=read(r/'completed_oblique.json')['wall_seconds'],
        main_peak_rss_kib=result['peak_rss_kib'],model_calls=0,paid_API_calls=0,training_updates=0,
        licensed_assets_downloaded=4,licensed_asset_download_bytes=7880372,
        download_wall_seconds=None,export_wall_seconds=None,
        total_research_wall_seconds=None,bundled_instance_billing_not_measured=True)
    dump(r/'cost_ledger.json',costs)
    qualifications=dict(current_split='exposed development, not independent held-out',
        inherited_source_manifest_independent_test_flag=True,
        source_manifest_flag_is_not_current_evaluation_claim=True,
        raw_brief_interpretation_automatic=False,
        raw_text_task_status='Specification-assisted development reference, not a deployable raw-text generation baseline',
        explicit_public_proxy_parameters_consumed=True,
        public_source_role_approaches_given=True,
        historical_model_run_is_fair_method_comparison=False,
        comparison_reason='Archived seven-call policy had different prompts, action permissions, observations and termination. This is a constructive ordinary baseline, not a controlled win over that policy.',
        future_fairness='Give every method the same public task/asset/validation interface and cumulative budget; either share this add-only action space or extend ordinary controls together with the Agent.',
        automatic_role_or_requirement_discovery_tested=False,
        dedicated_workspace_is_sufficient_not_necessary=True,
        shared_meal_work_solutions_not_excluded=True,
        case_learning_or_reasoning_effect_established=False,
        novel_method_established=False,whole_human_use_success=None,
        new_empty_house_generation=False,official_HTTP_calls=0,native_library_object_additions=2)
    dump(r/'comparison_qualification.json',qualifications)
    theory='''完整场景补全的普通参考求解器与下一步理论边界（Milan，2026-11-11）

一、本轮实际完成
已恢复71个原资源文件，全部旧SHA及大小匹配，原生HouseScene/RoomScene30个对象状态逐值一致。
保留13件原始真实HSSD家具及17个建筑对象，实际通过原生对象库增加一张桌和一把椅子，最终32个对象、15件家具。
这不同于只移动现有物体：新增实例来自冻结的同一16类资产库，保留源资产身份、真实三角形、尺寸与所有原始对象。
增加原生对象不是官方HTTP工具调用。没有新模型调用、训练、付费API或从空房的原生生成。

二、普通算法已经足够解决什么
从公共结构化任务与资产类别建立工作面/坐席角色，按真实三角形顶面检查40x30cm材料区。
匹配2个桌子模板与1个椅子模板，现有用餐高度的桌子优先于低茶几；这是一条普通设计启发，不是人类偏好标签。
在0.5m格点、四个朝向上枚举1976个联合候选，保守边界/包围盒筛出557个。
按当前沙发距离等固定规则排序，最多验证12个；第6个通过所有声明条件后停止，无质量重跑。
完整验证包含原14角色接近点的3种方形身体宽度、工作坐席侧/后接近、实际三角形相交、独立材料区及声明视线。
只有最终候选被实例化到原生场景；不能把1976个算成1976个生成场景，也不能把6次假设终态验证算独立房间。

三、真实执行给出的机制证据
进入完整验证的前5个候选都没有检出网格相交，却挡住了沙发/书架或冰箱的声明接近点；最后一个才全部保留。
所以当前材料确实出现了“物体几何不相交，但破坏既有用途接近条件”的区分。
这项区分已由普通用途检查和联合搜索解决，不支持再为同一可计算代理训练一个更慢的Critic。
14个原始接近点均通过、独立工作几何通过、15条声明视线清楚，不等于人体使用、工作舒适、隐私或真实材料占用全部验证。
声明点在保守网格中被阻挡，也不是任何人都无法使用该家具的证明。五个候选都来自同一房间，不能估计失败普遍性。

四、共同方法接口与理论
把角色/资产绑定记为z，坐标记为x，原始任务为u，几何/资产为G。普通后端已能求一个见证：C(u,G,z,x)=true。
本轮z由公共任务字段、资产名与普通规则形成，不是学习到的设计解释。C包含研究者公开定义的可计算代理，不包含全部人类用途真值T。
因此这是使用开发期形式化任务的参考解；原始文字到这些角色/检查参数的映射尚未自动完成，不能称可部署的原始文字生成基线。
C为真不能推出T为真。此处可用于训练的至多是带作用范围的执行/代理标签，不是完整场景偏好或专业设计正例。
求解器在有限预算内未找到见证，也不能直接推出该绑定不可行。若训练成功概率，应条件于搜索器和预算，不能把超时/未找到写成永久不可能。
若未来模型只提出相同z，再调用相同求解器，改善来自什么必须与普通规则、代码感知检索及同信息通用Agent分开测；这不是因为加了D或记忆就自动有新算法。
在没有独立空间选择/用途绑定收益之前，不启动LoRA/GRPO。已有模型的Task/Memory、标注视图和工具协议也必须先满足，不能用自定义JSON失败替代原生能力上限。

五、最近邻与训练目标
PlanIT已有关系图与神经引导实例化；Function2Scene已有用途/人群解析、功能约束和工具修复；FuncRoom-Agent v2已有递归可执行构造、条件SFT和阶段GRPO。
本轮复核FuncRoom最新版本仍为2026-09-17的v2，并读了既有完整补充审阅，不把v1页面或重复阅读包装成新的查新成果。
FuncRoom阶段内共同减父状态分在相同组标准化条件下没有新优势；该事实与势函数边界此前已有推导，不在本轮重新命名。
目前可研究的剩余位置是原始用途如何正确落到资产/角色/约束上，并在完整执行中保持要求；但普通匹配、解析和约束求解均是必须比较的强对照。
这只是剩余研究位置，不是已经证实的创新，也不代表另一篇功能生成论文没有覆盖。缺少真实残余错误时不应为了训练制造错误。

六、公平性和范围
本轮算法可读开发检查器的14个接近点、阈值和资产角色；旧模型运行的输入及权限不同，所以不能把32秒与旧7调用直接作优越性结论。
后续所有条件必须获得同样公开信息、资产、可改动作和验证接口。当前add-only普通方法不能去和可自由改整屋的Agent混成同权比较。
采用独立工作桌椅只是找到的充分解，未否定共享较大桌面等其他组织。没有自动原始自然语言解析、独立人类偏好或跨项目泛化。
该输入是10x7m研究迁移房间，不是建筑师原住宅测绘；旧源manifest的independent_test=true是当时分配，本轮已暴露，按开发材料处理。
本轮成功是“完整声明场景的增量构造及所测几何/功能代理通过”，full_scene_success仍为null，CVPR目标未完成。
'''
    (r/'algorithm_and_theory.txt').write_text(theory)
    reading=[dict(title='FuncRoom-Agent v2',url='https://arxiv.org/html/2608.29519v2',
                  scope='Latest-version confirmation and method reward/SFT interpretation, plus reread existing Nov03 full supplement analysis.',
                  prior_local='/root/autodl-tmp/agent_method_preparation_20261027/work/funcroom_v2_review_20261103/analysis.txt',
                  not_done='No new full page-by-page review, official code run or training reproduction.'),
             dict(title='Function2Scene',url='https://arxiv.org/html/2605.30819v1',
                  scope='Reread existing source-release/tool-contract findings; functional constraints are prior art.',
                  not_done='No new model, professional data or user-study reproduction.'),
             dict(title='Milan source and execution branch',commit='75fc783f86d4fea78714639e110fbe77476ee106',
                  scope='Complete build_milan_input, build_geneva_catalog, measure_milan_task, evaluate_milan_policy, export_catalog_geometry and hit_segment; frozen input, catalog and evaluation parameters.',
                  actual_execution='Exact native restore, ordinary additive scene construction and original numeric evaluator.',
                  not_read_as_solver_input='Old east/west proposed solutions and designer coordinates; no original output used to propose poses.'),
             dict(title='Existing scene-training objective studies',scope='Read saved decision_phase_study, supervision_and_branch_state, functional_preview and ordinal/critic audits.',
                  consequence='Do not repeat already established teacher-prefix, ordinary preview, tie or GRPO-shift results as a new algorithm.')]
    dump(r/'reading_ledger.json',reading)
    differences=[
        ['Ordinary implemented baseline','Public roles + catalog + joint finite search','Actually adds2 native objects and verifies preserved requirements','No learned method claim'],
        ['Archived Milan model trial','Different old prompt/action/termination','Failure motivates inspection only','Not a fair controlled comparison'],
        ['Function2Scene','Persona/activity constraints and typed tool repair','Closest common-method prior','Not locally reproduced'],
        ['FuncRoom-Agent v2','Recursive code SFT and stage ScenePRM/GRPO','Already covers staged learned construction','No new SFT/RL result here'],
        ['Candidate learned binding','Infer useful role/asset/constraint choices from raw brief and spatial evidence','Not yet implemented or shown better than ordinary matching','No novelty or training qualification yet']]
    with (r/'method_difference.csv').open('w',newline='') as f:
        out=csv.writer(f,lineterminator='\n');out.writerow(['condition','mechanism','actual_evidence','limit']);out.writerows(differences)
    with (r/'summary.csv').open('w',newline='') as f:
        out=csv.writer(f,lineterminator='\n');out.writerow(['condition','furniture','original_furniture_retained','original_role_access_at_each_width','dedicated_work_geometry','furniture_intersections','structure_intersections','model_calls','search_wall_seconds','full_scene_success'])
        out.writerow(['frozen_initial',13,13,'14/14;14/14;14/14','not_present_no_claim_about_shared_work',len(read(r/'initial_contacts.json')['furniture_pairs']),len(read(r/'initial_contacts.json')['structure_pairs']),0,'','unknown'])
        out.writerow(['ordinary_additive_completion',15,13,'14/14;14/14;14/14',True,0,0,0,result['wall_seconds'],'unknown'])
    repo=Path.cwd().resolve();commands=f'''Use the existing native interpreter and the already installed geometry dependency:
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH={repo}/runs/autodl/20261110_corpus_state_recovery/dependency/import:/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps
cd {repo}

All commands below were actually executed with the current result root, with the same native Python:
/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python -B scripts/autodl/prepare_milan_completion.py --root runs/autodl/20261111_milan_completion --scratch /dev/shm/milan_completion_prepare_check_20261111
/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python -B scripts/autodl/recover_milan_assets.py --root runs/autodl/20261111_milan_completion --scratch /dev/shm/milan_completion_20261111
/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python -B scripts/autodl/restore_milan_bundle.py --root runs/autodl/20261111_milan_completion --scratch /dev/shm/milan_completion_20261111 --native-source runs/autodl/20261110_native_state_runtime/source
/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python -B scripts/autodl/complete_milan_workspace.py --root runs/autodl/20261111_milan_completion --scratch /dev/shm/milan_completion_20261111 --native-source runs/autodl/20261110_native_state_runtime/source

The current search result has completion_protocol.json and refuses an overwrite. For a fresh reproduction, create a new result root, copy source/ and source_manifest.json, and use one fresh scratch consistently from preparation onward. The preparation check above only verifies the no-network input path and is not another scene search.
The4 licensed fixture GLBs are persisted at /root/autodl-tmp/agent_method_preparation_20261027/work/milan_completion_artifacts_20261111/raw_assets/. Existing12 original HSSD assets remain in official_assets; do not redistribute gated assets.
Only the already installed trimesh4.7.4 matches these old asset hashes. Wrong4.5.3 attempt and fixture sin(pi) normalization failures are retained; expected file hashes/geometry thresholds were never changed.
The complete final GLB is gzip-compressed at the absolute path in persisted_artifact_manifest.json. Native states, catalog and resource hashes remain authoritative. The full original native bundle can be restored by these commands without model calls or new downloads.
'''
    (r/'commands_and_reproduction.txt').write_text(commands)
    code=[p for p in Path('scripts/autodl').glob('*milan*.py')]
    dump(r/'code_manifest.json',[dict(path=str(p),sha256=sha(p)) for p in code])
    free=shutil.disk_usage(r).free
    report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Milan 完整场景普通补全</title>
<style>body{{font:17px/1.65 sans-serif;color:#182432;max-width:1150px;margin:30px auto;padding:20px}}table{{border-collapse:collapse}}td,th{{border:1px solid #aaa;padding:8px}}.note{{background:#fff1ce;padding:18px}}img{{width:49%;vertical-align:top}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:20px}}</style>
<h1>完整声明场景的普通补全参考已实际完成</h1>
<p>在Milan研究迁移输入中，原13件真实HSSD家具及17个建筑对象保持逐值不变，新增一组桌椅承担独立工作区。最终15件真实网格家具、32个原生对象。没有模型、训练或付费API调用。</p>
<table><tr><th>条件</th><th>家具实例</th><th>原实例保留</th><th>原14角色接近点（各宽度）</th><th>独立工作几何</th><th>网格相交</th></tr>
<tr><td>原始完整输入</td><td>13</td><td>13/13</td><td>14/14，14/14，14/14</td><td>未提供；共享办公可能性未判否</td><td>0</td></tr>
<tr><td>普通联合搜索</td><td>15</td><td>13/13且原姿态精确未变</td><td>14/14，14/14，14/14</td><td>已验证代理见证</td><td>0家具对、0结构对</td></tr></table>
<p>实际生成并便宜筛查1976个候选，557个通过保守包围盒筛查，完整验证6个后采纳173号。主循环含原数值评价器耗时{result['wall_seconds']:.3f}秒，峰值RSS约{result['peak_rss_kib']/1024**2:.3f}GiB。原71个资源哈希全部恢复匹配；完整GLB约23.9MB，61个几何节点回读通过。</p>
<p class="note">本求解器使用开发期形式化任务与检查参数，原始文字到这些参数的映射尚未自动完成。这是现有整屋的增量构造及所测条件通过，不是可部署的新描述完整生成，也不是Agent效果或CVPR创新证明。人体使用、人体工学、真实材料占用、隐私、光声及安装未验证，full_scene_success保持unknown。两视图的蓝色为新增工作桌椅、橙色为原有用餐组，仅作诊断高亮；结构在预览中隐藏，原生资源和GLB均保留。</p>
<img src="completed_topdown.png" alt="完整屋内补全俯视"><img src="completed_oblique.png" alt="同场景斜视">
<h2>比成功数更有解释力的观察</h2><p>前5个被完整检查的候选都未检出相交，却挡住了沙发/书架或冰箱的声明接近点。普通功能检查已能拒绝它们，第6个保留全部原接近点；这不是新Critic的独立贡献。六候选同属一个房间，不能当六个独立场景。</p>
<h2>为什么不是历史Agent的公平胜利</h2><p>本轮使用了公开开发检查参数和结构化角色；旧七调用模型的输入、工具权限及停止条件不同。详见<a href="comparison_qualification.json">资格说明</a>。今后的所有方法必须共享同样信息和动作范围。普通搜索只增加两个对象，不能拿它直接对比可自由移动/删除的Agent。</p>
<h2>训练与最近邻</h2><p>当前给出的是普通CSP完整见证。<a href="https://arxiv.org/html/2605.30819v1">Function2Scene</a>已有功能解析及工具修复，<a href="https://arxiv.org/html/2608.29519v2">FuncRoom-Agent v2</a>已有递归构造、SFT与阶段GRPO。这里尚未实现或证明一个超出普通匹配、检索和搜索的学习机制，不因成功补全就启动新训练。</p>
<h2>实际交付</h2><ul><li><a href="summary.csv">原始/补全指标</a>、<a href="completion_result.json">实际运行</a>、<a href="candidate_requirement_evidence.json">逐候选用途失败</a></li>
<li><a href="algorithm_and_theory.txt">算法、理论和剩余问题</a>、<a href="method_difference.csv">方法差异</a>、<a href="reading_ledger.json">阅读范围</a></li>
<li><a href="exact_resource_checks.json">71个原资源检查</a>、<a href="original_evaluation_source_audit.json">原检查器5个文件原哈希</a>、<a href="protected_originals_after.json">579个保护输入</a></li>
<li><a href="persisted_artifact_manifest.json">实际GLB与本地资产绝对路径/哈希</a>、<a href="commands_and_reproduction.txt">可运行命令</a>、<a href="cost_ledger.json">分别记录的耗时</a>、<a href="evidence.zip">证据包</a></li></ul>
<p>清理仅删除了3份旧重复导出/影子副本，保留相同内容原件、逐步记录及恢复路径，实际释放约51.1MB。没有删除模型权重或独有研究证据。当前数据盘空闲约{free/1024**3:.3f}GiB，保留5GiB余量。</p>
<pre>{html.escape(theory)}</pre></html>'''
    (r/'stage_report.html').write_text(report)
    with zipfile.ZipFile(r/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(r.rglob('*')):
            if p.is_file() and p.name not in ['evidence.zip','delivery_manifest.json']:
                z.write(p,p.relative_to(r))
        for p in code:z.write(p,'code/'+p.name)
    dump(r/'delivery_manifest.json',dict(files=[dict(path=str(r/n),sha256=sha(r/n),bytes=(r/n).stat().st_size) for n in ['stage_report.html','summary.csv','evidence.zip']],
        licensed_mesh_assets_not_in_Git_or_zip=True,mesh_paths='persisted_artifact_manifest.json',
        goal_complete=False,free_data_disk_bytes=shutil.disk_usage(r).free))
    print(json.dumps(dict(report=str(r/'stage_report.html'),evidence_bytes=(r/'evidence.zip').stat().st_size,main_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':main()
