"""Publish actual native-controller admission and its research boundaries."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();s=a.stage.resolve();repo=Path(__file__).resolve().parents[2]
    cpu=json.loads((s/'actual_v6/result.json').read_text())
    run=json.loads((s/'native_joint_actual_v2/result.json').read_text())
    ledger=run['model_ledger'];row=ledger['forwards'][0]
    assert cpu['status']=='whole_native_module_and_floor_tools_constructed'
    assert run['layout_changed'] and run['room_count_before']==0 and run['room_count_after']==1
    assert len(ledger['model_loads'])==len(ledger['forwards'])==1
    assert row['output_tokens']==141 and row['status']=='returned'
    assert all(c['exit_code'] is not None for c in run['owned_processes'])
    original=json.loads((s/'materialization.json').read_text())
    unchanged=all(sha(r['source'])==r['sha256'] for r in original['copied_sources'])
    assert unchanged
    proposal=json.loads((s/'native_joint_actual_v2/native_policy_proposal.json').read_text())
    geometry=json.loads((s/'actual_floor_geometry_v1/result.json').read_text())
    cold=json.loads((s/'packaged_geometry_cold_read.json').read_text())
    helpers=json.loads((s/'next_stage_helper_probe.json').read_text())
    assert geometry['drake_plant_finalized'] and cold['status']=='archive_restored_and_geometry_cold_parsed'
    assert proposal['tool_calls'][0]['name']=='generate_room_specs'
    summary=dict(status='original_joint_policy_to_real_native_tool_admitted',
        original_source_files_unchanged=len(original['copied_sources']),
        small_files_fetched=len(original['fetched']),small_download_bytes=sum(r['bytes'] for r in original['fetched']),
        CPU_native_tool_count=len(cpu['native_floor_tool_names']),
        CPU_material_lookup_successes=len(cpu['native_material_lookups']),
        CPU_construction_seconds=cpu['seconds'],actual_new_model_forwards=1,
        native_floor_geometry_handover=True,actual_architecture_wall_objects=4,
        zero_model_geometry_seconds=geometry['seconds'],archive_cold_parse_seconds=cold['seconds'],
        original_stage_helpers_whole_module_imported=True,actual_room_summary_produced=True,
        model_loads=1,input_tokens=row['input_tokens'],output_tokens=row['output_tokens'],
        forward_body_seconds=row['seconds'],parent_seconds=run['seconds'],
        peak_cuda_allocated_GiB=row['peak_cuda_allocated_bytes']/2**30,
        complete_scenes_generated=0,complete_scene_success=None,training_steps=0,paid_API_calls=0,
        whole_runtime_ready=False,learned_benefit_verified=False,
        original_phase_study_reopened=False,
        no_hidden_retry=True,all_owned_processes_terminal=True)
    dump(s/'summary.json',summary)
    with (s/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['measure','value','scope'])
        for k,v in summary.items():w.writerow([k,v,'Entry/tool qualification, not full generation quality'])
    methods=[
        ['A','Original action targets only; no added rationale labels','A(X)','No current training result'],
        ['AUX','Exactly the same recorded D and A mass; D auxiliary, A not conditioned on D','A(X)','Ordinary material-matched learning control'],
        ['J','Same recorded D/A in original autoregressive serialization','D then A in one native joint completion','Original native contract; this turn one frozen default-task step only'],
        ['P','Same recorded D/A, explicit two-stage conditioning','Own D then A(X,D)','Prospective candidate; phase instructions/framing differ from J'],
        ['SEARCH','Same tools, inputs, assets, edit permissions and measured total budget','Ordinary candidate search','Required computational control; not run in this entry check']]
    with (s/'method_differences.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['condition','supervision','deployment','evidence_boundary']);w.writerows(methods)
    theory='''原生整场入口接通后的算法与理论边界（2026-11-10）

本轮实做：完整导入原生控制器模块，正常构造 StatefulFloorPlanAgent 和19个SDK工具；
真实启动并回收Blender，查询作者原仓库两套本地材质。独立的新额度只允许1次前向，
原生默认卧室任务从空HouseLayout开始。冻结8B模型自行选择4.8×4.0米与
generate_room_specs，原SDK成功创建一个房间。原生观察、全部原始输出、token、
工具结果和前后布局均保留。没有手写工具答案，没有注入目标尺寸，没有训练。

随后零模型调用地重放实际模型输出的布局，调用原生地板finalize_stage和
derive_room_scene，生成真实地板、4面墙及SDF，Drake解析并完成plant构造。
再次从已归档字节在RAM恢复几何，配上原有3张木地板贴图，冷进程Drake解析成功。
墙面纹理已在缓冲中，地板仍引用伴随的官方材质文件；不能称归档单独完全自包含。
41.48MB新增输出只有4.22MB独有内容。已先在RAM构造并逐条验证归档，再移除
本轮新增重复缓存/buffer，落盘4.15MB无损包；没有删旧结果或使用文件系统硬链接。
需要使用原生几何时先还原buffer；JSON中的原始绝对字符串未改，未宣称任意目录
的完整RoomScene JSON搬迁已经通过。归档冷读直接检验恢复后的真实SDF。

这证明控制器/政策/工具及地板几何移交接通，不是床、床头柜与衣柜已经生成。未执行
整场终结、家具、壁挂、吊顶、小物体和完整语义验收；也没有做新的场景渲染。
Blender健康检查不能当成真实渲染通过。默认任务来自原集成常量，不是锁定测试样本。

1. 算法比较必须保留原生joint控制

令X包含原始要求、当前场景、多视图与真实工具收据。D是政策提出的决策，A是
工具动作。已有原生joint输出本来就是先生成think文本再生成工具调用。此次实际
141token中同时包含合法闭合的think和工具块，原生解析/执行成功。因此“先想再做”
不构成我们独有的机制。此前已经准备的同源joint控制不能从后续实验中消失。

四个学习条件：A只监督原动作；AUX使用相同D/A标签但动作不读D；J采用普通单次
自回归D+A；P显式分两阶段，部署时用自己生成的D条件化A。当前24原始单元只有
10项有D，14项缺D；缺失不能用模型补写理由当真值。保留原始token、家族和组件
系数；不能因为切成两条序列就各自重新归一化。可见图像覆盖与缺失仍以既有审计为准，
这次新空房任务没有扩大训练集，也没有解除缺失图像的限制。

2. 什么条件下只是同一个目标的两种执行

固定同一前缀X、相同D/A token、相同位置/掩码、相同参数、无随机扰动且精确
延续同一KV时，普通teacher-forced联合似然逐token分解为
 -log p(D,A|X) = -log p(D|X) -log p(A|X,D)。
同一套原token系数下逐token加和同样成立。这是链式法则，不是新目标或新理论。
当前显式阶段代码改变system后缀，动作前还使用指定framing，不能直接援用上式
宣布J与P的实际神经loss/梯度相同。之前实测D/A输入公共前缀仅143token，完整
KV复用未实现；本轮也没有实现。它们可以比较，但必须承认协议变化这一混杂。

对任意给定P的动作边缘分布 q(A|X)=sum_D p(D|X)p(A|X,D)，无限制的直接动作
模型原则上可表示q。故不能声称显式D带来新的可表示动作分布；有限容量、训练、
解码和计算分配才是可能收益来源。这里的表述不假设实际小模型达到了Bayes最优。
若D只由X与固定模型随机性生成，它没有额外观察到隐藏世界状态；若D选择了工具
查询，则新工具观察可能带来信息，但查询权限和费用必须给普通对照同等机会。
这些是普通概率边界，不是新三维定理，更不证明当前候选必然无效。

3. 真实本轮结果不能被用于什么结论

前次严格D/A输入是另一份带五视图的家具阶段观察；这次是空布局地板任务。
一个joint输出合法与前次阶段格式失败之间不是配对效果比较，不能据此宣布哪组
更好，也不能宣布改回原提示修好了所有格式问题。既有失败保留且额度关闭。
新v1在错误使用HouseLayout.rooms时、模型加载/前向之前退出；修正为真实
room_specs/placed_rooms字段，保存v1源码与零前向账本，再冻结v2独立1次额度。
v2只有1次前向，无输出修补、采样重试或额外隐式补调用。

4. 普通工具修正归共同基线

原集成引用的scenesmith.llm_provider不存在于两份固定官方源码树。新副本显式移除
这三处外部网关导入与启动，由受控本地RPC接管。没有造一个空模块冒充原生网关。
新副本只替换模型加载/远程政策和外部网关入口；所测ToolEnvironment、RolloutSession、
driver_loop、main_async、原生解析/提示/渲染帮助函数AST与原集成相同。
首次3.12 ast.unparse造成3.11 fstring语法错误、路径归一化错误及Popen注解包装错误
均保留，不把开发失败描述为模型或几何质量失败。

本地材质模式只提供作者仓库Plaster001/Wood094两套默认贴图，以关键词分支选择；
并未做完整语义CLIP材质检索。保留原提示中的CLIP文字也不代表实际后端就是CLIP。
正式完整运行须向所有条件明确声明实际材质能力，不能把任意材质满足算作已验证。
HSSD图像服务抑制、公共资产池过滤、RPC和这些材质修正均是工程/基线，不是创新。

5. 什么证据才支持研究投入

固定同一真实生成起点、同一原始要求、资产与操作范围，比较A/AUX/J/P和普通搜索。
先区分格式/入口成功、实际执行成功和完整任务成功。完整任务必须核查必需实例及
类别、原始关系、保护项、碰撞/越界/支撑，未知项不能算通过；默认卧室一个房间
被创建，尚未满足“床＋两个床头柜＋角落衣柜”。
记录prefill、输出、工具VLM、CLIP、渲染、模拟、重执行与模型加载，报告质量随
总预算的变化。当前52秒包括新控制器、模型冷加载、前向和清理，前向体约8.58秒，
不能拿它与前次其他输入的时间作效率比较。政策微调后工具基座的隔离尚未实现，
不能默认更新后的同一权重仍是公共冻结工具。没有授权的训练不会自动启动。

下一步是把已验证入口接到真正整场终结、阶段间移交和共同HSSD资产服务；还缺
受控共享服务启动、真实家具持久化和小物体完成。缺失room_summary/grpo两份
原始helper已按固定Git blob补取并整模块导入，并从实际布局生成4.8×4米摘要。
它们独立保留在下一阶段源码目录，没有回改已关闭推理协议或声称已执行家具保存。
这些缺口不等同于整个研究失败，但不补实际执行证据就不能称为整场准备完成。
若未来P只赢未适配输出合同的弱对手、额外标签更少的A，或收益全由普通搜索解释，
不能声称独立机制成立；若在同源AUX/J和搜索下仍有跨场景收益，再扩大研究。
本轮没有新完整场景、梯度更新、独立人类偏好标签或CVPR级贡献证明，目标继续。

6. 本轮近邻复核（定向重读，不冒充新增完整通读）

ThinkBLOX III方法已明确采用随场景更新的reason-then-act、CoT/结构化布局监督。
SceneReVis 3.1–3.2也把诊断与工具调用作为联合响应，使用多模态反馈及SFT/RL；
A.2的普通物理后处理还可能删除实体。因此仍须区分完整要求保持与碰撞改善。
这些机制已有直接先例，本轮接通入口不能据此提升新颖性结论。未运行这两篇的
官方实现，也未把此次定向重读称作新完整论文审查。
https://arxiv.org/html/2607.13539v1
https://arxiv.org/html/2602.09432v1
'''
    (s/'algorithm_and_theory.txt').write_text(theory)
    source=Path(original['native_root'])
    readings=[]
    for rel,scope in [
        ('scenesmith/floor_plan_agents/stateful_floor_plan_agent.py','Constructor, cleanup, native designer tool construction, material config call'),
        ('scenesmith/floor_plan_agents/tools/floor_plan_tools.py','generate_room_specs wrapper/implementation and material query; not every unrelated tool'),
        ('scenesmith/floor_plan_agents/tools/materials_resolver.py','Local-default vs server-mode material resolution'),
        ('scenesmith/agent_utils/blender/server_manager.py','Owned subprocess start, native command, health checks, shutdown'),
        ('scenesmith/agent_utils/house.py','HouseLayout state fields and serialization only')]:
        path=source/rel;readings.append(dict(path=str(path),sha256=sha(path),read_scope=scope))
    dump(s/'reading_ledger.json',dict(new_source_readings=readings,
        original_stage_helpers=dict(manifest='next_stage_sources/source_manifest.json',
            room_summary_read='Formatter, training adapter and inference placed-room adapter; not an upstream training run',
            grpo_read='Imports and complete _flush_furniture_json only; whole module imported, other methods not treated as fully read'),
        pinned_trees_checked_for_missing_gateway=['nepfaff/scenesmith e888297a8b0a2f3acbb8634edf9def59cceab836','ZJU-OmniAI/Scene-Designer a0565e50ed7d32c0e09132b9311c0b9b91165405'],
        prior_theory_sources_reused_not_newly_read=['../20261110_reasoning_execution_review/reading_ledger.json',
            '../20261110_shared_model_runtime/reading_ledger.json','../20261110_execution_supervision_review/reading_ledger.json'],
        targeted_primary_rereads=[dict(title='ThinkBLOX',url='https://arxiv.org/html/2607.13539v1',
            scope='III method, especially III-A.1; prior reading explicitly reused, not a new full read'),
            dict(title='SceneReVis',url='https://arxiv.org/html/2602.09432v1',
            scope='3.1–3.2 and A.2/B.2; no author code or runtime reproduction')],
        new_full_paper_reads=0,scope='Targeted source/integration and existing-objective synthesis; not a new whole-field novelty clearance'))
    body=html.escape(json.dumps(summary,ensure_ascii=False,indent=2))
    (s/'stage_report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>原生政策到真实工具接通</title>'
        '<style>body{max-width:1050px;margin:40px auto;font:17px/1.75 sans-serif}pre{white-space:pre-wrap;background:#f3f5f7;padding:20px}</style>'
        '<h1>原生政策到真实工具已接通；完整生成仍待完成</h1>'
        '<p>真实模型自行提议创建卧室，原生SDK把空布局变为一个房间。没有放置家具，不是完整场景成功；没有训练或付费API。</p>'
        '<h2>实测汇总</h2><pre>'+body+'</pre><h2>算法与对照边界</h2><pre>'+html.escape(theory)+'</pre>'
        '<p><a href="summary.csv">统计</a> · <a href="method_differences.csv">方法对照</a> · '
        '<a href="native_joint_actual_v2/result.json">真实执行</a> · <a href="native_joint_actual_v2/native_policy_proposal.json">原始政策输出</a> · '
        '<a href="actual_floor_geometry_v1/result.json">几何移交</a> · <a href="packaged_geometry_cold_read.json">归档冷读</a> · '
        '<a href="reading_ledger.json">实际阅读范围</a> · <a href="evidence.zip">证据包</a> · '
        '<a href="native_floor_geometry.zip">另附几何包</a></p></html>')
    # Do not duplicate the 308-file runtime and 5MB official textures in Git/zip.
    # Pins/hashes/URLs plus materializer retain the reconstruction record.
    (s/'.gitignore').write_text('/runtime/\n**/__pycache__/\n')
    skip={'artifact_manifest.json','delivery_audit.json','evidence.zip'}
    files=[f for f in s.rglob('*') if f.is_file() and 'runtime' not in f.relative_to(s).parts
        and '__pycache__' not in f.parts and f.name not in skip]
    manifest=[dict(path=str(f.relative_to(s)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(files)]
    for f in [repo/'scripts/autodl'/n for n in ('materialize_remote_controller.py','remote_native_control.py',
            'probe_remote_native_controller.py','native_joint_model_host.py','run_native_joint_admission.py',
            'finalize_actual_native_floor.py','package_native_floor.py','replay_packaged_native_floor.py',
            'recover_native_stage_helpers.py','probe_native_stage_helpers.py','report_remote_native_controller.py')]:
        manifest.append(dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size))
    dump(s/'artifact_manifest.json',dict(files=manifest,runtime_excluded_from_archive=True,
        native_floor_geometry_zip_separate_companion_not_duplicated_in_evidence_zip=True,
        reconstruction='materialize_remote_controller.py with qualified source closures; immutable initial attempts and source recoveries retained'))
    assert shutil.disk_usage(s).free>5*1024**3+4*1024**2
    with zipfile.ZipFile(s/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for r in manifest:
            if r['path']=='native_floor_geometry.zip':continue
            f=Path(r['path']);f=f if f.is_absolute() else s/f
            z.write(f,'scripts/'+f.name if Path(r['path']).is_absolute() else r['path'])
        z.write(s/'artifact_manifest.json','artifact_manifest.json')
    assert all(sha(Path(r['path']) if Path(r['path']).is_absolute() else s/r['path'])==r['sha256'] for r in manifest)
    with zipfile.ZipFile(s/'evidence.zip') as z:
        assert z.testzip() is None
        for r in manifest:
            if r['path']=='native_floor_geometry.zip':continue
            name='scripts/'+Path(r['path']).name if Path(r['path']).is_absolute() else r['path']
            assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
    protocol=json.loads((s/'native_joint_protocol_v2.json').read_text())
    assert all(sha(f)==h for f,h in protocol['code_sha256'].items())
    audit=dict(status='passed',manifest_files=len(manifest),zip_members=len(manifest),
        zip_sha256=sha(s/'evidence.zip'),zip_bytes=(s/'evidence.zip').stat().st_size,
        original_source_hashes_unchanged=unchanged,protocol_hashes_unchanged=True,
        actual_one_forward=True,owned_processes_terminal=True,goal_achieved=False)
    dump(s/'delivery_audit.json',audit);print(json.dumps(audit))


if __name__=='__main__':main()
