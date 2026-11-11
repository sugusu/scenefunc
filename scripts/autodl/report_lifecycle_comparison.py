"""Audit and package the actual whole-home comparison, without new search.

The report distinguishes completion under declared geometric proxies from
from-empty scene generation and expert/human functional acceptance.
"""
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time
import zipfile


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    repo=Path.cwd();v1=repo/'runs/autodl/20261111_lifecycle_case_comparison'
    r=repo/'runs/autodl/20261111_lifecycle_case_comparison_v2'
    a=read(v1/'summary.json');b=read(r/'summary.json')
    checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    inputs=read(v1/'protected_before.json')
    protected=[dict(**x,after=sha(x['path'])) for x in inputs]
    check('579 protected inputs unchanged',len(protected)==579 and all(x['after']==x['before']==x['expected'] for x in protected))
    write(r/'protected_before_after.json',dict(count=len(protected),all_unchanged=True,files=protected))
    source=read(v1/'source_manifest.json');check('All 6 historical source files unchanged',all(sha(x['local'])==x['sha256'] for x in source['files']))
    pools=[read(x/'common_candidates.json') for x in [v1,r]]
    posekeys=['candidate_id','remove_unneeded_bed','allocation','table_xy','chair_xy','table_yaw','chair_yaw','approach_xy']
    projection=lambda pool:[{k:x[k] for k in posekeys} for x in pool]
    check('Same 1120 geometric/action candidates across v1 and v2',len(pools[0])==len(pools[1])==1120 and projection(pools[0])==projection(pools[1]))
    check('v1 retained in full with no successful commit',all(x['native_candidates']==24 and x['office_conditional_geometry'] is False and x['additions']==0 for x in a['arms']))
    check('v2 two actual accepted endpoints',all(x['native_candidates']==3 and x['office_conditional_geometry'] and x['selected_candidate_id']==615 for x in b['arms']))
    check('Same v2 final state digest',b['same_final_state'])
    counter=[]
    for stage,pool in zip([v1,r],pools):
        orders=read(stage/'proposal_order_comparison.json')['orders']
        streams=list(orders.values());elig={x['candidate_id'] for x in pool if not x['cheap_task_violations'] and not x['cheap_overlap_count']}
        e=[[i for i in s if i in elig] for s in streams]
        check(stage.name+' equal admissible streams',e[0]==e[1])
        check(stage.name+' case penalty constant on admissible pool',all(x['case_compatibility_penalty']==0 for x in pool if x['candidate_id'] in elig))
        counter.append(dict(stage=stage.name,full_order_equal=streams[0]==streams[1],eligible_count=len(elig),eligible_order_equal=e[0]==e[1],
            first_full_order_difference=next((dict(rank=i,B0=x,B1=y) for i,(x,y) in enumerate(zip(*streams)) if x!=y),None)))
    write(r/'proposal_information_audit.json',dict(stages=counter,
        correction='Only admissible/evaluated streams are identical; full proposal lists reorder some rejected candidates.',
        no_general_impossibility_claim=True))
    initial=read(r/'initial_state.json');arms=[x['arm'] for x in b['arms']]
    for arm in arms:
        s=read(r/arm/'final_state.json')
        check(arm+' retained 28 records byte-equivalent as dictionaries',all(s['objects'][i]==o for i,o in initial['objects'].items() if i!='former_child_bed_0'))
        check(arm+' only permitted removal and two additions',set(initial['objects'])-set(s['objects'])=={'former_child_bed_0'} and set(s['objects'])-set(initial['objects'])=={'new_work_table_0','new_work_chair_0'})
    exported=read(r/'geometry_export_receipt.json');persisted=read(r/'persisted_geometry_manifest.json')
    check('Persistent licensed geometry matches both independently exported ends',sha(persisted['shared_geometry_path'])==persisted['sha256'] and exported['identical_arm_glbs'])
    for row in exported['outputs']:
        check(row['label']+' GLB mesh-node roundtrip bound tolerance',row['maximum_bound_error_m']<1e-5)
        if row['cold_restore']:
            check(row['label']+' native cold restoration',row['cold_restore']['object_ids_exact'] and row['cold_restore']['maximum_world_transform_error']<1e-12)
    current_initial=read('/dev/shm/lifecycle_case_comparison_v2_20261111/initial/bundle/house_state.json')['rooms']['michelet_lifecycle_transfer']
    check('Common native initial state unchanged',initial==current_initial)
    # Every accepted result is linked to its executed geometry measurements;
    # an unchanged object count or a plausible render is not the criterion.
    semantic=[]
    for arm in arms:
        rec=read(r/arm/'candidate_0615.json');access=rec['access'];result=rec['result']
        for width in [0.4,0.5,0.6]:
            sc=next(s for s in access['scenarios'] if s['square_body_width_m']==width)
            check(arm+f' office reachability body={width}',sc['targets']['new_office_seat']['reachable_from_entry'])
        fixed_passes=sum(t['reachable_from_entry'] for sc in access['scenarios'] for name,t in sc['targets'].items() if name!='new_office_seat')
        check(arm+' exactly 44 of 45 retained use approach witnesses pass',fixed_passes==44)
        check(arm+' accepted record has no new intersection or access regression',result['accepted'] and not result['new_furniture_intersections'] and not result['new_structure_intersections'] and not result['access_regressions'])
        semantic.append(dict(arm=arm,office_route_passes=3,office_route_total=3,
            retained_use_routes_passes=44,retained_use_routes_total=45,
            persistent_existing_warning='shower_front not reachable by the 0.6m square-body proxy',
            required_activities=['sleep','living_meal','kitchen','hygiene','storage_identity','office','lifecycle_tradeoff','restoration'],
            unknown=['Real sleep/cooking/hygiene use','Human sitting/reach and desk suitability','Door operation/privacy','Removed-bed destination'],
            measured_success_scope='Conditional geometry and object/use allocation; not complete human functional success'))
    write(r/'functional_coverage_audit.json',dict(arms=semantic,
        protocol_deviations=[dict(item='sampled ground support',actual='Acceptance checked native base bounds within 1mm of the floor, not a support polygon or physical stability.'),
            dict(item='desk minimum area/height',actual='Recorded in task metadata but not an explicit acceptance factor. No claim that ergonomic usable tabletop area was validated.'),
            dict(item='native operations',actual='Native RoomScene restore/addition/removal state operations, not the original LLM tool loop or a learned Agent.')],
        complete_generation_from_description=False,expert_quality_rating=False))
    operations=[]
    for arm in arms:
        s=read(r/arm/'final_state.json')
        operations.append(dict(arm=arm,removed=dict(object_id='former_child_bed_0',permitted_by_task=True,destination='unknown'),
            added=[dict(object_id=i,pose=s['objects'][i]['transform'],asset_registry=s['objects'][i]['metadata']['catalog_registry_id']) for i in ['new_work_table_0','new_work_chair_0']],
            unchanged_ids=sorted(set(initial['objects'])-{'former_child_bed_0'}),selection='candidate_615, same in both arms'))
    write(r/'actual_committed_edits.json',operations)
    unique={}
    for stage in [v1,r]:
        for arm in arms:
            for x in read(stage/arm/'candidate_evaluations.json'):
                k=x['candidate_id'];value=x['accepted']
                if k in unique:check(f'candidate {k} repeated feasibility consistent',unique[k]==value)
                unique[k]=value
    write(r/'training_target_admission.json',dict(unique_evaluated_candidates=len(unique),unique_accepted=sum(unique.values()),
        repeated_native_evaluations=54,unique_projects=1,researcher_reconstructed=True,
        public_expert_action_targets=0,professional_preference_pairs=0,original_model_ABI_action_targets=0,
        model_calls=0,training_updates=0,training_examples_admitted=0,
        reason='A deterministic state-level completion is not a human preference, original designer action or captured neural tool trajectory. Three v2 evaluations (one overlapping v1) do not establish learned cross-project improvement.'))
    reading=dict(records=[
        dict(source='Native original build_lifecycle_input.py',revision='75fc783f86d4fea78714639e110fbe77476ee106',read='Complete 184-line source, original task and build plan; native importer and relevant state/mesh routines inspected and executed.',executed=True),
        dict(source='Pine / Never Too Small',url='https://www.nevertoosmall.com/post/two-creative-maximalists-living-minimally',read='Project account; especially The Everything Room with existing and new desk coexistence.',scope='Co-produced reported decision; no raw brief or expert ranking of our result.'),
        dict(source='TM8 / DC.AD',url='https://www.archdaily.com/1001060/tm8-apartment-dd',read='Complete supplied project text; young-family rental, public/private split and three retained bedrooms.',scope='Case condition cannot override this task explicitly releasing one bed.'),
        dict(source='Function2Scene',url='https://arxiv.org/html/2605.30819v1',read='This turn rechecked §3, §4.1–4.2 and §5–6; previous continuous-session reading also covered tool signatures.',scope='Already personalizes functional criteria and preserves higher-priority constraints; no new first-use claim. Not rerun.'),
        dict(source='SceneExplorer institutional PDF',url='https://researchportal.bath.ac.uk/files/348054050/SceneExplorer_tvcg24.pdf',read='Full text not acquired; HTTP403 this turn.',scope='Do not claim full method reading.'),
        dict(source='SceneExplorer OpenReview mirror',url='https://openreview.net/pdf/60be75c6e99758a75771262b2ec18727d0ce3129.pdf',read='Returned browser challenge, not paper text.',scope='Do not claim full method reading.')],
        literature_search_scope='Targeted source verification for this comparison, not exhaustive novelty clearance',
        model_inference_this_stage=0,training_this_stage=0)
    write(r/'reading_ledger.json',reading)
    theory='''算法与证据边界：整屋用途分配的案例条件排序对照

1. 实际实现
输入是固定用途任务u、同一原生住宅s0、相同16项资产目录及两张其他住宅的案例卡K。
研究者预先指定一类桌椅资产及用途转换动作，未实现自由资产检索或自动语义解析。
候选x同时确定：是否移出获准释放的旧床、为办公室新增桌椅还是重分配原家庭餐位、桌椅的格点位置与旋转。
1120个联合候选由初态、房间范围和尺寸生成，无该住宅最终方案坐标。
公共普通因子检查保留用途、容纳与AABB相交；v2再检查同一人体代理的目标格点是否被占用。
B0按[v(x), overlap(x), edits(x), id(x)]排序。
B1按[v(x), overlap(x), case_incompatibility(x), edits(x), id(x)]排序。
两组均把相同的可进入候选送入原生状态、FCL表面相交和网格接近检查；至多24个，首次全满足声明检查即停止。
这是一种研究者编译的案例先验，不是实际神经Agent、学习算法或全新CSP。

2. 本例可验证的等价性质
设E为同一预筛后可进入原生检查的候选集合。本例在E上的case_incompatibility均为0。
插入一个常数排序键不改变E内相对次序。因此，在共享的确定性验证器V和停止预算下，两组访问同一候选序列，首次接受者及终态相同。
这是对本例实际实现的代数性质；证书逐个验证1120项，v1有172项进入域、v2有58项。
完整候选列表在域外并不完全同序，不能把“有效前缀相同”扩大成“全部排序相同”。
v2两组都在第3次原生检查选中615，两份状态及GLB哈希相同。更快的单次墙钟值不是受控效率收益，可能受导入、文件缓存与执行顺序影响。

3. 不成立的推广
不要求案例一定带来当前输入不存在的新事实才有价值。即使所有约束已知，较好的候选顺序也可能在同一预算下减少搜索；此次编码没有改变实际访问顺序。
也不能从24次失败证明空间无解：v1的普通提案器没有事先筛掉座位后方被堵的格点，有限前缀失败；保留v1后，v2公共普通修正完成了转换。
不能把任务明确释放一张床，与另一家庭项目保留三间卧室，构造为普遍的人类偏好反转标签。
不能从物体保留、无新增相交与代理路径通过推出完整语义、舒适、真实使用、隐私、稳定或CVPR创新。

4. 训练目标应该是什么，不应该是什么
若将来学习提案策略πθ(x|u,s0,K)，任务是提高固定后端和预算下首次有效候选的概率或降低完整求解成本。
可采用普通已验证动作交叉熵、有效候选集合的边际似然 -logΣ_{x∈Fverified}πθ(x|u,s0,K)，或包含全部提案/验证/模型成本的风险目标；这些都是已有通用目标，不是新损失。
Fverified只能包含实际通过相应检查的项；没评过的候选是未知，不是负例。当前有26个不同候选被测、其中1个接受；54次原生测量不是54个独立任务。
当前脚本操作不是原模型工具ABI的真实推理轨迹；没有原用户/设计师对比，也没有新调用或训练。不得把B1标成优于B0进行DPO/偏好训练；两组事实上相同。
若做动作SFT，还须捕获实际工具调用、先前状态、观察、结果、允许对象身份及完整上下文，验证重放与原读取器，再判断是否可作为几何教师数据。

5. 与近邻的明确关系
Function2Scene已经按住户与活动生成个性化约束并用工具迭代修复，且不破坏较高优先级要求。本例“功能需求+工具+保护旧用途”不是首次。
Pine与TM8是有出处的设计说明，提供候选条件和适用性讨论；当前把它们编译成已存在的任务因子，没有显示独立算法增益。
SceneExplorer全文本轮两条下载路径均未取得，不能以未读论文作排重结论。源码读取、可运行实现、实际整屋结果与方法创新分别报告。

6. 这轮完成与未完成
完成：另一套住宅的初态重建、两种原生终态、原先用途保留核对、共同普通修正及同权案例先验对照、冷恢复、整屋GLB和实际渲染。
未完成：从新描述自动生成全屋、真正的神经案例检索/分配决策、独立测试集、完整功能/人评、有效训练及可发表的新算法证据。
下一项算法必须改变一个可测的决策或搜索成本；不能只把已显式给定的任务条件重新命名为设计知识，也不应继续加大此住宅预算来制造案例组优势。
'''
    (r/'algorithm_and_theory.txt').write_text(theory)
    # Source code used for the actual runs remains readable after later edits.
    code=['prepare_lifecycle_comparison.py','compare_lifecycle_allocations.py','export_lifecycle_comparison.py','render_lifecycle_comparison.py','report_lifecycle_comparison.py']
    write(r/'code_manifest.json',[dict(path=str(repo/'scripts/autodl'/n),sha256=sha(repo/'scripts/autodl'/n)) for n in code])
    esc=html.escape
    body='''<!doctype html><meta charset="utf-8"><title>整屋用途转换：普通搜索与案例条件排序</title>
<style>body{font:16px/1.65 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#203040}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbc8d4;padding:9px}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap;background:#eef3f7;padding:16px}small{color:#596b7a}</style>
<h1>整屋用途转换已经实际完成；本例案例条件先验没有独立收益</h1>
<p>研究者重建住宅的用途转换开发实验。不是实测原住宅、从空房生成、神经Agent结果或已成立的CVPR贡献。目标住宅自身最终方案与动作未输入提案器。两组的原生终态和GLB完全相同。</p>
<h2>实际结果</h2><table><tr><th>实现</th><th>普通搜索</th><th>案例条件排序</th></tr>
<tr><td>v1有限检查后采纳</td><td>0 / 24</td><td>0 / 24</td></tr>
<tr><td>v2首次有效的原生检查序号</td><td>3</td><td>3</td></tr>
<tr><td>选中候选 / 总候选</td><td>615 / 1120</td><td>615 / 1120</td></tr>
<tr><td>原生对象：初态→终态</td><td>29→30</td><td>29→30</td></tr>
<tr><td>操作</td><td>允许移出1床，加入1桌1椅</td><td>相同</td></tr>
<tr><td>其余完整保留的对象记录</td><td>28</td><td>28</td></tr>
<tr><td>办公室代理接近检查</td><td>3/3</td><td>3/3</td></tr>
<tr><td>保留用途代理接近检查</td><td>44/45，无退化</td><td>44/45，无退化</td></tr>
<tr><td>最终家具 / 结构表面相交</td><td>0 / 0</td><td>0 / 0</td></tr></table>
<p>保留用途中，0.6米方形代理到淋浴点的原有失败未解决；目标住宅的设备、人体操作、隐私和旧床移出处未知。基座高度检查不是物理稳定性，桌面可用面积/人体工学没有完成验证。</p>
<h2>可查看的实际场景</h2><p>俯视图仅隐藏屋顶与外墙，保留内部隔墙；蓝色为新办公桌椅，不代表原材质。两组结果相同，分别列出仅为明确方法对应。</p>
<details><summary>共同初态</summary><img src="initial_topdown.webp" alt="共同原生住宅初态"></details>
<div class="pair"><section><h3>B0 普通任务搜索</h3><img src="final_topdown.webp" alt="普通搜索实际终态"></section><section><h3>B1 案例条件排序</h3><img src="final_topdown.webp" alt="案例条件排序的相同实际终态"></section></div>
<h2>为什么不能把本例写成案例方法成功</h2><p>Pine的已有工作面与新工作面共存原则、TM8的持续卧室需求核查，经过本任务条件解释后，落实成了普通任务已要求的“保留餐位、不保留明确释放的旧床”。在所有可进入原生验证的候选上，案例惩罚为常数，故实际访问顺序相同。完整1120项列表在被拒域中有重排，不能声称所有排序都相同。不同执行次序和缓存状态下的一次耗时也不构成效率优势。</p>
<p>原先v1保留：两组各24次均失败，没有采纳。v2是看到失败后一次明确记载的公共普通修正：预筛座位后方人体代理格点占用；没有改最终阈值、资产、候选空间或24次预算，也没有第三轮扩搜索。因此v2是开发修正结果，不是独立留出测试。</p>
<h2>阅读、理论与运行材料</h2><ul><li><a href="summary.csv">v2逐组表</a> / <a href="../20261111_lifecycle_case_comparison/summary.csv">保留v1逐组表</a></li><li><a href="algorithm_and_theory.txt">算法、等价性质、训练目标与证明边界</a></li><li><a href="reading_ledger.json">实际阅读范围及两次全文获取失败</a></li><li><a href="functional_coverage_audit.json">功能覆盖与协议偏差</a></li><li><a href="actual_committed_edits.json">实际采纳的对象与姿态</a></li><li><a href="persisted_geometry_manifest.json">持久几何、两组可打开的GLB路径和哈希</a></li><li><a href="geometry_export_receipt.json">原生冷恢复与GLB重新载入证据</a></li><li><a href="training_target_admission.json">训练数据资格</a></li><li><a href="protected_before_after.json">579份原输入前后哈希</a></li><li><a href="evidence.zip">文本/执行证据包（图与许可几何单列，不复制进ZIP）</a></li></ul>
<p>最近邻：<a href="https://arxiv.org/html/2605.30819v1">Function2Scene</a>已使用住户/活动条件和工具修复；<a href="https://www.nevertoosmall.com/post/two-creative-maximalists-living-minimally">Pine</a>与<a href="https://www.archdaily.com/1001060/tm8-apartment-dd">TM8</a>是设计报告，不是对本实验结果的人评。本轮未取得SceneExplorer全文，不声称已完成全领域排重。</p>
<h2>运行成本与后续边界</h2>'''
    body+='<pre>'+esc(json.dumps(dict(v1_search_wall_seconds=a['wall_seconds'],v2_search_wall_seconds=b['wall_seconds'],v2_arms_seconds={x['arm']:x['wall_seconds'] for x in b['arms']},native_export_and_cold_restore_seconds=exported['wall_seconds'],peak_search_rss_kib=b['peak_rss_kib'],extra_paid_API_calls=0,local_neural_inference_calls=0,training_updates=0),ensure_ascii=False,indent=2))+'</pre>'
    body+='''<p>当前阶段未开启新资源。几何持久文件是17.33MB的gzip，双方指向同一份已证实相同的内容；未用硬链接复制旧证据。可打开的未压缩GLB位于/dev/shm，属临时路径；重启后按命令从持久文件解压。相同几何仅保存一份，双方完整状态和原生资源清单各自保留。</p>
<p>研究继续，但这轮不增加该住宅预算、训练或把普通修正包装成创新。下一步应检验能改变有效决策/搜索代价的具体机制；从需求自动生成整屋、真实神经方法和独立房间效果仍未验证，目标尚未完成。</p>'''
    (r/'stage_report.html').write_text(body)
    commands='''已实际运行（repository根目录）：
python -B scripts/autodl/prepare_lifecycle_comparison.py --root runs/autodl/20261111_lifecycle_case_comparison --scratch /dev/shm/lifecycle_case_comparison_20261111 --catalog-bundle /dev/shm/milan_completion_20261111/bundle
nativePython -B runs/autodl/20261111_lifecycle_case_comparison/source/scripts/920/build_lifecycle_input.py --plan runs/autodl/20261111_lifecycle_case_comparison/local_build_plan.json --out /dev/shm/lifecycle_case_comparison_20261111/initial
nativePython -B scripts/autodl/compare_lifecycle_allocations.py --root runs/autodl/20261111_lifecycle_case_comparison --scratch /dev/shm/lifecycle_case_comparison_20261111
# 上一行v1执行时的真实代码保留在v1/executed_code/，当前脚本含明确的v2可选普通预筛。
# v2固定同一输入，初态路径通过RAM软链接共享；全部输入只读。它是公开开发修正，不是再次跑同配置挑结果。
nativePython -B scripts/autodl/compare_lifecycle_allocations.py --root runs/autodl/20261111_lifecycle_case_comparison_v2 --scratch /dev/shm/lifecycle_case_comparison_v2_20261111
nativePython -B scripts/autodl/export_lifecycle_comparison.py --root runs/autodl/20261111_lifecycle_case_comparison_v2 --scratch /dev/shm/lifecycle_case_comparison_v2_20261111
evalPython -B scripts/autodl/render_lifecycle_comparison.py --input /dev/shm/lifecycle_case_comparison_v2_20261111/geometry/initial.glb --output runs/autodl/20261111_lifecycle_case_comparison_v2/initial_topdown.png
evalPython -B scripts/autodl/render_lifecycle_comparison.py --input /dev/shm/lifecycle_case_comparison_v2_20261111/geometry/B0_task_aware_search.glb --output runs/autodl/20261111_lifecycle_case_comparison_v2/final_topdown.png
# 两PNG已转换为逐像素一致的lossless WebP；原PNG留在RAM，未作为模型输入。

nativePython=/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python
evalPython=/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH=<repository>/runs/autodl/20261110_corpus_state_recovery/dependency/import:/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps

程序拒绝覆盖已跑目录。复现应另建输出目录，使用source_manifest中原输入/代码及同样协议。catalog.json SHA为d3847eda41e338bbbf2b91904c3935c0a04405d268fa77fa26aa7838bb2f5320。
仅在已有RAM目录消失时，将持久GLB解压到新的RAM位置后用Blender打开；不需要模型、GPU或付费API：
python -c 'import gzip,pathlib; p=pathlib.Path("/root/autodl-tmp/agent_method_preparation_20261027/work/lifecycle_case_artifacts_20261111/shared_completed_scene.glb.gz"); pathlib.Path("/dev/shm/lifecycle_completed.glb").write_bytes(gzip.decompress(p.read_bytes()))'
该解压命令是已知标准操作说明，本报告未额外重复解压一次。两组相同GLB均由实际原生状态独立导出并重载验证。
完整新原生包目前在RAM；许可资产未随Git/ZIP重新发布。跨机器还需已获许可的16项资产；资产字节与重建路径见manifest，而不是一个已经完整移植的训练数据集。
'''
    (r/'commands_and_reproduction.txt').write_text(commands)
    write(r/'verification.json',dict(checks=checks,count=len(checks),all_passed=all(x['passed'] for x in checks),
        physics_independently_reexecuted=False,full_source_evaluator_independently_validated=False,
        hypothesis_or_goal_complete=False,large_artifact_full_portability=False))
    # ZIP is a convenience record bundle, explicitly not a geometry or image
    # dataset. Include both v1 and v2 so a successful revision cannot erase
    # its failed precursor. Large licensed geometry never enters Git.
    archive=r/'evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for stage in [v1,r]:
            for p in sorted(stage.rglob('*')):
                if p.is_file() and p.name not in ['delivery_manifest.json','package_verification.json'] and p.suffix in ['.json','.csv','.txt','.py','.html','.md']:
                    z.write(p,str(Path(stage.name)/p.relative_to(stage)))
        for n in code:z.write(repo/'scripts/autodl'/n,'code/'+n)
    data_check_count=len(checks)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            data=z.read(info)
            prefix,*rest=Path(info.filename).parts
            p=repo/'scripts/autodl'/Path(*rest) if prefix=='code' else repo/'runs/autodl'/prefix/Path(*rest)
            check('archive member '+info.filename,hashlib.sha256(data).hexdigest()==sha(p))
    write(r/'package_verification.json',dict(data_checks=data_check_count,archive_member_checks=len(checks)-data_check_count,
        total_checks=len(checks),all_passed=True,scope='Archive checks verify packaged bytes, not additional scene trials. Delivery/package manifests are excluded to avoid self-reference.'))
    files=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(r.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json']
    write(r/'delivery_manifest.json',dict(files=files,evidence_zip_bytes=archive.stat().st_size,
        evidence_zip_sha256=sha(archive),licensed_geometry_external=persisted['shared_geometry_path'],
        package_members_verified=True,side_by_side_images_external_to_zip=True,
        source_originals_unchanged=579,goal_complete=False,created_unix=time.time()))
    print(json.dumps(dict(status='audited_and_packaged',checks=len(checks),archive_bytes=archive.stat().st_size,report=str(r/'stage_report.html')),ensure_ascii=False))


if __name__=='__main__':
    main()
