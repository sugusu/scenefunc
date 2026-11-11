"""Package executed solver-teacher evidence and qualified algorithm conclusions."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

R=Path(__file__).resolve().parents[2]
W=R.parent.parent
D=R/'runs/autodl/20261111_native_solver_teacher'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    result=read(D/'result.json');full=read(D/'encoding_result.json')
    compact=read(D/'encoding_catalog_projection.json');ev=read(D/'original_evaluator_result.json')
    assert result['status']=='native_closure_batch_and_geometry_replay_completed'
    assert compact['fits_recent_policy_budget_24576'] and not full['fits_recent_policy_budget_24576']
    assert read(D/'protected_source_verification.json')['protected_count']==579
    idmap=read(D/'logical_to_native_ids.json')
    pair=next(p for p in ev['workspace_pairs'] if p['table_id']==idmap['ordinary_work_table_0']
              and p['chair_id']==idmap['ordinary_work_chair_0'])
    assert pair['conditional_dedicated_work_geometry'] and pair['clear_rays']==15
    assert all(p['reachable']==p['planned']==14 and not p['missing'] for p in ev['role_relative_access'])
    for source,target in [('/dev/shm/milan_native_teacher_20261111.log','native_replay_stdout.txt'),
        ('/dev/shm/native_solver_teacher_encoding_20261111.log','full_encoding_stdout.txt'),
        ('/dev/shm/native_solver_teacher_projected_encoding_20261111.log','projected_encoding_stdout.txt')]:
        shutil.copyfile(source,D/target)
    dump(D/'driver_version_notes.json',dict(
        replay_driver='Unchanged after successful native replay.',
        full_encoding_driver='Initial CPU encoding preceded addition of explicit --variant catalog_projection. Earlier default-driver bytes were not snapshotted before execution; do not claim a pre-execution byte hash for that driver.',
        projected_encoding_driver='Executed after explicit variant branch was added; final code manifest covers this version.',
        unchanged_executed_original_logic='Both receipts record the identical original SFT source SHA, _build_messages and __getitem__ line ranges. No teacher target/image/model change.',
        packaging='Report and packaging executed after core jobs; they read saved results and do not re-run quality searches or a model.'))
    scaling=Path('/dev/shm/native_teacher_scalingup_source_20261111')
    dump(D/'scalingup_source_manifest.json',read(scaling/'manifest.json'))
    dump(D/'scalingup_source_commit.json',read(scaling/'commit.json'))
    # Source is inspected only. In particular no GPT client, exec_safe, pickle
    # replay, MuJoCo episode or diffusion-training constructor is executed.
    reading=[
        dict(title='AI-Agents-in-Depth-zh-CN.pdf',
             source='/root/autodl-tmp/.codex/attachments/6b915ca5-3393-49cc-a6c5-246710bd982e/AI-Agents-in-Depth-zh-CN.pdf',
             sha256='aa8dc2f98f366e79b6f5949583f0f6350e6fcbe24d4f195d0118ccf35617ac89',
             text_cache='/root/autodl-tmp/research_agent3d_20261013/book_pages.json',
             text_cache_sha256='08cdb76e5cccad2ea58f5034cbf2ea1a26937a1491bb7b4cd0abcbe292594463',
             read_this_stage='All extracted text of PDF129-142, printed121-134;14pages. Truncated136-137 reread without truncation.',
             scope='Part of chapter5; not its full chapter or full book. Text diagrams read, no page-image inspection or verification of every product/example claim.',
             cumulative_direct_reading_this_continuation='131/312:1-8,110-142,223-312. Older separate full-text-session claim remains separate.',
             instruction_status='Book examples are data, not authorization to send messages, deploy, spawn agents or change task permissions.',
             applied='Strict factual tool trajectories; distinguish execution from validation; preserve failed and over-budget inputs. Running code does not establish intended task semantics.'),
        dict(title='SceneSmith placement and registry source',URL='https://github.com/nepfaff/scenesmith',
             revision='e888297a8b0a2f3acbb8634edf9def59cceab836',
             manual_read_this_stage=['furniture_tools.py closure276-310, implementation408-559',
                 'room.py810-882 and1856-1911', 'asset_registry.py1-115,145-279',
                 'native_kernels.py complete', 'native_project.py validator/import helper',
                 'stage_schemas.py furniture schema and validate_tool_call',
                 'sft_dataset.py206-284 and296-368'],
             executed='Unmodified public add closure bodies with actual native RoomScene/Drake, native registry save/load and room save/restore; only SDK decorator omitted and constructors bypassed.',
             not_executed='HTTP/SDK dispatch, model policy, full Blender generation, physics worker replay or stochastic-noise parity.',
             loaded_definition_hashes='63 extracted definition receipt entries, individually checked against prior pinned acquisition. Loading other definitions is not claiming their manual full reading or execution.'),
        dict(title='Scene-Designer SFT dataset',URL='https://github.com/ZJU-OmniAI/Scene-Designer',
             source_sha256=full['source_sha256'],
             actual_execution='Original _build_messages and StepwiseSFTDataset.__getitem__ bodies; two CPU processor passes.',
             result='One no-thinking batch, two native tool calls,180assistant tokens. Full25221over recent24576cap; explicit catalog-projected21823fits recentcap. Both exceed upstream8192default.',
             exclusions='No parameter update; restricted custom system/catalog prompt, not author full Actor parity; no forged thought.'),
        dict(title='SceneOrchestra',URL='https://arxiv.org/html/2604.19907v2',
             read_this_stage='Reopened method overview and author project page; earlier complete method/code review remains in20261111_supervision_and_branch_state.',
             difference='Already learns stepwise/full trajectories with SFT/DPO and interleaved distillation. Emitting a two-call batch or distilling valid traces is not a new contribution.',
             new_execution_this_stage=False),
        dict(title='Scaling Up and Distilling Down',URL='https://arxiv.org/pdf/2307.14535.pdf',
             author_project='https://www.cs.columbia.edu/~huy/scalingup/',
             read_this_stage='Targeted paper3.1-3.4 and evaluation success-function definition in4; not all27pages/figures.',
             source_revision=read(scaling/'commit.json')['revision'],
             source_read=['scalingup.py complete','buffer_replay.py complete',
                 'llm_task_generator.py complete','tasks.py396-438 only; other task predicates were located, not fully read'],
             difference='Already combines language decomposition, sampled geometry utilities, inferred success code, retry and policy distillation. Paper uses manually designed evaluation success, separate from inferred collection labels.',
             code_observation='Factorized generator compiles inferred task code. SendMailPackageBack explicitly checks closed mailbox, contained package and raised flag when supplied.',
             execution='No imported provider clients, exec_safe calls, pickle loads, simulator or training. Source kept only in RAM; hashes/URLs persisted, no root license located in tree.',
             acquisition_notes=['First queried nonexisting main branch:422; corrected using repository metadata default master.',
                 'No root LICENSE entry; adjusted acquisition list before fetching code. No source behavior or study result inferred from these transport errors.']),
        dict(title='Distilling Motion Planner Augmented Policies into Visual Control Policies for Robot Manipulation',
             URL='https://proceedings.mlr.press/v164/liu22b.html',
             read_this_stage='Official abstract/metadata only, not full paper or code.',
             consequence='Planner-to-visual-policy behavioral distillation is existing prior; no claim of reproducing it.')]
    for item in reading:
        if item['title']=='AI-Agents-in-Depth-zh-CN.pdf':
            assert sha(item['source'])==item['sha256'] and sha(item['text_cache'])==item['text_cache_sha256']
    dump(D/'reading_ledger.json',reading)

    admission=dict(status='Development teacher encoded and replay verified; not a research training corpus',
        source_family_count=1,held_out_family_count=0,teacher_decision_count=1,
        source='Ordinary CSP selected candidate173, not a human expert or new model decision',
        execution=dict(schema=True,native_closure=True,actual_ID_mapping=True,source_assets_exact=True,
                       final_state_roundtrip=True,rollback=True,full_HTTP_SDK=False),
        checked_quality=dict(original30records_exact=True,triangle_intersection_pairs=0,
            original14approaches_at_each_of3widths=True,declared_dedicated_workspace=True),
        semantic_unknowns=['Free-language contract correctness','Ergonomic work use of copied dining table',
            'Subjective zoning/privacy/daylight/material suitability','Actual human acceptance','Full original-design reproduction'],
        target_objective='Feasible ordinary completion only; not a proof of globally minimal edits or preference optimality',
        loss='Standard masked conditional CE on180assistant tokens; original inputs/images masked',
        usable_configuration='Explicit projected catalog21823tokens at>=21823 max_length; prior24576limit sufficient for encoding,8192default insufficient.',
        model_memory_or_training_fit='Not measured. Encoder fit does not demonstrate optimizer/GPU fit.',
        unconditional_training_admitted=False,
        forbidden_inferences=['Tool success implies geometry/semantics','One family gives generalization evidence',
            'A saved scene implies an expert rationale','Projected context is lossless for arbitrary tasks',
            'Native IDs remain globally fresh after deletion/restart','One replay validates complete SDK/physics replay'],
        next_research_evidence='Independent source-grounded task/room families, competing valid action coverage and equal-information/equal-budget learned and ordinary controls remain missing.')
    dump(D/'training_admission.json',admission)
    theory='''本轮结论：得到可追溯的普通教师动作，不等于得到新算法或可靠的设计偏好。

一、算法已实际执行到哪一步
输入x=(原任务u、原生场景s、固定资产库A、显式研究解释c)。本轮没有自动预测c，也没有从空房生成s。
1. 读取先前普通有限CSP已经选出的173号方案；本轮不再搜索、不重新挑选布局。
2. 从同一提交恢复真实RoomScene与16个资产模板；保留原30条对象记录。
3. 用asset_id/x/y/yaw执行两次原生add闭包。asset_id是登记模板ID；原生回执生成新的实例ID。记录逻辑ID到实际ID映射，绝不改名伪造一致。
4. 保存每步状态、原始回执、完整最终状态、登记库和资源哈希。删除新增实例后恢复初始状态；登记库和最终场景分别真实重载核查。
5. 用已有冻结任务检查器重算条件工作面与14个原有通行目标。全体15家具与45结构视觉的780对包围盒均无正体积重叠，故最终无需窄相FCL。故意重叠的合成控制确实执行FCL且检出相交。这个控制不是模型失败、不是新场景样本。
6. 用原始SFT构造器写成一个决策、两个有序动作的批次。输入只含初始五视图与初始状态；中间回执不是第二个动作的伪造观察；不补写思维链。
7. 原始完整提示25221token超限，保留失败资格。另存普通资产字段投影：不删除任何初始对象或资产ID、不缩图，只省略提示中的文件路径和已断言为空/中性的登记字段。21823token可编码，180个动作目标token不变。不是新压缩算法，也不证明任意任务语义无损。

二、三层真值必须分开
schema(a)=true只说明接口参数满足格式；exec(s,a)=success只说明实现接受并创建对象；V_c(s')=true只说明声明检查集合c通过。它们都不直接推出用户的真实设计目标H_u(s')。
本轮原生工具接受两张完全重叠的桌子，而FCL检出相交，是exec成功不能推出几何合格的实际反例。
同理，检查器全部运行成功也不证明c正确表达u。此前4个可执行资源程序在一个正例上均可通过，需要有标签的任务族检查才能区别。多数票没有补充外部语义信息；工具回执也没有。
因此整体标签应保留(格式、执行、已核验几何、已核验语义、未知项)向量。未知项不能写成True，也不能误写成False。当前full_scene_success仍为unknown。

三、可主张的条件性正确性
在同一源文件、同一资产、零噪声、相同有序输入与固定数值容差下，本次动作重放得到的两个新对象与普通解几何姿态一致，旧对象状态严格相等。登记库与持久元数据足够重建这份最终bundle，已实际重建。
这个结论只涉及本次有界状态转换，不涉及全部工具、随机状态、SDK调度、物理服务、任意序列或删除后的全局ID唯一性。native generate_unique_id只检查当前对象字典，因此本轮不宣称分支恢复后的身份语义已完整解决。

四、训练目标能与不能说明什么
一份可执行教师tau可以按普通目标L_CE=-sum_t log pi_theta(a_t|x,a_<t)学习，所有提示与图像token的labels=-100。本轮监督180个assistant token，原始__getitem__已经实际执行核对。
这是行为克隆，不需要把普通解伪装成专家思考。选出一个可行解也不代表它在用户未指定的新增物体数、改动规模或风格上最优。
如果确有多个经过来源核验的可行程序，可使用标准集合边际似然-log sum_{p in P_valid(x)}pi_theta(p|x)。这也是已有目标，不是新损失；集合P_valid仍必须来自可信标签和声明检查，不能由无标签投票杜撰。
没有偏好依据时不构造DPO胜负；同一房间的尺寸变体、重放、视角和模板分支均留在同一开发来源族，不算独立泛化证据。

五、与最近邻的实际边界
SceneOrchestra已学习逐步及整条工具轨迹，普通两调用批次不构成方法差异。
Scaling Up and Distilling Down已把语言分解、几何采样、成功函数、重试与蒸馏结合；其训练采集成功和评测成功函数明确分开。把求解器结果重放成数据也不是新贡献。
书中第五章强调工程验证、轨迹完整性和恢复；它们用于提高证据可靠性，不会自动给三维研究增加新颖性。书中可运行代码的论述不能替代任务语义与检查器有效性审查。

六、仍值得验证、但没有证明的新贡献候选
候选问题仍是三维用途解释如何约束实际资源占用：持续保留的物品、不同时间的活动与真实表面/通道共同决定可用设计。需要证明针对这项交互的监督，在未见房间和活动组合上优于普通结构化解析+CSP、直接条件SFT、整轨迹学习和同工具反馈。任务条件变化本身、显式字段、过滤正确解、日志与回滚均已有先例。
目前一份来源族的教师记录只能支撑管线开发；没有证据宣布上述候选已是CVPR主创新，也不能据此否定三维智能体整体目标。
继续工作的必要对象是外部语义依据、竞争方案覆盖与多来源场景上的可检验差异，而不是继续在这份房间上加搜索预算、添加日志名称或立即训练一轮求分数。
'''
    (D/'algorithm_and_theory.txt').write_text(theory)
    controls=[
        ['普通结构化解析＋共同CSP','当前已能生成本例普通可行解','同任务依据、真实资产和检查器','先隔离语义解释与几何执行；本例不证明新方法胜出'],
        ['原生工具回执','实际执行两次增补及三个合成控制','相同固定源码与资产','重叠桌成功回执不能作为正质量标签'],
        ['标准条件SFT／普通教师蒸馏','已完成一个批次的原始编码；未训练','相同180目标token和来源族划分','这是应具备的学习基线，不是新算法'],
        ['SceneOrchestra式整轨迹监督','已有论文与此前源码对照；本轮未训练','同动作库、输入与总执行预算','不可只凭批量工具调用宣称独有全局规划'],
        ['同工具通用反馈','已有开发痕迹，不是本轮重新比较','同观测、工具、资产与累计开销','不能让候选看到额外几何答案再声称策略胜出'],
        ['候选用途／资源交互监督','仍缺多来源的可信语义标签和训练比较','与上述方法共享证据和执行能力','只有泛化收益独立存在才扩大；不得用本轮编码通过冒充效果']]
    with (D/'strong_controls.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['method','actual_status','fairness','interpretation']);w.writerows(controls)
    rows=[['positive_replay',result['wall_seconds'],result['peak_rss_kib'],0,0,0],
          ['full_context_CPU_encoding',full['wall_seconds'],full['peak_rss_kib'],0,0,0],
          ['catalog_projection_CPU_encoding',compact['wall_seconds'],compact['peak_rss_kib'],0,0,0]]
    with (D/'costs.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['phase','wall_seconds','peak_rss_kib','model_forwards','training_updates','paid_API_calls']);w.writerows(rows)
    dump(D/'cost_ledger.json',dict(phases=[dict(zip(['phase','wall_seconds','peak_rss_kib','model_forwards','training_updates','paid_API_calls'],r)) for r in rows],
        measured_phase_seconds_sum=sum(r[1] for r in rows),sum_is_not_end_to_end_research_time=True,
        GPU_model_use=False,training=False,new_downloaded_model_weights=0,new_large_assets=0,
        source_retrieval='Four small ScalingUp Python files in RAM; no root license located, source not copied into deliverable',
        disk_free_bytes_at_report=shutil.disk_usage(R).free,
        cleanup='None this stage: no unique old results or active weights removed; large copies stayed in RAM.'))
    commands='''# Run from repository root. These commands do not run a model or train.
# Initial71resource bundle must first be restored with the existing Milan preparation/asset recovery scripts.
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/runs/autodl/20261110_corpus_state_recovery/dependency/import:/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps"
# Use NEW destinations; never overwrite the recorded run.
/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python scripts/autodl/replay_milan_native_teacher.py --root /dev/shm/teacher_replay_NEW --bundle /dev/shm/milan_completion_20261111/bundle --scratch /dev/shm/teacher_replay_scratch_NEW
/root/autodl-tmp/agent_method_preparation_20261027/work/v100_runtime_20261104/runtime/bin/python scripts/autodl/encode_native_solver_teacher.py --root /dev/shm/teacher_replay_NEW
/root/autodl-tmp/agent_method_preparation_20261027/work/v100_runtime_20261104/runtime/bin/python scripts/autodl/encode_native_solver_teacher.py --root /dev/shm/teacher_replay_NEW --variant catalog_projection
# Reconstruct from persisted metadata without running the two actions again:
python scripts/autodl/restore_native_solver_teacher.py --root runs/autodl/20261111_native_solver_teacher --source-bundle /dev/shm/milan_completion_20261111/bundle --destination /dev/shm/teacher_persisted_restore_NEW
# Initial five images are referenced by SHA and pinned git paths in sft_sample_provenance.json.
# If RAM is lost, restore those same PNG blobs from the listed git revision before encoding.
'''
    (D/'reproduce.txt').write_text(commands)
    links=''.join('<li><a href="'+n+'">'+label+'</a></li>' for n,label in [
        ('actual_tool_trace.json','实际两次放置调用与原始回执'),('tool_acceptance_controls.json','三个明确标注的合成边界控制'),
        ('state_002.json','实际32对象原生状态'),('reconstruction_receipt.json','从持久元数据实际重建的结果'),
        ('original_evaluator_result.json','冻结任务检查器的本轮复算'),('fresh_mesh_contacts.json','实际网格包围盒／接触检查范围'),
        ('training_admission.json','训练标签资格与未知项'),('encoding_result.json','原完整输入超限记录'),
        ('encoding_catalog_projection.json','普通字段投影后的原生数据集编码'),
        ('sft_catalog_projection_full_messages.json','一个真实普通教师批次，两个动作，无思维链'),
        ('reading_ledger.json','实际阅读范围与未完成项'),('strong_controls.csv','强对照及当前完成度'),
        ('algorithm_and_theory.txt','详细算法与条件性结论'),('reproduce.txt','重放、编码与恢复命令'),
        ('protected_source_verification.json','579个旧输入及原生源码哈希核查'),('evidence.zip','本轮小型证据包（不复制大资产和原图）')])
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>原生工具轨迹与普通教师资格</title>
<style>body{{font:17px/1.8 system-ui;max-width:1080px;margin:36px auto;padding:0 24px;color:#17202a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd6dd;padding:9px;text-align:left}}.note{{background:#fff3d6;padding:14px}}a{{color:#125ea0}}pre{{white-space:pre-wrap;font-size:15px}}</style>
<h1>已有完整家具场景的普通解，已通过真实原生放置函数重放并编码</h1>
<p>2026-11-11 · 一个已暴露的Milan开发来源族 · 本轮零模型前向、零训练、零付费API。解决的是数据来源与工具执行衔接，不是新算法效果。</p>
<table><tr><th>项目</th><th>本轮实际结果</th><th>结论边界</th></tr>
<tr><td>真实增补</td><td>两次原生add闭包：dining_table_0、dining_chair_1_0</td><td>保留真正生成ID；仅移除SDK包装，未跑HTTP／完整Actor。</td></tr>
<tr><td>整场保留</td><td>原30对象逐项不变；最终32对象、15家具、17建筑节点</td><td>从已有家具场景增补；没有新的从文字／空房完整生成。</td></tr>
<tr><td>声明检查</td><td>14个原有通行目标，在0.4/0.5/0.6m宽度均可达；独立工作面代理通过</td><td>真实人体使用、材料、隐私与主观分区仍未知。</td></tr>
<tr><td>几何核查</td><td>15家具×45结构及家具间，共780对包围盒均无正重叠</td><td>最终无需窄相FCL；不是物理稳定、实体包含或门扇操作证明。</td></tr>
<tr><td>重叠反例</td><td>原生工具回执success=true；实际FCL相交=true</td><td>故意构造的合成控制，只证明回执不足以作为质量奖励。</td></tr>
<tr><td>保存与恢复</td><td>登记库重载、最终状态重载和撤销均逐项一致；又从持久元数据重建bundle</td><td>未证明全部SDK／随机状态／工作进程恢复，也未解决删除后的全局ID复用。</td></tr>
<tr><td>训练编码</td><td>原始25221token超限；普通目录字段投影21823token，目标均为180token</td><td>后者满足近期24576预算，二者均超原数据集8192默认值；不是训练显存测试。</td></tr></table>
<p>核心重放与几何检查{result['wall_seconds']:.3f}s，峰值进程内存约{result['peak_rss_kib']/1024**2:.2f}GiB。两次CPU编码分别{full['wall_seconds']:.3f}s、{compact['wall_seconds']:.3f}s；这些不包括此前普通解搜索、文献阅读和报告时间，不能当作方法效率优势。</p>
<h2>对算法方向的判断</h2>
<p>可重放的普通示范已成为真实的数据单元，但它仍只有一个开发来源族。必须区分“调用格式合法”“工具接受”“声明的几何检查通过”和“用户真实用途完成”。本轮没有独立的人类偏好或跨场景泛化证据，因此没有宣布CVPR主创新成立。</p>
<p><a href="https://arxiv.org/html/2604.19907v2">SceneOrchestra</a>已有整轨迹学习；<a href="https://arxiv.org/pdf/2307.14535.pdf">Scaling Up and Distilling Down</a>已有语言分解、几何工具、成功函数、重试与蒸馏。原生重放和普通CE应列作共同基线。候选研究点仍需落在用途解释与实际三维资源交互，证明它在新来源房间上超过普通编译／搜索和直接轨迹学习。</p>
<p class="note">本轮既没有证明新机制有效，也没有因此终止完整三维生成研究。当前不需新开计费资源：待补的是可信语义依据与竞争方案覆盖；不是再对这份房间加预算。旧结果、未知项和完整输入超限记录均保留。</p>
<h2>阅读、实现和可复查证据</h2><p>新增完整读出书PDF129–142的14页文字，当前持续会话可追溯直接阅读累计131/312页；不把此前另一会话的全书文字阅读记录冒充本轮重读。未逐页看图，也未逐篇核实书中所有引用。原生工具／数据集源码和ScalingUp源码的具体阅读边界见清单。</p>
<ul>{links}</ul><h2>算法与理论全文</h2><pre>{html.escape(theory)}</pre></html>'''
    (D/'stage_report.html').write_text(body)
    script_names=['replay_milan_native_teacher.py','encode_native_solver_teacher.py',
                  'restore_native_solver_teacher.py','report_native_solver_teacher.py']
    dump(D/'code_manifest.json',[dict(path=str(R/'scripts/autodl'/n),sha256=sha(R/'scripts/autodl'/n)) for n in script_names])
    files=[p for p in sorted(D.iterdir()) if p.is_file() and p.name not in ['evidence.zip','delivery_verification.json']]
    manifest=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in files]
    zpath=D/'evidence.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        for n in script_names:z.write(R/'scripts/autodl'/n,'code/'+n)
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(x['path'])).hexdigest()==x['sha256'] for x in manifest)
        assert all(hashlib.sha256(z.read('code/'+n)).hexdigest()==sha(R/'scripts/autodl'/n) for n in script_names)
    dump(D/'delivery_verification.json',dict(status='verified',files=manifest,
        zip_sha256=sha(zpath),zip_bytes=zpath.stat().st_size,scripts_checked=len(script_names),
        no_large_asset_duplication=True,goal_complete=False))
    print(json.dumps(dict(report=str(D/'stage_report.html'),files=len(files),zip_bytes=zpath.stat().st_size,
        compact_tokens=compact['full_tokens'],goal_complete=False)),flush=True)


if __name__=='__main__':main()
