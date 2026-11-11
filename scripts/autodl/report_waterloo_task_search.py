"""Assemble the second-project CPU result and preserve its research limits."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args(); root = a.root.resolve()
    s = read(root/'summary.json'); bounds = read(root/'action_count_lower_bound.json')
    qualifications = dict(
        independent_agent_comparison_qualified=False,
        reason='B1 executes the same task-aware solver with a manually declared equivalent case factor set; no automatic case interpretation, binding or Agent policy was run.',
        ordinary_baseline_actually_executed=True,
        repeated_execution_on_identical_constraints=True,
        case_information_value_identified=False,
        task_bindings='Researcher-authored table IDs, seat assignments and functional thresholds are public to both arms.',
        inference_allowed='The explicit task already entails this declared case strategy, and ordinary search reaches the action-count floor of this finite protocol.',
        inference_forbidden=['Case learning is useless', 'Agents cannot plan',
                             'Novel learned allocation method tested', 'Original Waterloo residence functional success',
                             '55 exact mesh/access evaluations per arm', 'Complete new scene generated from text'])
    dump(root/'comparison_qualification.json', qualifications)
    reading = [
        dict(title='Motion Planning for Convertible Indoor Scene Layout Design',
             url='https://johnfu1988.github.io/publication/TVCG20/index.html',
             read_scope='Author project abstract: source/reference correspondence, target layout, joint space-time motion optimization.',
             limit='PDF fetched into RAM; browser access and local pdftotext failed. Do not claim full-paper reading or numerical reproduction.',
             consequence='Furniture merging, convertible layouts and their motion planning already have direct prior art.'),
        dict(title='Function2Scene: 3D Indoor Scene Layout from Functional Specifications',
             url='https://arxiv.org/html/2605.30819v1',
             read_scope='Sections3-4, 5.1-5.2, conclusion; appendix C.3 tool signatures. Not all long appendix prompts/figures inspected.',
             consequence='Persona/activity-conditioned constraints, typed tools and priority-based repair are established baselines; professional briefs and human study are different evidence from this synthetic transfer.',
             limit='No source repository executed, original scenes reproduced, or independent human evaluation.'),
        dict(title='Waterloo historical development source',
             commit='75fc783f86d4fea78714639e110fbe77476ee106',
             read_scope='Source record, protocol, build plan/reference, complete builder, complete table-mode evaluator; existing access and visual-mesh functions.',
             split='train_source_candidate/development, not held-out',
             limit='Public professional strategy; scene geometry, occupancy, task binding and paths are researcher-authored.'),
        dict(title='Training target follow-through',
             source='runs/autodl/20261111_native_contract_calibration/algorithm_and_evidence.txt',
             read_scope='Read current saved source/qualification review; no claim of a new full Scene-Designer source audit.',
             consequence='A hard-constraint optimum supplies no independent human preference label. No new learned ranking objective is justified by this example.')]
    dump(root/'reading_ledger.json', reading)
    theory = '''算法与理论边界：Waterloo 第二开发项目（CPU 实跑）

1. 实际完成
从冻结源码与原始资产恢复完整声明场景，原初始状态逐值一致；27个对象全部保留。
原生RoomScene中实际提交桌子/椅子的2次变换，再反向提交2次，返回状态逐值一致。
两个执行标签分别独立完成相同基线，不是两个不同的学习方法。初态和待客态已导出真实SDF视觉三角形GLB并CPU渲染。
原生对象赋值不等于官方HTTP工具调用；本轮没有Agent、模型推理、训练或从文字新生成整间场景。

2. 强普通方法
公共任务已明确两张桌子合并、座位绑定、通行点与阈值。用当前资产边界产生55个有限联合候选，
共同便宜筛查全部55个，用几何违反数/改动量排序，再对入选候选做原冻结模式、通行、网格相交、运动扫掠检查。
两种单对象移动顺序均检查；联合候选不要求每个中间状态改善目标，但路径和活动可达必须保留。
实际每个标签只需1个完整验证，两者均选42号，最终状态哈希一致。
55个便宜评分是共享预处理，不能省略；也不能写成每组55次完整网格/通行求解。

3. 可证明的局部下界
对固定55个候选逐一枚举，12个满足便宜的待客条件，每个都改变桌子B和椅子B两个对象。
在本协议的一次只修改一个对象的原子操作下，到这些待客终态至少2次；要求精确返回再至少2次。
因此有限候选池上的提交次数下界为4，普通方法已经达到4。不是任意旋转、连续路径或真实物理推拉的全局最优证明。
每个独立、尚未认证的入选终态必须做至少1次协议要求的完整验证；当前已达到这项计数下界。
这说明此例不再能检验“更少提交动作/更少完整终态验证”的收益；便宜筛查开销仍可优化，但不自动构成新机制。

4. 案例信息什么时候才可能有价值
设公共任务和几何约束为C(u,G)，案例解释给出附加条件phi(z)。若在候选集合中C蕴含phi，
则{(z,x):C且phi}与{(z,x):C}相同。没有新增可行性信息，仍可能改变搜索顺序；必须相对强普通传播/启发式计总成本。
若C不蕴含phi，案例会排除原来允许的设计；需要用途证据或独立偏好确认，不能把设计师某个结果直接当作唯一真值。
这是普通约束满足的逻辑事实，不是新定理或新算法。本例的“可分合桌子”已经在公共任务中明说，故手工案例因子重复。
当前B1没有独立实现自动案例抽取/绑定，不能把两次同算法运行称为Agent对照，也不能据此否定案例学习。

5. 训练目标为何暂不启动
本例可提供几何条件标签，不能提供“更适合真实使用者”的独立偏好。全部硬检查能够决定的比较，不足以识别软设计取舍。
需要在同等可行的完整场景之间，确认用途和空间条件带来的差别，或证明自动任务绑定比普通匹配/检索更好；当前均未具备。
不能把人工作用角色、几何阈值和普通规则标成训练创新。现有证据不足以训练新Critic/策略，或扩大昂贵生成批次。

6. 下一步的具体证据要求
保留该场景作为强普通基线和完整资源包，不继续在它上面调整到某个Agent获胜。
下一项可检验环节是：从原始用途与真实资产中自动形成对象—活动绑定，再由共同几何后端生成完整场景。
先找已有真实生成中“几何可完成，但绑定/用途解释不同”的记录；没有这种记录就不制造错误或训练新模块。
对照必须包括明确任务条件的普通规则/约束传播、普通代码感知检索、同模型通用Agent；相同资产和上游动作，计完整成本。
Function2Scene已经覆盖用途解析与功能约束修复，PlanIT式关系约束和TVCG2020转换规划也已存在；上述候选位置尚不是新颖性声明。

限制：研究者迁移房间8x7m，不是原住宅；7个网格实例含6个生成桌椅替代物，11个家具为代理，9个建筑对象。
FCL是表面相交检查且有1e-4m正AABB重叠预筛，非实体穿透深度/包含关系证明。平方身体网格、座位矩形及间隙是操作约定。
未验证脚轮刹车、载荷、坐姿、舒适、材料语义、光声环境或用户偏好。完整场景生成成功保持未知。
'''
    (root/'algorithm_and_theory.txt').write_text(theory)
    dump(root/'training_label_audit.json', dict(
        finite_geometric_cheap_labels=55, hosting_cheap_pass=12,
        independent_human_preference_labels=0,
        independent_heldout_cases=0, automatic_activity_binding_labels=0,
        new_SFT_or_RL_dataset_created=False, training_qualified=False,
        reason='A synthetic proxy-feasibility witness does not establish human-conditioned design preference or a new policy advantage.'))
    reproducibility = '''运行位置：repository 根目录。原有文件只读，不安装新环境、不下载模型。
原生解释器：/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python
环境变量：PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH=/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps

已实际执行：
python scripts/autodl/prepare_waterloo_task_search.py --root runs/autodl/20261111_waterloo_task_search --scratch /dev/shm/waterloo_task_search_reproduction_20261111 --native-source runs/autodl/20261110_native_state_runtime/source
python scripts/autodl/waterloo_task_search.py --root runs/autodl/20261111_waterloo_task_search --bundle /dev/shm/waterloo_task_search_20261111/native_build/bundle --native-source runs/autodl/20261110_native_state_runtime/source
python scripts/autodl/export_waterloo_task_search.py --root runs/autodl/20261111_waterloo_task_search --bundle /dev/shm/waterloo_task_search_20261111/native_build/bundle --out /root/autodl-tmp/agent_method_preparation_20261027/work/waterloo_task_artifacts_20261111

原结果目录含protocol.json，执行器拒绝覆盖/自动重跑。复现比较应复制source和source_manifest到新结果目录，指定新的scratch。
prepare脚本仅依赖已在本地的原始HSSD沙发（固定SHA）与Git恢复的两类生成桌椅，不需要GPU。
完整原生输入包和两种终态GLB位于geometry_artifact_manifest.json / native_bundle_archive.json的绝对路径。
GLB是Y-up可视几何预览；原生状态是Z-up米制。CPU渲染器已处理Blender导入时的Y→Z转换。
最早两个侧视图保留为initial_preview/hosting_preview，纠正相机后另存topdown，没有改动布局或检查结果。
'''
    (root/'commands_and_reproduction.txt').write_text(reproducibility)
    rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(x[k]))+'</td>' for k in [
        'arm','evaluated_candidates','committed_single_object_moves','preserved_objects',
        'hosting_conditional_pass','return_state_exact','cpu_wall_seconds'])+'</tr>' for x in s['arms'])
    content = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Waterloo 第二开发项目 CPU 实跑</title>
<style>body{{font:17px/1.65 sans-serif;max-width:1120px;margin:35px auto;padding:20px;color:#18212b}}table{{border-collapse:collapse;font-size:14px}}td,th{{border:1px solid #ccc;padding:8px}}img{{width:48%;vertical-align:top}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:20px}}.limit{{background:#fff1ce;padding:15px}}</style>
<h1>完整声明场景已恢复；普通方法达到本协议动作下界</h1>
<p>这是第二个完整三维开发项目的CPU执行结果。原始27对象逐值恢复，实际原生变换及精确返回已完成，初态与待客态已导出真实视觉三角形GLB。新增场景生成、模型调用、训练均为0。</p>
<p class="limit">B1是人工声明等价案例因子的同一普通算法复跑，不是实际Agent或学习方法对照。本轮支持普通基线足够处理这条明确任务；没有识别案例信息的独立价值，不能推广为“案例/Agent无用”。</p>
<table><tr><th>执行标签</th><th>完整候选验证</th><th>提交对象变换</th><th>保留对象</th><th>待客几何条件</th><th>精确返回</th><th>CPU秒</th></tr>{rows}</table>
<p>共同预处理实际评分55个候选，其中12个满足便宜待客谓词；每标签1个入选候选进入完整检查。两种顺序的中间可达与扫掠也计入实际本地计算。总研究执行器耗时{s['wall_seconds']:.3f}秒，峰值RSS约{s['peak_rss_kib']/1024:.1f}MiB。另计冷恢复、导出与CPU渲染，见各receipt。</p>
<p>有限候选池中的4次原子变换下界：待客终态的桌子B与椅子B都必须改变，精确恢复也必须分别返回。普通方法达到下界，两个终态状态SHA一致。本证明只适用于当前池、原子动作和操作约定。</p>
<h2>实际几何预览</h2><p>左：原始分桌工作；右：合桌待客。代理/材料简化，视图不代表真实使用或偏好验证。结构和顶灯仅在切开预览中隐藏，原生资源与GLB均保留。</p>
<img src="initial_topdown.png" alt="分桌工作"><img src="hosting_topdown.png" alt="合桌待客">
<h2>可复核证据</h2><ul>
<li><a href="summary.csv">逐标签CSV</a>、<a href="common_candidates.json">55候选</a>、<a href="action_count_lower_bound.json">有限动作下界</a></li>
<li><a href="comparison_qualification.json">比较资格与不允许的推论</a>、<a href="algorithm_and_theory.txt">算法与理论边界</a>、<a href="reading_ledger.json">实际阅读记录</a></li>
<li><a href="reproduction_input_checks.json">34个原生资源复现逐字节一致</a>、<a href="protected_input_verification.json">579个旧输入未变</a></li>
<li><a href="geometry_artifact_manifest.json">两个完整GLB路径/哈希</a>、<a href="native_bundle_archive.json">自包含原生输入压缩包</a>、<a href="commands_and_reproduction.txt">运行命令</a></li></ul>
<h2>最近邻与研究判断</h2><p>家具合并及转换运动已有<a href="https://johnfu1988.github.io/publication/TVCG20/index.html">TVCG2020直接前作</a>；用途解析、功能约束与检查修复已有<a href="https://arxiv.org/html/2605.30819v1">Function2Scene</a>。仅增加案例提示或用途角色绑定，不足以承担新贡献。</p>
<p>下一项机制必须先在既有真实生成中找到“正确绑定仍未由普通方法解释”的可执行证据，再检验自动决策。当前不应在此简单样例上增加训练或GPU投入，CVPR目标仍未达成。</p>
<h2>全部操作边界</h2><pre>{html.escape(theory)}</pre></html>'''
    (root/'stage_report.html').write_text(content)
    code = [p for p in Path('scripts/autodl').glob('*waterloo*.py')]
    dump(root/'code_manifest.json', [{'path':str(p),'sha256':sha(p)} for p in code])
    with zipfile.ZipFile(root/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name not in ['evidence.zip','delivery_manifest.json']:
                z.write(p,p.relative_to(root))
        for p in code:
            z.write(p,'code/'+p.name)
    selected = ['stage_report.html','summary.csv','evidence.zip']
    dump(root/'delivery_manifest.json', dict(files=[dict(path=str((root/n).resolve()),
        bytes=(root/n).stat().st_size,sha256=sha(root/n)) for n in selected],
        large_geometry_in_zip=False,
        geometry_absolute_paths_in='geometry_artifact_manifest.json and native_bundle_archive.json',
        free_data_disk_bytes=shutil.disk_usage(root).free,
        overall_research_goal_completed=False))
    print(json.dumps(dict(report=str(root/'stage_report.html'), evidence_bytes=(root/'evidence.zip').stat().st_size,
                         ordinary_four_move_baseline=True, independent_agent_comparison=False)), flush=True)


if __name__ == '__main__':
    main()
