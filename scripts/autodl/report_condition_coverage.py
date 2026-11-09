"""Render the actual source/data audit and state what it does not establish."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

from audit_material_routing import write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args(); root = args.root
    d = json.loads((root/'actual/condition_coverage.json').read_text())
    verification = json.loads((root/'verification.json').read_text())
    assert verification['status'] == 'passed'
    sources = json.loads((root/'source_review/gala_sources.json').read_text())
    for source in sources['files']:
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    pipeline = root/'source_review/gala/agent/pipeline.py'
    tree = ast.parse(pipeline.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_infer_and_validate')
    assignments = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'max_instruction_retries' for t in n.targets)]
    assert len(assignments) == 1 and isinstance(assignments[0].value, ast.Constant)
    guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                 and 'should_regenerate and instr_attempt < max_instruction_retries - 1' == ast.unparse(n.test))
    limit = assignments[0].value.value
    # A finite check of this loop bound/guard, not execution of GALA or its tools.
    reachable = [i for i in range(limit) if i < limit-1]
    code_facts = {'pin': sources['pin'], 'source_sha256': hashlib.sha256(pipeline.read_bytes()).hexdigest(),
                  'method': fn.name, 'instruction_retry_limit': limit,
                  'assignment_line': assignments[0].lineno, 'regeneration_guard_line': guard.lineno,
                  'loop_indices_passing_regeneration_guard': reachable,
                  'interpretation': 'At this pinned source version the instruction regeneration branch is unreachable under its constant bound. This is not evidence about the paper experiment configuration.',
                  'source_execution_performed': False, 'model_calls': 0}
    write(root/'gala_source_control_flow.json', code_facts)
    readings = [
        {'title': 'GALA: Geometry-Aware Language Model for Controllable Object Arrangement',
         'url': 'https://arxiv.org/html/2610.28280v1',
         'read': 'Abstract, introduction, §§2–3.4, §4.1 and the beginning of §4.2; targeted code ranges listed separately.',
         'not_read': 'Full experimental tables, all appendices and prompts; no model execution.',
         'finding': 'Learns numeric 6D insertion from point clouds and relation instructions using ordinary autoregressive likelihood; planner and instruction generator provide a generation loop. Its primary quantitative task is insertion, not professional activity-dependent whole-home preference.',
         'method_implication': 'Point-cloud grounding, geometry-conditioned placement and plan/place/verify are existing capabilities. Our hypothesis must concern when a design decision applies, beyond better pose prediction.',
         'source_pin': sources['pin'],
         'code_read': {'agent/pipeline.py': ['640–1085'], 'agent/placement_validator.py': ['87–152', '322–361'],
                       'dataset_pipeline/generation/01_generate_training_data.py': ['1–58'],
                       'utils/message.py': ['396–410'],
                       'third_party/LLaMA-Factory/examples/train_full/qwen3-0_6b/3dish_separate_pointcloud_331k_train/base.yaml': ['complete'] }},
        {'title': 'Scene layout via conceptual design',
         'url': 'https://xccelephant.github.io/assets/pdf/scene-layout-via-conceptual-design.pdf',
         'read': '§§3–6 (PDF pp. 3–6), related-work passage and beginning of §7; equation/algorithm text.',
         'not_read': 'All quantitative results, supplement, implementation, figures as images.',
         'finding': 'Hierarchical requirements become geometric constraints. ComputeIIS-derived modification plans are explained semantically and an LLM selects a conflict resolution; forced constraint removal supports termination.',
         'method_implication': 'Selecting a constraint or upstream decision to change after feedback already has direct scene-layout precedent. Termination by removing constraints must not be confused with retaining all user requirements.'},
        {'title': 'Scene-Designer: Learning Design Intelligence for Agentic 3D Scene Generation',
         'url': 'https://arxiv.org/html/2610.23121v1',
         'read': 'This turn: targeted §3.1 paragraphs on trajectory consolidation, action-conditioned reasoning and perturbation supervision. Earlier wider reads are not claimed again.',
         'not_read': 'No new full-paper or full-repository re-read, no native execution.',
         'finding': 'The published method already uses observation–reasoning–action supervision; reasoning is constructed with the retained target action available.',
         'method_implication': 'Legitimate teacher supervision is not test-time evidence. Merely adding a reason before a tool action is insufficient differentiation.'},
        {'title': 'DesignAgent3D: Interactive 3D Scene Editing via Designer-like Multimodal Reasoning',
         'url': 'https://arxiv.org/html/2608.21438v1',
         'read': 'Abstract, introduction and related-work overview.',
         'not_read': 'Full method, experiments, code and publication status verification.',
         'finding': 'The stated task is interactive, geometry-preserving appearance editing of reconstructed scenes, using clarification and object grounding.',
         'method_implication': 'Relevant for intent grounding, but this overview does not establish it as a direct trained furniture-layout decision baseline.'}
    ]
    write(root/'reading_ledger.json', {'recorded_at': datetime.now(timezone.utc).isoformat(), 'entries': readings,
          'scope': 'Targeted primary-source review, not exhaustive novelty clearance; public Git metadata/code downloads are not paid model calls.'})
    method = {
        'status': 'Existing paired-supervision hypothesis retained; no new method name or unsupported novelty claim.',
        'variables': {'U': 'Current requested activities and non-negotiable requirements.',
                      'S': 'Actual 3D state and available measurements; history is a separate variable.',
                      'K': 'The same attributed source knowledge available to every information-matched arm.',
                      'D': 'Proposed applicability/decision record, not a verified tool receipt.',
                      'A': 'Tool action or stop proposal.', 'E': 'Post-execution requirement and preservation checks.'},
        'core_training_control': {'separate': 'L(D|U,S,K)+L(A|U,S,K)',
                                  'paired': 'L(D|U,S,K)+L(A|U,S,K,D)',
                                  'implementation': '../20261110_decision_phase_study/algorithm_and_theory.json',
                                  'weights': 'Preserve all original target tokens and coefficients; missing reasons remain missing.',
                                  'test': 'Generate D without target action or reference endpoint; count every pass and evaluate executed consequences.'},
        'supported_data_inference': 'One existing task×state block rules out task-only or state-only deterministic classification of every edit/stop label. It does not imply a learned reason is necessary: the ordinary input-measurement rule already classifies supported states.',
        'not_identified': ['Whether pairing improves deployment over separate learning.',
                           'Whether a different demonstrated action is wrong; multiple valid layouts and paths are possible.',
                           'Professional preference or complete-home success from source attribution and local proxy checks.',
                           'Cross-project generalization from repeated states in one family.'],
        'next_data_requirement': 'Use existing source-linked families to find both successful and limited applications under independently checked spatial conditions. Preserve ordinary-rule-solvable cases; do not manufacture reversal labels or import hindsight into inputs.',
        'next_execution': ['Obtain the final receipt of the already started 920 LEARN run before any duplicate launch.',
                           'Recover exact missing source views and validate full training accumulation and generated-decision deployment.',
                           'Run the same-material control only when complete inputs and whole-room endpoint checks are available; include same-knowledge feedback and ordinary measurement/search controls.'],
        'resource_status': 'No new GPU request, model call, training update or weight download in this turn.',
        'full_room_evaluation': ['required objects and identities', 'current activity completion', 'preservation of still-active requirements',
                                 'geometry failures and unknowns', 'total inference/execution cost'],
        'stop_expansion_if': 'Improvement remains only task-word mapping, termination formatting, or a gain fully explained by ordinary declared rules. Keep the data and execution foundation without declaring the entire 3D research invalid.'}
    write(root/'method_position.json', method)
    rows = []
    for key, ctl in d['controls'].items():
        count = ctl.get('correct_at_most', ctl.get('correct'))
        rows.append(f'<tr><td>{escape(key)}</td><td>{count}/{ctl["n"]}</td><td>{"实际规则复算" if key == "ordinary_declared_gate" else "使用标签计算的有限记录上限，不是模型成绩"}</td></tr>')
    pairs = ''.join('<li>'+escape(' ↔ '.join(p['ids']))+'</li>' for p in d['same_geometry_pairs'])
    papers = ''.join(f'<li><a href="{escape(r["url"])}">{escape(r["title"])}</a>：{escape(r["method_implication"])}</li>' for r in readings)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>设计决定的条件与执行证据</title>
<style>body{{font:17px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#152332}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border:1px solid #ccd4dc}}code{{font-size:14px}}aside{{background:#edf4f9;padding:18px}}a{{color:#135799}}</style>
<h1>设计决定的条件与执行证据</h1><p>2026-11-10 · 固定 Git 来源 <code>{d['pin']}</code> · 本轮零模型调用、零训练、零场景编辑。</p>
<aside>现有材料确实包含用途与状态的交互，但普通量测规则已解释已支持的编辑／停止标签。下一步仍应检验同材料配对学习能否改善实际完整场景；这里没有确立新的算法或专业偏好收益。</aside>
<h2>实际完成的核查</h2><p>逐项读取并绑定全部 24 个状态、6 个项目家族。8 处来源理由对应 5 项原则；22 个动作状态链接到 8 份已存执行后果记录。2 个终止状态另有输入中的当前量测与终态来源引用。复核器重新校验 81 份来源哈希，并从原始输入重算条件表。专业原则有出处，具体位姿及操作次序仍主要是研究者重建。</p>
<p>来源结果保存了实际执行与副作用，但整段执行后果不能归因于其中任意一步。是否存在原生回执、是否通过局部规则、是否获得专业整屋认可，是不同字段。本轮没有重跑物理引擎。</p>
<h2>用途与状态是否真的都需要</h2><p>找到 3 对对象表完全相同、活动条件不同的记录。比较使用对象 ID、包围盒和变换的原始数值，不做近似合并；这不代表图像、历史和完整环境均相同。</p><ul>{pairs}</ul>
<p>其中 Waterloo 有一个完整的两用途×两状态表。4 个端点记录中，仅看用途或仅看状态最多符合 2/4 个编辑／停止标签。连同 2 个中间状态，下表的分母为 6。米兰那对仅说明示范动作不同，不能推出另一动作无效。</p>
<table><tr><th>条件</th><th>符合记录标签</th><th>性质</th></tr>{''.join(rows)}</table>
<p>普通规则读取与模型相同的公开要求和量测，不读取训练答案；6/6 是实际复算。其余 18 个状态返回未知。它没有产生坐标，也没有完成房间。这个结果支持设置联合条件对照，同时反对把这 6 个标签当作配对学习必要性的证明。</p>
<h2>近邻方法与源码边界</h2><ul>{papers}</ul>
<p>GALA 源码固定在 <code>{sources['pin']}</code>。本次读到的放置分支把指令重试上限固定为 1，随后重生成守卫要求循环索引小于 0，因此此版本该分支不可达。这里只核了控制流；没有运行 GALA，也不能由此否定论文成绩。保留 Apache-2.0 LICENSE 与 NOTICE。</p>
<h2>算法和理论的实际位置</h2><p>当前可执行假设仍是相同理由、动作和信息下，比较分开学习与正确配对学习。监督损失是普通自回归交叉熵，几何与规则是共同工具。拟议收益必须来自把经验适用条件落实到当前对象和真实执行结果，而不能来自更多监督、答案前缀、不同工具或更宽的预算。</p>
<p>有限表格只能证明此处两类输入都与标签有关；它不能证明学习、理由生成或新的因果机制不可替代。测试时的理由必须由模型生成，原任务与实际回执保持可见，最终用执行后的完整需求和保留情况评分。模型学到一种示范，不等于其他合法设计错误。</p>
<h2>尚未完成</h2><p>扩展公开图片查找在约 40.8 MB 的读取预算处结束，仍为 41/109 个唯一图像精确匹配、5/24 个状态五视图齐全。扫描未覆盖所有公开图像，不能断言其他图像不存在。新训练器完整集成、两阶段模型推理、独立项目整屋收益均未完成；本轮没有启动重复训练或要求开启资源。</p>
<p>下一个实质动作是收齐已启动训练的结果，并补足同材料实验的真实输入与执行闭环；不再用同一房间的更多接口通过数代替学习证据。</p>
<h2>可复核文件</h2><ul><li><a href="actual/condition_coverage.json">24 状态及执行后果记录</a> · <a href="actual/condition_coverage.csv">逐项 CSV</a></li><li><a href="verification.json">独立复算与来源哈希核验</a></li><li><a href="reading_ledger.json">论文和源码实际阅读范围</a> · <a href="method_position.json">方法与继续条件</a></li><li><a href="gala_source_control_flow.json">GALA 控制流检查</a></li><li><a href="published_image_search/image_availability.json">公开图像查找记录</a></li></ul></html>'''
    (root/'stage_report.html').write_text(html)
    (root/'COMMANDS.txt').write_text('''PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/audit_condition_coverage.py --repo . --output runs/autodl/20261110_conditional_evidence/actual
PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/verify_condition_coverage.py --repo . --audit runs/autodl/20261110_conditional_evidence/actual/condition_coverage.json --output runs/autodl/20261110_conditional_evidence/verification.json
PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/report_condition_coverage.py --root runs/autodl/20261110_conditional_evidence
''')
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    for name in ('audit_condition_coverage.py','verify_condition_coverage.py','report_condition_coverage.py'):
        p = Path('scripts/autodl')/name; manifest[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    write(root/'manifest.json', {'files': manifest, 'old_experiment_files_modified': False,
                               'model_calls': 0, 'training_updates': 0})
    print(json.dumps({'report': str(root/'stage_report.html'), 'manifest_files': len(manifest), 'gala_branch_reachable': bool(reachable)}))


if __name__ == '__main__':
    main()
