"""Report frozen primary outcomes and explicitly post-hoc literal selections.

Never changes an output, dispatches a choice, or replaces prospective scores.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from shared_action_contract import strict_loads


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def literal_selection(raw):
    result = dict(payload_readable=False)
    try:
        if raw.count('<tool_call>') != 1:
            raise ValueError('Not exactly one explicit selection block')
        text = raw.split('<tool_call>', 1)[1].lstrip()
        _, end = json.JSONDecoder().raw_decode(text)
        call = strict_loads(text[:end])
        tail = text[end:].strip()
        if tail not in ('', '</tool_call>', '<|im_end|>', '</tool_call><|im_end|>'):
            raise ValueError('Unexpected trailing material')
        if set(call) != {'name', 'arguments'} or call['name'] != 'select_layout':
            raise ValueError('Unexpected JSON fields or tool name')
        arg = call['arguments']
        if not isinstance(arg, dict) or set(arg) != {'candidate_id', 'reason', 'unknowns'}:
            raise ValueError('Unexpected arguments')
        if arg['candidate_id'] not in ('A', 'B', 'undetermined') or any(not isinstance(arg[k], str) for k in ('reason', 'unknowns')):
            raise ValueError('Invalid selection values')
        result.update(payload_readable=True, **arg, closing_tag_present='</tool_call>' in tail, eos_marker_present='<|im_end|>' in tail)
    except (ValueError, TypeError, KeyError) as exc:
        result['error'] = str(exc)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args(); p = a.run.resolve()
    protocol = read(p / 'PROTOCOL.json'); summary = read(p / 'SUMMARY.json'); status = read(p / 'STATUS.json')
    assert status['status'] in ('completed', 'failed')
    assert sha(Path(__file__))
    primary = {r['id']: r for r in summary['results']}
    rows, model_calls, elapsed = [], 0, 0.
    for cell in protocol['cells']:
        directory = Path(cell['plan']['path']).parent
        if not (directory / 'model/RESULT.json').exists(): continue
        model = read(directory / 'model/RESULT.json')
        model_calls += model['model_calls']
        receipt = read(directory / 'receipt/receipt.json')
        elapsed += receipt['elapsed_seconds']
        rawpath = directory / 'model/raw_output.txt'
        row = dict(id=cell['id'], arm=cell['arm'], activity=cell['activity'], order=cell['order'], label_map=cell['label_map'],
            primary_correct=primary.get(cell['id'], {}).get('correct', False), primary_valid=primary.get(cell['id'], {}).get('valid_response', False),
            model_calls=model['model_calls'], exit_code=receipt['exit_code'], input_tokens=model.get('encoding', {}).get('input_tokens'),
            output_tokens=model.get('output_tokens'), generation_seconds=model.get('generation_seconds'))
        if rawpath.exists():
            literal = literal_selection(rawpath.read_text())
            literal['selected_endpoint'] = cell['label_map'].get(literal.get('candidate_id'))
            literal['matches_frozen_activity'] = literal['selected_endpoint'] == cell['expected_endpoint']
            row.update(raw_output=str(rawpath), raw_output_sha256=sha(rawpath), posthoc_literal=literal)
        rows.append(row)
    assert model_calls == summary['actual_model_calls']
    arms = {}
    for arm in summary['arms']:
        selected = [r for r in rows if r['arm'] == arm]
        by_key = {(r['activity'], r['order']): r.get('posthoc_literal', {}) for r in selected}
        consistent = {activity: len([by_key.get((activity, order), {}).get('selected_endpoint') for order in ['forward', 'reversed'] if by_key.get((activity, order), {}).get('selected_endpoint')]) == 2
            and by_key[(activity, 'forward')]['selected_endpoint'] == by_key[(activity, 'reversed')]['selected_endpoint'] for activity in ['reserve', 'everyday']}
        arms[arm] = dict(primary=summary['arms'][arm], posthoc_literal_correct=sum(r.get('posthoc_literal', {}).get('matches_frozen_activity', False) for r in selected),
            posthoc_payload_readable=sum(r.get('posthoc_literal', {}).get('payload_readable', False) for r in selected),
            label_order_consistency=consistent, total_cells=len(selected))
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), status=status['status'], actual_model_calls=model_calls,
        model_subprocess_seconds=elapsed, native_edits=0, training_updates=0, arms=arms, rows=rows,
        protocol_sha256=sha(p / 'PROTOCOL.json'), primary_summary_sha256=sha(p / 'SUMMARY.json'),
        posthoc_review_plan=read(p / 'SECONDARY_REVIEW_PLAN.json'), report_script_sha256=sha(Path(__file__)),
        claim='Training-exposed supplied-endpoint recognition only; no new scene, professional label, independent project or generation benefit.')
    with (p / 'REVIEW.json').open('x') as f: json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    lines = ['# 固定活动与选项顺序的方案识别诊断', '',
        '在已用于训练的 Paper 研究重建中，固定当前房间五视图及两份已实际执行的终态，仅改变当前活动和选项顺序。各模型均得到同一六条来源知识、完整对象几何和候选替换记录。没有原生编辑或新训练。', '',
        '| 模型 | 严格主判据正确 | 主格式合法 | 事后读取的明确选项正确 | 两种活动各自换序一致 |',
        '| --- | ---: | ---: | ---: | --- |']
    for arm, result in arms.items():
        n = result['total_cells']; primary_arm = result['primary']
        lines.append(f"| {arm} | {primary_arm['correct']}/{n} | {primary_arm['valid']}/{n} | {result['posthoc_literal_correct']}/{n} | {result['label_order_consistency']} |")
    lines += ['', '**严格判据与事后文字读取分开。** 冻结主判据要求完整闭合的工具块，但提示未明确写出闭合标记；不能把这种格式不一致解释成专业知识缺失。事后读取只记录输出 JSON 中原样写出的选项，不根据解释替模型改答案。SECONDARY_REVIEW_PLAN.json 记录追加此审阅时已完成的调用数；它不替代主判据。', '',
        '| 条件 | 已测日常终态 | 已测腾空终态 |', '| --- | --- | --- |',
        '| 两人同时用餐 | 2 个声明餐位通过 | 0 个声明餐位通过 |',
        '| 访客预留区无家具 | 桌占用约 0.842 平方米 | 预留区清空 |', '',
        '两终态来自原 Paper 第 6、8 步。候选名称与上述评分没有提供给模型；输入提供候选几何、世界包围盒及研究者规定的几何阈值。保留对象与端点包络条件相同；实际搬运、沙发床展开、人体舒适及完整住宅质量仍未知。当前五视图只展示第 6 步，不代表两个候选均提供了五视图。', '',
        f"真实调用 {model_calls}/12，模型子进程累计 {elapsed:.3f} 秒，批次状态 {status['status']}；每次上限 180 秒、768 新 token，0 重试、0 新优化、0 场景动作。真实启动与退出见 run_receipt/receipt.json。", '',
        '这是一个项目的四种输入条件，不是四个独立项目。两适配器都见过其布局和动作；给定候选及数值条件的识别远易于自主设计。即使全部选对，也不能证明完整房间生成或专业经验学习有效。不同任务格式、标签绑定和语法均可能影响结果，不能凭本诊断确定旧整屋失败的单一原因。', '',
        '按冻结规则，本批到此停止，不调提示、不换种子、不追加 Paper 生成。原十二房间主试验保持不变。原始输出、逐项主判分、事后明文读取与哈希见 cells/、SUMMARY.json 和 REVIEW.json。', '']
    with (p / 'README.md').open('x') as f: f.write('\n'.join(lines))
    print(json.dumps(dict(actual_model_calls=model_calls, arms=arms, model_subprocess_seconds=elapsed)))


if __name__ == '__main__': main()
