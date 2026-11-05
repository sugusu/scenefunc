"""Summarize completed room attempts without inventing a quality score."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--summary-plan', type=Path)
    args = parser.parse_args()
    public, runtime = args.public, args.runtime
    assert digest(Path(__file__)) == read(args.summary_plan or public / 'SUMMARY_PLAN.json')['sha256']
    batch = read(runtime / 'BATCH.json')
    assert [x['arm'] for x in batch['arms']] == ['base', 'separate', 'paired']
    for stage in ['batch_execution', 'postprocess_execution']:
        receipt = read(runtime / stage / 'receipt.json')
        assert receipt['status'] == 'completed' and receipt['exit_code'] == 0
    rows, bindings, trajectories = [], {}, {}
    for arm in ['base', 'separate', 'paired']:
        folder = public / arm
        assert digest(runtime / arm / 'rollout/round_000/request.json') == batch['shared_initial_request_sha256']
        report = read(folder / 'REPORT.json')
        schema = read(folder / 'ACTION_SCHEMA_REVIEW.json')
        restore = read(folder / 'cold_restore/RESULT.json')
        visual = read(folder / 'VISUAL_REVIEW.json')
        assert report['actual_exit_code'] == 0 and len(report['requirements']) == 9
        assert restore['status'] == 'terminal_native_session_restored_without_edits'
        assert restore['receipt_count'] == report['tool_calls']
        assert visual.get('all_five_views_inspected', visual.get('views_all_inspected')) is True
        assert len(visual['view_paths']) == 5
        roles, access = {}, {}
        for requirement in report['requirements']:
            roles[requirement['id']] = requirement.get('available_catalog_role_counts', {})
            for scenario in requirement.get('conditional_access', []):
                width = str(scenario['square_body_width_m'])
                for target, measurement in scenario['targets'].items():
                    key = target + '@' + width
                    value = measurement['reachable_from_entry']
                    assert key not in access or access[key] == value
                    access[key] = value
        row = {key: report[key] for key in [
            'arm', 'actual_exit_code', 'actual_started_at', 'actual_ended_at',
            'actual_seconds', 'stop_reason', 'model_calls', 'tool_calls',
            'generation_seconds', 'output_tokens', 'peak_allocated_bytes',
            'furniture_count', 'inventory_counts', 'triangle_contacts',
            'outside_nominal_interior', 'whole_room_accepted', 'glb']}
        row.update(
            schema_mismatch_calls=schema['schema_mismatch_calls'],
            action_counts=schema['counts'],
            immutable_records_unchanged=all(report['immutable_records_unchanged'].values()),
            absent_catalog_roles_by_requirement={k: [r for r, n in v.items() if n == 0]
                                                for k, v in roles.items()},
            conditional_access=access,
            native_restore_passed=True,
            requirement_statuses={x['id']: x['overall_status'] for x in report['requirements']},
        )
        rows.append(row)
        trajectories[arm] = [x['requested'] for x in report['actions']]
        for name in ['REPORT.json', 'ACTION_SCHEMA_REVIEW.json', 'cold_restore/RESULT.json',
                     'VISUAL_REVIEW.json', 'EVIDENCE_MANIFEST.json']:
            bindings[f'{arm}/{name}'] = digest(folder / name)
    prefixes = {}
    for left, right in [('base', 'separate'), ('base', 'paired'), ('separate', 'paired')]:
        count = 0
        for a, b in zip(trajectories[left], trajectories[right]):
            if a != b:
                break
            count += 1
        prefixes[left + '_vs_' + right] = count
    summary = {
        'run_id': public.name, 'summarized_at': datetime.now(timezone.utc).isoformat(),
        'status': 'three_terminal_attempts_compared', 'arms': rows,
        'shared_initial_request_sha256': batch['shared_initial_request_sha256'],
        'source_sha256': bindings,
        'identical_requested_action_prefix_lengths': prefixes,
        'trajectory_note': 'Descriptive requested-action equality only; no causal or quality inference.',
        'training_updates_in_this_comparison': 0,
        'limits': [
            'One exposed research reconstruction, not a sealed test or a sample of independent homes.',
            'Each adapter has only two approved updates; no learning/generalization inference.',
            'Training contained ten furniture moves in furnished states; this task starts empty.',
            'Presence, fewer objects, accepted tools and unknown requirements are not success.',
            'Access uses declared grid/body proxies; no independent designer or actual-use evaluation.',
            'Strong prompting, same-knowledge retrieval and action-only training controls remain outstanding.'
        ],
    }
    (public / 'COMPARISON.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    columns = ['arm', 'model_calls', 'tool_calls', 'furniture_count', 'schema_mismatch_calls',
               'actual_seconds', 'generation_seconds', 'output_tokens', 'whole_room_accepted']
    with (public / 'COMPARISON.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    lines = ['# Casa 三组整屋比较', '',
             '三组运行及终态证据已完成。下表报告实际结果，不将过程成功或未知用途计作设计成功。', '',
             '| 条件 | 模型/工具次数 | 家具数 | 网格相交对数 | 参数不合 schema | 已取得整屋验收 |',
             '|---|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['model_calls']}/{row['tool_calls']} | "
                     f"{row['furniture_count']} | {len(row['triangle_contacts'])} | "
                     f"{row['schema_mismatch_calls']} | {row['whole_room_accepted']} |")
    lines += ['', '“未取得整屋验收”是证据状态，不是具有完整金标准的二元成功率；'
              '当前报告没有可自动赋予整屋合格的规则。明确缺失的设施和测得的相交单独报告，'
              '安装、隐私与实际使用等未测项保留未知。', '',
              '各组的九项要求、实际动作、五视图与负结果说明见对应目录 REPORT.md、'
              'INTERPRETATION.md、VISUAL_REVIEW.json；GLB 留服务器，路径及哈希见 COMPARISON.json。', '',
              '本次只能检验这两个小规模适配器在同一研究房间中的实际行为。'
              '训练只有已有房间内的十个移动步骤，评测要求从空房开始；此任务差异为事后诊断。'
              '结果不能证明配对学习有效或无效，也不能证明跨项目专业设计能力。', '',
              '后续数据应同时补充两组共有的整屋建设动作，并仅在有来源的决定处连接专业理由；'
              '独立局部决定任务与整屋生成分别评价。充分提示、同知识检索、动作微调和专业盲评仍待完成。'
              '新增训练须另行审定。', '']
    (public / 'COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps({'status': summary['status'], 'arms': len(rows)}))


if __name__ == '__main__':
    main()
