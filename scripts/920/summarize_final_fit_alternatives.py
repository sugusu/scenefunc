"""Publish verifiable single-step native consequences, not coordinate-match scores."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from native_checkpoint import canonical_hash
from native_project import sha256, validate_bundle
from score_activity_transition import score_transition
from replay_final_fit_alternatives import write, check_inputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args(); pub = a.run.resolve()
    plan = json.loads((pub/'PLAN.json').read_text()); check_inputs(plan)
    root = Path(plan['runtime_directory']); result = json.loads((root/'RESULT.json').read_text())
    receipt = json.loads((pub/'execution/receipt.json').read_text())
    if (result['status'] != 'completed' or result['native_calls'] != 6 or result['completed'] != 6
            or result['model_calls'] != 0 or result['training_updates'] != 0 or receipt['exit_code'] != 0
            or result['plan_sha256'] != sha256(pub/'PLAN.json')):
        raise ValueError('Batch incomplete or accounting mismatch')
    review = pub/'review'; review.mkdir(exist_ok=True)
    rows = []; recovery = []
    for planned, row in zip(plan['cases'], result['rows'], strict=True):
        if any(row[k] != v for k, v in planned.items()): raise ValueError('Plan/result mismatch')
        cid = row['id']; source = Path(plan.get('retained_case_directories', {}).get(cid, str(root/cid)))
        target = review/cid; target.mkdir(exist_ok=True)
        before = json.loads((source/'before_state.json').read_text())
        after = json.loads((source/'after_state.json').read_text())
        be = json.loads((source/'before_evaluation.json').read_text())
        ae = json.loads((source/'after_evaluation.json').read_text())
        if canonical_hash(before) != row['before_sha256'] or canonical_hash(after) != row['after_sha256']:
            raise ValueError('State hash mismatch')
        manifest, house, _, _ = validate_bundle(source/'checkpoint')
        if canonical_hash(house['rooms'][manifest['selected_room_id']]) != row['after_sha256']:
            raise ValueError('Checkpoint does not contain actual endpoint')
        calls = json.loads((source/'HTTP_REQUESTS.json').read_text())
        edits = [x for x in calls if x['route'] == '/session/step']
        if len(edits) != 1 or edits[0]['payload'] != {'tool_calls': [row['command']]}:
            raise ValueError('Actual native call differs from proposal')
        score = score_transition(be['review'], ae['review'], attempts=1, native_calls=1,
                                 state_changes=int(row['before_sha256'] != row['after_sha256']))
        if score != row['transition']: raise ValueError('Transition not reproducible')
        accepted = row['native_reply']['results'][0]['output'].get('success')
        error_type = row['native_reply']['results'][0]['output'].get('error_type')
        compact = dict(id=cid, state_id=row['state_id'], arms=row['arms'], command=row['command'],
                       tool_accepted=accepted, native_error_type=error_type, transition=score,
                       before_table_geometry=be['measurements']['table_geometry'],
                       after_table_geometry=ae['measurements']['table_geometry'],
                       before_seat_alignment=be['measurements']['seat_alignment'],
                       after_seat_alignment=ae['measurements']['seat_alignment'],
                       after_geometry=ae['measurements']['geometry'])
        rows.append(compact)
        for name in ['before_state.json','after_state.json','before_evaluation.json','after_evaluation.json',
                     'TRANSITION.json','HTTP_REQUESTS.json','observation.json']:
            shutil.copy2(source/name, target/name)
        shutil.copy2(row['image_paths'][0], target/'after.topdown.png')
        recovery.append(dict(id=cid, checkpoint=str(source/'checkpoint'),
            manifest_sha256=sha256(source/'checkpoint/manifest.json'),
            source_bundle=row['bundle'], source_manifest_sha256=sha256(Path(row['bundle'])/'manifest.json'),
            final_native_sha256=row['after_sha256'], actual_native_call=row['command']))
    prior_receipts = [json.loads(Path(p).read_text()) for p in plan.get('prior_receipts', [])]
    total_seconds = receipt['elapsed_seconds'] + sum(r['elapsed_seconds'] for r in prior_receipts)
    if total_seconds > 1200: raise ValueError('Cumulative budget exceeded')
    report = dict(status='verified_completed', recorded_at=datetime.now(timezone.utc).isoformat(),
        unique_native_replays=6, arm_rows=7, excluded_outputs=plan['excluded_outputs'],
        model_calls=0, training_updates=0, rows=rows, execution=receipt, prior_receipts=prior_receipts,
        cumulative_execution_seconds=total_seconds,
        scope=plan['interpretation'], unknowns=plan['limits'])
    write(pub/'RESULT.json', report)
    write(pub/'RECOVERY.json', dict(rows=recovery, method='Open existing native checkpoint with session/existing using frozen runtime. Large meshes remain on server; before/after states and receipts published. Rerunning noisy moves is not bit-identical checkpoint recovery.'))
    text = ['# 已有替代动作的实际后果', '',
        '对训练内已有自由输出做事后单步回放：7条合法移动对应6个不同状态/动作，重复项共享一次执行；没有新生成或训练。固定对象违例、未提供工具及格式错误不修补；6条合法查询不按移动失败计分，也未继续生成。', '',
        '| 状态 | 方法 | 引擎回执 | 当前用途之前→之后 | 新增失败项 |',
        '|---|---|---|---|---|']
    for row in rows:
        s = row['transition']
        text.append('| '+ ' | '.join([row['state_id'], '/'.join(row['arms']), 'accepted' if row['tool_accepted'] else str(row['native_error_type']),
            s['initial_status']+' → '+s['final_status'], ', '.join(s['newly_failed']) or '无'])+' |')
    text += ['', '6次原生调用只有4次实际变化；另2次返回 no_movement。联合组在初态办公任务的 pass→pass 是请求原位移动被引擎拒绝、场景未变，并非成功停止或新增用途收益。底座/动作组在待客任务的相同提案亦为原位移动，既有缺口未解。', '',
        '评价采用既有 conditional-endpoint-v1 的已验收 V3 实现，参考对象身份不变，收益/损害相对于各自原始状态计算。yaw 原值按原生度数单位执行，保留引擎位姿噪声；并非按示范坐标相同与否评分。', '',
        '一次动作失败不等于后续不能修复；一次保持不等于学到专业设计。这里只有一个曝光项目的3个编辑起点，不能证明跨项目收益。坐席、间距、方形身体通行是声明代理，真实舒适、搬运和完整房间质量仍未知。', '',
        '累计执行耗时 %.3f 秒（包含首批错误），续行退出码 %s。首批重复注册服务在第2条动作前停止；首条结果复用，只续行剩余5条。完整检查向量、前后状态、原生回执和俯视图见 review/；服务器检查点及哈希见 RECOVERY.json。'%(total_seconds, receipt['exit_code'])]
    if (pub/'REFERENCE_CONTEXT.json').exists():
        context = json.loads((pub/'REFERENCE_CONTEXT.json').read_text())
        text += ['', '## 已有示范中间态，事后解释用', '',
            '读取历史检查点，没有重跑示范。待客示范下一步通过全部43项；恢复办公示范下一步先拉开桌子，此时椅子未跟上，仍有1项坐席失败，后续才移动椅子。因此不能要求每个中间动作立即完成整屋。模型恢复办公提案中，底座新增桌体重叠，联合组保留分桌间距不足；它们的问题依据实际几何，不依据精确坐标匹配。初态办公的参考停止保持全部34项。详见 REFERENCE_CONTEXT.json。']
        report['reference_context'] = dict(path='REFERENCE_CONTEXT.json', sha256=sha256(pub/'REFERENCE_CONTEXT.json'),
            status='posthoc_read_only_saved_reference_intermediates', rows=[dict(state_id=x['state_id'],transition=x['transition']) for x in context['rows']])
        write(pub/'RESULT.json', report)
    (pub/'SUMMARY.md').write_text('\n'.join(text)+'\n')
    files = sorted(p for p in review.rglob('*') if p.is_file()) + [pub/'RESULT.json', pub/'RECOVERY.json']
    if (pub/'REFERENCE_CONTEXT.json').exists(): files.append(pub/'REFERENCE_CONTEXT.json')
    write(pub/'ARTIFACT_MANIFEST.json', dict(files=[dict(path=str(p.relative_to(pub)),bytes=p.stat().st_size,sha256=sha256(p)) for p in files]))
    print(json.dumps(dict(status='verified_completed', files=len(files), rows=[dict(id=x['id'], transition=x['transition']['outcome']) for x in rows])))


if __name__ == '__main__': main()
