"""Attach explicitly reconstructed decisions to two already executed return steps.

No new source, action, model call, episode mutation, or training admission.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from audit_edit_decision_coverage import read, sha, write


ANNOTATIONS = {
    'paper_table_modes_build_09': {
        'paired_existing_position': 'paper_table_modes_build_08',
        'condition': '当前活动已从访客预留转回日常用餐。',
        'application': '移动同一桌实例恢复日常桌面位置；随后仍需归位坐凳。活动改变后，不将腾空访客区域作为永久目标。',
        'cost_and_unknowns': '访客预留区将重新被桌占用；本步未单独完成两餐位，源沙发床展开及真实睡眠未知。旧回放的坐凳直线扫掠与沙发存在冲突，不能称完整物理搬运示范。',
        'sequence_end': 'paper_table_modes_build_10',
        'measured_result_scope': '既有第8步预留区清空；第10步恢复2个声明餐位和原实例/位姿。仅端点几何与声明用途代理，保留通用接近探针失败。',
    },
    'waterloo_edit_03': {
        'paired_existing_position': 'waterloo_edit_02',
        'condition': '共享待客阶段结束，当前需要恢复两人同时独立工作。',
        'application': '将第二张桌从组合布局移开，保留桌实例；随后调整第二把椅子，恢复独立工作模式。可分可合的经验在这里支持分开。',
        'cost_and_unknowns': '四人围坐共享模式暂时失效；本步桌移动后两种完整模式均未通过，需下一步坐椅归位。桌宽扰动、拼缝、轮载和实际搬运未知。',
        'sequence_end': 'waterloo_edit_04',
        'measured_result_scope': '既有第2步共享模式通过，第3步两种模式均不通过，第4步独立工作模式通过；7个固定接近点在3种声明人体宽度下均保持。',
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise ValueError('Do not overwrite candidates')
    print(datetime.now(timezone.utc).isoformat(), 'begin two existing return-step annotation candidates', flush=True)
    repo = Path(__file__).resolve().parents[2]
    dataset = read(a.study / 'DATASET.json')
    byid = {r['id']: r for r in dataset['rows']}
    ledger_path = repo / 'research/920/decision_packets_v1/SOURCED_DECISIONS.json'
    ledger = {r['decision_id']: r for r in read(ledger_path)['records']}
    records = []
    for key, annotation in ANNOTATIONS.items():
        row = byid[key]
        if row['decision_target'] is not None:
            raise ValueError('Expected a previously action-only step')
        for ref in [*row['documents'].values(), row['source'], row['consequences']]:
            path = Path(ref['path'])
            path = path if path.is_absolute() else repo / path
            if sha(path) != ref['sha256']:
                raise ValueError('Evidence hash mismatch: ' + str(path))
        for view in row['images']:
            if sha(view['path']) != view['sha256']:
                raise ValueError('View hash mismatch')
        original_decision = byid[annotation['paired_existing_position']]['decision_target']
        source = ledger[row['decision_id']]
        records.append(dict(id=key, family_id=row['family_id'], decision_id=row['decision_id'],
            status='researcher_application_candidate_not_training_admitted',
            condition_and_application_author='researcher, not quoted designer reasoning',
            source_principle=original_decision['principle_applicability']['sourced_principle'],
            source_evidence=source['professional_evidence'], source_scope=source['decision_scope'],
            annotation=annotation, actual_action=row['command'],
            existing_prestate_episode=row['documents']['action_only'],
            existing_five_views=row['images'], existing_consequences=row['consequences'],
            dependency=row['source'], whole_room_professional_positive=False,
            new_independent_project=False, original_episode_modified=False))
    write(a.out, dict(recorded_at=datetime.now(timezone.utc).isoformat(), records=records,
        script_sha256=sha(__file__), source_ledger_sha256=sha(ledger_path),
        dataset_sha256=sha(a.study / 'DATASET.json'), counts=dict(candidate_application_annotations=2,
        new_source_principles=0, new_project_families=0, new_actions=0, model_calls=0, training_updates=0),
        scope='Data candidates only. They do not retroactively increase the completed study\'s seven decision targets; no candidate episodes have been encoded or trained.'))
    print('completed 2 candidates; existing source/action/view hashes verified; 0 new model/native/training calls', flush=True)


if __name__ == '__main__':
    main()
