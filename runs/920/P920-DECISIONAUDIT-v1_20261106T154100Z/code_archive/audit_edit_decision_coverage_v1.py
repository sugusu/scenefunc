"""Audit the completed study, without modifying episodes or running a model.

Semantic reviews are explicit researcher annotations, not automated quality scores.
Exact-state grouping is a narrow reproducible count, not a test of all possible
semantic contrasts. The knowledge export is a candidate, not deployed data.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


REVIEWS = {
    'clothing_build_07': '用途没有指定挂杆，但给定目录只有开放挂杆这一衣物专用资产；没有已核可用的封闭衣柜对照。不能据此评价开放/封闭策略选择。',
    'storage_build_07': '任务指定三件柜体，位置未指定；可学习集中布置，但柜体数量不是模型选择。同族另一决定位置不是独立项目。',
    'paper_table_modes_build_08': '需求指定访客预留区和活动阶段，没有指定移动对象或坐标；有恢复阶段动作，但恢复阶段没有专业决定目标，且起始场景不同。',
    'storage_edit_01': '只允许移动三柜，目标位置未指定；仍有布局选择空间。目标说明的“本步指定柜体”不是输入事实，输入未指定本步移动哪一柜。',
    'waterloo_edit_02': '需求指定共享活动模式，可编辑对象限于第二桌和椅；有恢复独立工作模式的动作，但该模式没有专业决定目标，且起始场景不同。',
    'lifecycle_01': '任务解除儿童床需求并允许移出，但没有强制删除；已有示范仅覆盖移除这一方案，未建立床位仍需保留时的对应决定监督。',
    'fhouse_perimeter_01': '需求给出交流和保留家具约束，没有指定靠边摆柜；训练只包含共同交流条件，已有减少干扰的研究者条件未纳入专业决定监督。',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(), 'begin existing 22-step decision-coverage audit', flush=True)
    dataset = read(a.study / 'DATASET.json')
    rows, states, pools = [], defaultdict(list), {}
    checked = 0
    for row in dataset['rows']:
        docs = {}
        for arm, ref in row['documents'].items():
            if sha(ref['path']) != ref['sha256']:
                raise ValueError('Episode hash mismatch: ' + ref['path'])
            docs[arm] = read(ref['path'])
            checked += 1
        context = json.loads(docs['action_only']['task_context']['scene_description'])
        other = json.loads(docs['decision_action']['task_context']['scene_description'])
        if context != other:
            raise ValueError('Arms do not share input facts')
        pool = context['available_source_knowledge']
        pools[digest(pool)] = pool
        objects = context['objects']
        if isinstance(objects, list):
            objects = {obj['object_id']: obj for obj in objects}
        # Asset catalog and exact object records are fixed within each counted group.
        state = digest({'family': row['family_id'], 'objects': objects,
                        'catalog': context.get('asset_catalog')})
        condition = {k: context[k] for k in ('design_request', 'request', 'revision_request',
                     'activity_context', 'current_requested_mode') if k in context}
        item = dict(id=row['id'], family_id=row['family_id'], decision_id=row['decision_id'],
                    has_decision_target=row['decision_target'] is not None,
                    documents=row['documents'], exact_state_hash=state,
                    condition=condition, condition_hash=digest(condition),
                    catalog_names=[v['name'] for v in context.get('asset_catalog', [])],
                    system_operation_constraints=docs['action_only']['system'].split('Available executable tools:')[0],
                    command=row['command'], researcher_review=REVIEWS.get(row['id']),
                    action_is_reconstructed_not_historical=True)
        rows.append(item)
        states[state].append(item)
    if len(rows) != 22 or checked != 44 or len(pools) != 1:
        raise ValueError('Unexpected dataset scope or differing knowledge pools')
    pool = next(iter(pools.values()))
    decision_rows = [r for r in rows if r['has_decision_target']]
    if {r['id'] for r in decision_rows} != set(REVIEWS):
        raise ValueError('Manual annotations do not cover the actual decision rows')
    contrasts = [[r['id'] for r in group] for group in states.values()
                 if len({r['condition_hash'] for r in group}) > 1]
    # Whitelist source knowledge fields; retain the removed metadata alongside it.
    # No claim that all retained principles were independently rechecked here.
    export = [{k: item[k] for k in ('decision_id', 'condition', 'principle')} for item in pool]
    metadata = [{'decision_id': item['decision_id'],
                 'excluded_from_candidate_knowledge': {k: v for k, v in item.items()
                     if k not in ('decision_id', 'condition', 'principle')}} for item in pool]
    write(a.out / 'KNOWLEDGE_CANDIDATE.json', dict(status='not_deployed_not_trained',
          records=export, original_pool_sha256=digest(pool),
          limitation='Original source/rights ledger and scene unknowns remain required. This export only separates existing experiment annotations; it is not a new source verification.'))
    write(a.out / 'KNOWLEDGE_METADATA.json', metadata)
    result = dict(status='completed', recorded_at=datetime.now(timezone.utc).isoformat(),
        dataset=dict(path=str(a.study / 'DATASET.json'), sha256=sha(a.study / 'DATASET.json')),
        script_sha256=sha(__file__), checked_documents=checked, steps=len(rows),
        families=len({r['family_id'] for r in rows}), decision_positions=len(decision_rows),
        professional_principles=len({r['decision_id'] for r in decision_rows}),
        knowledge_records=len(pool), identical_knowledge_pool_in_all_44_documents=True,
        exact_same_state_multiple_condition_groups=contrasts,
        exact_state_grouping_limit='Includes all object fields, not geometry alone; absence does not prove absence of useful conditional learning.',
        input_metadata_finding='Knowledge scope includes researcher supervision/sequence annotations. Gold-field mutation checks do not detect information copied into the input beforehand. Shared exposure does not establish an arm-specific advantage.',
        rows=rows, model_calls=0, training_updates=0, scene_edits=0,
        research_conclusion='No design benefit demonstrated. Data has some genuine layout discretion, but sparse decision supervision and asymmetric condition coverage; not all tasks dictate the answer. This audit does not identify the causal reason for training failure.')
    write(a.out / 'RESULT.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows',)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
