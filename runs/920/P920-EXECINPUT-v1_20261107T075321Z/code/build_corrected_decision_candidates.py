"""Materialize reviewed data corrections without changing historical training data."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from edit_request import from_episode_inputs, validate_action
from shared_action_contract import schemas_from_observation


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise ValueError('Use a fresh candidate directory')
    a.out = a.out.resolve()
    print(datetime.now(timezone.utc).isoformat(), 'materialize corrected 44 documents; no model', flush=True)
    data = read(a.study / 'DATASET.json')
    knowledge = read(a.audit / 'audit_v2/KNOWLEDGE_CANDIDATE.json')['records']
    returns = {x['id']: x for x in read(a.audit / 'CONDITIONAL_COMPLETIONS.json')['records']}
    correction = read(a.audit / 'FHOUSE_DATA_USE_CORRECTION.json')
    if correction['future_data_use']['admit_original_as_positive_incremental_visibility_decision']:
        raise ValueError('Expected explicit F-House data-use correction')
    first = read(data['rows'][0]['documents']['action_only']['path'])
    native = schemas_from_observation({'system': first['system']})
    rows = []
    for original in data['rows']:
        row = copy.deepcopy(original)
        identifier = row['id']
        target = copy.deepcopy(row['decision_target'])
        changes = ['Remove experiment scope from shared source knowledge; preserve all other input facts.']
        if identifier in returns:
            record = returns[identifier]
            if record['actual_action'] != row['command'] or target is not None:
                raise ValueError('Return annotation does not match existing action')
            ref = record['existing_prestate_episode']
            if sha(ref['path']) != ref['sha256']:
                raise ValueError('Return prestate changed')
            annotation = record['annotation']
            # Retrospective measurements stay in the evidence record, not in a
            # purported pre-action prediction by the future policy.
            prospective_cost = {
                'paper_table_modes_build_09': '恢复餐桌将占用访客预留区；仍需归位坐凳并检查两餐位及移动路径。沙发床展开和真实睡眠尚未知，不能仅凭端点位置承诺完整物理搬运可行。',
                'waterloo_edit_03': '分开桌面将暂时失去共享围坐布局；还需调整第二把椅子并检查两人工位和通路。桌宽扰动、拼缝、轮载与实际搬运未知。',
            }[identifier]
            target = dict(current_condition=annotation['condition'],
                principle_applicability=dict(source_decision_id=row['decision_id'],
                                             sourced_principle=record['source_principle']),
                proposed_decision=dict(description=annotation['application'],
                    object_or_asset_references=row['command'][0]['arguments']['object_id']),
                expected_cost_and_unknowns=prospective_cost)
            row['candidate_evaluation_only'] = {
                'retrospective_annotation': annotation,
                'existing_consequences': record['existing_consequences'],
                'source_evidence': record['source_evidence'],
            }
            changes.append('Add researcher-authored reverse-phase application; not a new source decision.')
        elif identifier == 'fhouse_perimeter_01':
            target = None
            changes.append('Remove unsupported incremental-visibility decision target; keep optional feasible action only, not an optimal-action label.')
        elif identifier == 'storage_edit_01':
            target['proposed_decision']['description'] = '选择移动已有柜体 recon_storage_a_0，作为集中收纳布局的首个重建动作；这不是用户指定的柜体选择。'
            changes.append('Remove false input attribution that the next cabinet was specified.')
        row.update(decision_target=target, documents={}, requests={},
                   original_documents=original['documents'], candidate_changes=changes,
                   candidate_status='not_training_authorized_not_professionally_accepted',
                   decision_scope='Source principle paraphrase plus explicitly reconstructed state/action application; not historical designer reasoning.')
        input_pair = []
        for arm, ref in original['documents'].items():
            if sha(ref['path']) != ref['sha256']:
                raise ValueError('Historical training document changed')
            doc = read(ref['path'])
            context = json.loads(doc['task_context']['scene_description'])
            old_knowledge = context['available_source_knowledge']
            if [{k: x[k] for k in ('decision_id', 'condition', 'principle')} for x in old_knowledge] != knowledge:
                raise ValueError('Clean knowledge changes more than experiment metadata')
            context['available_source_knowledge'] = knowledge
            doc['task_context']['scene_description'] = json.dumps(context, ensure_ascii=False, sort_keys=True)
            doc['turns'][0]['content'] = json.dumps(target, ensure_ascii=False) if arm == 'decision_action' and target else ''
            doc['provenance'].update(training_authorized=False,
                candidate_data_only=True, original_document=ref,
                decision_text_authorship='Only sourced_principle is an attributed paraphrase; applications and costs are researcher annotations.')
            if doc['turns'][0]['tool_calls'] != row['command']:
                raise ValueError('Actual historical action changed')
            path = a.out / 'dataset' / arm / identifier / 'episode.json'
            write(path, doc)
            request = from_episode_inputs(doc, native)
            for action in row['command']:
                validate_action(action, request)
            poisoned = copy.deepcopy(doc)
            poisoned['turns'][0].update(content='TARGET_ONLY_SENTINEL', tool_calls=[])
            poisoned['evaluation_only'] = {'sentinel': 'TARGET_ONLY_SENTINEL'}
            if from_episode_inputs(poisoned, native) != request:
                raise ValueError('Target or evaluation metadata changes model input')
            request_path = path.with_name('request.json')
            write(request_path, request)
            row['documents'][arm] = {'path': str(path), 'sha256': sha(path)}
            row['requests'][arm] = {'path': str(request_path), 'sha256': sha(request_path)}
            input_pair.append(request)
        if input_pair[0] != input_pair[1]:
            raise ValueError('The arms do not receive the same input')
        rows.append(row)
    decisions = [r for r in rows if r['decision_target'] is not None]
    if len(rows) != 22 or len(decisions) != 8:
        raise ValueError('Unexpected action or decision coverage')
    result = dict(schema='scenefunc-corrected-decision-candidates-v1',
        created_at=datetime.now(timezone.utc).isoformat(), rows=rows,
        counts=dict(actions=22, documents=44, source_families=6, decision_positions=8,
                    source_principles_with_decision_targets=len({r['decision_id'] for r in decisions}),
                    knowledge_records=len(knowledge), new_actions=0, new_sources=0, new_projects=0),
        original_dataset={'path': str(a.study / 'DATASET.json'), 'sha256': sha(a.study / 'DATASET.json')},
        correction_inputs={str(p): sha(p) for p in [a.audit / 'CONDITIONAL_COMPLETIONS.json',
            a.audit / 'FHOUSE_DATA_USE_CORRECTION.json', a.audit / 'audit_v2/KNOWLEDGE_CANDIDATE.json']},
        script_sha256=sha(__file__), training_authorized=False, model_calls=0, training_updates=0,
        whole_room_professional_acceptance=False,
        limits=['Ordinary F-House movement remains a feasible optional demonstration, not evidence that movement is needed; no no-edit target is encoded.',
                'Reverse-phase annotations expose intended subsequent steps, not measured future outcomes; they are training targets only.',
                'Candidate encoding does not approve training, rights, human preference or generalization.'])
    write(a.out / 'DATASET.json', result)
    print(json.dumps(result['counts']), flush=True)


if __name__ == '__main__':
    main()
