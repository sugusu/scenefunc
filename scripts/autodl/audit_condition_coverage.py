"""Connect fixed supervision to task/state contrasts and archived consequences.

No model, generated annotation, geometry edit, or new training is used. The
label-based oracles describe ambiguity on the observed records, not learned
test performance. An identical object table is not identical interaction history.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

from audit_material_routing import CORPUS, PIN, GitInputs, digest, write
from audit_declared_stop_gate import decide, extract_payload


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def hashed(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def object_signature(context):
    objects = context.get('objects')
    if isinstance(objects, dict):
        if any(key != value.get('object_id') for key, value in objects.items()):
            raise ValueError('Object dictionary key differs from its identity')
        objects = list(objects.values())
    if not isinstance(objects, list):
        return None
    ids = [obj['object_id'] for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated object identity')
    # No rounding or tolerance. This compares the published transform and box
    # records, not the unavailable complete meshes or private environment state.
    keys = ('object_id', 'object_type', 'bbox_min', 'bbox_max', 'transform')
    return hashed([{key: obj.get(key) for key in keys}
                   for obj in sorted(objects, key=lambda obj: obj['object_id'])])


def public_task(context):
    if context.get('current_requested_mode'):
        return 'current_requested_mode', context['current_requested_mode']
    activity = context.get('activity_context', {})
    if activity.get('phase'):
        return 'activity_context.phase', activity['phase']
    return None, None  # Never infer a missing input field from a target rationale.


def source_path(path):
    if '/scenefunc/' in path:
        return path.split('/scenefunc/', 1)[1]
    if path.startswith('scenefunc/'):
        return path[len('scenefunc/'):]
    if path.startswith('runs/') or path.startswith('research/'):
        return path
    return None


def oracle(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    details = []
    correct = 0
    for key, group in groups.items():
        labels = Counter(row['edit_stop_reference'] for row in group)
        best = max(labels.values())
        correct += best
        details.append({'key': list(key), 'ids': [r['unit_id'] for r in group],
                        'label_counts': dict(labels), 'best_count': best})
    return {'fields': fields, 'correct_at_most': correct, 'n': len(rows),
            'groups': details, 'uses_reference_labels': True,
            'scope': 'Empirical representational ceiling on these records; not a trained or deployable policy.'}


def audit(repo):
    started = time.perf_counter()
    src = GitInputs(repo)
    corpus = src.json(CORPUS)
    rows, consequences, decisions = [], {}, []
    for unit in corpus['rows']:
        inp, inp_path = src.reference(unit['input'])
        request, req_path = src.reference(unit['request'])
        episode, ep_path = src.reference(unit['documents']['decision_action'])
        context = inp['context']
        turn = episode['turns'][0]
        mode_field, mode = public_task(context)
        expected = 'stop_with_scoped_unknowns' if unit['target_kind'] == 'terminal_control' else 'edit_required'
        payload = extract_payload(request)
        gate = decide(payload)
        if payload is not None:
            for name in ('available_measurement', 'declared_activity_requirements', 'current_requested_mode'):
                if payload.get(name) != context.get(name):
                    raise ValueError('Input/request context differs: ' + unit['id'] + '/' + name)
        source = unit.get('source', {})
        ev = episode.get('evaluation_only', {})
        consequence = source.get('consequences')
        outcome = None
        if consequence:
            p = source_path(consequence['path'])
            if p is not None:
                b = src.blob(p)
                if digest(b) != consequence['sha256']:
                    raise ValueError('Consequence reference hash mismatch: ' + p)
                result = json.loads(b)
                consequences[p] = {'sha256': digest(b), 'result': result}
                outcome = {'path': p, 'sha256': digest(b),
                           'scope': 'Record/episode consequences, not proof this one action caused every reported improvement.'}
        text = turn.get('content', '')
        reason = None
        if unit['target_kind'] == 'tool_action' and text:
            reason = json.loads(text)
            if not isinstance(reason, dict) or 'principle_applicability' not in reason:
                raise ValueError('Unrecognized sourced-decision schema')
            decisions.append({'unit_id': unit['id'], 'family_id': unit['family_id'],
                              'reason': reason,
                              'authorship': episode.get('provenance', {}).get('decision_text_authorship'),
                              'scope': source.get('decision_scope', ev.get('decision_scope'))})
        calls = turn.get('tool_calls', [])
        row = {'unit_id': unit['id'], 'family_id': unit['family_id'],
               'target_kind': unit['target_kind'], 'edit_stop_reference': expected,
               'input_path': inp_path, 'input_sha256': unit['input']['sha256'],
               'request_path': req_path, 'request_sha256': unit['request']['sha256'],
               'episode_path': ep_path, 'episode_sha256': unit['documents']['decision_action']['sha256'],
               'object_table_sha256': object_signature(context),
               'object_count': len(context.get('objects', [])),
               'task_field': mode_field, 'task_value': mode,
               'has_sourced_decision': reason is not None,
               'decision_id': reason['principle_applicability']['source_decision_id'] if reason else None,
               'tool_calls': calls, 'action_signature': hashed(calls),
               'source_consequence': outcome,
               'evaluation_only_fields': sorted(ev),
               'inline_receipt_field_present': any(k in ev for k in ('actual_reply', 'actual_receipt', 'outcome')),
               'terminal_measurement_ref': source.get('measured_evidence') if unit['target_kind'] == 'terminal_control' else None,
               'execution_evidence_keys': sorted(source.get('execution_evidence', {})),
               'gate': gate,
               'gate_agrees': gate['decision'] == expected if gate['decision'] != 'unknown' else None}
        rows.append(row)
    pairs = []
    for left, right in itertools.combinations(rows, 2):
        if left['family_id'] != right['family_id'] or left['task_value'] is None or right['task_value'] is None:
            continue
        if left['object_table_sha256'] is None or right['object_table_sha256'] is None:
            continue
        if left['object_table_sha256'] == right['object_table_sha256'] and left['task_value'] != right['task_value']:
            pairs.append({'ids': [left['unit_id'], right['unit_id']],
                          'family_id': left['family_id'],
                          'object_table_sha256': left['object_table_sha256'],
                          'tasks': [left['task_value'], right['task_value']],
                          'edit_stop_changes': left['edit_stop_reference'] != right['edit_stop_reference'],
                          'demonstrated_action_changes': left['action_signature'] != right['action_signature'],
                          'limits': 'Same published boxes/transforms only. History and wording can differ; different demonstrations do not prove the other action is invalid.'})
    supported = [r for r in rows if r['gate_agrees'] is not None]
    families = sorted({r['family_id'] for r in supported})
    controls = {key: oracle(supported, fields) for key, fields in {
        'family_only': ['family_id'],
        'task_only': ['family_id', 'task_value'],
        'object_table_only': ['family_id', 'object_table_sha256'],
        'task_and_object_table': ['family_id', 'task_value', 'object_table_sha256']}.items()}
    controls['ordinary_declared_gate'] = {'correct': sum(r['gate_agrees'] for r in supported),
        'n': len(supported), 'unknown': len(rows)-len(supported),
        'uses_reference_labels_to_predict': False,
        'scope': 'Existing input measurements and public thresholds; no action generation or full-room certification.'}
    # Find complete binary task x state tables from archived data, without
    # choosing a new state, perturbing geometry, or relabeling a failed result.
    factorials = []
    for family in families:
        group = [r for r in supported if r['family_id'] == family]
        modes = sorted({r['task_value'] for r in group})
        states = sorted({r['object_table_sha256'] for r in group})
        for tasks in itertools.combinations(modes, 2):
            for geom in itertools.combinations(states, 2):
                cells = [[r for r in group if r['task_value'] == t and r['object_table_sha256'] == g]
                         for t, g in itertools.product(tasks, geom)]
                if not all(len(c) == 1 for c in cells):
                    continue
                chosen = [c[0] for c in cells]
                factorials.append({'family_id': family, 'tasks': list(tasks), 'states': list(geom),
                                   'ids': [r['unit_id'] for r in chosen],
                                   'labels': [r['edit_stop_reference'] for r in chosen],
                                   'task_only': oracle(chosen, ['task_value']),
                                   'state_only': oracle(chosen, ['object_table_sha256'])})
    return {'pin': PIN, 'source_corpus': CORPUS, 'source_sha256': src.manifest,
            'counts': {'units': len(rows), 'families': len({r['family_id'] for r in rows}),
                       'sourced_reason_units': len(decisions),
                       'unique_source_principles': len({r['decision_id'] for r in rows if r['decision_id']}),
                       'linked_outcome_records': len(consequences),
                       'units_with_linked_outcome': sum(r['source_consequence'] is not None for r in rows),
                       'same_object_table_different_task_pairs': len(pairs),
                       'complete_two_task_two_state_blocks': len(factorials)},
            'rows': rows, 'sourced_decisions': decisions, 'consequences': consequences,
            'same_geometry_pairs': pairs, 'controls': controls, 'factorial_blocks': factorials,
            'scope': 'Fixed development corpus: source principles with researcher-reconstructed actions and archived proxy consequences. No professional preference, new-room performance, or causal identification claim.',
            'new_model_calls': 0, 'new_scene_edits': 0, 'training_updates': 0,
            'wall_seconds': time.perf_counter()-started}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = audit(args.repo)
    write(args.output/'condition_coverage.json', result)
    fields = ['unit_id', 'family_id', 'target_kind', 'task_value', 'object_table_sha256',
              'has_sourced_decision', 'decision_id', 'inline_receipt_field_present', 'gate_agrees']
    with (args.output/'condition_coverage.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader(); writer.writerows(result['rows'])
    print(canonical({'counts': result['counts'], 'controls': {
        k: {kk: vv for kk, vv in v.items() if kk not in ('groups', 'scope')}
        for k, v in result['controls'].items()}, 'wall_seconds': result['wall_seconds']}))


if __name__ == '__main__':
    main()
