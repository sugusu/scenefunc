"""Read the actual frozen decision targets and their matched observations.

This reports supervision content and confounds; it supplies no new targets,
scores, semantic truth labels or model predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scene_evidence import HOST_MARKER, text_slots


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evidence(row):
    found = [json.loads(s[k][len(HOST_MARKER):]) for s, k in text_slots(row['messages'])
             if s[k].startswith(HOST_MARKER)]
    if len(found) != 1:
        raise ValueError('Exactly one actual host-state envelope required')
    return found[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--material', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    before = sha(a.material)
    rows = [json.loads(line) for line in a.material.read_text().splitlines()]
    units = sorted({r['unit_id'] for r in rows})
    records = []; conditions = {}
    for unit in units:
        choices = [r for r in rows if r['unit_id'] == unit and r['stage'] == 'decision']
        if not choices:
            records.append({'unit_id': unit, 'status': 'no_recorded_decision_target'})
            continue
        if len({r['target_sha256'] for r in choices}) != 1:
            raise ValueError('Decision supervision differs across arms')
        d = choices[0]
        target = d['target_text'].strip()
        if not target.startswith('<think>') or not target.endswith('</think>'):
            raise ValueError('Unexpected decision serialization')
        parsed = json.loads(target[len('<think>'):-len('</think>')].strip())
        if 'proposed_decision' in parsed:
            if set(parsed) != {'current_condition', 'principle_applicability', 'proposed_decision', 'expected_cost_and_unknowns'}:
                raise ValueError('Decision schema changed; reassess geometry-target statement')
            if set(parsed['proposed_decision']) != {'description', 'object_or_asset_references'}:
                raise ValueError('New structured decision fields need review')
            if not all(isinstance(v, str) for v in parsed['proposed_decision'].values()):
                raise ValueError('Structured action/geometry values require new audit')
        elif set(parsed) != {'requested_mode', 'measured_scope', 'unmet_in_measured_scope', 'unknowns', 'whole_room_success'}:
            raise ValueError('Unexpected STOP summary schema')
        action = next(r for r in rows if r['unit_id'] == unit and r['stage'] == 'action'
                      and r['arm'] == 'paired_phase_v3')
        obs = evidence(d)
        action_obs = evidence(action)
        if obs != action_obs:
            raise ValueError('Decision/action state differs')
        match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', action['target_text'], re.S)
        call = json.loads(match.group(1)) if match else None
        ref = parsed.get('proposed_decision', {}).get('object_or_asset_references')
        action_ref = (call['arguments'].get('object_id') or call['arguments'].get('asset_id')) if call else None
        oid_set = {o['object_id'] for o in obs['actual_scene_state']['objects']}
        record = {'unit_id': unit, 'family_id': d['family_id'],
                  'status': 'decision_target_audited', 'decision_fields': sorted(parsed),
                  'decision_target_sha256': d['target_sha256'],
                  'action_target_sha256': action['target_sha256'],
                  'decision_reference': ref, 'action_reference': action_ref,
                  'same_reference': ref == action_ref, 'tool': call['name'] if call else 'STOP',
                  'current_object_reference_exists': ref in oid_set if call and 'object_id' in call['arguments'] else None,
                  'source_principle_id': parsed.get('principle_applicability', {}).get('source_decision_id'),
                  'structured_geometry_constraint_fields': [],
                  'schema_scope': 'Existing string-valued proposed_decision / principle fields or measured STOP summary; not an executable geometric predicate target.',
                  'actual_state_digest': digest(obs['actual_scene_state'])}
        records.append(record); conditions[unit] = (parsed, record)
    pairs = []
    for first, second in [('paper_table_modes_build_08', 'paper_table_modes_build_09'),
                          ('waterloo_edit_02', 'waterloo_edit_03')]:
        x, y = conditions[first], conditions[second]
        pairs.append({'units': [first, second],
                      'same_source_principle': x[1]['source_principle_id'] == y[1]['source_principle_id'],
                      'same_object_reference': x[1]['decision_reference'] == y[1]['decision_reference'],
                      'same_observed_state': x[1]['actual_state_digest'] == y[1]['actual_state_digest'],
                      'condition_texts': [x[0]['current_condition'], y[0]['current_condition']],
                      'interpretation': 'Activity-transition examples with different scene states; not same-scene preference reversal or a controlled intervention on decision semantics.'})
    supervised = [r for r in records if r['status'] == 'decision_target_audited']
    result = {'units': len(units), 'decision_supervised': len(supervised),
              'no_recorded_decision': len(units)-len(supervised),
              'operation_selection_decisions': sum(r['tool'] != 'STOP' for r in supervised),
              'stop_summaries': sum(r['tool'] == 'STOP' for r in supervised),
              'source_principles': len({r['source_principle_id'] for r in supervised if r['source_principle_id']}),
              'decision_and_action_refs_agree': sum(r['same_reference'] for r in supervised if r['tool'] != 'STOP'),
              'condition_pairs_with_fixed_scene': sum(r['same_observed_state'] for r in pairs),
              'material_sha256': before, 'material_unchanged': sha(a.material) == before,
              'model_calls': 0, 'new_targets': 0,
              'implication': 'A gain can reflect principle-conditioned action selection, repeated object/action supervision, prompt/deployment changes or improved stopping. The current material does not isolate a new 3D geometric reasoning algorithm.'}
    for name, value in [('records.json', records), ('condition_pairs.json', pairs), ('summary.json', result)]:
        (a.out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
