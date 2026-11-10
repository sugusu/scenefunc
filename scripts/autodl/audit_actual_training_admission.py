"""Audit real native receipts without promoting them to successful-room targets."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from trajectory_supervision_contract import ClauseEvidence, terminal_admission


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(repo, out):
    out.mkdir(parents=True, exist_ok=False)
    prior = repo / 'runs/autodl/20261110_complete_native_runtime'
    source = repo / 'runs/autodl/20261110_supervision_horizon/segments_v2/corpus_ledger.json'
    local_units = json.loads(source.read_text())
    components = repo / 'runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl'
    # These are audited existing source components, not fresh model completions.
    decision_ids = {d['unit_id'] for d in map(json.loads, components.read_text().splitlines())
                    if d['stage'] == 'decision'}
    rows = []
    for stage in ('floor', 'furniture', 'wall'):
        trace = prior / 'review_traces' / f'{stage}_trace.json'
        for turn in json.loads(trace.read_text()):
            proposal = turn['proposal']
            results = turn.get('results', [])
            calls = proposal.get('tool_calls', [])
            if len(calls) != len(results) and turn['status'] != 'native_policy_stop':
                raise ValueError('Unmatched native call and result')
            for i, call in enumerate(calls):
                result = results[i]
                assert result['name'] == call['name']
                output = result['output']
                native_flag = output.get('success') if isinstance(output, dict) else None
                rows.append(dict(stage=stage, turn=turn['turn'], call_index=i,
                    name=call['name'], arguments=call.get('arguments', {}),
                    native_success_flag=native_flag,
                    output_kind=type(output).__name__,
                    outcome='native_success' if native_flag is True else
                            'native_failure' if native_flag is False else 'no_boolean_success_certificate',
                    original_trace=str(trace.relative_to(repo)), source_sha256=sha(trace),
                    source_pointer=f'/turn={turn["turn"]}/results/{i}',
                    rationale_present=bool(proposal.get('think')),
                    eligible_expert_action_positive=False,
                    eligible_complete_room_positive=False,
                    reason='Self-generated action and reply: execution fact, not independently accepted design.'))
    requirements_path = prior / 'required_instances_review.csv'
    review = list(csv.DictReader(requirements_path.open()))
    ids = [f'original_clause_{i:02d}' for i in range(len(review))]
    evidence = [ClauseEvidence(k, 'unknown',
                 str(requirements_path.relative_to(repo)) + f'#row={i+2}', False)
                for i, k in enumerate(ids)]
    # Full3D success also requires basic validity. Do not silently make a positive
    # room label depend only on the five explicitly mentioned inventory/relations.
    # No new geometric inference is made: both checks remain unverified here.
    original_ids = list(ids)
    mandatory_ids = ['common_geometry_validity', 'common_functional_usability']
    ids.extend(mandatory_ids)
    evidence.extend(ClauseEvidence(k, 'unknown', 'CPU_ADMISSION_V2_PROTOCOL.json#'+k, False)
                    for k in mandatory_ids)
    admission = terminal_admission(ids, evidence, terminal_reason='actor_deadline',
                                   execution_errors=sum(x['native_success_flag'] is False for x in rows))
    terminal = dict(required_clause_denominator=len(ids),
                    original_clause_denominator=len(original_ids),
                    mandatory_validity_denominator=len(mandatory_ids),
                    original_required_instances=4,
                    evidence=[dict(clause_id=k, original_text=row['original_requirement'],
                                   value='unknown', independently_verified=False)
                              for k, row in zip(original_ids, review)] +
                             [dict(clause_id=k, original_text='Common full3D validity requirement',
                                   value='unknown', independently_verified=False) for k in mandatory_ids],
                    **admission)
    (out / 'native_calls.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    (out / 'terminal_admission.json').write_text(json.dumps(terminal,indent=2)+'\n')
    unit_rows = [dict(id=x['id'], family_id=x['family_id'],
                provenance='project_reconstruction_from_sourced_design_project',
                rationale_supported_by_source=x['has_sourced_reason'],
                original_local_decision_target_present=x['id'] in decision_ids,
                target_kind=x['target_kind'],
                original_author_tool_trajectory=False,
                complete_room_acceptance_proved=x['whole_room_professional_target_proved'])
                for x in local_units]
    (out / 'local_units_provenance.json').write_text(json.dumps(unit_rows,indent=2)+'\n')
    result = dict(status='actual_receipts_and_original_components_audited',
                  native_tool_calls=len(rows),
                  native_call_outcomes=dict(Counter(x['outcome'] for x in rows)),
                  local_units=len(unit_rows), families=len({x['family_id'] for x in unit_rows}),
                  source_supported_rationale_units=sum(x['rationale_supported_by_source'] for x in unit_rows),
                  existing_local_decision_targets=len(decision_ids),
                  missing_local_decision_targets=len(unit_rows)-len(decision_ids),
                  complete_positive_rooms_admitted=int(admission['eligible_complete_positive']),
                  licensed_SceneChain_samples_acquired=0,
                  heldout_critique_benchmark_used_for_training=False,
                  unknown_not_converted_to_preference_negatives=True,
                  paid_API_calls=0, model_forwards=0, training_updates=0,
                  source_files={str(p.relative_to(repo)):sha(p) for p in
                                [source, components, requirements_path]})
    # Count rationale turns from the actual receipts, including the floor stop.
    result['native_rationale_turns'] = sum(bool(t['proposal'].get('think'))
         for stage in ('floor','furniture','wall')
         for t in json.loads((prior/'review_traces'/f'{stage}_trace.json').read_text()))
    (out / 'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.out.resolve())
