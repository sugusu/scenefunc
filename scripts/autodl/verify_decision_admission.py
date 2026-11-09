"""Check the saved audit against pinned inputs; no inference or native replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from audit_material_routing import CORPUS, PIN, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.audit/'summary.json').read_text())
    verified = {}
    for path, expected in summary['source_sha256'].items():
        data = subprocess.check_output(['git', 'show', PIN+':'+path])
        actual = hashlib.sha256(data).hexdigest()
        assert actual == expected, path
        verified[path] = actual
    for path, expected in summary['local_source_sha256'].items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert actual == expected, path
        verified[path] = actual
    corpus = json.loads(subprocess.check_output(['git', 'show', PIN+':'+CORPUS]))['rows']
    roles = json.loads((args.audit/'unit_roles.json').read_text())
    reasons = json.loads((args.audit/'reason_evidence.json').read_text())
    opportunities = json.loads((args.audit/'compression_candidates.json').read_text())
    trials = json.loads((args.audit/'compression_pose_trials.json').read_text())
    assert {r['id'] for r in roles} == {r['id'] for r in corpus}
    assert len(roles) == len({r['id'] for r in roles}) == 24
    assert len(reasons) == 8
    assert len({r['source_decision_id'] for r in reasons}) == 5
    assert all(not r['existing_training_authorization_reinterpreted'] for r in roles)
    assert all(r['reason_is_not_a_positive_quality_label'] for r in reasons)
    assert len(opportunities) == 12 and sum(r['recorded_decisions'] for r in opportunities) == 72
    segments = [s for r in opportunities for s in r['candidate_segments']]
    assert len(segments) == 4
    assert sum(len(s['deletion_candidate_ids']) for s in segments) == 3
    allowed = {c for s in segments for c in s['deletion_candidate_ids']}
    assert len(trials) == 4
    assert dict(Counter(r['status'] for r in trials)) == summary['offline_pose_trial_status']
    assert sum(bool(r['changed_inputs']) for r in trials) == 3
    for row in trials:
        assert set(row['removed_candidate_ids']) <= allowed
        assert not row['training_ready'] and not row['full_compression_executed']
        assert not row['observation_rebinding_performed']
        assert row['no_new_label_or_training_admission']
        assert row['tolerances'] == {'position_m': .03, 'rotation_deg': 3., 'scale_absolute': .0001}
        if row['status'] == 'consistent_under_model':
            assert row['original_vs_recorded'] == row['differences'] == []
        else:
            assert row['status'] == 'unverifiable' and 'NO_MOVEMENT' in row['reason']
    loop = next(r for r in trials if len(r['removed_candidate_ids']) == 2)
    assert loop['can_reuse_observations'] and not loop['changed_inputs']
    assert loop['archived_boundary_comparisons'][0]['recorded_boundary_hash_equal']
    transitions = {r['id']: r for r in map(json.loads, Path(
        'runs/autodl/20261110_executed_effects/actual_v2/transitions.jsonl').read_text().splitlines())}
    fragments = {r['id']: r for r in map(json.loads, Path(
        'runs/autodl/20261110_executed_effects/actual_v2/source_fragments.jsonl').read_text().splitlines())}
    for row in trials:
        for gap in row['archived_boundary_comparisons']:
            left, right = gap['preceding_retained_action'], gap['next_retained_action']
            actual = transitions[left]['after_state_sha256'] == transitions[right]['before_state_sha256']
            assert actual == gap['recorded_boundary_hash_equal']
    # This exact recorded loop supports a narrower statement than pose tolerance.
    assert (fragments['desk_work_17_action_only__01']['parent']['objects'] ==
            fragments['desk_work_17_action_only__03']['parent']['objects'])
    report = {'status': 'passed', 'source_files_verified': len(verified),
        'source_sha256': verified, 'corpus_units_preserved': len(roles),
        'reason_units_checked': len(reasons), 'deletion_trials_checked': len(trials),
        'recorded_loop_boundary_and_object_table_equal': True,
        'original_sources_mutated': 0, 'native_scene_calls': 0,
        'model_calls': 0, 'training_updates': 0,
        'scope': 'Saved records, mapping and consistency checks; not full upstream compression, image rebinding, native physics or room-quality validation.'}
    write(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
