"""Validate and summarize the closed STORAGEGATE evidence; never run inference."""
import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from native_checkpoint import canonical_hash


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    public = args.public.resolve()
    runtime = root.parent / 'runtime' / public.name
    arms = ['frozen_base_knowledge_feedback', 'action_only', 'decision_action']
    counts = dict(archive_copies=0, observation_bindings=0, model_image_hashes=0,
                  frozen_code_hashes=0, frozen_input_hashes=0, terminal_restores=0,
                  geometry_exports=0, execution_receipts=0)
    prep = read(runtime / 'prepare/RESULT.json')
    rows, claim_checks = [], []
    table = []
    common_copies = []
    for name in ['initial_state.json', 'initial_measurement.json', 'initial_request.json',
                 'initial_observation.json', 'RESULT.json', 'checkpoint/task.json']:
        source = runtime / 'prepare' / name
        target = public / 'prepare/evidence' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        common_copies.append(dict(source=str(source), destination=str(target.relative_to(public)), sha256=sha(target)))
    obs = read(runtime / 'prepare/initial_observation.json')
    binding = read(Path(obs['state_binding']))
    assert canonical_hash(binding['native_state']) == canonical_hash(read(runtime / 'prepare/initial_state.json'))
    for img in prep['encoding']['images']:
        assert sha(Path(img['origin'])) == img['sha256']
    top = Path(prep['encoding']['images'][0]['origin'])
    target = public / 'prepare/initial.topdown.png'
    shutil.copy2(top, target)
    common_copies.append(dict(source=str(top), destination=str(target.relative_to(public)), sha256=sha(target)))
    write(public / 'prepare/ARCHIVE.json', dict(copies=common_copies, images=prep['encoding']['images'],
          state_binding_verified=True, initial_input_sha256=prep['initial_request_sha256']))
    for arm in arms:
        row = read(public / arm / 'REVIEW_ROW.json')
        result = read(runtime / arm / 'RESULT.json')
        plan = read(public / arm / 'PLAN.json')
        archive = read(public / arm / 'ARCHIVE.json')
        for item in archive['copies']:
            assert sha(public / item['destination']) == item['sha256'], item['destination']
            assert sha(Path(item['source'])) == item['sha256'], item['source']
            counts['archive_copies'] += 1
        for item in archive['views']:
            state_file = item['observation'].replace('_observation', '_state')
            state = read(runtime / arm / state_file)
            binding = read(public / arm / 'evidence/bindings' / item['observation'])
            assert canonical_hash(state) == item['state_sha256'] == canonical_hash(binding['native_state'])
            assert len(item['images']) == 5
            for img in item['images']:
                assert sha(Path(img['path'])) == img['sha256']
            counts['observation_bindings'] += 1
        for name, expected in plan['code_sha256'].items():
            assert sha(root / 'scripts/920' / name) == expected, name
            counts['frozen_code_hashes'] += 1
        for item in plan['inputs'].values():
            assert sha(Path(item['path'])) == item['sha256']
            counts['frozen_input_hashes'] += 1
        assert result['plan_sha256'] == sha(public / arm / 'PLAN.json')
        assert result['initial_request_sha256'] == row['first_input_sha256'] == prep['initial_request_sha256']
        assert sha(runtime / arm / 'round_000/request.json') == prep['initial_request_sha256']
        assert len(result['rounds']) == row['model_calls'] == 6
        assert sum('reply' in r for r in result['rounds']) == row['native_calls']
        assert sum('rejection' in r for r in result['rounds']) == row['rejected']
        assert sum(r['before_state_sha256'] != r['after_state_sha256'] for r in result['rounds'] if 'reply' in r) == row['actual_changes']
        for turn in result['rounds']:
            rd = runtime / arm / f"round_{turn['turn']:03d}"
            assert sha(rd / 'model/raw_output.txt') == turn['model']['raw_output_sha256']
            assert sha(rd / 'request.json') == turn['model']['request_sha256']
            for img in read(rd / 'model/ENCODING.json')['images']:
                assert sha(Path(img['origin'])) == img['sha256']
                counts['model_image_hashes'] += 1
            assert read(rd / 'receipt/receipt.json')['exit_code'] == 0
            counts['execution_receipts'] += 1
        for suffix in ['_receipt', '_restore_receipt', '_export_receipt']:
            assert read(runtime / (arm + suffix) / 'receipt.json')['exit_code'] == 0
            counts['execution_receipts'] += 1
        cp = Path(row['checkpoint']['directory'])
        assert sha(cp / 'manifest.json') == row['checkpoint']['manifest_sha256']
        assert canonical_hash(read(cp / 'house_state.json')) == row['checkpoint']['state_sha256']
        assert row['restore']['status'] == 'passed'
        assert row['restore']['identities'] == 26 and row['restore']['task_and_asset_bindings_exact']
        counts['terminal_restores'] += 1
        export = read(runtime / (arm + '_export') / 'RESULT.json')
        glb = Path(export['glb']['path'])
        if not glb.is_absolute():
            glb = root / glb
        assert sha(glb) == row['export']['sha256'] and glb.stat().st_size == row['export']['bytes']
        assert export['status'] == 'geometry_export_roundtrip_checked' and export['geometry_nodes'] == 42
        counts['geometry_exports'] += 1
        assert row['score'] == result['final_score']
        if arm != arms[0]:
            initial = read(runtime / 'prepare/initial_state.json')['objects']
            final = read(runtime / arm / 'final_state.json')['objects']
            unchanged = [key for key in ['recon_storage_b_0', 'recon_storage_c_0'] if initial[key] == final[key]]
            assert len(unchanged) == 2
            assert result['rounds'][0]['call']['arguments']['object_id'] == 'recon_storage_a_0'
            claim_checks.append(dict(arm=arm, evidence=f'{arm}/evidence/round_001/model/raw_output.txt',
                                     claim='All three storage units were moved successfully.',
                                     actual_unchanged_object_ids=unchanged,
                                     first_executed_object='recon_storage_a_0'))
        score = row['score']
        table.append(dict(arm=arm, model_calls=row['model_calls'], native_calls=row['native_calls'],
                          actual_changes=row['actual_changes'], rejections=row['rejected'],
                          fixed_access_counts=score['fixed_access_counts'],
                          storage_front_counts=score['storage_front_counts'],
                          living_free_gain_m2=score['living_free_gain_m2'],
                          AABB_overlap_pairs=len(score['geometry']['positive_3d_AABB_pairs']),
                          geometric_rule_passed=score['conditional_geometric_rule_passed']))
        rows.append(row)
    assert sum(r['model_calls'] for r in rows) == 18
    assert sum(r['native_calls'] for r in rows) == 7
    assert sum(r['actual_changes'] for r in rows) == 6
    assert sum(r['rejected'] for r in rows) == 11
    comparison = read(runtime / 'comparison_receipt/receipt.json')
    assert comparison['exit_code'] == 0
    assert all(read(public / 'SCORING_CHECKS.json').values())
    now = datetime.now(timezone.utc).isoformat()
    summary = dict(run_id=public.name, status='completed_reviewed_geometric_failure', reviewed_at=now,
        model_calls=18, native_calls=7, actual_state_changes=6, rejected_outputs=11,
        recovered_terminal_rooms=3, geometric_rule_passed_rooms=0, training_updates=0,
        gpu_subprocess_seconds=sum(r['gpu_subprocess_seconds'] for r in rows),
        gpu_time_scope='Elapsed inference subprocess time including loading/encoding; not measured GPU utilization.',
        first_input_sha256=prep['initial_request_sha256'], first_input_tokens=prep['encoding']['input_tokens'],
        execution_receipt=comparison, rows=rows, reference_gate=prep['reference_gate'],
        false_execution_claims=claim_checks,
        conclusion='The existing adapters do not establish a benefit on this exposed task. All three arms fail the frozen geometric rule.',
        limitations=['One training-exposed source and one seed sequence; not independent generalization.',
          'Only move_furniture_tool advertised. Seven adapter rejections requested unadvertised query tools; continuation is interface-confounded.',
          'Common current input differs from the old training representation. Corrected candidate_v3 was not trained.',
          'AABB and square-body reachability are declared proxies, not physical usability or independent professional room quality.',
          'No causal comparison with historical rejection rates and no claim of pairing-mechanism superiority.'],
        decision='Close this finite gate with no reruns. Current adapters are not ready for broader room evaluation. Any new learning study needs a matched training/deployment contract and separate concrete authorization.')
    write(public / 'SUMMARY.json', summary)
    write(public / 'RESULT_TABLE.json', table)
    with (public / 'RESULT_TABLE.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    write(public / 'VALIDATION.json', dict(status='passed', checked_at=now, counts=counts,
        scoring_checks=read(public / 'SCORING_CHECKS.json'), actual_counts_verified=True,
        actual_common_first_request_verified=True, false_claim_state_checks=claim_checks,
        visual_review='Initial and three final topdowns reviewed in preceding execution session; no new render.',
        scope='Evidence consistency and recovery/export acceptance; scientific geometric task failed in all arms.'))
    print(json.dumps(dict(status='passed', counts=counts, results=table), ensure_ascii=False))


if __name__ == '__main__':
    main()
