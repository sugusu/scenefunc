"""Audit completed fixed-state recovery and unchanged training supervision.

This verifies artifacts and actual encoder receipts, not learned policy quality.
It never renders, changes a state, starts inference, or admits training.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args(); root = a.run
    read = lambda name: json.loads((root / name).read_text())
    checks = []
    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
        if not value:
            raise AssertionError(name)
    batch = read('views_v2/result.json')
    check('five_fixed_states_completed', batch['status'] == 'completed' and
          {r['unit_id'] for r in batch['rows']} == {f'storage_edit_{i:02}' for i in range(1,6)})
    check('inputs_unchanged', batch['inputs_unchanged'] and
          all(sha(p) == h for p,h in batch['inputs_sha256'].items()))
    resources = read('native_v2/resource_checks.json')
    check('24_exact_native_resources', len(resources) == 24 and all(
          sha(root / 'native_v2/native_build/bundle' / r['path']) == r['expected_sha256'] for r in resources))
    raw = read('native_v2/raw_asset_checks.json')
    check('two_exact_original_meshes', len(raw) == 2 and all(sha(r['path']) == r['sha256'] for r in raw))
    states = {r['unit_id']: r for r in json.loads(Path(
        'runs/autodl/20261110_corpus_state_recovery/state_audit_v2/units.json').read_text())}
    for row in batch['rows']:
        uid = row['unit_id']
        ex = json.loads(Path(row['export']).read_text())
        ann = json.loads(Path(row['annotation']).read_text())
        check(uid+':source_bound', ex['source_room_sha256'] == ann['source_room_sha256'] == states[uid]['state_canonical_sha256'])
        actual = json.loads((Path(row['export']).parent/'actual_room_state.json').read_text())
        actual_hash = hashlib.sha256(json.dumps(actual, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        check(uid+':runtime_bound', actual_hash == ex['runtime_room_sha256'] == ann['runtime_room_sha256'])
        if ex['source_room_sha256'] != actual_hash:
            proof = json.loads(Path(row['serialization_proof']).read_text())
            check(uid+':exact_native_roundtrip', proof['status'] == 'exact_native_serialization' and
                  proof['source_sha256'] == ex['source_room_sha256'] and proof['runtime_sha256'] == actual_hash and
                  proof['all_other_fields_exact'] and proof['scene_edits'] == 0)
        check(uid+':five_images', ann['status'] == 'native_five_view_observation_completed' and
              len(ann['images']) == 5 and all(sha(im['path']) == im['sha256'] for im in ann['images']))
    old_path = Path('runs/autodl/20261110_remaining_observation_recovery/material_v3/components.jsonl')
    old = [json.loads(s) for s in old_path.read_text().splitlines()]
    new = [json.loads(s) for s in (root/'material_v4/components.jsonl').read_text().splitlines()]
    check('68_components_unchanged_except_observations', len(old) == len(new) == 68 and all(
        {k:v for k,v in x.items() if k != 'messages'} == {k:v for k,v in y.items() if k != 'messages'}
        for x,y in zip(old,new)))
    joint = [json.loads(s) for s in (root/'joint_v4/components.jsonl').read_text().splitlines()]
    source = {(r['unit_id'], r['stage']):r for r in new if r['arm'] == 'paired_phase_v3'}
    check('24_joint_units', len(joint) == 24 and len({r['unit_id'] for r in joint}) == 24)
    for row in joint:
        components = [source[(row['unit_id'], stage)] for stage in ['decision','action'] if (row['unit_id'], stage) in source]
        check(row['unit_id']+':same_targets_and_weights', all(row[key] == sum([c[key] for c in components], [])
              for key in ['target_token_ids','token_objective_weights','source_token_origins']))
    check('2975_original_targets', sum(len(r['target_token_ids']) for r in joint) == 2975)
    encoding = read('joint_multimodal_v4/summary.json')
    encoded = [r for r in encoding['records'] if r['status'] == 'encoded']
    missing = {r['unit_id'] for r in encoding['records'] if r['status'] != 'encoded'}
    check('19_encoded_5_retained_missing', len(encoded) == 19 and missing == {
        'air_edit_01','waterloo_edit_02','waterloo_edit_04','lifecycle_02','lifecycle_03'})
    check('matched_vision_and_joint_prefix', all(r['common_vision_exact'] and r['joint_deployment_prefix_exact'] for r in encoded))
    check('encoding_sources_unchanged', all(sha(p) == h for p,h in encoding['source_sha256'].items()))
    check('no_model_training_or_generation', all(encoding[k] == 0 for k in
        ['actual_model_calls','optimizer_updates','paid_api_calls','new_complete_scenes']) and not encoding['training_ready'])
    result = {'status':'passed','checks':checks,'count':len(checks),
              'scope':'Input identity, original supervision, actual processor receipts. Not human preference, physical correctness, policy quality or whole-scene success.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','checks':len(checks),'encoded_units':len(encoded),'missing_units':sorted(missing)}))


if __name__ == '__main__':
    main()
