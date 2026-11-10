"""Verify recovered inputs, not learned performance or whole-scene quality."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args(); root = a.run
    read = lambda name: json.loads((root/name).read_text())
    checks = []
    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)
    for p, expected in read('protected_source_hashes.json').items():
        check('protected:'+p, sha(p) == expected)
    for r in read('fhouse_v3/resource_checks.json'):
        check('geometry:'+r['path'], sha(root/'fhouse_v3/native_build/bundle'/r['path']) == r['expected_sha256'])
    for r in read('fhouse_v3/raw_asset_checks.json'):
        check('raw:'+r['asset_id'], sha(r['actual_path']) == r['sha256'])
    for p, r in read('frozen_renderer_fingerprint.json')['files'].items():
        check('renderer:'+p, sha(Path('runs/autodl/20261110_native_observation_export/checkpoint_renderer')/p) == r['expected'])
    ex = read('fhouse_export/result.json'); ann = read('fhouse_views/annotation_result.json')
    check('same_recorded_state', ex['source_room_sha256'] == ex['runtime_room_sha256'] == ann['runtime_room_sha256'] == ann['source_room_sha256'])
    check('no_frozen_state_edits', ex['frozen_override']['changed_from_bundle'] == [])
    check('five_native_views', len(ann['images']) == 5 and ann['status'] == 'native_five_view_observation_completed')
    for r in ann['images']:
        check('annotated_image:'+r['view'], sha(r['path']) == r['sha256'])
    old = [json.loads(s) for s in Path('runs/autodl/20261110_stage_alignment_review/material_v2/components.jsonl').read_text().splitlines()]
    new = [json.loads(s) for s in (root/'material_v3/components.jsonl').read_text().splitlines()]
    check('all_components_retained', len(old) == len(new) == 68)
    check('only_observation_messages_changed', all({k:v for k,v in x.items() if k != 'messages'} == {k:v for k,v in y.items() if k != 'messages'} for x,y in zip(old,new)))
    joint = read('joint_v3/summary.json'); enc = read('joint_multimodal_v3/summary.json')
    check('24_units_2975_original_targets', joint['units'] == 24 and joint['source_tokens_once'] == 2975 and joint['every_joined_target_token_verified'])
    check('14_encoded_10_still_missing', enc['encoded'] == 14 and enc['missing_views'] == 10)
    check('common_pixels_and_deployment_prefix', all(r['common_vision_exact'] and r['joint_deployment_prefix_exact'] for r in enc['records'] if r['status'] == 'encoded'))
    check('no_models_training_or_scenes', all(enc[k] == 0 for k in ['actual_model_calls','optimizer_updates','paid_api_calls','new_complete_scenes']))
    check('training_not_admitted', not enc['training_ready'])
    for p, expected in enc['source_sha256'].items():
        check('encoding_source:'+p, sha(p) == expected)
    for name in ['check_joint_multimodal_control.py','audit_frozen_decision_schema.py','prepare_frozen_reference_asset.py','recover_fhouse_observation_bundle.py']:
        ast.parse((Path('scripts/autodl')/name).read_text())
    result = {'status':'passed','checks':checks,'count':len(checks),
              'scope':'Resource/observation/material consistency, actual processor receipts. Not independent geometry correctness, human preference, model quality, or full generation.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','checks':len(checks)}))


if __name__ == '__main__':
    main()
