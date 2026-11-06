"""Bind historical supervision to existing initial/end-state evidence; no inference.

This produces a retrospective data-use overlay, never rewrites trained episodes.
It does not infer whole-room quality or rerun the historical geometry evaluator.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    refs = {}

    def read(path):
        path = Path(path)
        if not path.is_absolute():
            path = repo / path
        key = str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)
        refs[key] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        return json.loads(path.read_text())

    print(datetime.now(timezone.utc).isoformat(), 'bind historical F-House evidence', flush=True)
    f = Path('runs/920/P920-FCHOICE-v2_20261105T193428Z')
    p = Path('runs/920/P920-FPOLICY-v1_20261105T195352Z')
    s = Path('runs/920/P920-EDITSTUDY-v1_20261106T074752Z')
    episode_path = s / 'dataset/decision_action/fhouse_perimeter_01/episode.json'
    episode = read(episode_path)
    dataset = read(s / 'DATASET.json')
    row = next(x for x in dataset['rows'] if x['id'] == 'fhouse_perimeter_01')
    require(row['documents']['decision_action']['sha256'] == sha(repo / episode_path),
            'Historical training episode changed')
    initial = read(f / 'perimeter/native/initial_state.json')
    before = read(f / 'perimeter/native/step_01_before.json')
    after = read(f / 'perimeter/native/step_01_after.json')
    final = read(f / 'perimeter/native/final_state.json')
    steps = read(f / 'perimeter/native/STEPS.json')
    require(initial == before and after == final, 'Initial/terminal linkage mismatch')
    require(canonical(initial) == steps[0]['before_sha256'] and
            canonical(after) == steps[0]['after_sha256'], 'Replay state hash mismatch')
    context = json.loads(episode['task_context']['scene_description'])
    require(set(context['objects']) == set(initial['objects']), 'Object inventory mismatch')
    for oid, obj in context['objects'].items():
        require(all(initial['objects'][oid][k] == v for k, v in obj.items()),
                'Training prestate mismatch: ' + oid)
    bindings = [read(f / 'perimeter/bindings/step_000.state_binding.json'),
                read(p / 'shared_visual_contact/bindings/step_000.state_binding.json')]
    images = episode['turns'][0]['images']
    require(len(images) == 5, 'Missing five-view observation')
    for binding in bindings:
        require(binding['state_sha256'] == canonical(initial) and
                binding['native_state'] == initial, 'Observation state mismatch')
        require(images == [x['path'] for x in binding['images']], 'Image ordering mismatch')
        for image in binding['images']:
            path = Path(image['path'])
            require(sha(path) == image['sha256'], 'Observation image hash mismatch')
            refs[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    plan = read(p / 'shared_visual_contact/PLAN.json')
    evaluation = read(p / 'shared_visual_contact/INITIAL_EVALUATION.json')
    receipt = read(p / 'execution/shared_visual_contact/initial_evaluation_execution/receipt.json')
    bundle = Path(plan['bundle'])
    manifest = read(bundle / 'manifest.json')
    require(sha(bundle / 'manifest.json') == plan['manifest_sha256'] ==
            evaluation['input_manifest_sha256'], 'Evaluation input manifest mismatch')
    house_path = bundle / manifest['inputs']['native_state']
    house = read(house_path)
    require(sha(house_path) == manifest['files'][manifest['inputs']['native_state']]['sha256'],
            'Evaluation state file hash mismatch')
    require(house['rooms'][manifest['selected_room_id']] == initial,
            'Evaluated initial state is not the training prestate')
    require(receipt['exit_code'] == 0 and
            receipt['command'][receipt['command'].index('--bundle') + 1] == str(bundle),
            'Historical execution receipt mismatch')
    spec_path = p / 'EVALUATION_PLAN.json'
    read(spec_path)
    require(sha(repo / spec_path) == evaluation['plan_sha256'], 'Evaluation plan mismatch')
    measurements = read(f / 'MEASUREMENT.json')
    endpoint = measurements['candidates']['perimeter']
    require(endpoint['initial_state_sha256'] == canonical(initial) and
            endpoint['final_state_sha256'] == canonical(after), 'Measurement state mismatch')
    # Same ordered rays are required before comparing their clear counts.
    require(len(evaluation['rays']) == len(endpoint['rays']) == 15, 'Ray count mismatch')
    for a, b in zip(evaluation['rays'], endpoint['rays']):
        for key in ('origin', 'target'):
            require(all(abs(x-y) < 1e-12 for x, y in zip(a[key], b[key])),
                    'Ray definitions differ')
        require(a['clear'] == b['clear'], 'Unexpected ray outcome difference')
    oid = 'recon_tall_storage_0'
    moved = [k for k in initial['objects'] if initial['objects'][k] != after['objects'][k]]
    require(moved == [oid], 'Unexpected changed object')
    source = read(f / 'SOURCE_RECORD.json')
    result = {
        'status': 'completed_retrospective_data_use_correction',
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'episode_id': row['id'], 'family_id': row['family_id'],
        'source_decision_id': source['decision_id'],
        'original_target': json.loads(episode['turns'][0]['content']),
        'original_command': steps[0]['command'],
        'prestate_canonical_sha256': canonical(initial),
        'poststate_canonical_sha256': canonical(after),
        'binding': {'training_object_records': len(context['objects']), 'views': 5,
                    'exact_evaluated_prestate': True, 'same_15_ray_definitions': True},
        'observed': {
            'initial_clear_rays': evaluation['clear_rays'],
            'final_clear_rays': endpoint['clear_rays'],
            'clear_ray_gain': endpoint['clear_rays'] - evaluation['clear_rays'],
            'initial_conditional_proxy_pass': evaluation['conditional_proxy_pass'],
            'initial_four_declared_places': evaluation['four_declared_places'],
            'initial_role_access': [{'body_m': a['square_body_width_m'],
                                     'passed': sum(t['reachable_from_entry'] for t in a['targets'].values()),
                                     'total': len(a['targets'])} for a in evaluation['access']],
            'final_fixed_access_all_pass': endpoint['all_declared_access'],
            'access_comparison_limit': 'Different role-relative and fixed approach definitions; do not claim equal accessibility or equal route lengths.',
            'initial_storage_transform': initial['objects'][oid]['transform'],
            'final_storage_transform': after['objects'][oid]['transform'],
            'actual_translation_distance_m': math.dist(initial['objects'][oid]['transform']['translation'],
                                                       after['objects'][oid]['transform']['translation']),
            'edits': 1, 'other_object_records_unchanged': True},
        'future_data_use': {
            'status': 'candidate_overlay_not_encoded_not_trained',
            'retain_source_principle': True,
            'retain_actual_move_as_feasible_action_record': True,
            'admit_original_as_positive_incremental_visibility_decision': False,
            'reason': 'Claimed reduction of separation is unsupported relative to its own initial state under the recorded visibility proxy.',
            'candidate_decision': '初态已满足声明的共同交流视线、四餐位及对象相对接近代理；在这些指标下可维持现状。现有靠边移柜是另一可行端点，未显示视线增益，且增加一次编辑。',
            'candidate_authorship': 'Researcher interpretation of historical execution evidence, not an original designer statement or expert preference.',
            'no_edit_scope': 'Admissible for measured initial shared-contact proxies only; not professionally optimal or whole-room accepted.',
            'endpoint_contrast_retained': 'Perimeter vs divider remains 15 vs 0 clear rays; this does not establish improvement over the initial layout.',
            'training_episode_and_adapters_unchanged': True},
        'limits': ['No causal attribution of training failure to this one target.',
                   'No new professional decision, project, human rating, model call, training update or scene edit.',
                   'Physical moving effort, comfort, acoustic privacy and complete-room success remain unknown.',
                   'Existing evaluation reused; no geometry or model rerun.'],
        'sources': refs, 'script_sha256': sha(Path(__file__))}
    require(result['observed']['clear_ray_gain'] == 0 and evaluation['conditional_proxy_pass'],
            'Finding not supported')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as file:
        file.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'sources': len(refs),
                      'clear_ray_gain': 0, 'historical_training_unchanged': True}), flush=True)


if __name__ == '__main__':
    main()
