"""Archive a terminal structured-policy attempt without editing or resampling."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from PIL import Image
from native_project import sha256, validate_bundle
from native_checkpoint import canonical_hash, rotation_roundtrip_differences
from shared_action_contract import ActionContract, context_from_request, schemas_from_observation


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    public = repo / 'runs/920' / args.run
    plan = read(public / 'PLAN.json')
    runtime = Path(plan['runtime'])
    receipt = read(runtime / 'rollout_stage/receipt.json')
    assert receipt['status'] != 'running'
    result = read(runtime / 'rollout/RESULT.json')
    selection = read(public / 'TERMINAL_SELECTION.json')
    bundle = Path(selection['bundle'])
    assert sha256(bundle / 'manifest.json') == selection['manifest_sha256']
    manifest, state, _, _ = validate_bundle(bundle)
    native = state['rooms'][manifest['selected_room_id']]
    print(datetime.now(timezone.utc).isoformat(), 'archive terminal structured policy', flush=True)
    copies = []
    def cp(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha256(source) == sha256(target)
        copies.append({'path': str(target.relative_to(public)), 'source': str(source), 'sha256': sha256(source)})
    initial = read(runtime / 'prepare/observation.json')
    tools = schemas_from_observation(initial)
    observations = [('initial', initial)]
    models, validations, dispatched = [], [], []
    for folder in sorted((runtime / 'rollout').glob('round_*')):
        if (folder / 'model/RESULT.json').exists():
            models.append(read(folder / 'model/RESULT.json'))
        if (folder / 'VALIDATED_ACTION.json').exists():
            ctx = context_from_request(read(folder / 'request.json'))
            contract = ActionContract(tools, ctx)
            checked = contract.validate((folder / 'model/action_output.txt').read_text(), ctx)
            assert checked == read(folder / 'VALIDATED_ACTION.json') == read(folder / 'model/VALIDATED_ACTION.json')
            validations.append({'round': folder.name, **checked})
        if (folder / 'tool_reply.json').exists():
            dispatched.append({'round': folder.name, 'calls': read(folder / 'parsed_calls.json'),
                               'reply': read(folder / 'tool_reply.json')})
        if (folder / 'observation.json').exists():
            observations.append((folder.name, read(folder / 'observation.json')))
        for f in list(folder.glob('*.json')) + list((folder / 'model').glob('*.json')) + list((folder / 'model').glob('*.txt')):
            cp(f, public / 'rollout' / f.relative_to(runtime / 'rollout'))
    bindings, unique_states, unique_images = [], set(), set()
    for label, obs in observations:
        binding = read(Path(obs['state_binding']))
        assert canonical_hash(binding['native_state']) == binding['state_sha256']
        assert len(obs['image_paths']) == len(binding['images']) == 5
        for path, image in zip(obs['image_paths'], binding['images']):
            assert path == image['path'] and sha256(Path(path)) == image['sha256']
            with Image.open(path) as picture:
                picture.verify()
            unique_images.add(image['sha256'])
        cp(Path(obs['state_binding']), public / 'bindings' / (label + '.json'))
        if binding['state_sha256'] not in unique_states:
            cp(Path(obs['image_paths'][0]), public / 'views' / (label + '.topdown.png'))
        unique_states.add(binding['state_sha256'])
        bindings.append({'label': label, 'state_sha256': binding['state_sha256'], 'images': binding['images']})
    final_obs = read(Path(selection['observation']))
    final_binding = read(Path(final_obs['state_binding']))
    differences = rotation_roundtrip_differences(final_binding['native_state'], native)
    for index, path in enumerate(final_obs['image_paths']):
        cp(Path(path), public / 'views' / ('final.' + str(index) + '.png'))
    for folder in ['prepare', 'rollout', 'prepare/encode', 'export', 'cold_restore']:
        for f in (runtime / folder).glob('*.json'):
            cp(f, public / folder / f.name)
    for name in ['FINAL_AUDIT.json', 'INITIAL_AUDIT.json']:
        cp(runtime / name, public / name)
    for name in ['manifest.json', 'house_state.json', 'runtime.json', 'task.json', 'catalog.json', 'session_checkpoint.json']:
        if (bundle / name).exists():
            cp(bundle / name, public / 'checkpoint_metadata' / name)
    for f in sorted(runtime.rglob('receipt.json')):
        rec = read(f)
        if rec['status'] == 'running':
            assert any(Path(x).name == Path(__file__).name for x in rec['command'])
            continue
        cp(f, public / 'execution' / f.relative_to(runtime))
        cp(f.parent / 'stage.log', public / 'execution' / f.parent.relative_to(runtime) / 'stage.log')
    audit = read(runtime / 'FINAL_AUDIT.json')
    evaluation = read(Path(plan['evaluation']['plan']))
    assert sha256(Path(plan['evaluation']['plan'])) == plan['evaluation']['sha256']
    counts = Counter(row['name'] for row in audit['inventory'])
    roles = {key: {'present': sum(counts[name] for name in rule['catalog_names']),
                   'minimum': rule['minimum'], 'presence_pass': sum(counts[name] for name in rule['catalog_names']) >= rule['minimum'],
                   'functional_use': 'unknown'} for key, rule in evaluation['role_counts'].items()}
    access = [{'body_width_m': row['square_body_width_m'], 'entry_clear': row['entry_cell_clear'],
               'targets': len(row['targets']), 'reachable': sum(x['reachable_from_entry'] for x in row['targets'].values()),
               'unreachable': [name for name, value in row['targets'].items() if not value['reachable_from_entry']]}
              for row in audit['access']['scenarios']]
    summary = {'run_id': args.run, 'status': 'terminal_archived_pending_publication',
               'rollout': {key: receipt[key] for key in ['started_at', 'ended_at', 'elapsed_seconds', 'exit_code', 'first_log']},
               'policy_status': result['status'], 'stop_reason': result.get('stop_reason'), 'error': result.get('error'),
               'model_calls': result['model_calls'], 'tool_attempts': result['tool_calls'],
               'recorded_native_dispatches': len(dispatched), 'validated_outputs': len(validations),
               'state_changes': sum(x.get('after_sha256') is not None and x['before_sha256'] != x['after_sha256'] for x in result['rounds']),
               'furniture_count': len(audit['inventory']), 'roles': roles,
               'all_role_presence_pass': all(x['presence_pass'] for x in roles.values()),
               'immutable_records_unchanged': all(audit['immutable_records_unchanged'].values()),
               'mesh_contacts': audit['surface_intersection_pairs'],
               'outside_count': sum(x['outside_nominal_room_xy'] for x in audit['bounds']), 'access': access,
               'input_tokens': sum(x.get('encoding', {}).get('input_tokens', 0) for x in models),
               'output_tokens': sum(x.get('output_tokens', 0) for x in models),
               'constraint_setup_seconds': sum(x.get('constraint_setup_seconds', 0) for x in models),
               'model_load_seconds': sum(x.get('load_seconds', 0) for x in models),
               'generation_seconds': sum(x.get('generation_seconds', 0) for x in models),
               'observations': len(observations), 'image_instances': len(observations)*5,
               'unique_states': len(unique_states), 'unique_image_hashes': len(unique_images),
               'final_observation_roundtrip_differences': differences,
               'training_updates': 0, 'professional_review': False, 'independent_test': False,
               'whole_room_success': 'not_accepted', 'learning_or_generalization_gain': 'not_tested'}
    write(public / 'SUMMARY.json', summary)
    write(public / 'ACTIONS.json', {'validated': validations, 'dispatched': dispatched})
    write(public / 'OBSERVATIONS.json', bindings)
    write(public / 'COPY_RECORD.json', copies)
    print(json.dumps({'furniture': summary['furniture_count'], 'roles_present': summary['all_role_presence_pass'],
                      'contacts': summary['mesh_contacts'], 'model_calls': summary['model_calls']}), flush=True)


if __name__ == '__main__':
    main()
