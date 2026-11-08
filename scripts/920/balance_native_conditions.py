"""Uniform pre-action measurements and condition wording for six Waterloo records.

No target, reference endpoint or proposed action enters the measurement function
or shared deployment request builder. Old corpus is preserved as evidence.
"""
import argparse
import copy
import json
from pathlib import Path

from materialize_native_completion import read, write, sha, ref, require, locked
from native_completion_supervision import build_native_request
from conditional_room_metrics import measure_endpoint
from native_checkpoint import canonical_hash
from native_project import validate_bundle
from edit_request import normalize_context


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    args = ap.parse_args()
    plan = read(args.plan)
    for path, digest in plan['inputs'].items():
        require(sha(path) == digest, 'Frozen source changed: ' + path)
    require(plan['model_calls'] == plan['training_updates'] == plan['scene_edits'] == 0, 'CPU-only scope')
    output = Path(plan['output'])
    require(not (output / 'corpus').exists(), 'Refuse to overwrite candidate')
    data = read(plan['source_dataset'])
    evaluation = read(plan['measurement_plan'])
    root = Path(evaluation['initial_bundle'])
    require(sha(root / 'manifest.json') == evaluation['initial_manifest_sha256'], 'Initial bundle changed')
    manifest, initial, _, _ = validate_bundle(root)
    room_id = manifest['selected_room_id']
    initial = initial['rooms'][room_id]
    requests = plan['activity_requests']
    rows = []
    measurements = {}
    input_by_id = {}
    audit = []
    print('Start uniform pre-action condition materialization;0model/0sceneedit', flush=True)
    for source in data['rows']:
        row = copy.deepcopy(source)
        if row['id'] not in plan['modes']:
            rows.append(row)
            continue
        inputs = locked(row['input'])
        context = inputs['context']
        terminal = row['target_kind'] == 'terminal_control'
        evidence = row['source']['state'] if terminal else row['source']['execution_evidence']['pre_action_state']
        state = locked(evidence)
        native_objects = normalize_context({'objects': state['objects']})['objects']
        observed = normalize_context(context)['objects']
        require(set(native_objects) == set(observed), 'Current object identities differ')
        require(all(all(key in native_objects[oid] and value == native_objects[oid][key]
                        for key, value in obj.items()) for oid, obj in observed.items()),
                'Current observed object fields differ from measured state')
        mode = plan['modes'][row['id']]
        measured = measure_endpoint(state, initial, root / ('room_' + room_id), root, evaluation['program'], mode)
        # This complete result is evaluation-only provenance. The request gets
        # raw pre-action measurements and declared requirements, never checks,
        # their aggregate, a stop flag, target poses or reference commands.
        mpath = output / 'measurements' / (row['id'] + '.json')
        write(mpath, dict(state_sha256=canonical_hash(state), source_state=evidence, **measured))
        context['available_measurement'] = measured['measurements']
        context['declared_activity_requirements'] = evaluation['program']
        context['current_requested_mode'] = mode
        context['current_activity_request'] = dict(request=requests[mode],
            authorship='researcher_specified_pre_action_activity',
            scope='Common current-condition wording for the same saved reconstruction; not a historical resident quote.')
        inputs['additional_text'] = plan['common_additional_text']
        request, accounting = build_native_request(**inputs)
        folder = output / 'corpus' / row['id']
        write(folder / 'inputs.json', inputs)
        write(folder / 'request.json', request)
        require(build_native_request(**read(folder / 'inputs.json'))[0] == request, 'Deployment roundtrip changed')
        documents = {}
        for arm in ('action_only', 'decision_action'):
            doc = locked(row['documents'][arm])
            doc['task_context']['scene_description'] = json.dumps(context, ensure_ascii=False)
            doc['provenance']['conditional_input_version'] = 'same-state-condition-v1'
            path = folder / arm / 'episode.json'
            write(path, doc)
            documents[arm] = ref(path)
            require(not doc['turns'][0]['content'] or doc['turns'][0]['content'] not in json.dumps(request, ensure_ascii=False), 'Target leaked to request')
        row.update(input=ref(folder / 'inputs.json'), request=ref(folder / 'request.json'),
            documents=documents, history_accounting=accounting, balanced_pre_action_measurement=ref(mpath))
        rows.append(row)
        input_by_id[row['id']] = inputs
        measurements[row['id']] = measured
        audit.append(dict(id=row['id'], native_state_sha256=canonical_hash(state), requested_mode=mode,
            current_measured_status=measured['review']['measured_requirements_status'],
            target_kind=row['target_kind'], original_input=source['input'], new_input=row['input']))
        print(row['id'], 'uniform current-state measurements prepared', flush=True)
    pairs = []
    for edit_id, stop_id in plan['same_state_pairs']:
        a, b = copy.deepcopy(input_by_id[edit_id]), copy.deepcopy(input_by_id[stop_id])
        require(measurements[edit_id]['measurements'] == measurements[stop_id]['measurements'], 'Raw measurements depend on requested outcome')
        changed = []
        for field in ('current_activity_request', 'current_requested_mode'):
            if a['context'][field] != b['context'][field]:
                changed.append(field)
            a['context'].pop(field)
            b['context'].pop(field)
        require(a == b, 'Same-state pair differs beyond activity condition')
        require(changed == ['current_activity_request', 'current_requested_mode'], 'No actual condition contrast')
        pairs.append(dict(edit_id=edit_id, stop_id=stop_id, same_state_and_raw_measurements=True,
            only_input_differences=changed, input_identity_after_removing_condition=True))
    require(len(audit) == 6 and len(rows) == 24, 'Candidate count changed')
    write(output / 'corpus/DATASET.json', dict(schema='native-completion-matched-condition-v1', rows=rows,
        counts=dict(examples=24, tool_actions=22, terminal_states=2, updated_inputs=6, same_state_pairs=2),
        previous_dataset=ref(plan['source_dataset']), model_calls=0, training_updates=0, scene_edits=0,
        source_decision_or_action_targets_changed=False,
        limits='Two matched pairs and two intermediate steps in one exposed reconstruction; not independent professional or generalization evidence.'))
    write(output / 'PAIR_AUDIT.json', dict(status='matched_inputs_verified', pairs=pairs, rows=audit,
        new_model_calls=0, new_training_updates=0, new_scene_edits=0,
        limits='Removes identified differential input cues; does not prove a model used them or eliminate all dataset shortcuts. Current review values are not included in requests.'))


if __name__ == '__main__':
    main()
