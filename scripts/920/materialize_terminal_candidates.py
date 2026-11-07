"""Bind two prospective no-edit targets to saved states; no scene execution."""
import argparse
import copy
import json
from pathlib import Path

from executed_edit_request import build_executed_request
from edit_request import normalize_context
from materialize_executed_edit_corpus import read, ref, require, sha, write
from native_checkpoint import canonical_hash
from native_project import validate_bundle
from terminal_supervision import terminal_text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    a = ap.parse_args(); p = read(a.plan); out = Path(p['output'])/'terminal_corpus'
    require(not out.exists(), 'Refuse to replace terminal candidates')
    for path, digest in p['input_hashes'].items():
        require(sha(path) == digest, 'Frozen input changed: '+path)
    data = read(p['base_dataset']); root = Path(p['repo_root']); parent = root.parent
    source = root/'runs/920/P920-WATERLOO-v1_20261105T074801Z/SUMMARY.json'
    summary = read(source); rows = []
    for case in p['cases']:
        old = next(x for x in data['rows'] if x['id'] == case['input_id'])
        input_ref = old['execution_inputs']; require(sha(input_ref['path']) == input_ref['sha256'], 'Input changed')
        args = read(input_ref['path']); measured = read(case['measurement'])
        require(measured['review']['measured_requirements_status'] == 'pass', 'Unmet condition cannot supply satisfied stop')
        require(measured['review']['requested_mode'] == case['mode'], 'Wrong mode')
        evidence = next(x for x in summary['rows'] if x['state'] == case['state'])
        state_path = parent/evidence['state_ref']['path']; state = read(state_path)
        require(sha(state_path) == evidence['state_ref']['sha256'], 'State changed')
        bundle = Path(case['bundle'])
        require(sha(bundle/'manifest.json') == case['bundle_manifest_sha256'], 'Bundle changed')
        manifest, house, _, _ = validate_bundle(bundle)
        bound_state = house['rooms'][manifest['selected_room_id']]
        require(canonical_hash(bound_state) == measured['state_sha256'], 'Review belongs to another bundle')
        # JSON int/float serialization differs across portable snapshots. Exact
        # recursive numeric equality, not a pose tolerance, binds observations.
        require(state == bound_state, 'Saved observation differs from measured bundle')
        native = normalize_context({'objects': state['objects']})['objects']
        represented = normalize_context(args['context'])['objects']
        require(set(native) == set(represented), 'Input inventory differs')
        require(all(all(k in native[oid] and native[oid][k] == value for k, value in obj.items())
                    for oid, obj in represented.items()), 'Represented object field differs')
        binding_path = parent/evidence['binding_ref']['path']
        require(sha(binding_path) == evidence['binding_ref']['sha256'], 'Binding changed')
        binding = read(binding_path)
        require(binding['state_sha256'] == canonical_hash(state), 'View binding state changed')
        require(args['images'] == [str(parent/x) for x in evidence['view_paths']], 'Wrong saved views')
        for image in old['images']:
            require(sha(image['path']) == image['sha256'], 'Image changed')
        if args['history']:
            require(args['history'][-1]['after_sha256'] == canonical_hash(state), 'History not current')
        context = args['context']
        context['current_activity_request'] = dict(request=case['request'],
            authorship='researcher_specified_pre_action_activity',
            scope='Prospective current condition on a saved reconstruction, not a historical resident utterance.')
        context['current_requested_mode'] = case['mode']
        # Both edit and terminal candidates may observe already acquired tool
        # measurements. Never provide the aggregate stop label or target prose.
        context['available_measurement'] = copy.deepcopy(measured['measurements'])
        context['declared_activity_requirements'] = measured['feedback']['declared_requirements']
        args['additional_text'] = ('Current supplied measurements are declared geometric proxies. '
            'If they show no unmet requirement for the current activity, preserving the current state '
            'is allowed; otherwise address deficits. Stopping never certifies whole-room usability.')
        request, accounting = build_executed_request(**args)
        target = dict(requested_mode=case['mode'],
            measured_scope='Declared object identity, fixed geometry, assigned tabletop places and proxy access checks for the current activity.',
            unmet_in_measured_scope=[], unknowns=context['unknowns'], whole_room_success='unknown')
        content = terminal_text(target)
        require(content not in json.dumps(request, ensure_ascii=False), 'Target leaked to input')
        doc_ref = old['documents']['action_only']
        require(sha(doc_ref['path']) == doc_ref['sha256'], 'Template changed')
        doc = read(doc_ref['path'])
        doc['task_context']['scene_description'] = json.dumps(context, ensure_ascii=False)
        doc['turns'] = [dict(role='assistant', content=content, tool_calls=[], images=args['images'])]
        doc.pop('evaluation_only', None)
        doc['provenance'] = dict(family_id=old['family_id'],
            kind='researcher_measured_scope_terminal_policy', original_designer_action=False,
            source_decision_target=False, new_professional_decisions=0, training_authorized=False,
            serialization='terminal-control-v1', official_ota_serialization_overridden=True)
        folder = out/case['id']
        for name, value in [('inputs.json', args), ('request.json', request), ('target.json', target), ('episode.json', doc)]:
            write(folder/name, value)
        require(build_executed_request(**read(folder/'inputs.json'))[0] == request, 'Request roundtrip differs')
        rows.append(dict(id=case['id'], family_id=old['family_id'], target_kind='terminal_control',
            input=ref(folder/'inputs.json'), request=ref(folder/'request.json'),
            target=ref(folder/'target.json'), document=ref(folder/'episode.json'),
            source_input=input_ref, measured_evidence=ref(case['measurement']),
            state=ref(state_path), view_binding=ref(binding_path), images=old['images'],
            history_accounting=accounting, state_sha256=canonical_hash(state),
            measured_bundle_state_sha256=canonical_hash(bound_state),
            observation_bundle_binding='exact_recursive_numeric_equality; JSON integer/float spelling can differ',
            identical_in_arms=['action_only', 'decision_action']))
    write(out/'DATASET.json', dict(schema='scenefunc-terminal-candidates-v1', rows=rows,
        counts=dict(terminal_states=len(rows), source_families=1, new_professional_decisions=0,
                    new_scene_edits=0, arm_records=2*len(rows)),
        base_dataset=ref(p['base_dataset']), plan=ref(a.plan),
        limits=['Prospective researcher stopping policy; no historical designer or model stop observed.',
                'The satisfied label concerns declared current-activity proxies only; whole-room success remains unknown.',
                'Two correlated states from a single already exposed training family. Not new generalization evidence.',
                'No rejection/recovery targets added. No new training authorization.']))
    print('Materialized', len(rows), 'terminal states for both arms', flush=True)


if __name__ == '__main__':
    main()
