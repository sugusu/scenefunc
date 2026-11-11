"""Recover one sourced decision example without running models or a simulator.

Reads a fixed existing Git commit. Keeps the Pine development exclusion and
separates deployment observations, proposed supervision and later outcomes.
The historical 22-action loader/training receipts are reused, never rerun.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
PINE = 'runs/920/P920-PINE-v1_20261106T002523Z/'
STUDY = 'runs/920/P920-EDITSTUDY-v1_20261106T074752Z/'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Retain original outputs; use a new directory')
    args.out.mkdir(parents=True)
    started = time.monotonic()
    sources = []

    def blob(path):
        raw = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show',
                                       PIN + ':' + path])
        sources.append(dict(path=path, bytes=len(raw), sha256=sha(raw)))
        return raw

    def read(path):
        return json.loads(blob(path))

    task = read(PINE + 'TASK_INPUT.json')
    attribution = read(PINE + 'SOURCE_INTERPRETATION.json')
    packet = read(PINE + 'DECISION_PACKET.json')
    binding = read(PINE + 'bindings/step_000.state_binding.json')
    initial = read(PINE + 'native/initial_state.json')
    final = read(PINE + 'native/final_state.json')
    consequences = read(PINE + 'CONSEQUENCES.json')
    final_audit = read(PINE + 'FINAL_AUDIT.json')
    dataset = read(STUDY + 'DATASET.json')
    split = read(STUDY + 'SPLIT.json')
    preflight = read(STUDY + 'PREFLIGHT_RESULT.json')
    dry = {arm: read(STUDY + fname) for arm, fname in
           [('action', 'DRY_ACTION.json'), ('joint', 'DRY_JOINT.json')]}
    training = {arm: read(STUDY + fname) for arm, fname in
                [('action', 'ACTION_TRAINING_RESULT.json'),
                 ('joint', 'JOINT_TRAINING_RESULT.json')]}
    messages = read(STUDY +
                    'evaluation/Pine_action_only_17/recorded/round_000/request.json')
    target = packet['records'][0]['command']
    assert target == dict(name='move_furniture_tool',
                          arguments=dict(object_id='old_desk_0',
                                         x=2.09, y=0.25, yaw=90))
    assert packet['records'][0]['before_state_sha256'] == binding['state_sha256']
    assert initial == binding['native_state']
    assert len(dataset['rows']) == 22
    assert sum(bool(x.get('decision_target')) for x in dataset['rows']) == 7
    family = packet['source']['family_id']
    assert family in split['excluded_from_training_and_retrieval']
    assert family not in {x['family_id'] for x in dataset['rows']}

    image_refs = []
    for image in binding['images']:
        relative = PINE + 'views/' + Path(image['path']).name
        raw = blob(relative)
        assert sha(raw) == image['sha256']
        image_refs.append(dict(git_path=relative, sha256=sha(raw),
                               bytes=len(raw), content_verified_now=True))
    texts = []
    deployed_images = []
    for m in messages:
        content = m.get('content', [])
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content:
            if part.get('type') == 'text':
                texts.append(part['text'])
            elif part.get('type') in ['image', 'image_url']:
                deployed_images.append(part)
    joined = '\n'.join(texts)
    assert task['researcher_authored_brief'] in joined
    # Numeric values elsewhere in scene metadata are NOT answer leakage.
    # Test the actual target object/argument assignment, not raw substring 1.65.
    prescribed_command = json.dumps(target, ensure_ascii=False)
    leakage = dict(
        original_brief_present=True,
        old_desk_retention_explicit_in_brief=True,
        target_command_literal_present=prescribed_command in joined,
        target_east_x_2_09_present='2.09' in joined,
        target_adjacent_wall_instruction_present=any(
            x in joined for x in ['相邻墙', 'adjacent wall', 'L-shaped', 'L形']),
        target_Pine_source_card_present='pine_existing_desk_with_window' in joined,
        input_views=len(deployed_images),
        future_final_audit_supplied='pine_transfer_consequences_measured' in joined,
        manual_scope='Reviewed the original task, scene/catalog and six non-Pine '
        'knowledge cards: they permit edits and show existing poses, but do not '
        'prescribe the adjacent-wall/window pair. Literal checks alone do not '
        'establish absence of every semantic shortcut.',
    )
    assert not any(leakage[k] for k in [
        'target_command_literal_present', 'target_east_x_2_09_present',
        'target_adjacent_wall_instruction_present',
        'target_Pine_source_card_present', 'future_final_audit_supplied'])

    # One positive demonstration can supervise imitation without a preferred
    # negative design. It does not identify a unique best or necessary strategy.
    unit = dict(
        id='pine_adjacent_old_desk_before_second_workstation',
        role='Audited development demonstration; NOT inserted into any train split',
        input=dict(task=task, before_state_git_path=PINE+'native/initial_state.json',
                   before_state_sha256=packet['records'][0]['before_state_sha256'],
                   five_view_references=image_refs,
                   deployment_request_git_path=STUDY+
                   'evaluation/Pine_action_only_17/recorded/round_000/request.json'),
        target=dict(
            decision='在保留原桌和原椅的前提下，将旧桌移向相邻墙，'
                     '为窗前另一个工作位留出位置；随后再安置第二张桌和椅子。',
            first_native_action=target,
            follow_through_actions=[x['command'] for x in packet['records'][1:]],
            decision_text_author='Research assistant retrospective summary, not '
                                 'a recovered historical designer thought',
            source_supported='Window desk by Gurney plus residents reusing an '
                             'IKEA desk on the adjacent wall for concurrent work',
            researcher_supplied='Shell, substitute assets, initial layout, '
                                'coordinates, action ordering and chair operations'),
        targets_are_not_deployment_input=True,
        old_desk_retention_is_input_constraint_not_learned_insight=True,
        unique_optimum=False, counterfactual_preference_label=None,
        joint_4_action_outcome_not_attributed_to_first_move_alone=True,
        training_admitted=False,
        exclusion_reason='Pine is an exposed development family explicitly '
                         'excluded from the existing 22-action training and retrieval.',
    )
    write(args.out/'one_decision_example.json', unit)
    write(args.out/'deployment_input_audit.json', leakage)
    ids0, ids1 = set(initial['objects']), set(final['objects'])
    unchanged = [i for i in sorted(ids0 & ids1)
                 if initial['objects'][i] == final['objects'][i]]
    measured = dict(
        evidence_role='Reanalysis of saved native execution; no new execution',
        original_native_steps=len(packet['records']),
        objects_before=len(ids0), objects_after=len(ids1),
        retained_all_original_ids=ids0 <= ids1,
        added_ids=sorted(ids1-ids0), removed_ids=sorted(ids0-ids1),
        exactly_unchanged_original_ids=unchanged,
        desktop_probe_count_before=consequences['work']['before']['separate_desktop_probe_count'],
        desktop_probe_count_after=consequences['work']['after']['separate_desktop_probe_count'],
        joint_static_access=consequences['common_access'],
        endpoint_surface_intersection_pairs=final_audit['surface_intersection_pairs'],
        immutable_records_unchanged=final_audit['immutable_records_unchanged'],
        tradeoffs_and_unknowns=consequences['tradeoffs'],
        whole_room_success=consequences['whole_room_functional_success'],
        expert_preference=False,
        alternative_shared_worktable_tested=False,
        interpretation='One source-linked positive strategy, with conditional '
                       'static geometry evidence. Does not prove two desks necessary '
                       'or human-use success; not a new whole-scene generation run.')
    write(args.out/'existing_execution_evidence.json', measured)
    reused = dict(
        original_source_pin=PIN, preflight=preflight,
        dry_run_existing_results={k: dict(status=v['status'],
                                         normalization=v['normalization'])
                                  for k, v in dry.items()},
        old_training_results={k: dict(status=v['status'],
            optimizer_updates=v['training']['optimizer_updates'],
            backwards=v['training']['backward_completed'],
            design_or_generalization_gain=v.get('design_or_generalization_gain'))
                              for k, v in training.items()},
        Pine_not_in_22_training_rows=True,
        new_loader_or_optimizer_runs=0, new_model_calls=0,
        new_native_scene_executions=0,
        reuse_scope='Previously saved loader and training receipts inspected; '
                   'not an independent re-execution or a new training claim.')
    write(args.out/'existing_loader_training_reuse.json', reused)
    write(args.out/'source_receipts.json', dict(pin=PIN, files=sources,
          source_attribution=attribution, source_urls=packet['source']['source_urls']))
    write(args.out/'result.json', dict(
        status='One positive condition-to-object decision recovered and qualified',
        audited_positive_demonstrations=1, new_expert_demonstrations=0,
        required_successful_neural_preference_pairs=0,
        source_decision_observed=True,
        reconstruction_action_sequence_recorded=True,
        exact_initial_five_view_bytes_verified=5,
        target_selection_not_prescribed_in_deployment_request=True,
        imitation_can_use_single_positive_in_principle=True,
        unchanged_development_exclusion=True,
        no_new_generalization_evidence=True,
        first_remaining_material='A source-independent recorded application of '
          'this selection to a different brief/geometry, with its pre-action '
          'observations and native result. The current one positive does not '
          'establish conditional transfer or expert preference.',
        next_one_acquisition_step='Inspect the already-published Pine reference '
          'bundle and other existing source-family demonstrations for such a '
          'record before collecting anything new. Do not relabel same-project '
          'reconstructions as independent professional demonstrations.',
        elapsed_seconds=time.monotonic()-started,
        new_model_calls=0, new_paid_API_calls=0, new_training=0,
        new_native_scene_runs=0))
    print(json.dumps(dict(status='audited',positive=1,new_models=0,
                           old_training_actions=22,initial_images_verified=5)))


if __name__ == '__main__':
    main()
