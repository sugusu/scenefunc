"""Freeze a recoverable, source-exposed whole-home development comparison.

No original final states/actions are copied. Historical assumptions and a byte-
identical catalog are retained. Case cards paraphrase other projects only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--catalog-bundle', type=Path, required=True)
    a = ap.parse_args()
    if a.root.exists() or a.scratch.exists():
        raise FileExistsError('Use fresh output paths; never overwrite an experiment')
    a.root.mkdir(parents=True); a.scratch.mkdir(parents=True)
    pin = '75fc783f86d4fea78714639e110fbe77476ee106'
    base = 'runs/920/P920-LIFECYCLE-v1_20261105T221829Z/'
    paths = [base+'TASK_INPUT.json', base+'BUILD_PLAN.json',
             base+'input_metadata/manifest.json',
             'scripts/920/build_lifecycle_input.py',
             'scripts/920/build_reference_room.py', 'scripts/920/native_project.py']
    refs = []
    for path in paths:
        data = subprocess.check_output(['git', 'show', pin+':'+path])
        out = a.root/'source'/path
        out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(data)
        refs.append(dict(git_path=path, local=str(out.resolve()), sha256=sha(out)))
    write(a.root/'source_manifest.json', dict(pin=pin, files=refs,
        final_state_or_action_program_read=False, source_already_exposed=True))
    m = json.loads((a.root/'source'/base/'input_metadata/manifest.json').read_text())
    expected = m['files']['catalog.json']['sha256']
    assert sha(a.catalog_bundle/'catalog.json') == expected
    # Catalog geometry comes from the already restored bundle. Its placed
    # room state is never read by the builder; original builder validates that
    # every asset template has an identity pose.
    plan = json.loads((a.root/'source'/base/'BUILD_PLAN.json').read_text())
    local_task = a.root/'source'/base/'TASK_INPUT.json'
    plan['inputs']['task']['path'] = str(local_task.resolve())
    plan['inputs']['catalog_manifest'] = dict(path=str((a.catalog_bundle/'manifest.json').resolve()),
                                             sha256=sha(a.catalog_bundle/'manifest.json'))
    plan['scenesmith'] = str(Path('runs/autodl/20261110_native_state_runtime/source').resolve())
    plan['code_hashes'] = {str((a.root/'source'/p).resolve()): sha(a.root/'source'/p)
                           for p in paths if p.endswith('.py')}
    write(a.root/'local_build_plan.json', plan)
    cards = [dict(id='pine_separate_concurrent_work', family='sydney_pine_nicholas_gurney_2022',
        url='https://www.nevertoosmall.com/post/two-creative-maximalists-living-minimally',
        locator='The Everything Room, paragraph describing existing IKEA desk and simultaneous work',
        condition='Multiple continuing activities need separate usable work surfaces; existing useful furniture retained.',
        reported_decision='An existing desk and an added window desk coexist for two simultaneous workers.',
        permissible_transfer='Prefer a separate work allocation when the existing table still has a required use.',
        excluded_transfer='No exact L shape, window pose, capacity, target coordinates or historical action supervision.'),
        dict(id='tm8_preserved_bedrooms', family='lisbon_tm8_dc_ad',
        url='https://www.archdaily.com/1001060/tm8-apartment-dd',
        locator='Architect-supplied text, long-term young-family rental and three retained bedrooms',
        condition='Continuing family bedroom needs in a long-term rental.',
        reported_decision='Three existing bedrooms retained while shared areas were reorganized.',
        permissible_transfer='Check continuing sleeping needs before reassigning a bedroom.',
        excluded_transfer='Do not convert this observed strategy into a universal prohibition on removing an explicitly unneeded bed.')]
    write(a.root/'common_case_cards.json', dict(cards=cards,
        target_family_excluded='paris_michelet_nomadic', annotation_author='research assistant',
        expert_preference_labels=False, original_training_license_established=False,
        neural_case_retrieval_run=False, source_read_online_before_freeze=True))
    protocol = dict(frozen_unix=time.time(), project='michelet_lifecycle_transfer',
        source_split='already_exposed_researcher_reconstruction_development',
        selection='First remaining recoverable complete dwelling with explicit discontinued/retained uses, after input review and before candidate evaluation.',
        rejected_before_run=dict(budapest='Single preselected clothes rail and explicit quick-access brief are insufficient to test case-based allocation.',
                                 fhouse='Only shared living/dining room; excludes cooking/sleep/hygiene.'),
        inputs='Unmodified historical task, initial construction recipe and byte-identical 16-entry catalog; no target final state/program.',
        arms=['B0_task_aware_search', 'B1_case_condition_order'],
        common_information=['task', 'initial complete native state', 'all 16 catalog entries', 'both case cards', 'same geometric tools'],
        only_difference='B1 inserts a source-condition/use-allocation compatibility tie breaker into the ordinary proposal order; same backend selector and feasibility tests.',
        common_candidate_space=dict(bed_action=['keep', 'remove_explicitly_unneeded'],
            work_allocation=['add_catalog_table_and_chair', 'reuse_family_table_and_one_chair'],
            workspace='Frozen convertible-room bounds; generated lattice, no source target coordinates',
            table_yaw_degrees=[0,90,180,270], chair_sides=[-1,1], lattice_spacing_m=0.30,
            wall_margin_m=0.10, chair_edge_gap_m=0.15),
        common_budget=dict(native_candidate_cap=24, accepted_single_object_operations_cap=3,
            wall_seconds_cap_per_arm=180, local_model_calls=0, paid_API_calls=0, training_updates=0),
        selector='First candidate in common ranked stream passing allocation, object preservation, containment, sampled ground support, surface-intersection and access checks; no improvement required at intermediate edits.',
        access=dict(grid_m=0.05, assumed_body_height_m=1.8, square_body_widths_m=[0.4,0.5,0.6],
                    preservation='No fixed use endpoint loses reachability relative to initial state; new office approach must pass all declared widths.'),
        contact_threshold_m=0.0001, ground_tolerance_m=0.001,
        task_change_allowed=False, no_outcome_dependent_retry=True,
        resource_scope='CPU only; no new asset downloads, model loading, training or paid resources',
        retained_uncertainties=['Furniture serving as desk is a catalog substitute, not verified equipment/ergonomics.',
            'Body widths, approach points, room and assets are research assumptions.',
            'Removed bed destination, carrying, privacy, daylight, utilities and true functional use remain unknown.'],
        full_scene_generation_success=None, claim='Finite whole-dwelling completion diagnostic, not a novel Agent or from-empty generation result')
    write(a.root/'protocol.json', protocol)
    protected=json.loads(Path('runs/autodl/20261111_feedback_information_control/source_before_after.json').read_text())
    rows=[dict(path=r['path'], expected=r['before'], before=sha(r['path'])) for r in protected['original_inputs']]
    assert len(rows)==579 and all(r['before']==r['expected'] for r in rows)
    write(a.root/'protected_before.json', rows)
    write(a.root/'preparation_result.json', dict(source_files=len(refs), protected_files=len(rows),
         catalog_sha256=expected, catalog_entries=16, scratch=str(a.scratch.resolve()),
         actual_scene_searches=0, model_calls=0, training_updates=0,
         build_command=['python', '-B', str((a.root/'source/scripts/920/build_lifecycle_input.py').resolve()),
                        '--plan', str((a.root/'local_build_plan.json').resolve()), '--out', str(a.scratch/'initial')]))
    print(json.dumps(dict(status='prepared_before_search', root=str(a.root), scratch=str(a.scratch))))


if __name__ == '__main__':
    main()
