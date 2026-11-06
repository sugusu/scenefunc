"""Archive the closed Flinders reference failure without executing another edit.

This is evidence packaging, not training admission or a model experiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--public', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    a = ap.parse_args(); p = a.public; t = a.runtime
    if (p/'SUMMARY.json').exists():
        raise ValueError('Closed delivery exists; do not overwrite')
    initial = read(p/'INITIAL_AUDIT.json'); final = read(p/'CANDIDATE_AUDIT.json')
    gate = read(p/'FEASIBILITY_GATE.json')
    if gate['model_rollout_allowed'] or gate['status'] != 'failed':
        raise ValueError('This archive describes a rejected reference only')
    native = t/'candidate/native'; steps = read(native/'STEPS.json')
    assert len(steps) == 3 and all(s['reply']['results'][0]['output']['success'] for s in steps)
    now = datetime.now(timezone.utc).isoformat()
    copies = []

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('Archive target exists: '+str(target))
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        copies.append(dict(source=str(source), destination=str(target.relative_to(p)),
                           bytes=target.stat().st_size, sha256=sha(target)))

    for source in sorted(native.glob('*.json')):
        copy(source, p/'evidence/native'/source.name)
    for name in ['build', 'candidate/preflight', 'restore', 'export']:
        copy(t/name/'RESULT.json', p/'evidence'/name/'RESULT.json')
    for label, bundle in [('initial', t/'build/bundle'), ('final', native/'checkpoint')]:
        for name in ['house_state.json', 'catalog.json', 'manifest.json', 'task.json',
                     'provenance.json', 'runtime.json', 'session_checkpoint.json']:
            if (bundle/name).exists():
                copy(bundle/name, p/'evidence'/label/name)
    views = []
    for i in range(4):
        observation = read(native/('initial_observation.json' if i == 0 else f'step_{i:02d}_observation.json'))
        assert observation['step_idx'] == i and len(observation['image_paths']) == 5
        binding = Path(observation['state_binding'])
        copy(binding, p/'evidence/bindings'/binding.name)
        for name in observation['image_paths']:
            image = Path(name)
            entry = dict(step=i, path=str(image), bytes=image.stat().st_size, sha256=sha(image),
                         state_binding=str(binding))
            if i in (0, 3) or image.name.endswith('.topdown.png'):
                copy(image, p/'views'/image.name)
                entry['public_path'] = 'views/'+image.name
            views.append(entry)
    write(p/'VIEW_INDEX.json', dict(views=views, visual_review='Initial and final five views inspected by assistant; no independent professional review'))
    code = Path(__file__).parent
    for name in ['build_flinders_input.py', 'evaluate_storage_room.py', 'storage_partition_metrics.py',
                 'run_edit_policy_live.py', 'archive_flinders_result.py']:
        copy(code/name, p/'code'/name)
    rows = []
    for label, audit in [('initial_open', initial), ('executed_partition', final)]:
        for scenario in audit['activity']['scenarios']:
            rows.append(dict(state=label, width_m=scenario['width_m'],
                reachable_required_points=sum(scenario['role_access'].values()),
                required_points=len(scenario['role_access']),
                two_dining_place_proxy_pass=scenario['two_dining_place_proxy_pass'],
                preserved_ids=scenario['all_required_ids_preserved'],
                clear_segments=scenario['visible_segments'], total_segments=scenario['total_segments'],
                surface_intersection_pairs=audit['surface_intersection_pairs'],
                outside_count=sum(b['outside_nominal_room_xy'] for b in audit['bounds']),
                declared_visual_condition_pass=scenario['visual_proxy_pass']))
    write(p/'RESULT_TABLE.json', rows)
    contacts = [x for x in final['positive_aabb_pairs'] if x['triangle_surface_intersection_detected']]
    native_receipt = read(p/'candidate_native_receipt/receipt.json')
    summary = dict(run_id=p.name, recorded_at=now, status='closed_reference_rejected_no_model_rollout',
        actual_native_edits=3, observations=4, images=20, model_calls=0, training_updates=0,
        completed_model_cells=0, planned_model_cells=2,
        initial_clear_segments=27, final_clear_segments=18, total_segments=27,
        actual_surface_contacts=contacts, all_ids_assets_dimensions_preserved=True,
        whole_room_human_use='unknown', professional_positive_demonstration=False,
        source_reproduction=False, independent_test=False,
        native_execution=dict(start=native_receipt['started_at'], end=native_receipt['ended_at'],
                              elapsed_seconds=native_receipt['elapsed_seconds'], exit_code=native_receipt['exit_code']),
        restore=read(t/'restore/RESULT.json'), geometry_preview=read(t/'export/RESULT.json')['glb'],
        inference_status='Both frozen cells not started because the prospective feasibility gate failed',
        conclusion='This particular open-frame shelf substitution failed the sampled screening requirement. No model failure, training benefit or impossibility of all shelf arrangements is established.')
    write(p/'SUMMARY.json', summary)
    source = read(p/'SOURCE.json')
    write(p/'DECISION_PACKET.json', dict(
        family_id=source['family_id'], source=source,
        original_evidence='Architect-provided description supports custom full-height storage partitioning; source is not evidence for these shelf assets or this placement.',
        researcher_reconstruction=dict(task='TASK_INPUT.json', revision='INPUT_REVISION.json',
            initial='evidence/initial/house_state.json', assumptions='7.6 x 5.6 x 2.7 m room, ordinary open-frame shelves, sampled sight lines, declared static approach and meal proxies'),
        attempted_decision=dict(objects=['storage_0', 'storage_1', 'storage_2'],
            action_source='CANDIDATE_PLAN.json', actual_actions='evidence/native/STEPS.json',
            rationale='Researcher attempted to combine storage access with visual separation; not a historical designer trajectory'),
        executed_room='evidence/final/house_state.json', measured_results='RESULT_TABLE.json',
        local_benefit='9 of 27 sampled segments blocked, compared with zero initially',
        costs_and_failures=['18 sampled sight segments remain clear', 'One shelf pair has actual mesh surface contact',
                            'Three edits consumed; no tested continuous carrying path', 'Reachable floor-center area and several routes worsen despite endpoint reachability'],
        unknowns=final['activity']['limits'],
        permitted_evidence_use=['Source knowledge', 'Research reconstruction and execution evidence', 'Objective consequence/limitation record'],
        not_admitted_as=['Positive professional action demonstration', 'Validated full-room solution', 'Model comparison', 'New training sample'],
        next_research_decision='Close this fixture. Reuse feasible existing decision records for a coherent study; verify required asset properties before scheduling another reference rollout. Do not add another layout or training run to rescue this result.'))
    write(p/'COPY_MANIFEST.json', dict(created_at=now, files=copies))
    print(json.dumps({k:summary[k] for k in ['status', 'actual_native_edits', 'model_calls', 'initial_clear_segments', 'final_clear_segments']}), flush=True)


if __name__ == '__main__':
    main()
