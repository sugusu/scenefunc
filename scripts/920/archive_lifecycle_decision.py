"""Archive the executed lifecycle transfer without promoting proxy success to design quality."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from jsonschema import validate
from native_checkpoint import canonical_hash


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--public', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    a = ap.parse_args(); p = a.public.resolve(); t = a.runtime.resolve()
    print(datetime.now(timezone.utc).isoformat(), 'archive actual lifecycle transfer and probe limitations', flush=True)
    steps = read(t/'native/STEPS.json'); assert len(steps) == 3
    receipts = {}
    for directory in sorted(t.glob('*_stage')):
        if directory.name.startswith('archive_'):
            continue
        receipt = read(directory/'receipt.json'); assert receipt['exit_code'] == 0
        receipts[directory.name] = receipt
        shutil.copytree(directory, p/directory.name, dirs_exist_ok=True)
    for name in ['INITIAL_AUDIT.json', 'FINAL_AUDIT.json', 'CONSEQUENCES.json']:
        shutil.copy2(t/name, p/name)
    for folder in ['native', 'cold_restore', 'export', 'preflight']:
        (p/folder).mkdir(exist_ok=True)
        for f in (t/folder).glob('*.json'):
            shutil.copy2(f, p/folder/f.name)
    shutil.copytree(t/'dataset', p/'dataset', dirs_exist_ok=True)
    bindings = []
    for i in range(4):
        label = 'initial' if i == 0 else f'step_{i:02d}'
        obs = read(t/'native'/f'{label}_observation.json')
        binding_path = Path(obs['state_binding']); b = read(binding_path)
        state = read(t/'native'/f'{label}_state.json')
        assert canonical_hash(state) == canonical_hash(b['native_state']) == b['state_sha256']
        context = json.loads(obs['room_prompt'].split('\nCURRENT NATIVE STATE:\n', 1)[1])
        assert canonical_hash(context) == b['current_context_sha256']
        assert len(b['images']) == len(obs['image_paths']) == 5
        for item, path in zip(b['images'], obs['image_paths']):
            f = Path(path); assert f == Path(item['path']) and sha(f) == item['sha256']
            with Image.open(f) as im:
                im.verify()
            if i in (0, 3) or '.topdown.' in f.name:
                (p/'views').mkdir(exist_ok=True); shutil.copy2(f, p/'views'/f.name)
        (p/'bindings').mkdir(exist_ok=True)
        shutil.copy2(binding_path, p/'bindings'/binding_path.name)
        bindings.append(dict(step=i, state_sha256=b['state_sha256'], images=b['images']))
        if i < 3:
            tools, _ = json.JSONDecoder().raw_decode(obs['system'].split('Available executable tools:\n', 1)[1])
            schemas = {x['name']: x['parameters'] for x in tools}
            command = steps[i]['command']; validate(command['arguments'], schemas[command['name']])
            assert steps[i]['before_sha256'] == b['state_sha256']
            assert steps[i]['reply']['results'][0]['output']['success'] is True
        if i > 0:
            assert steps[i-1]['after_sha256'] == b['state_sha256']
    audits = {k: read(t/f'{k}_AUDIT.json') for k in ['INITIAL', 'FINAL']}
    generic = {}
    for k, audit in audits.items():
        assert audit['surface_intersection_pairs'] == 0
        assert all(audit['immutable_records_unchanged'].values())
        assert not any(x['outside_nominal_room_xy'] for x in audit['bounds'])
        generic[k] = [dict(width_m=s['square_body_width_m'], reachable=sum(x['reachable_from_entry'] for x in s['targets'].values()), total=len(s['targets']), failed=[key for key, v in s['targets'].items() if not v['reachable_from_entry']]) for s in audit['access']['scenarios']]
    erratum = dict(frozen_probe='former_bed_side', point_xy_m=[2.45, 1.25],
        actual_interpretation='West bed-end approach, not lateral bedside: the former bed has yaw 90 degrees.',
        former_bed_lateral_gaps_m=[0.32265767455106187, 0.3226576745510195],
        generic_lateral_probes_reachable=False,
        consequence='former_bed_approach_before=true establishes only a reachable west-end point. Initial three-person sleeping usability is unverified.',
        original_measurements_preserved=True, geometry_changed_after_review=False,
        whole_room_positive_demonstration=False, generic_access=generic)
    for s in audits['INITIAL']['access']['scenarios']:
        for side in ['left_side', 'right_side']:
            assert not s['targets']['former_child_bed_0:'+side]['reachable_from_entry']
    write(p/'PROBE_INTERPRETATION.json', erratum)
    write(p/'VISUAL_REVIEW.json', dict(reviewed_images=sorted(f.name for f in (p/'views').glob('*.png')),
        reviewer='AI visual inspection; not independent professional evaluation',
        observation='Initial/final five views and intermediate topdowns show only the former bed removal and office table/chair additions; original shelf and rest of dwelling retained. Partition occlusion expected.',
        limitations=['Single-plane transfer excludes source stairs and vertical interleaving.', 'No operable doors or verified privacy.', 'Former bed lateral access fails; see PROBE_INTERPRETATION.json.', 'Static chair/table fit is not verified work ergonomics.']))
    c = read(t/'CONSEQUENCES.json'); nr = receipts['native_stage']
    summary = dict(run_id=p.name, status='executed_transfer_with_qualified_consequences', native_actions=3, observations=4, images=20,
        source_decisions=1, source_families=1, new_knowledge_records=0, new_source_family_registrations=0,
        initial_furniture=16, final_furniture=17, preserved_furniture=15,
        initial_and_final_mesh_contacts=0, initial_and_final_outside=0,
        fixed_access=c['fixed_access_comparison'], generic_access=generic,
        model_calls=0, training_updates=0, professional_evaluations=0, professional_positive_demonstration=False,
        training_admitted=False, official_loader_encoding_this_run=False,
        native_started_at=nr['started_at'], native_ended_at=nr['ended_at'], native_seconds=nr['elapsed_seconds'], actual_exit_code=nr['exit_code'],
        cold_restore=read(t/'cold_restore/RESULT.json')['status'], geometry_export=read(t/'export/RESULT.json')['glb'],
        interpretation='Executable researcher transfer of a sourced room-use transition, with conserved fixed objects and explicit loss of one bed instance. Not historical designer actions, verified complete dwelling, learning gain or professional endorsement.')
    write(p/'SUMMARY.json', summary)
    write(p/'ARCHIVE_CHECKS.json', dict(state_bound_observations=4, decoded_hashed_images=20, schema_valid_successful_actions=3, all_prior_stage_exit_codes_zero=True,
        public_episode_files_exact=all(sha(f)==sha(p/'dataset'/f.relative_to(t/'dataset')) for f in (t/'dataset').rglob('*.json')),
        build_log_label_erratum='Inherited build first-log says held-out; this is prospectively train-allocated Michelet. Original log preserved.', bindings=bindings))
    status = read(p/'STATUS.json'); now = datetime.now(timezone.utc).isoformat()
    status.update(status=summary['status'], updated_at=now, completed=3, planned=3, ended_at=nr['ended_at'], actual_exit_code=0,
        latest_verifiable_artifact=str(p/'SUMMARY.json'), failure_reason=None,
        scientific_limitations='Initial bedside probe was mislabeled; generic access 26/33 in both states. Whole-room usability and positive professional label not established.',
        next_step='Keep qualified source/action record; broaden justified decision data and prepare fair study, without more tuning of this room or new training.')
    write(p/'STATUS.json', status)
    write(p.parent/'latest.json', dict(run_id=p.name, stage=status['status'], updated_at=now, run_directory=str(t)))
    print(json.dumps({k:summary[k] for k in ['status','native_actions','images','training_updates']}), flush=True)


if __name__ == '__main__':
    main()
