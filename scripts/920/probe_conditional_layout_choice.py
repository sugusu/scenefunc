"""Finite, training-exposed endpoint recognition diagnostic; never edits a room.

Candidate states are previously executed complete endpoint states. The model
receives their geometry, not their measured scores or historical phase labels.
This is a lower-demand recognition task, not an OTC-trained model or rollout.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

R = Path(__file__).resolve().parents[2]
D = R.parent
STUDY = R / 'runs/920/P920-EDITSTUDY-v1_20261106T074752Z'
PAPER_ID = 'P920-PAPER-v2_20261105T182424Z'
ARMS = ['frozen_base_knowledge_feedback', 'action_only', 'decision_action']


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def packed(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def ref(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))


def assess(raw, model, label_map, expected):
    from shared_action_contract import strict_loads
    row = dict(valid_response=False, correct=False, selected_endpoint=None)
    try:
        blocks = re.findall(r'<tool_call>(.*?)</tool_call>', raw, re.S)
        if len(blocks) != 1 or raw.count('<tool_call>') != 1 or model.get('output_rejected_reason'):
            raise ValueError('Incomplete or non-single selection')
        # No repair/coercion of malformed model output.
        call = strict_loads(blocks[0])
        if set(call) != {'name', 'arguments'} or call['name'] != 'select_layout':
            raise ValueError('Unexpected tool or fields')
        arg = call['arguments']
        if not isinstance(arg, dict) or set(arg) != {'candidate_id', 'reason', 'unknowns'}:
            raise ValueError('Unexpected selection arguments')
        if arg['candidate_id'] not in ('A', 'B', 'undetermined') or any(not isinstance(arg[k], str) for k in ('reason', 'unknowns')):
            raise ValueError('Invalid selection value or evidence type')
        endpoint = label_map.get(arg['candidate_id'])
        row.update(valid_response=True, choice=arg, selected_endpoint=endpoint, correct=endpoint == expected)
    except (ValueError, TypeError, KeyError) as exc:
        row['error'] = str(exc)
    return row


def prepare(out):
    if (out / 'PROTOCOL.json').exists(): raise ValueError('Refusing to overwrite frozen inputs')
    out.mkdir(parents=True, exist_ok=True)
    source = R / 'runs/920' / PAPER_ID
    native = D / 'runtime' / PAPER_ID / 'native'
    states = {name: read(native / f'step_{step:02d}_after.json') for name, step in [('everyday', 6), ('reserve', 8)]}
    # Keep the existing native geometry dependency in its environment; the VLM
    # processor environment deliberately does not install collision packages.
    geometry_command = [str(D / 'envs/scene_designer311_native/bin/python'), '-c',
        'import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from preflight_sourced_trajectory import boxes; '
        'print(json.dumps({str(i):{k:[lo.tolist(),hi.tolist()] for k,(lo,hi) in boxes(json.loads((Path(sys.argv[2])/f"step_{i:02d}_after.json").read_text())).items()} for i in [6,8]}))',
        str(Path(__file__).parent), str(native)]
    all_bounds = json.loads(subprocess.check_output(geometry_command, text=True, timeout=60))
    write(out / 'GEOMETRY_PREPARATION.json', dict(command=geometry_command, bounds=all_bounds,
        code={n: sha(Path(__file__).with_name(n)) for n in ['preflight_sourced_trajectory.py', 'inspect_native_mesh_contacts.py']}))
    measurements = read(source / 'MODE_MEASUREMENT.json')
    scores = {}
    scene_objects = {}
    for name, step in [('everyday', 6), ('reserve', 8)]:
        row = next(r for r in measurements['rows'] if r['step'] == step)
        scores[name] = dict(reserve_clear=row['reserve_clear'], dining_places=sum(p['within_declared_alignment_and_distance'] for p in row['declared_dining_places']),
                            occupancy_m2=row['reserve_occupants_overlap_m2'], envelope_pairs=row['positive_envelope_pairs'])
        bounds = all_bounds[str(step)]
        scene_objects[name] = {oid: dict(object_id=oid, object_type=obj['object_type'], name=obj['name'],
            immutable=obj['immutable'], bbox_min=obj['bbox_min'], bbox_max=obj['bbox_max'], transform=obj['transform'],
            **(dict(world_aabb_min=bounds[oid][0], world_aabb_max=bounds[oid][1]) if oid in bounds else {})) for oid, obj in states[name]['objects'].items()}
    assert scores['everyday']['dining_places'] == 2 and not scores['everyday']['reserve_clear']
    assert scores['reserve']['dining_places'] == 0 and scores['reserve']['reserve_clear']
    assert all(not x['envelope_pairs'] for x in scores.values())
    assert set(scene_objects['everyday']) == set(scene_objects['reserve'])
    changed = [k for k in scene_objects['everyday'] if scene_objects['everyday'][k] != scene_objects['reserve'][k]]
    assert set(changed) == {'mobile_table_0', 'chair_right_0'}
    episode = STUDY / 'dataset/action_only/paper_table_modes_build_07/episode.json'
    document = read(episode)
    context = json.loads(document['task_context']['scene_description'])
    assert context['objects'].keys() == states['everyday']['objects'].keys()
    assert all(context['objects'][k]['transform'] == v['transform'] for k, v in states['everyday']['objects'].items())
    images = document['turns'][0]['images']
    knowledge_path = R / 'runs/920/P920-DECISIONAUDIT-v1_20261106T154100Z/audit_v2/KNOWLEDGE_CANDIDATE.json'
    knowledge = read(knowledge_path)['records']
    # All arms receive exactly the same already-known principles. No experiment scope metadata.
    common = dict(current_objects=scene_objects['everyday'], available_source_knowledge=knowledge,
        physical_unknowns=context['physical_unknowns'], candidate_representation='Each candidate replaces only listed complete object records; every omitted object is unchanged. Five views depict the current room, not both candidates.',
        common_limits='Both endpoint states retain every object and have zero positive furniture envelope intersections in prior measurements. Full human use, physical movement and professional quality are unknown. No score, phase label, original trajectory or target action is provided.')
    tasks = dict(
        reserve='Tonight reserve the specified floor rectangle for a visitor activity; no dining takes place concurrently. Choose the candidate that leaves this rectangle unoccupied by furniture. Rectangle [xmin,ymin,xmax,ymax] in metres: [-2.6,-0.55,-1.0,0.75]. This is a declared floor-use task, not a verified sleeping or sofa-conversion requirement.',
        everyday='The visitor activity is over. Two residents now need to sit at the existing mobile table together for a meal; no empty visitor rectangle is required at this time. Choose the candidate that supports the two existing stools at the table. For this diagnostic only, each stool centre must lie 0.2 to 0.75 metres beyond its respective left/right table edge and its full Y extent must align within the tabletop Y extent. These are geometric proxies, not ergonomic standards.')
    tool = dict(name='select_layout', description='Record a review choice only. Does not execute or modify a scene.', parameters=dict(type='object', additionalProperties=False,
        properties=dict(candidate_id=dict(type='string', enum=['A', 'B', 'undetermined']), reason=dict(type='string'), unknowns=dict(type='string')),
        required=['candidate_id', 'reason', 'unknowns']))
    system = ('Review two supplied room layouts for the CURRENT activity. You may reason briefly in <think>. '
        'Use the actual object geometry and explain which objects support the choice and what the alternative loses. '
        'Neither moving nor retaining furniture is inherently preferred. If the supplied facts cannot distinguish them, select undetermined. '
        'Do not claim complete room quality or human usability. Finish with exactly one <tool_call> JSON object with name and arguments. '
        'Available review tool:\n' + packed(tool))
    template = read(STUDY / 'evaluation/Milan_frozen_base_knowledge_feedback_17/PLAN.json')
    sys.path.insert(0, str(Path(template['encoder']).parent))
    from local_vlm import LocalQwenEngine
    engine = LocalQwenEngine(template['model_directory'], out / 'unused', max_pixels=65536, max_input_tokens=18432, max_new_tokens=768)
    cells = []
    requests = {}
    for order, names in [('forward', ['everyday', 'reserve']), ('reversed', ['reserve', 'everyday'])]:
        label_map = dict(zip(['A', 'B'], names))
        candidates = [dict(candidate_id=label, replacements={k: scene_objects[name][k] for k in changed}) for label, name in label_map.items()]
        for activity in ['reserve', 'everyday']:
            ident = activity + '_' + order
            text = dict(common, current_activity=tasks[activity], candidates=candidates)
            messages = [dict(role='system', content=system), dict(role='user', content=[*[dict(type='image', image=p) for p in images], dict(type='text', text=packed(text))])]
            request = out / 'inputs' / ident / 'request.json'
            write(request, messages)
            _, encoding = engine.encode(messages)
            assert len(encoding['images']) == 5
            write(request.with_name('ENCODING.json'), encoding)
            requests[ident] = dict(text=text, request=ref(request), label_map=label_map)
            for arm in ARMS:
                plan = read(STUDY / f'evaluation/Milan_{arm}_17/PLAN.json')
                for old in ['bundle', 'manifest_sha256', 'prepared_directory', 'prompt_supplement', 'evaluation', 'rejection_feedback_policy']:
                    plan.pop(old, None)
                cell_id = ident + '_' + arm
                plan.update(run_id=out.name, cell_id=cell_id, purpose='training_exposed_conditional_endpoint_recognition_no_execution',
                            public_directory=str(out / 'cells' / cell_id), runtime_directory=str(out / 'cells' / cell_id))
                plan['budget'].update(model_calls=1, tool_calls=0, max_new_tokens=768, seconds_per_model_call=180, gpu_seconds_per_cell=180)
                plan['inference']['seed'] = 17
                planpath = out / 'cells' / cell_id / 'PLAN.json'
                write(planpath, plan)
                cells.append(dict(id=cell_id, arm=arm, activity=activity, order=order, label_map=label_map, expected_endpoint=activity,
                                  plan=ref(planpath), request=ref(request)))
    # Prospective comparison invariants: only activity differs, then only candidate label/order differs.
    for order in ['forward', 'reversed']:
        x, y = [copy.deepcopy(requests[a + '_' + order]['text']) for a in ['reserve', 'everyday']]
        x.pop('current_activity'); y.pop('current_activity'); assert x == y
    for activity in tasks:
        x, y = [copy.deepcopy(requests[activity + '_' + order]['text']) for order in ['forward', 'reversed']]
        cx, cy = x.pop('candidates'), y.pop('candidates'); assert x == y
        assert cx[0]['replacements'] == cy[1]['replacements'] and cx[1]['replacements'] == cy[0]['replacements']
    write(out / 'PRIVATE_SCORING.json', dict(measured_scores=scores, source=ref(source / 'MODE_MEASUREMENT.json'),
        states={name: ref(native / f'step_{step:02d}_after.json') for name, step in [('everyday', 6), ('reserve', 8)]},
        inputs_omit_measured_scores=True, knowledge=ref(knowledge_path), observation_source=ref(episode), images=[ref(p) for p in images]))
    protocol = dict(run_id=out.name, frozen_at=now(), question='Can models distinguish the activity-appropriate supplied endpoint independently of native tool execution?',
        scope='One training-exposed Paper reconstruction; two activity conditions and reversed option labels, NOT four independent projects. Recognition with supplied candidates and world AABBs, not generation or professional evaluation.',
        budgets=dict(model_calls=12, subprocess_seconds=2160, max_new_tokens_each=768, training_updates=0, native_edits=0, retries=0),
        code_sha256={name: sha(Path(__file__).with_name(name)) for name in ['probe_conditional_layout_choice.py', 'edit_study_model_step.py', 'run_stage.py']},
        scoring_sha256=sha(out / 'PRIVATE_SCORING.json'), cells=cells,
        primary='Per arm: both opposing activities correct under both label orders (4/4); report malformed/undetermined and label consistency separately. No significance claim.',
        gates=['If base is 4/4, conditional recognition is available on this scaffold; no need for more training on this task is established.',
               'If joint exceeds controls, record exposed recognition only; do not infer generation or professional-learning benefit.',
               'If all fail, diagnose recognition/prompt geometry limitations; do not automatically add training or rerun.',
               'Regardless of result: stop after the frozen batch; no prompt tuning, alternate seed or additional Paper rollout.'],
        limitations=['Researchers supplied both complete endpoints, world AABBs and quantitative task criteria; substantially easier than autonomous design.',
                     'Both adapters saw this family and its actions; seen layouts are not independent generalization.',
                     'No source-gold reason, measured per-candidate success, or phase labels enter input. Source knowledge remains shared.',
                     'Prompt and output task differ from edit training. Success or failure cannot isolate training, interface or knowledge causally.',
                     'Static endpoint selection does not validate sofa conversion, sweeps, comfort, full residence or independent professional preference.'])
    write(out / 'PROTOCOL.json', protocol)
    write(out / 'STATUS.json', dict(question=protocol['question'], status='frozen_not_started', updated_at=now(), completed=0, planned=12,
        configuration_and_inputs_sha256=dict(protocol=sha(out / 'PROTOCOL.json'), scoring=protocol['scoring_sha256']), latest_verifiable_artifact=str(out / 'PROTOCOL.json'),
        failure_reason=None, next_step='Execute the single frozen 12-call batch on GPU0; no scene edits or training.'))
    print(packed(dict(status='frozen', run_id=out.name, cells=len(cells))), flush=True)


def run(out):
    protocol = read(out / 'PROTOCOL.json')
    for name, digest in protocol['code_sha256'].items():
        if sha(Path(__file__).with_name(name)) != digest: raise ValueError('Frozen code changed: ' + name)
    scoring = read(out / 'PRIVATE_SCORING.json')
    if sha(out / 'PRIVATE_SCORING.json') != protocol['scoring_sha256']: raise ValueError('Scoring changed')
    for item in scoring['images']:
        if sha(item['path']) != item['sha256']: raise ValueError('Image changed')
    status = read(out / 'STATUS.json')
    if status['status'] != 'frozen_not_started': raise ValueError('Refusing restart')
    locks, results = [], []
    first = read(protocol['cells'][0]['plan']['path'])
    status.update(status='running', started_at=now(), first_log=str(out / 'run_receipt/stage.log'), next_step='Complete frozen calls; stop on infrastructure error.')
    write(out / 'STATUS.json', status)
    exit_code = 0
    try:
        for p in first['gpu_lock_paths']:
            lock = Path(p).open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
        gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
        if gpu[0].strip() != first['gpu_uuid'] or int(gpu[1]) >= 512 or int(gpu[2]) <= 23000: raise RuntimeError('GPU unavailable')
        for cell in protocol['cells']:
            for key in ['plan', 'request']:
                if sha(cell[key]['path']) != cell[key]['sha256']: raise ValueError('Frozen cell changed')
            directory = Path(cell['plan']['path']).parent
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=first['gpu_uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            command = [first['model_python'], str(Path(__file__).with_name('run_stage.py')), '--receipt-dir', str(directory / 'receipt'), '--timeout', '180', '--',
                first['model_python'], str(Path(__file__).with_name('edit_study_model_step.py')), '--plan', cell['plan']['path'], '--request', cell['request']['path'], '--out', str(directory / 'model'), '--mode', 'generate']
            code = subprocess.call(command, env=env)
            if code: raise RuntimeError(cell['id'] + ': model subprocess failed; no retry')
            model = read(directory / 'model/RESULT.json')
            raw = (directory / 'model/raw_output.txt').read_text()
            result = dict(id=cell['id'], arm=cell['arm'], activity=cell['activity'], order=cell['order'],
                model_result=ref(directory / 'model/RESULT.json'), raw_output=ref(directory / 'model/raw_output.txt'),
                **assess(raw, model, cell['label_map'], cell['expected_endpoint']))
            results.append(result); write(directory / 'CHECK.json', result)
            status.update(completed=len(results), updated_at=now(), latest_verifiable_artifact=str(directory / 'CHECK.json'))
            write(out / 'STATUS.json', status)
            print(packed(dict(cell=cell['id'], correct=result['correct'], valid=result['valid_response'])), flush=True)
    except BaseException as exc:
        exit_code = 1; status['failure_reason'] = str(exc)
        raise
    finally:
        # Count actual generation attempts independently of parsing/validation success.
        actual = []
        for cell in protocol['cells']:
            p = Path(cell['plan']['path']).parent / 'model/RESULT.json'
            if p.exists(): actual.append(read(p))
        status.update(status='completed' if exit_code == 0 else 'failed', ended_at=now(), updated_at=now(), exit_code=exit_code,
            actual_model_calls=sum(m['model_calls'] for m in actual), next_step='Review frozen results; do not repeat or extend this diagnostic.')
        summary = dict(status=status['status'], results=results, actual_model_calls=status['actual_model_calls'], training_updates=0, native_edits=0,
            arms={arm: dict(scored=sum(r['arm'] == arm for r in results), correct=sum(r['arm'] == arm and r['correct'] for r in results),
                           valid=sum(r['arm'] == arm and r['valid_response'] for r in results)) for arm in ARMS})
        write(out / 'SUMMARY.json', summary); write(out / 'STATUS.json', status)
        for lock in locks: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else run)(args.out.resolve())
