"""Public-task intake and score-semantics audit, never a scene generator.

Official source is pinned and hashed before import. The only imported source
modules here are pure arithmetic modules, not provider or Blender entrypoints.
Organizer scoring files are never included in candidate packets.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/autodl/20261110_frontier_scene_evaluation'
SOURCE = OUT / 'source'
PIN = '1462133985213af9bae14344d2b938e18868350b'

# Human scope annotations from all twenty public briefs, before any outcomes.
# These are not ground truth scores or model capability measurements.
SCOPE = {
    1: ('outdoor', 'harbour_ship_pier'), 2: ('outdoor', 'planetary_outpost'),
    3: ('outdoor', 'suburban_street'), 4: ('outdoor', 'castle_courtyard'),
    5: ('outdoor', 'cemetery'), 6: ('outdoor', 'minigolf'),
    7: ('outdoor', 'lighthouse_island'), 8: ('outdoor', 'canyon'),
    9: ('outdoor', 'lunar_terrain'), 10: ('outdoor', 'construction_site'),
    11: ('interior_workstation', 'cocktail_robot'),
    12: ('greenhouse', 'picking_robot'),
    13: ('interior', 'elevator_delivery_robot'),
    14: ('interior_section', 'library_retrieval_robot'),
    15: ('interior', 'tool_room_robot'),
    16: ('interior', 'laundry_robot'),
    17: ('interior_workstation', 'coffee_robot'),
    18: ('interior', 'garage_charging_robot'),
    19: ('industrial_test_area', 'humanoid_obstacles'),
    20: ('outdoor', 'drone_swap_dock'),
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def manifest():
    rows = {}
    for name in ['source_manifest.json', 'source_manifest_supplement.json']:
        record = json.loads((OUT/name).read_text())
        if record['pin'] != PIN:
            raise ValueError('Unexpected source pin')
        for r in record['files']:
            if r['path'] in rows and rows[r['path']] != r:
                raise ValueError('Conflicting source entry')
            rows[r['path']] = r
    return rows


def checked_bytes(relative, rows):
    raw = (SOURCE/relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != rows[relative]['sha256']:
        raise ValueError('Changed source '+relative)
    return raw


def task_packet(number, rows):
    if number not in SCOPE:
        raise ValueError('Task outside the fixed population')
    prefix = f'data/3dbench/tasks/L2/Brief2Scene/task_L2_Brief2Scene_{number:03d}/'
    meta = tomllib.loads(checked_bytes(prefix+'task.toml', rows).decode())
    raw = checked_bytes(prefix+'instruction.md', rows)
    if meta['instruction'] != 'instruction.md' or meta['environment_files'] != []:
        raise ValueError('New input dependencies need explicit review')
    # Do not copy task.toml: it also points at organizer solution/harness paths.
    return {'task_id': meta['id'], 'source_pin': PIN,
            'instruction_sha256': hashlib.sha256(raw).hexdigest(),
            'instruction': raw.decode(), 'input_files': [],
            'submission_files': meta['submission_files'],
            'status': 'public_input_only_not_a_generated_scene'}


def load_pure(relative, rows):
    checked_bytes(relative, rows)
    spec = importlib.util.spec_from_file_location('frontier_audit_'+Path(relative).stem, SOURCE/relative)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def conjunction(report, expected):
    """Extra audit endpoint, not a change to the published official score."""
    items = report['requirements']
    ids = [r['id'] for r in items]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Fixed requirement population changed')
    states = [r['status'] for r in items]
    if 'violated' in states:
        return False
    if any(s != 'satisfied' for s in states):
        return None
    return True


def audit():
    tick = time.monotonic(); rows = manifest()
    for p in rows:
        checked_bytes(p, rows)
    directory = OUT/'audit_v1'
    directory.mkdir(exist_ok=False)
    tasks = []; packets = []
    for number in SCOPE:
        packet = task_packet(number, rows); packets.append(packet)
        prefix = f'data/3dbench/tasks/L2/Brief2Scene/task_L2_Brief2Scene_{number:03d}/'
        meta = tomllib.loads(checked_bytes(prefix+'task.toml', rows).decode())
        context, subject = SCOPE[number]
        tasks.append({'task_id': packet['task_id'], 'title': meta['title'],
            'context_review': context, 'subject_review': subject,
            'review_kind': 'researcher_scope_annotation_not_model_result',
            'instruction_sha256': packet['instruction_sha256'],
            'robot_frozen_pose_required': number >= 11,
            'evaluation_route': 'native_then_detail' if meta['harness_status'] in ('mvp','implemented') else 'extension_with_agent_numeric_audit',
            'current_complete_generator_qualified': False,
            'qualification_reason': 'Current audited furniture-stage tools do not establish '+
                ('outdoor structural/terrain generation and Blender delivery' if number <= 10 else 'required robot geometry/contact and Blender delivery'),
            'source_instruction': prefix+'instruction.md',
            'requirements_removed': 0})
    with (directory/'task_coverage.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(tasks[0]),lineterminator='\n');writer.writeheader();writer.writerows(tasks)
    # Exact payloads are transient here; save only a receipt. Export one on demand.
    dump(directory/'public_packet_receipts.json', [
        {k:p[k] for k in ('task_id','source_pin','instruction_sha256','input_files','status')}
        for p in packets])
    plan_path = 'configs/release/addition-config/task_L2_Brief2Scene_013/plan.json'
    plan = json.loads(checked_bytes(plan_path, rows))
    numeric = load_pure('src/evaluator/addition_brief_numeric.py', rows)
    detail = load_pure('src/evaluator/legacy_detail.py', rows)
    ids = [c['id'] for c in plan['native_content_checks']]
    initial = {'status':'audited','case_id':plan['task_id'],
               'requirements':[{'id':i,'status':'satisfied'} for i in ids]}
    cases = []
    for status in ('satisfied','violated','partial','observed','not_measurable'):
        candidate = copy.deepcopy(initial)
        next(r for r in candidate['requirements'] if r['id']=='N24')['status']=status
        result = numeric.numeric_data_score(candidate, plan)
        n = len(ids)
        expected = None if status=='not_measurable' else 1 if status=='satisfied' else (n-1)/n
        assert result['score']==expected
        cases.append({'changed_check':'N24','status':status,'official_numeric_output':result,
            'extra_required_conjunction':conjunction(candidate, ids),
            'algebraic_total_if_vlm_and_detail_equal_one':None if expected is None else .6+.3*expected+.1,
            'not_a_scene_result':True})
    rejections=[]
    for label in ('missing','duplicate','wrong_case','unknown_status'):
        bad=copy.deepcopy(initial)
        if label=='missing':bad['requirements'].pop()
        elif label=='duplicate':bad['requirements'].append(copy.deepcopy(bad['requirements'][0]))
        elif label=='wrong_case':bad['case_id']='another_task'
        else:bad['requirements'][0]['status']='unrecognized'
        try:numeric.numeric_data_score(bad,plan)
        except ValueError as e:rejections.append({'case':label,'error':str(e)})
        else:raise AssertionError('Invalid audit accepted: '+label)
    variants=plan['detail_variants']
    fixture_plan={'measurements':[{'id':'shape','method':'visual_coverage_deviation','weight':1}]}
    values={v:{'shape':s} for v,s in zip(variants,[1,.8,.5,.9])}
    minimum=detail.detail_from_criteria(fixture_plan,['shape'],values,variants)
    assert minimum['score']==.5
    absent=copy.deepcopy(values);del absent[variants[2]]
    missing=detail.detail_from_criteria(fixture_plan,['shape'],absent,variants)
    assert missing['score'] is None
    dump(directory/'official_score_semantics.json',{'fixture_kind':'synthetic audit rows using actual frozen 32-check plan; no mesh or judge executed',
         'numeric_cases':cases,'rejected_invalid_rows':rejections,'detail_minimum':minimum,'missing_detail':missing,
         'full_finalizer_executed':False,'official_benchmark_scores_produced':0})
    summary={'source_pin':PIN,'source_files_verified':len(rows),'public_tasks_read':20,
        'public_task_packets_verified':len(packets),'robot_pose_briefs':10,
        'first_ten_outdoor_briefs':10,'current_generator_qualified_tasks':0,
        'qualification_is_not_proven_impossibility':True,
        'native_route_tasks':sum(t['evaluation_route']=='native_then_detail' for t in tasks),
        'extension_route_tasks':sum(t['evaluation_route'].startswith('extension') for t in tasks),
        'official_numeric_checks_in_example':len(ids),'scene_generations':0,
        'model_calls':0,'training_updates':0,'wall_seconds':time.monotonic()-tick}
    dump(directory/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true')
    p.add_argument('--task',type=int);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.audit:audit()
    elif a.task is not None and a.out:
        packet=task_packet(a.task,manifest())
        with a.out.open('x') as f:json.dump(packet,f,ensure_ascii=False,indent=2)
        print(json.dumps({'public_task_exported':packet['task_id'],'scene_generated':False}))
    else:p.error('Use --audit or --task NUMBER --out NEW_FILE')
