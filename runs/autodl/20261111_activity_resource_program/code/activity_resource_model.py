"""Offline allocation component probe, explicitly NOT native Actor reproduction.

Both construction skills are ordinary, previously verified scene operations.
The model chooses a resource binding; it does not generate new furniture geometry
or discover new operations. A correct choice is not a new Agent mechanism.
"""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import sys
import time

W=Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
SOURCE=W/'native_execution_inventory_20261104/source/Designer/agent_rollout.py'
R=Path(__file__).resolve().parents[2]

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def source_functions():
    chosen=[]
    for n in ast.parse(SOURCE.read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['_THINK_RE','_TOOL_CALL_RE'] for t in n.targets):chosen.append(n)
        if isinstance(n,ast.FunctionDef) and n.name in ['parse_assistant_text','build_messages']:chosen.append(n)
    scope={'re':re,'json':json,'Path':Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=chosen,type_ignores=[])),str(SOURCE),'exec'),scope)
    return scope['build_messages'],scope['parse_assistant_text']

def evaluate(calls,case):
    """Same tool/checker for model and ordinary policy, no hidden preference label."""
    if len(calls)!=1 or calls[0].get('name')!='allocate_resources':
        return dict(valid=False,reason='Exactly one allocate_resources tool call required')
    a=calls[0].get('arguments',{})
    keys={'work_surface','work_seat','meal_surface','meal_seats','material_persists_during_meal','resident_works_while_meal'}
    if set(a)!=keys:return dict(valid=False,reason='Tool fields do not match published schema')
    if a['meal_surface']!='meal_table_0' or a['meal_seats']!=['meal_chair_n_0','meal_chair_s_0']:
        return dict(valid=False,reason='Original two meal seats and surface must be retained')
    if a['material_persists_during_meal'] is not True or a['resident_works_while_meal'] is not False:
        return dict(valid=False,reason='Persistent materials and sequential people modes incorrectly represented')
    pair=(a['work_surface'],a['work_seat'])
    if pair==('meal_table_0','meal_chair_n_0'):
        ok=case['shared']['status']=='witness'
        return dict(valid=ok,plan='shared',new_furniture=0,reason='Constructive packing witness' if ok else case['shared'],
                    native_state='original' if ok else None,physical_material_objects_created=False)
    if pair==('new_work_table','new_work_chair'):
        ok=case['independent_top_capacity_verified']
        return dict(valid=ok,plan='independent',new_furniture=2,reason='Existing verified add-only joint placement and individual table capacity' if ok else 'No capacity witness',
                    native_state='ordinary_completed' if ok else None,physical_material_objects_created=False)
    return dict(valid=False,reason='Unsupported binding or seat not paired with its verified work surface; no silent remapping')

SYSTEM='''You allocate activities to resources in an existing furnished 3D room.
This is a custom TWO-SKILL allocation component, not the author native scene-generation API.
Keep every original object and pose unchanged. Never invent an extra requirement.
There are two available, ordinary construction skills:
1. Reuse meal_table_0 for work, with meal_chair_n_0 as the work seat. Two original meal seats remain available during the meal mode. The geometry tool searches non-overlapping surface regions; feasibility is not guaranteed.
2. Request new_work_table and new_work_chair. The backend copies the SAME dining-table and chair assets from the public catalog, using an ordinary, preverified collision/access-preserving placement. This adds TWO furniture instances. Work capacity is still checked against the actual table surface. No sizing or deletion is allowed.
People do not work and eat simultaneously. Materials DO persist on their work region while both people eat. Only regions simultaneously occupied must be disjoint.
The explicit benchmark objective is to satisfy the published planar/mode contract, then minimize added furniture WITHIN THESE TWO SKILLS. This is not an independently established human preference.
Respond with one <tool_call>{"name":"allocate_resources","arguments":{...}}</tool_call>. You may precede it with <think>...</think>.
Exactly these argument fields are accepted:
work_surface: meal_table_0 or new_work_table;
work_seat: meal_chair_n_0 or new_work_chair;
meal_surface: meal_table_0;
meal_seats: [meal_chair_n_0,meal_chair_s_0];
material_persists_during_meal: true;
resident_works_while_meal: false.
Choose a binding, not a verbal recommendation. All geometry/optimality feedback uses the same ordinary checker for every policy.'''

def prepare(root):
    protocol=read(root/'protocol.json');surface=read(root/'measured_table_surface.json')
    src=R/'runs/autodl/20261111_milan_completion/source/runs/920/P920-MILAN-TASK-v1_20261105T224015Z/input_metadata'
    house=read(src/'house_state.json');task=read(src/'task.json');views=read(root/'view_manifest.json')
    rid=next(iter(house['rooms']));state=house['rooms'][rid]
    build,_=source_functions()
    requests={}
    for case in protocol['cases']:
        if case['id'] not in protocol['model_trials']['task_ids']:continue
        prompt=(task['design_request']+'\n\nThis is a NEW SYNTHETIC numeric subtask, not real user evidence or the old frozen score. '
            'The resident uses one work seat; during meals the resident and guest use the existing two meal seats. '
            f"Materials occupy a persistent {case['work_m'][0]} by {case['work_m'][1]} metre rectangle. "
            f"Each of two meal settings occupies {case['meal_m'][0]} by {case['meal_m'][1]} metre rectangle. "
            'All three regions must lie on valid table surfaces and be pairwise disjoint during meals. '
            'Meal centres must be on the corresponding north/south half of the dining table. '
            'Rectangles may rotate 0 or90degrees; tool search uses a1cm grid and1micrometre inset. '
            'Do not add an extra simultaneous worker. Both construction skills preserve original geometry/access; '
            'other ergonomic, functional and aesthetic requirements are not being scored here. '
            'Use the minimum furniture count among the two skills when feasible.')
        public=dict(initial_full_room_state=state,actual_table_top_area_m2=surface['projected_top_area_m2'],
            actual_table_world_bounds=surface['bounds'],measurement_method=surface['method'],
            feedback=None,native_observation_images='Original frozen five views of this exact initial scene, not newly rendered',
            covered_scope='Two-skill resource allocation component, not complete raw-text generation')
        requests[case['id']]=build(SYSTEM,prompt,json.dumps(public,ensure_ascii=False),[Path(v['local_path']) for v in views])
    return requests

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    root=a.root.resolve();requests=prepare(root)
    if a.prepare_only:
        for name,messages in requests.items():dump(root/(name+'_request.json'),messages)
        dump(root/'model_contract.json',dict(native_message_builder_and_parser_sha256=sha(SOURCE),
            native_actor_reproduction=False,custom_two_skill_component=True,
            fixed_existing_checkpoint='XuanyuLiu/Scene-Builder-Qwen3-VL-8B-Instruct joint SFT, revision63e235a0c0d6e5bc2f8f3485214df4457fd2e672',
            max_pixels=16777216,max_input_tokens=24576,max_new_tokens=1200,
            per_forward_alarm_seconds=180,total_process_cap_seconds=480,maximum_forwards=4,
            observations_and_tools_shared=True,independent_agent_method_comparison=False,
            feedback_rule='One new feedback turn only for invalid or nonminimal first result; no identical-request retry',
            ordinary_optimality_scope='Only the two published construction skills under declared planar proxies',
            initial_requests={n:sha(root/(n+'_request.json')) for n in requests},
            physical_material_objects_created=False,training=False,paid_API_calls=0))
        return
    result_path=root/'model_probe.json'
    if result_path.exists() or (root/'model_started.json').exists():raise FileExistsError('No implicit model rerun')
    lock=(root/'exclusive_model.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(root/'protocol.json');compiled=read(root/'compiled_programs.json');rows={x['id']:x for x in compiled['rows']}
    assert all(sha(R/k)==h for k,h in protocol['source_inputs'].items())
    contract=read(root/'model_contract.json')
    for n in requests:
        assert sha(root/(n+'_request.json'))==contract['initial_requests'][n]
        assert read(root/(n+'_request.json'))==requests[n]
    calls=0;network=[];start=time.monotonic();outputs=[]
    dump(root/'model_started.json',dict(pid=os.getpid(),start_unix=time.time(),no_rerun=True))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    def deny(*args,**kwargs):network.append(True);raise RuntimeError('Offline model forbids network')
    socket.socket.connect=deny;socket.create_connection=deny
    def expired(*_):raise TimeoutError('Prospectively bounded forward timed out; no retry')
    signal.signal(signal.SIGALRM,expired)
    try:
        sys.path.insert(0,str(W/'paris_local_run_results_20261104'))
        from local_vlm_sdpa import LocalQwenEngine
        import torch
        torch.manual_seed(0)
        engine=LocalQwenEngine(W/'critic_public_runtime_20261104/checkpoint',root/'model_calls',
            max_pixels=16777216,max_input_tokens=24576,max_new_tokens=1200)
        _,parser=source_functions()
        for name,messages in requests.items():
            case=rows[name];minimal=0 if case['shared']['status']=='witness' else 2
            for turn in range(2):
                assert calls<4
                calls+=1;signal.alarm(180)
                # complete writes the exact request before encoding/model load;
                # call budget includes any failed forward attempt.
                try:result=engine.complete(messages,purpose='custom_activity_allocation_component',metadata=dict(task=name,turn=turn))
                finally:signal.alarm(0)
                thought,parsed=parser(result['raw']);feedback=evaluate(parsed,case)
                feedback['minimum_in_published_skill_library']=minimal
                feedback['minimal']=bool(feedback['valid'] and feedback['new_furniture']==minimal)
                outputs.append(dict(task=name,turn=turn,parsed=parsed,feedback=feedback,receipt_dir=result['receipt_dir']))
                dump(result_path,dict(status='running',attempts=outputs,model_attempts=calls))
                if feedback['minimal']:break
                # Native Task/Memory layout retained; this explicit custom tool
                # feedback is new evidence, never an identical request retry.
                messages=copy.deepcopy(requests[name])
                messages[-1]['content'][-1]['text']+='\n\n## Tool feedback\n'+json.dumps(dict(previous_proposal=parsed,feedback=feedback),ensure_ascii=False)+'\nRevise the binding once using the same published tool.'
        dump(result_path,dict(status='completed',attempts=outputs,model_attempts=calls,
            elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            network_connection_attempts=len(network),paid_API_calls=0,training_updates=0,
            native_actor_reproduction=False,new_full_scene_generations=0))
    except BaseException as e:
        dump(result_path,dict(status='failed',attempts=outputs,model_attempts=calls,error=f'{type(e).__name__}: {e}',
            elapsed_seconds=time.monotonic()-start,network_connection_attempts=len(network),no_retry=True))
        raise

if __name__=='__main__':main()
