"""Text-only interpretation gate on certified activity/geometry interactions.

Not native Actor reproduction, visual reasoning, a new method, or training.
Four direct decisions first; only if one fails are the same four inputs passed
through the same model's ordinary program parser plus a common geometry binder.
Prior closed room-generation budgets are neither read as permission nor reset.
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

R=Path(__file__).resolve().parents[2]
W=R.parent.parent
BASE=R/'runs/autodl/20261111_same_task_geometry'
ROLLOUT=W/'native_execution_inventory_20261104/source/Designer/agent_rollout.py'
ENGINE=W/'paris_local_run_results_20261104/local_vlm_sdpa.py'
CHECKPOINT=W/'critic_public_runtime_20261104/checkpoint'


def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def upstream_functions():
    chosen=[]
    for n in ast.parse(ROLLOUT.read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['_THINK_RE','_TOOL_CALL_RE'] for t in n.targets):chosen.append(n)
        if isinstance(n,ast.FunctionDef) and n.name in ['parse_assistant_text','build_messages']:chosen.append(n)
    env={'Path':Path,'re':re,'json':json}
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(ROLLOUT),'exec'),env)
    return env['build_messages'],env['parse_assistant_text']


COMMON='''You interpret use requirements for an existing furnished 3D room.
This is a TEXT-ONLY component, not full scene generation or visual perception.
The task text and the geometry tool observations are the complete common input.
The geometry observations report the capacities of specified footprint combinations,
not whether those combinations are required by the user's activity schedule.
Infer the actual required combination from the text. Distinguish a person from
materials left on a surface. Do not introduce unrequested simultaneous people,
independent tables, preference scores, or assumptions contradicting the text.
Return exactly one tool call and no chain of thought, commentary or markdown.
evidence_quotes must be a nonempty list of exact substrings of the Task text.
Quotes are source citations, not a fabricated expert rationale.
'''
DIRECT=COMMON+'''
Tool: assess_shared_surface(feasible: boolean or null, evidence_quotes: list[str]).
Judge only whether the designated existing table can support all stated activity
modes. null means unresolved; do not use null for a case the observations settle.
Respond: <tool_call>{"name":"assess_shared_surface","arguments":{...}}</tool_call>.
'''
PROGRAM=COMMON+'''
Tool: compile_activity_program(people_by_phase: object, materials_active_phases:
list[str] or null, evidence_quotes: list[str]).
Available phase IDs: work, meal. Available person IDs: resident, guest.
people_by_phase maps each phase ID to the person IDs actually using that phase.
materials_active_phases lists phases in which WORK MATERIALS occupy table area;
it does not describe when a person is working. Use null only if unresolved.
A common deterministic binder will combine this program with the SAME geometry
facts supplied here. It does not repair wrong interpretations or add people.
Respond: <tool_call>{"name":"compile_activity_program","arguments":{...}}</tool_call>.
'''


def prepare(root):
    if root.exists() and any(p.name!='preparation_v1' for p in root.iterdir()):
        raise FileExistsError('Fresh diagnostic directory required except archived pre-inference preparation')
    root.mkdir(parents=True,exist_ok=True)
    protocol=read(BASE/'protocol.json');worlds=read(BASE/'final_result.json')['rows']
    assert protocol['task_text'].count('资料持续占用')==1
    # Neutralize only the archival provenance sentence, not the active task.
    # Otherwise the clear variant would retain a stale keep-provenance mention.
    keep=protocol['task_text'].replace('资料持续占用','资料跨阶段的处理方式')
    old1='办公资料可以留在工作面上，用餐时无需收走；'
    new1='住户同意用餐前收走全部工作资料，工作与用餐先后使用桌面；'
    old2='资料用餐时仍留在同一工作面。'
    new2='资料在用餐前全部收走，用餐阶段不占桌面。'
    assert keep.count(old1)==keep.count(old2)==1
    clear=keep.replace(old1,new1).replace(old2,new2)
    assert old1 not in clear and old2 not in clear and clear!=keep
    texts={'keep':keep,'clear':clear};byworld={r['world']:r for r in worlds}
    build,_=upstream_functions();records=[]
    # Fixed before any encoding or generation. One negative and three positive
    # component labels: always-True and geometry-only are both75% controls.
    order=[('keep','original_table'),('clear','smaller_table'),
           ('keep','smaller_table'),('clear','original_table')]
    for index,(intent,world) in enumerate(order):
        w=byworld[world];geometry=dict(
            tool_origin='Previously executed real-mesh reservation checks; same facts to both methods',
            units='metres',footprints=dict(work_materials=[.4,.4],each_meal=[.35,.25]),
            geometry_fact=dict(one_work_region_fits=True,two_meal_regions_fit=True,
                               work_materials_plus_two_meal_regions_fit=w['correct_shared_surface_feasibility']),
            evidence_limits='Projected top-band reservations, not full ergonomics, subjective design or a visual observation',
            original_objects_retained=30)
        case_id=f'case_{index:02d}'
        for arm,system in [('direct',DIRECT),('program',PROGRAM)]:
            messages=build(system,texts[intent],json.dumps(geometry,ensure_ascii=False,allow_nan=False),[])
            dump(root/f'{case_id}_{arm}_request.json',messages)
        records.append(dict(id=case_id,intent=intent,world=world,
            task_sha256=hashlib.sha256(texts[intent].encode()).hexdigest(),
            expected_material_phases=['work','meal'] if intent=='keep' else ['work'],
            expected_people=dict(work=['resident'],meal=['resident','guest']),
            expected_shared_feasible=w['correct_shared_surface_feasibility'] if intent=='keep' else True,
            common_geometry=geometry,source_world_evidence_sha256=sha(BASE/'final_result.json')))
    assert [x['expected_shared_feasible'] for x in records]==[True,True,False,True]
    dump(root/'cases_and_gold.json',records)
    requests={str(p):sha(p) for p in sorted(root.glob('*_request.json'))}
    dump(root/'protocol.json',dict(
        scope='Raw-text activity interpretation with common exact geometry observations, on one exposed furnished source family',
        semantic_truth='Explicit research-authored keep/clear task clauses; not original homeowner or professional labels',
        pre_inference_edit='Neutralized archival provenance mention of material persistence in BOTH tasks; old preparation archived before any model call. Active keep/clear clauses remain explicit.',
        cases=4,source_families=1,new_world_geometries=0,new_whole_scene_generations=0,
        methods=['Direct task-conditioned judgement','Same-model program extraction plus ordinary geometry binding'],
        common_input_excludes_gold=True,model_receives_no_expected_active_phase_values=True,
        text_only=True,images=0,actual_vision_parity=False,native_actor_parity=False,
        execution_order=[x['id'] for x in records],max_model_generate_calls=8,
        stop_after_direct_all_correct=True,
        continuation_rule='If any direct case is invalid/incorrect/unknown, run program arm on ALL four same cases, not just the failures',
        no_feedback_or_second_try=True,no_quality_retry=True,
        checkpoint=str(CHECKPOINT),checkpoint_revision='63e235a0c0d6e5bc2f8f3485214df4457fd2e672',
        model_role='Public joint-SFT8B checkpoint; not trained Critic v10 or author full pipeline',
        max_input_tokens=8192,max_new_tokens=768,max_pixels=16777216,
        per_call_alarm_seconds=120,total_wall_cap_seconds=600,
        requests=requests,source_hashes={str(ROLLOUT):sha(ROLLOUT),str(ENGINE):sha(ENGINE),
            str(BASE/'protocol.json'):sha(BASE/'protocol.json'),str(BASE/'final_result.json'):sha(BASE/'final_result.json')},
        costs='Count all model attempts/tokens and compiler time. If program arm runs, compare all four cases and all costs.',
        limits=['No claim of generalization from4component cases in1source family',
            'Exact substring citation is not an automatic semantic-entailment test',
            'Program fields still require source evidence; no human rationale invented',
            'No closed prior whole-room or allocation inference budget is reopened'],
        training_updates=0,paid_API_calls=0,new_model_downloads=0))
    print(json.dumps(dict(status='prepared',cases=4,request_files=8,model_calls=0)),flush=True)


def score(calls,arm,case,task):
    expected_name='assess_shared_surface' if arm=='direct' else 'compile_activity_program'
    if len(calls)!=1 or calls[0].get('name')!=expected_name:
        return dict(valid=False,reason='Exactly one declared tool call required',prediction=None,correct=False)
    args=calls[0].get('arguments',{})
    fields={'feasible','evidence_quotes'} if arm=='direct' else {'people_by_phase','materials_active_phases','evidence_quotes'}
    if set(args)!=fields:return dict(valid=False,reason='Tool fields differ from fixed schema',prediction=None,correct=False)
    quotes=args.get('evidence_quotes')
    quotes_valid=(isinstance(quotes,list) and bool(quotes) and all(isinstance(q,str) and bool(q) and q in task for q in quotes))
    if not quotes_valid:return dict(valid=False,reason='Missing/nonliteral source citation',prediction=None,correct=False,arguments=args)
    if arm=='direct':
        pred=args['feasible'];valid=pred is None or type(pred) is bool
        return dict(valid=valid,prediction=pred,correct=bool(valid and pred is case['expected_shared_feasible']),
            source_quotes_literal=True,source_quotes_entailment='not automatically established',arguments=args)
    people=args['people_by_phase'];active=args['materials_active_phases']
    valid=(isinstance(people,dict) and set(people)=={'work','meal'} and all(isinstance(x,list) and all(isinstance(y,str) for y in x) for x in people.values())
           and (active is None or isinstance(active,list) and all(x in ['work','meal'] for x in active)))
    if not valid:return dict(valid=False,reason='Malformed activity program',prediction=None,correct=False,arguments=args)
    semantic=(all(sorted(people[k])==sorted(case['expected_people'][k]) for k in people) and
              active is not None and sorted(active)==sorted(case['expected_material_phases']))
    common=case['common_geometry']['geometry_fact']
    if active is None:pred=None
    elif set(people['work'])!={'resident'} or set(people['meal'])!={'resident','guest'}:pred=None
    elif set(active)=={'work','meal'}:pred=common['work_materials_plus_two_meal_regions_fit']
    elif active==['work']:pred=common['one_work_region_fits'] and common['two_meal_regions_fit']
    else:pred=None
    return dict(valid=True,prediction=pred,correct=bool(pred is case['expected_shared_feasible'] and semantic),
        semantic_fields_exact=semantic,source_quotes_literal=True,
        source_quotes_entailment='not automatically established',arguments=args)


def encode(root):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,str(ENGINE.parent));from local_vlm_sdpa import LocalQwenEngine
    engine=LocalQwenEngine(CHECKPOINT,root/'unused_encoding_log',max_pixels=16777216,max_input_tokens=8192,max_new_tokens=768)
    rows=[]
    for case in read(root/'cases_and_gold.json'):
        for arm in ['direct','program']:
            _,receipt=engine.encode(read(root/f"{case['id']}_{arm}_request.json"))
            assert not receipt['images']
            rows.append(dict(case=case['id'],arm=arm,encoding=receipt))
    assert engine._model is None
    dump(root/'encoding_preflight.json',dict(rows=rows,model_weights_loaded=False,model_calls=0,all_without_truncation=True))
    print(json.dumps(dict(status='encoding_ready',max_input_tokens=max(x['encoding']['input_tokens'] for x in rows),model_weights_loaded=False)),flush=True)


def run(root):
    p=read(root/'protocol.json');cases=read(root/'cases_and_gold.json')
    if (root/'model_started.json').exists() or (root/'model_result.json').exists():raise FileExistsError('No implicit model rerun')
    for path,h in {**p['requests'],**p['source_hashes']}.items():assert sha(path)==h
    assert read(root/'encoding_preflight.json')['all_without_truncation']
    assert int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text())>=16*1024**3
    # This global lock is merely cooperative; actual device usage is also
    # checked by the caller. A stale lock file is not treated as a live job.
    lock=(W/'local_intent_diagnostic_serial.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();outputs=[];attempts=0;network=[]
    dump(root/'model_started.json',dict(pid=os.getpid(),time_unix=time.time(),maximum_calls=8,no_rerun=True))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def deny(*_,**__):network.append(True);raise RuntimeError('No network during offline model probe')
    socket.socket.connect=deny;socket.create_connection=deny
    def expire(*_):raise TimeoutError('Fixed local diagnostic time budget; no retry')
    signal.signal(signal.SIGALRM,expire)
    sys.path.insert(0,str(ENGINE.parent));from local_vlm_sdpa import LocalQwenEngine
    import torch
    assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info();assert free>=26*1024**3,'Existing device must be idle; do not displace another job'
    engine=LocalQwenEngine(CHECKPOINT,root/'model_calls',max_pixels=16777216,max_input_tokens=8192,max_new_tokens=768)
    _,parse=upstream_functions();torch.manual_seed(0)
    try:
        arm='direct'
        while True:
            for case in cases:
                if time.monotonic()-start>p['total_wall_cap_seconds']:raise TimeoutError('Total local probe budget')
                assert attempts<p['max_model_generate_calls'];attempts+=1
                request=read(root/f"{case['id']}_{arm}_request.json")
                signal.alarm(p['per_call_alarm_seconds'])
                try:r=engine.complete(request,purpose='text_only_intent_geometry_gate',metadata=dict(case=case['id'],arm=arm))
                finally:signal.alarm(0)
                _,calls=parse(r['raw'])
                text=request[-1]['content'][-1]['text'];task=text[len('## Task\n'):].split('\n\n## Memory\n',1)[0]
                t=time.monotonic();graded=score(calls,arm,case,task);compiler_seconds=time.monotonic()-t
                outputs.append(dict(case=case['id'],arm=arm,parsed=calls,score=graded,
                    receipt_dir=r['receipt_dir'],compiler_and_schema_seconds=compiler_seconds))
                dump(root/'model_result.json',dict(status='running',attempts=outputs,model_calls=attempts))
            if arm=='program':break
            if all(x['score']['correct'] for x in outputs):break
            arm='program'
        dump(root/'model_result.json',dict(status='completed',attempts=outputs,model_calls=attempts,
            program_arm_ran=any(x['arm']=='program' for x in outputs),
            direct_all_correct=all(x['score']['correct'] for x in outputs if x['arm']=='direct'),
            early_stop_if_direct_correct=p['stop_after_direct_all_correct'],
            elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            network_attempts=len(network),paid_API_calls=0,training_updates=0,
            text_only=True,new_scenes_generated=0,native_actor_reproduction=False))
    except BaseException as e:
        dump(root/'model_result.json',dict(status='failed',attempts=outputs,model_calls=attempts,
            error=f'{type(e).__name__}: {e}',elapsed_seconds=time.monotonic()-start,no_retry=True))
        raise
    finally:signal.alarm(0)
    print(json.dumps(dict(status='completed',model_calls=attempts,direct_correct=sum(x['score']['correct'] for x in outputs if x['arm']=='direct'),program_arm_ran=any(x['arm']=='program' for x in outputs))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--mode',choices=['prepare','encode','run'],required=True);a=ap.parse_args()
    {'prepare':prepare,'encode':encode,'run':run}[a.mode](a.root.resolve())
