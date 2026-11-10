"""Run native four-stage and adapted five-stage controllers with explicit doubles.

Verifies stage/model ownership and cost recording, NOT 3D generation. No weights,
SDK servers, renderers, assets or model calls are used. Token counts are fixture
values and must never enter an experimental efficiency table.
"""
import argparse
import ast
import asyncio
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from check_native_phase_host import Tokenizer
from full_generation_bridge import GenerationLedger, run_native_trial
from phase_host import StageBudget, digest


class Shape:
    def __init__(self, length):self.shape=(1,length)


class ImageReference:
    def copy(self):return self


async def case(args, name, arm, with_adapter, fail_stage=None, admitted=True, completion=False):
    out=args.out/name;out.mkdir(parents=True,exist_ok=False)
    events=[];loads=[];models=[]
    tokenizer=Tokenizer(args.tokenizer)
    eos=tokenizer.convert_tokens_to_ids('<|im_end|>')

    class Model:
        def __init__(self):self.adapter_enabled=False
        @contextmanager
        def disable_adapter(self):
            prior=self.adapter_enabled;self.adapter_enabled=False
            try:yield
            finally:self.adapter_enabled=prior
        def generate(self,*,input_ids,max_new_tokens,completion_ids,stage):
            events.append(dict(event='fixture_model_generate',stage=stage,adapter_enabled=self.adapter_enabled))
            if stage==fail_stage:raise RuntimeError('deliberate backend failure')
            assert len(completion_ids)<=max_new_tokens
            return Shape(input_ids.shape[1]+len(completion_ids))

    class BasePolicy:
        def __init__(self,*a,**kw):
            loads.append('one_base_model');self.model=Model();models.append(self.model)
            self.processor=SimpleNamespace(tokenizer=tokenizer)
        def generate(self,obs):
            self.model.generate(input_ids=Shape(32),max_new_tokens=128,
                                completion_ids=[eos],stage=obs['stage'])
            return dict(raw='<|im_end|>',think=None,tool_calls=[])

    class Backend:
        def __init__(self,model,processor):
            self.model=model;self.processor=processor;self.eos_ids=[eos]
        def generate(self,messages,images,*,stage,token_limit,**kwargs):
            text='<think>Controller fixture only.</think>' if stage=='decision' else '<|im_end|>'
            if stage=='joint':text='<think>Controller fixture only.</think><|im_end|>'
            ids=tokenizer.encode(text)
            self.model.generate(input_ids=Shape(32),max_new_tokens=token_limit,
                                completion_ids=ids,stage='furniture')
            return dict(output_ids=ids,input_tokens=32,scope='scripted lifecycle fixture')

    state={'objects':{}}
    class Env:
        def __init__(self,stage,scene_dir):
            self.stage=stage;self.room_prompt='Controller fixture, not a scene task'
            self.mem={};self.steps_dir=scene_dir/('rollout_'+stage);self.steps_dir.mkdir()
            self.scene=SimpleNamespace(to_state_dict=lambda:state)
            self.tool_map={'move_furniture_tool':SimpleNamespace(params_json_schema={'type':'object'})}
        def observation(self):
            return dict(stage=self.stage,step_idx=0,memory_str='',image_path=None,
                        messages=[{'role':'system','content':'Controller fixture, no task labels.'}])
        async def step(self,calls):raise AssertionError('No action in this controller fixture')
        def finalize_stage(self):
            events.append(dict(event='fixture_finalize',stage=self.stage))
            if fail_stage==self.stage+'_finalizer':
                raise RuntimeError('deliberate finalizer failure')
        def cleanup(self):events.append(dict(event='fixture_cleanup',stage=self.stage))

    class Session:
        def __init__(self,scene_dir,*a):
            self.scene=SimpleNamespace(scene_dir=scene_dir)
        def start(self):events.append(dict(event='fixture_start'))
        def make_floor_plan_env(self):return Env('floor_plan',self.scene.scene_dir)
        def handover_after_floor_plan(self,room):
            events.append(dict(event='fixture_handover'))
            return fail_stage!='handover'
        def make_roomscene_env(self,stage,*a):return Env(stage,self.scene.scene_dir)
        def close(self):events.append(dict(event='fixture_close'))

    ns=dict(Path=Path,datetime=datetime,time=time,sys=sys,json=json,argparse=argparse,
            RolloutSession=Session,ModelPolicy=BasePolicy,
            ConsoleLogger=lambda **kw:None,write_run_metrics=lambda *a:None)
    tree=ast.parse(args.native_source.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name in ('driver_loop','main_async')]
    assert {n.name for n in nodes}=={'driver_loop','main_async'}
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[])),str(args.native_source),'exec'),ns)
    # Functions resolve ModelPolicy/driver_loop in ns; this object writes through
    # to that dictionary so it exercises the same names as the real module.
    class Module:
        def __getattr__(self,k):return ns[k]
        def __setattr__(self,k,v):ns[k]=v
    module=Module();original_driver=module.driver_loop
    def adapter_loader(model):
        model.adapter_enabled=True
        events.append(dict(event='fixture_adapter_attached'))
        return model
    native_args=SimpleNamespace(results_dir=out/'native',config='fixture',prompt='fixture',
        model_dir=Path('/not-loaded'),max_new_tokens=128,max_steps_per_stage=1,
        room=None,skip_furniture=False,skip_wall=False,skip_ceiling=False,render_gpu_id=None)
    ledger=GenerationLedger(out/'model_calls.json')
    admission={k:{'verified':True,'evidence':'TEST DOUBLE ONLY; not production qualification'}
               for k in ('full_stage_assets','tool_side_no_paid_api','common_evaluation','common_action_scope')}
    if not admitted:admission['tool_side_no_paid_api']['verified']=False
    error=None
    async def complete(native,session,policy,arguments):
        assert not any(e['event']=='fixture_close' for e in events)
        events.append(dict(event='fixture_manipuland_before_close'))
        policy.generate(dict(stage='manipuland'))
        if fail_stage=='manipuland_finalizer':
            raise RuntimeError('deliberate manipuland finalizer failure')
        return dict(status='fixture_returned',full_scene_success=None)
    try:
        await run_native_trial(module,native_args,runtime_admission=admission,
            required_stages=('floor_plan','furniture','wall','ceiling')+(
                ('manipuland',) if completion else ()),
            completion_driver=complete if completion else None,
            service_preflight=lambda *_: events.append(dict(event='fixture_service_preflight',
                scope='Service double only; not real configuration qualification')),
            phase_settings=dict(arm=arm,total_output_tokens=128,decision_output_tokens=64,
                action_output_tokens=64,total_seconds=10),
            stage_budget=StageBudget(1,1 if arm=='joint_same_material_v1' else 2,1,128,30),
            allowed_tools=('move_furniture_tool',),load_images=lambda _: [ImageReference() for _ in range(5)],
            acceptance=lambda _:dict(state_sha256=digest(state),full_scene_success=None),
            evidence_format='canonical_v2',ledger=ledger,
            adapter_loader=adapter_loader if with_adapter else None,backend_factory=Backend)
    except (RuntimeError,ValueError) as exc:error=type(exc).__name__+': '+str(exc)
    assert module.ModelPolicy is BasePolicy and module.driver_loop is original_driver
    assert module.RolloutSession is Session
    for model in models:
        assert 'generate' not in model.__dict__ and not hasattr(model,'_scene_generation_meter_active')
    data=json.loads(ledger.path.read_text())
    if not admitted:
        assert error and not loads and not events and not data['calls']
    else:
        assert len(loads)==1 and events[-1]['event']=='fixture_close'
        for e in events:
            if e['event']=='fixture_model_generate':
                assert e['adapter_enabled']==(with_adapter and e['stage']=='furniture')
        expected=['floor_plan','furniture','wall','ceiling']
        if completion:expected.append('manipuland')
        if fail_stage=='wall':expected=expected[:3]
        if fail_stage in ('floor_plan_finalizer','handover'):expected=expected[:1]
        assert [r['stage'] for r in data['stages']]==expected
        assert bool(error)==bool(fail_stage)
        assert data['accounting_complete']==(fail_stage!='wall')
        assert data['trial']['all_stage_finalizers_returned']==(fail_stage is None)
        assert data['trial']['wall_seconds']>0
        assert data['full_scene_success'] is None
        if fail_stage in ('floor_plan_finalizer','handover'):
            assert 'before all stage finalizers' in error
        assert data['lifecycle'][-1]['operation']=='session_close'
        assert data['lifecycle'][-1]['status']=='returned'
    result=dict(case=name,status='passed',model_constructions=len(loads),arm=arm,
        entrypoint='adapted_five_stage' if completion else 'original_four_stage',
        fixture_generate_calls=len(data['calls']),stage_order=[r['stage'] for r in data['stages']],
        trial=data['trial'],
        error=error,restored_native_bindings=True,events=events,
        actual_language_model_calls=0,actual_native_scene_actions=0,actual_generated_scenes=0)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


async def main(a):
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    rows=[]
    for params in [('paired','paired_phase_v3',True,None,True),
                   ('direct','separate_direct_v3',True,None,True),
                   ('wall_failure','paired_phase_v3',True,'wall',True),
                   ('floor_finalize_failure','paired_phase_v3',True,'floor_plan_finalizer',True),
                   ('handover_failure','paired_phase_v3',True,'handover',True),
                   ('ceiling_finalize_failure','paired_phase_v3',True,'ceiling_finalizer',True),
                   ('unqualified','paired_phase_v3',False,None,False),
                   ('complete_five_stages','paired_phase_v3',True,None,True,True),
                   ('joint_complete_five_stages','joint_same_material_v1',True,None,True,True),
                   ('manipuland_failure','paired_phase_v3',True,'manipuland_finalizer',True,True)]:
        rows.append(await case(a,*params))
    summary=dict(status='passed',cases=[{k:v for k,v in r.items() if k!='events'} for r in rows],
        native_controller_source=str(a.native_source),
        native_controller_sha256=hashlib.sha256(a.native_source.read_bytes()).hexdigest(),
        seconds=time.monotonic()-start,actual_language_model_calls=0,
        actual_generated_scenes=0,scope='Original four-stage main/driver and adapted five-stage entrypoint with explicit service/model doubles; NOT a model, native SDK or full-scene experiment.')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('native-source','tokenizer','out'):ap.add_argument('--'+k,type=Path,required=True)
    asyncio.run(main(ap.parse_args()))
