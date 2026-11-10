"""Use one native model instance across a complete generation trial.

This is an integration control, not a new policy or authorization to run one.
The caller owns native services, assets, weights and evaluation. No imports of
the native SDK, weight loading or service startup happen at module import.
Only furniture uses the chosen phase policy; other stages use the native base
policy with any experimental adapter disabled. All actual model.generate calls
are counted, including failures. Tool-side LLMs are NOT covered by this meter.
"""
from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path
import time

from phase_host import furniture_driver
from phase_policy import PhasePolicy, TorchPhaseBackend


class GenerationLedger:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.stages = []
        self.lifecycle = []
        self.trial = {'status': 'not_started', 'all_stage_finalizers_returned': False}
        self.save()

    def save(self):
        data = dict(model_calls=len(self.rows), calls=self.rows, stages=self.stages,
            lifecycle=self.lifecycle, trial=self.trial,
            accounting_complete=all(r.get('input_tokens') is not None and
                                    r.get('output_tokens') is not None for r in self.rows),
            known_input_tokens=sum(r.get('input_tokens') or 0 for r in self.rows),
            known_output_tokens=sum(r.get('output_tokens') or 0 for r in self.rows),
            full_scene_success=None,
            scope='Policy model.generate only; tool-side model calls and GPU kernel time are not measured. Driver wall times include observation/tool work. Finalizers and session lifecycle are timed separately. Returned finalizers do not prove scene quality.')
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
        tmp.replace(self.path)

    @contextmanager
    def operation(self, name, stage=None):
        row = dict(operation=name, stage=stage, status='running')
        self.lifecycle.append(row); self.save(); tick=time.monotonic()
        try:
            yield row
            row['status']='returned'
        except BaseException as exc:
            row.update(status='raised',error=type(exc).__name__)
            raise
        finally:
            row['wall_seconds']=time.monotonic()-tick
            self.save()

    @contextmanager
    def stage(self, name):
        row = dict(stage=name, status='running', first_model_call=len(self.rows))
        self.stages.append(row); self.save(); tick=time.monotonic()
        try:
            yield
            row['status']='returned'
        except BaseException as exc:
            row.update(status='raised', error=type(exc).__name__)
            raise
        finally:
            row.update(wall_seconds=time.monotonic()-tick, model_calls=len(self.rows)-row['first_model_call'])
            self.save()

    @contextmanager
    def meter(self, model, stage, adapter_enabled):
        if getattr(model, '_scene_generation_meter_active', False):
            raise RuntimeError('Nested/concurrent inference metering is unsupported')
        original = model.generate
        own = model.__dict__.get('generate')
        had_own = 'generate' in model.__dict__

        def generate(*args, **kwargs):
            row = dict(stage=stage, adapter_enabled=adapter_enabled, status='started',
                       input_tokens=None, output_tokens=None,
                       output_cap=kwargs.get('max_new_tokens'))
            self.rows.append(row); self.save(); tick=time.monotonic()
            try:
                inputs = kwargs.get('input_ids', args[0] if args else None)
                if inputs is not None and len(inputs.shape)==2 and inputs.shape[0]==1:
                    row['input_tokens']=int(inputs.shape[1])
                output=original(*args, **kwargs)
                sequence=getattr(output,'sequences',output)
                if (row['input_tokens'] is not None and hasattr(sequence,'shape') and
                        len(sequence.shape)==2 and sequence.shape[0]==1 and
                        sequence.shape[1]>=row['input_tokens']):
                    row['output_tokens']=int(sequence.shape[1])-row['input_tokens']
                row['status']='returned'
                return output
            except BaseException as exc:
                row.update(status='raised',error=type(exc).__name__)
                raise
            finally:
                row['wall_seconds']=time.monotonic()-tick
                self.save()
        model._scene_generation_meter_active=True
        model.generate=generate
        try:
            yield
        finally:
            if had_own:model.generate=own
            else:delattr(model,'generate')
            delattr(model,'_scene_generation_meter_active')


class BaseStagePolicy:
    def __init__(self, native_policy, ledger, has_adapter):
        self.native=native_policy
        self.ledger=ledger
        self.has_adapter=has_adapter

    @property
    def model(self):return self.native.model

    @property
    def processor(self):return self.native.processor

    def generate(self, obs):
        scope=self.model.disable_adapter() if self.has_adapter else nullcontext()
        with scope, self.ledger.meter(self.model,obs['stage'],False):
            return self.native.generate(obs)


class MeasuredPhasePolicy(PhasePolicy):
    def __init__(self, backend, *, ledger, model, has_adapter, **kwargs):
        super().__init__(backend,**kwargs)
        self.ledger=ledger;self.model=model;self.has_adapter=has_adapter

    def generate(self, obs):
        with self.ledger.meter(self.model,'furniture',self.has_adapter):
            return super().generate(obs)


class _ForwardingProxy:
    """Keep scene/memory mutations on the original native instance."""
    def __init__(self, target, ledger):
        object.__setattr__(self,'_target',target)
        object.__setattr__(self,'_ledger',ledger)

    def __getattr__(self, key):return getattr(self._target,key)

    def __setattr__(self, key, value):setattr(self._target,key,value)


class _ObservedEnvironment(_ForwardingProxy):
    def finalize_stage(self):
        with self._ledger.operation('finalize',self.stage):
            return self._target.finalize_stage()

    def cleanup(self):
        with self._ledger.operation('cleanup',self.stage):
            return self._target.cleanup()


class _ObservedSession(_ForwardingProxy):
    def start(self):
        with self._ledger.operation('session_start'):
            return self._target.start()

    def make_floor_plan_env(self):
        return _ObservedEnvironment(self._target.make_floor_plan_env(),self._ledger)

    def make_roomscene_env(self,*args,**kwargs):
        return _ObservedEnvironment(self._target.make_roomscene_env(*args,**kwargs),self._ledger)

    def handover_after_floor_plan(self,*args,**kwargs):
        with self._ledger.operation('floor_plan_handover') as row:
            result=self._target.handover_after_floor_plan(*args,**kwargs)
            row['accepted']=bool(result)
            return result

    def close(self):
        with self._ledger.operation('session_close'):
            return self._target.close()


@contextmanager
def shared_model_driver(native_module, *, phase_settings, stage_budget,
                        allowed_tools, load_images, acceptance, evidence_format,
                        ledger, adapter_loader=None, backend_factory=TorchPhaseBackend,
                        tool_vlm_binding=None):
    """Wrap native main_async without a second model constructor.

    adapter_loader, if supplied, attaches an already available local adapter to
    the loaded base model; it must not merge it. Its disable_adapter context is
    required so floor/wall/ceiling remain a common base policy. Provenance and
    loaded tensor equivalence remain the experiment's responsibility.
    """
    if evidence_format not in ('append_v1','canonical_v2'):
        raise ValueError('Explicit known evidence format required')
    if phase_settings.get('arm') not in ('paired_phase_v3','separate_direct_v3','separate_phase_v3'):
        raise ValueError('Unknown phase condition')
    factory=native_module.ModelPolicy; original_driver=native_module.driver_loop
    session_factory=native_module.RolloutSession
    loaded=[]

    def make_policy(*args,**kwargs):
        if loaded:raise RuntimeError('Native trial attempted to load a second policy model')
        policy=factory(*args,**kwargs)
        if adapter_loader is not None:
            policy.model=adapter_loader(policy.model)
            if not callable(getattr(policy.model,'disable_adapter',None)):
                raise ValueError('Unmerged adapter with reversible disable_adapter required')
        wrapper=BaseStagePolicy(policy,ledger,adapter_loader is not None)
        loaded.append(wrapper)
        if tool_vlm_binding is not None:
            tool_vlm_binding.bind(policy.model,policy.processor,adapter_loader is not None)
        return wrapper

    async def driver(*,env,policy,max_steps):
        if len(loaded)!=1 or policy is not loaded[0]:
            raise RuntimeError('Native stage did not use the common loaded policy')
        with ledger.stage(env.stage):
            if env.stage!='furniture':
                return await original_driver(env=env,policy=policy,max_steps=max_steps)
            phase=MeasuredPhasePolicy(backend_factory(policy.model,policy.processor),
                ledger=ledger,model=policy.model,has_adapter=adapter_loader is not None,
                **copy.deepcopy(phase_settings))
            with furniture_driver(native_module,policy=phase,budget=stage_budget,
                                  allowed_tools=allowed_tools,load_images=load_images,
                                  acceptance=acceptance,evidence_format=evidence_format):
                return await native_module.driver_loop(env=env,policy=policy,max_steps=max_steps)

    native_module.ModelPolicy=make_policy
    native_module.driver_loop=driver
    native_module.RolloutSession=lambda *a,**kw: _ObservedSession(session_factory(*a,**kw),ledger)
    try:
        # Install before session.start creates service objects; bind only after
        # native main loads its single model. Tool receipts are a separate ledger.
        with tool_vlm_binding.scope() if tool_vlm_binding is not None else nullcontext():
            yield
    finally:
        native_module.ModelPolicy=factory
        native_module.driver_loop=original_driver
        native_module.RolloutSession=session_factory
        ledger.save()


async def run_native_trial(native_module, args, *, runtime_admission, **bridge):
    """Execute only an explicitly qualified runtime; not a static readiness test.

    Qualification must cover tool-side services, not just this local model.
    This declaration is an entry contract, NOT proof that the runtime is free
    from external requests. Its cited evidence must be independently inspected.
    """
    required={'full_stage_assets','tool_side_no_paid_api','common_evaluation','common_action_scope'}
    if (set(runtime_admission)!=required or
            any(not isinstance(v,dict) or v.get('verified') is not True or
                not isinstance(v.get('evidence'),str) or not v['evidence'].strip()
                for v in runtime_admission.values())):
        raise ValueError('Full runtime has not been qualified; no native startup performed')
    if args.skip_furniture or args.skip_wall or args.skip_ceiling:
        raise ValueError('Full-generation trial cannot silently skip room stages')
    ledger=bridge['ledger'];tick=time.monotonic()
    ledger.trial.update(status='running')
    ledger.save()
    try:
        with shared_model_driver(native_module,**bridge):
            await native_module.main_async(args)
        # Native main_async catches floor geometry exceptions and returns;
        # a zero process exit / normal return is therefore not completion.
        finalized=[r['stage'] for r in ledger.lifecycle
                   if r['operation']=='finalize' and r['status']=='returned']
        complete=finalized==['floor_plan','furniture','wall','ceiling']
        ledger.trial.update(all_stage_finalizers_returned=complete,
                            returned_finalizers=finalized)
        if not complete:
            raise RuntimeError('Native controller returned before all stage finalizers completed')
        ledger.trial['status']='native_stages_returned_quality_unassessed'
    except BaseException as exc:
        ledger.trial.update(status='incomplete_or_failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        ledger.trial['wall_seconds']=time.monotonic()-tick
        ledger.save()
    return copy.deepcopy(ledger.trial)
