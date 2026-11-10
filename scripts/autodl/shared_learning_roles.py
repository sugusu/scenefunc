"""Serialized ordinary LoRA role isolation, with no loader or optimizer.

Supports an already-created single-device, unmerged, standard two-adapter PEFT
model. The caller must use this SAME lock for every model operation and keep
student forward/backward/optimizer within student_train. Checkpoint identity,
quantized/offloaded/distributed models, VLM numerics and cache semantics are not
qualified by this controller or by the small CPU probe.
"""
from contextlib import ExitStack,contextmanager
import threading

from local_tool_vlm import require_reversible_adapter


ROLES=('student_train','policy_eval','teacher_eval','tool_eval')


def require_student_only_optimizer(optimizer,roles):
    """Inspect ownership only; never create or step an optimizer.

    requires_grad=False alone does not remove a parameter from an optimizer or
    clear a stale gradient. The optimizer must own exactly the declared student
    adapter parameters, with no shared base, teacher or duplicate entries.
    """
    params=dict(roles.model.named_parameters())
    expected={id(params[name]) for name in roles.student_names}
    actual=[p for group in optimizer.param_groups for p in group['params']]
    ids=[id(p) for p in actual]
    if len(ids)!=len(set(ids)) or set(ids)!=expected:
        raise ValueError('Optimizer must contain exactly student adapter parameters once')
    return dict(student_parameter_tensors=len(expected),teacher_or_base_parameter_tensors=0,
                optimizer_step_executed=False)


class SharedLearningRoles:
    def __init__(self,model,*,lock,student_adapter='student',teacher_adapter='teacher'):
        if not isinstance(lock,type(threading.RLock())):
            raise ValueError('One explicit shared threading.RLock is required')
        if not student_adapter or not teacher_adapter or student_adapter==teacher_adapter:
            raise ValueError('Distinct explicit student and teacher adapters required')
        self.model=model;self.lock=lock;self.student=student_adapter;self.teacher=teacher_adapter
        self.events=[];self.active=None
        self._validate_configuration()
        student,teacher='.'+self.student+'.','.'+self.teacher+'.'
        params=dict(model.named_parameters())
        self.student_names={n for n in params if student in n}
        self.teacher_names={n for n in params if teacher in n}
        if not self.student_names or not self.teacher_names:raise ValueError('Both adapter parameter sets required')
        for name in self.student_names:
            counterpart=name.replace(student,teacher,1)
            if counterpart not in self.teacher_names or params[counterpart].shape!=params[name].shape:
                raise ValueError('Student and teacher adapter parameter shapes differ')
        if any(p.requires_grad for n,p in params.items() if n not in self.student_names):
            raise ValueError('Shared base and teacher must already be frozen outside roles')
        devices={p.device for p in params.values()}
        if len(devices)!=1 or next(iter(devices)).type not in ('cpu','cuda'):
            raise ValueError('Only an already materialized single-device model is supported')
        self.device=next(iter(devices))

    def _validate_configuration(self):
        require_reversible_adapter(self.model)
        status=self.model.get_model_status()
        if status.enabled is not True:raise ValueError('Enter roles only from fully enabled adapter state')
        cfgs=getattr(self.model,'peft_config',{})
        if hasattr(self.model,'_orig_mod'):raise ValueError('Compiled model state has not been qualified')
        for cfg in cfgs.values():
            kind=getattr(cfg.peft_type,'value',cfg.peft_type)
            if kind!='LORA':raise ValueError('Only standard LoRA is qualified')
            if (cfg.bias!='none' or getattr(cfg,'lora_bias',False) or cfg.modules_to_save or
                getattr(cfg,'trainable_token_indices',None) or getattr(cfg,'layer_replication',None)):
                raise ValueError('Adapted shared bias/modules/tokens/layers cannot be a common frozen tool base')
            if getattr(cfg,'use_dora',False) or getattr(cfg,'use_qalora',False):
                raise ValueError('Adapter variant not qualified by this controller')
            if cfg.init_lora_weights not in (True,'gaussian'):
                raise ValueError('Potentially base-modifying initialization is not qualified')
        if set(cfgs)!={self.student,self.teacher}:raise ValueError('Exactly the two declared adapters are required')
        self._active_adapter()

    def _active_adapter(self):
        active=self.model.get_model_status().active_adapters
        if not isinstance(active,list) or len(active)!=1 or active[0] not in (self.student,self.teacher):
            raise ValueError('One consistent declared active adapter required')
        return active[0]

    @contextmanager
    def role(self,name):
        if name not in ROLES:raise ValueError('Unknown model role')
        import torch
        with self.lock:
            if self.active is not None:raise RuntimeError('Nested role scopes are not supported')
            self._validate_configuration()
            previous=self._active_adapter()
            modules=[(n,m,m.training) for n,m in self.model.named_modules()]
            params=[(n,p,p.requires_grad) for n,p in self.model.named_parameters()]
            self.active=name
            event=dict(role=name,thread_id=threading.get_ident(),status='entered',
                       previous_active_adapter=previous,checkpoint_identity_checked=False)
            self.events.append(event)
            try:
                with ExitStack() as stack:
                    if name!='tool_eval':
                        self.model.set_adapter(self.teacher if name=='teacher_eval' else self.student)
                    self.model.train(name=='student_train')
                    for n,p,_ in params:p.requires_grad_(name=='student_train' and n in self.student_names)
                    if name=='tool_eval':stack.enter_context(self.model.disable_adapter())
                    if name!='student_train':
                        devices=[]
                        if self.device.type=='cuda':
                            if not torch.cuda.is_initialized():raise RuntimeError('This scope must not initialize CUDA')
                            devices=[self.device.index]
                        stack.enter_context(torch.random.fork_rng(devices=devices))
                        stack.enter_context(torch.no_grad())
                    # Eval is an execution-mode choice, not just no_grad. The
                    # snapshot restores EACH module flag, including mixed modes.
                    event.update(active_adapter=self._active_adapter(),
                                 adapters_enabled=self.model.get_model_status().enabled,
                                 module_training_flags=[bool(m.training) for _,m,_ in modules],
                                 trainable_parameter_names=[n for n,p,_ in params if p.requires_grad],
                                 grad_mode=bool(torch.is_grad_enabled()))
                    yield self.model
                event['status']='returned'
            except BaseException as exc:
                event.update(status='raised',error=type(exc).__name__+': '+str(exc));raise
            finally:
                # set_adapter itself changes requires_grad. Restore those flags
                # AFTER switching back, not before it. Do not normalize mixed
                # child training modes with a final recursive model.train().
                self.model.set_adapter(previous)
                for _,m,flag in modules:m.training=flag
                for _,p,flag in params:p.requires_grad_(flag)
                self.active=None
                event['restored_active_adapter']=self._active_adapter()
                event['mode_and_trainability_restored']=all(m.training==flag for _,m,flag in modules) and all(p.requires_grad==flag for _,p,flag in params)
                if not event['mode_and_trainability_restored']:raise RuntimeError('Role state restoration failed')
