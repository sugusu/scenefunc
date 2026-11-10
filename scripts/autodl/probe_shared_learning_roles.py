"""Actual two-adapter PEFT CPU role probe, not a learned scene experiment.

Explicit diagnostic tensors, small linear forward/autograd, zero optimizer
updates or pretrained/VLM calls. Includes the pinned original HinT-SD adapter
context body without importing its trainer/API/vLLM modules.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace


def digest(data):return hashlib.sha256(data).hexdigest()


def main(repo,package,hint_source,out):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=False);result={}
    sys.path.insert(0,str(package))
    import torch
    import peft
    from torch import nn
    from peft import LoraConfig,get_peft_model
    from shared_learning_roles import SharedLearningRoles,require_student_only_optimizer
    from local_tool_vlm import require_reversible_adapter
    assert peft.__version__=='0.17.1' and not torch.cuda.is_initialized()
    torch.set_num_threads(1);torch.manual_seed(0)
    forwards=0
    class Probe(nn.Module):
        def __init__(self,bias=False):
            super().__init__();self.config={'model_type':'custom'};self.proj=nn.Linear(8,8,bias=bias)
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(8))
                if self.proj.bias is not None:self.proj.bias.zero_()
        def forward(self,x):
            nonlocal forwards
            forwards+=1;return self.proj(x)
    x=torch.arange(1.,9.).reshape(1,8)
    base=Probe().eval()
    with torch.no_grad():base_reference=base(x).clone()
    cfg=lambda:LoraConfig(r=2,lora_alpha=2,target_modules=['proj'],lora_dropout=.05,bias='none')
    model=get_peft_model(base,cfg(),adapter_name='student');model.add_adapter('teacher',cfg())
    model.set_adapter('student')
    with torch.no_grad():
        for n,p in model.named_parameters():
            if '.student.' in n:p.fill_(.125 if 'lora_B' in n else .25)
            elif '.teacher.' in n:p.fill_(.25 if 'lora_B' in n else .375)
            p.requires_grad_('.student.' in n)
    model.train()
    # A mixed child mode must survive every normal/exceptional role exit.
    for n,m in model.named_modules():
        if n.endswith('lora_dropout.student'):m.eval()
    def state():return dict(active=model.get_model_status().active_adapters,
        enabled=model.get_model_status().enabled,
        modes={n:m.training for n,m in model.named_modules()},
        requires_grad={n:p.requires_grad for n,p in model.named_parameters()})
    def weights():return {n:digest(p.detach().cpu().numpy().tobytes()) for n,p in model.named_parameters()}
    original_state=state();original_weights=weights()
    # Execute the exact original context manager; not its trainer or models.
    tree=ast.parse(hint_source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Trainer')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_use_adapter')
    ns=dict(contextmanager=contextmanager)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(hint_source),'exec'),ns)
    owner=SimpleNamespace(_unwrap_adapter_model=lambda:model,
                          _get_active_adapter_name=lambda m:m.get_model_status().active_adapters[0])
    source_outputs=[]
    for seed in range(32):
        torch.manual_seed(seed)
        with ns['_use_adapter'](owner,'teacher'),torch.no_grad():
            source_outputs.append(model(x).clone())
    source_unique=len({digest(v.numpy().tobytes()) for v in source_outputs})
    assert state()==original_state and source_unique>1
    roles=SharedLearningRoles(model,lock=threading.RLock())
    teacher_outputs=[]
    for seed in range(32):
        torch.manual_seed(seed);before_rng=torch.get_rng_state().clone()
        with roles.role('teacher_eval'):
            assert not torch.is_grad_enabled() and not any(m.training for m in model.modules())
            assert not any(p.requires_grad for p in model.parameters())
            teacher_outputs.append(model(x).clone())
        assert state()==original_state and torch.equal(before_rng,torch.get_rng_state())
    controlled_unique=len({digest(v.numpy().tobytes()) for v in teacher_outputs})
    assert controlled_unique==1
    with roles.role('tool_eval'):
        tool_output=model(x).clone();assert torch.equal(tool_output,base_reference)
        assert model.get_model_status().enabled is False
    assert state()==original_state
    torch.manual_seed(37);before_rng=torch.get_rng_state().clone()
    with roles.role('policy_eval'):
        policy_output=model(x).clone();torch.rand(7)
    assert state()==original_state and torch.equal(before_rng,torch.get_rng_state())
    student_params=tuple(p for n,p in model.named_parameters() if n in roles.student_names)
    optimizer=torch.optim.SGD(student_params,lr=.01)
    optimizer_receipt=require_student_only_optimizer(optimizer,roles)
    teacher_param=next(p for n,p in model.named_parameters() if n in roles.teacher_names)
    optimizer.add_param_group(dict(params=[teacher_param]))
    try:require_student_only_optimizer(optimizer,roles)
    except ValueError:pass
    else:raise AssertionError('Teacher entered the student optimizer')
    optimizer.param_groups.pop()
    original_group_params=optimizer.param_groups[0]['params']
    optimizer.param_groups[0]['params']=list(student_params[:1])
    try:require_student_only_optimizer(optimizer,roles)
    except ValueError:pass
    else:raise AssertionError('Incomplete student optimizer accepted')
    optimizer.param_groups[0]['params']=original_group_params
    require_student_only_optimizer(optimizer,roles)
    with roles.role('student_train'):
        y=model(x).sum()
        expected=torch.autograd.grad(y,student_params,retain_graph=True)
    assert state()==original_state
    # Real mode/adapter switches between forward and the retained graph's
    # second gradient. No parameter step is performed, and all weights stay fixed.
    with roles.role('teacher_eval'):model(x)
    with roles.role('tool_eval'):model(x)
    with roles.role('student_train'):actual=torch.autograd.grad(y,student_params)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected)) and any(g.abs().sum()>0 for g in actual)
    assert state()==original_state and weights()==original_weights
    for role in ('teacher_eval','tool_eval','student_train','policy_eval'):
        try:
            with roles.role(role):raise RuntimeError('deliberate role exit probe')
        except RuntimeError as exc:assert str(exc)=='deliberate role exit probe'
        assert state()==original_state
    # Known unsupported counterexample: adapter-disabled is not an ORIGINAL
    # base if a shared bias was adapted. Explicit assignment, not optimization.
    bad_base=Probe(bias=True).eval()
    with torch.no_grad():before_bias=bad_base(x).clone()
    bad=get_peft_model(bad_base,LoraConfig(r=2,lora_alpha=2,target_modules=['proj'],bias='all'))
    with torch.no_grad():
        for n,p in bad.named_parameters():
            if n.endswith('base_layer.bias'):p.fill_(.5)
    require_reversible_adapter(bad)
    with bad.disable_adapter(),torch.no_grad():after_bias=bad(x).clone()
    assert not torch.equal(before_bias,after_bias)
    try:SharedLearningRoles(bad,lock=threading.RLock())
    except ValueError as exc:
        rejected_reason=str(exc);assert 'shared bias' in rejected_reason
    else:raise AssertionError('Adapted shared-bias configuration admitted')
    assert not torch.cuda.is_initialized() and weights()==original_weights
    package_root=Path(peft.__file__).parent
    sources=[package_root/'peft_model.py',package_root/'tuners/tuners_utils.py',package_root/'tuners/lora/layer.py',hint_source]
    result=dict(status='real_PEFT_two_adapter_mode_and_gradient_probe_completed_CPU_only',
        peft=peft.__version__,torch=torch.__version__,tiny_network='8x8 linear; rank2 LoRA student+teacher',
        dropout_probability=.05,source_context_repetitions=32,
        source_context_teacher_distinct_outputs=source_unique,
        source_context_scope='Original HinT-SD _use_adapter AST with no_grad on this deliberately train-mode CPU fixture; not author checkpoint reproduction',
        controlled_teacher_distinct_outputs=controlled_unique,controlled_teacher_no_grad_and_eval=True,
        all_role_mode_trainability_and_active_adapter_states_restored=True,
        frozen_tool_output_exactly_equals_unadapted_fixture_base=True,
        teacher_and_policy_role_CPU_rng_restored=True,
        retained_student_graph_gradient_unchanged_by_intervening_role_switches=True,
        student_gradient_norms=[float(g.norm()) for g in actual],autograd_gradient_evaluations=2,
        optimizer_parameter_ownership=optimizer_receipt,
        real_optimizer_instance_created_without_step=True,
        teacher_contamination_and_missing_student_parameters_rejected=True,
        actual_small_linear_forward_calls=forwards,all_main_fixture_parameter_hashes_unchanged=True,
        explicit_diagnostic_tensor_assignments_not_training=True,
        shared_bias_counterexample=dict(old_state_only_gate_accepts=True,base_changed_when_adapter_disabled=True,new_guard_rejects=rejected_reason),
        source_hashes=[dict(path=str(p),sha256=digest(p.read_bytes())) for p in sources],
        original_hint_method_lines=[node.lineno,node.end_lineno],
        VLM_forwards=0,model_generate_requests=0,pretrained_weights_loaded=False,
        optimizer_updates=0,paid_API_calls=0,new_native_SDK_calls=0,new_complete_scenes=0,
        actual_8B_forward_or_memory_qualification=False,CUDA_initialized=False,
        role_controller_sha256=digest((repo/'scripts/autodl/shared_learning_roles.py').read_bytes()),
        wall_seconds=time.monotonic()-tick)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'role_events.json').write_text(json.dumps(roles.events,indent=2)+'\n')
    (out/'main_fixture_parameter_hashes.json').write_text(json.dumps(original_weights,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--hint-source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.package.resolve(),a.hint_source.resolve(),a.out.resolve())
