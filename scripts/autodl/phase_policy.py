"""Local two-phase policy adapter, without a model loader, API or tool executor.

An explicitly supplied model runs real generate calls. The native host retains
scene observation, action validation, dispatch and completion checks. All arms
use the same adapter; no saved target is accepted by the public policy API.
"""
import copy
import json
import math
import re
import time

from phase_deployment import deployment_input, decision_stop_ids
from routed_multimodal import frame_completed_decision, tensor_digest


def strict_json(text):
    def pairs(values):
        out={}
        for key,value in values:
            if key in out:raise ValueError('Duplicate JSON key: '+key)
            out[key]=value
        return out
    def invalid(value):raise ValueError('Non-finite JSON: '+value)
    value=json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)
    def check(x):
        if isinstance(x,float) and not math.isfinite(x):raise ValueError('Non-finite JSON number')
        if isinstance(x,dict):
            for child in x.values():check(child)
        elif isinstance(x,list):
            for child in x:check(child)
    check(value)
    return value


def classify_stage(tokenizer, ids, *, stage, eos_ids):
    """Use the actual generated-token boundary, not regex absence, to stop.

    EOS-only is the NEW action-stage STOP target. The legacy classifier's
    nonempty completion rule remains unchanged in the old experiments.
    """
    if stage not in ('decision','action'):raise ValueError('Unknown phase')
    if not ids or any(not isinstance(t,int) for t in ids):
        return {'kind':'invalid','reason':'Empty or noninteger generated sequence'}
    raw=tokenizer.decode(ids,skip_special_tokens=False)
    if stage=='decision':
        close=tokenizer.convert_tokens_to_ids('</think>')
        if ids[-1]!=close or any(t in set(eos_ids)|{close} for t in ids[:-1]):
            return {'kind':'invalid','reason':'Decision did not end at its first closing delimiter','raw':raw}
        if '<|' in raw:
            return {'kind':'invalid','reason':'Role/control token inside decision proposal','raw':raw}
        try:prefix=frame_completed_decision(raw)
        except ValueError as exc:return {'kind':'invalid','reason':str(exc),'raw':raw}
        return {'kind':'decision','raw':raw,'framed_prefix':prefix,'semantics_verified':False}
    if ids[-1] not in eos_ids or any(t in eos_ids for t in ids[:-1]):
        return {'kind':'invalid','reason':'Action lacks a single verified native EOS boundary','raw':raw}
    text=tokenizer.decode(ids[:-1],skip_special_tokens=False).strip()
    if not text:
        return {'kind':'stop_proposed','raw':raw,'full_scene_success':None}
    if any(tag in text for tag in ('<think','</think','<|','<done','</done')):
        return {'kind':'invalid','reason':'Unexpected control or decision text in action phase','raw':raw}
    match=re.fullmatch(r'<tool_call>(.*?)</tool_call>',text,re.DOTALL)
    if not match or text.count('<tool_call>')!=1 or text.count('</tool_call>')!=1:
        return {'kind':'invalid','reason':'Expected one complete tool block or EOS-only STOP','raw':raw}
    try:call=strict_json(match.group(1))
    except (ValueError,TypeError) as exc:return {'kind':'invalid','reason':str(exc),'raw':raw}
    if (not isinstance(call,dict) or set(call)!={'name','arguments'} or
            not isinstance(call['name'],str) or not isinstance(call['arguments'],dict)):
        return {'kind':'invalid','reason':'Malformed tool envelope','raw':raw}
    return {'kind':'tool_proposed','raw':raw,'call':call,'native_validation_required':True}


class TorchPhaseBackend:
    """Uses an ALREADY loaded inference model and the validated shared encoder.

    Construction never loads weights. Outer host owns GPU authorization/locks,
    hard subprocess timeout and exact adapter/dtype validation. Here max_time
    is only Transformers' soft generation limit, not an interrupt guarantee.
    """
    def __init__(self,model,processor,*,max_context_tokens=16384,sampling=None):
        if model.training:raise ValueError('Model must already be in evaluation mode')
        if getattr(model.generation_config,'forced_eos_token_id',None) is not None:
            raise ValueError('Forced EOS would confound a model-proposed STOP')
        if (getattr(model.generation_config,'num_beams',1)!=1 or
                getattr(model.generation_config,'num_return_sequences',1)!=1):
            raise ValueError('This shared budget supports one generated sequence only')
        self.model=model;self.processor=processor;self.max_context_tokens=max_context_tokens
        self.sampling=dict(sampling or {'do_sample':False})
        allowed={'do_sample','temperature','top_p','top_k'}
        if set(self.sampling)-allowed:raise ValueError('Uncontrolled generation configuration')
        eos=model.generation_config.eos_token_id
        self.eos_ids=list(eos) if isinstance(eos,list) else [eos]
        if not self.eos_ids or any(not isinstance(i,int) or i<0 for i in self.eos_ids):
            raise ValueError('Native generation EOS IDs required')

    def generate(self,messages,images,*,arm,stage,decision,token_limit,seconds_limit,seed):
        import torch
        batch=deployment_input(self.processor,messages,images,arm=arm,stage=stage,
                               generated_decision=decision,max_length=self.max_context_tokens)
        input_count=int(batch['input_ids'].shape[1])
        cap=min(token_limit,self.max_context_tokens-input_count)
        if cap<1:raise ValueError('No context capacity remains for generation; no truncation')
        if any(k in batch for k in ('labels','token_objective_weights','action_mask','decision_mask')):
            raise ValueError('Training-only fields reached inference')
        hashes={k:tensor_digest(v) for k,v in batch.items() if isinstance(v,torch.Tensor)}
        stop=list(dict.fromkeys(decision_stop_ids(self.processor.tokenizer)+self.eos_ids)) if stage=='decision' else self.eos_ids
        device=self.model.device
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        torch.manual_seed(seed)
        pad=self.processor.tokenizer.pad_token_id
        with torch.inference_mode():
            output=self.model.generate(**batch,max_new_tokens=cap,max_time=seconds_limit,
                eos_token_id=stop,pad_token_id=pad if pad is not None else self.eos_ids[0],
                use_cache=True,return_dict_in_generate=False,**self.sampling)
        if output.shape[0]!=1 or not torch.equal(output[:,:input_count],batch['input_ids']):
            raise ValueError('Unexpected generation shape or changed prompt')
        ids=output[0,input_count:].detach().cpu().tolist()
        if len(ids)>cap:raise ValueError('Backend exceeded its generated-token cap')
        return {'output_ids':ids,'input_tokens':input_count,'input_tensor_sha256':hashes,
                'allowed_output_tokens':cap,'generation_eos_ids':stop,'seed':seed,
                'sampling':self.sampling,'cache_scope':'fresh per generate call; no D-stage cache passed to A'}


class PhaseOutcomeError(RuntimeError):
    """A failed policy attempt with evidence; MUST NOT be counted as STOP."""
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__(receipt['status'])


class PhasePolicy:
    """One native-style generate(obs) call, sharing the D+A cumulative budget.

    obs must contain actual messages, decoded images, and a native state hash.
    The host must catch PhaseOutcomeError as a failed attempt, not success.
    Result has no dispatch privilege. D is kept in the audit receipt, NEVER
    spliced into the host's action log or treated as an execution receipt.
    """
    def __init__(self,backend,*,arm,total_output_tokens,decision_output_tokens,
                 action_output_tokens,total_seconds,seed=0):
        if arm not in ('paired_phase_v3','separate_phase_v3','separate_direct_v3'):
            raise ValueError('Unknown prospective deployment condition')
        limits=(total_output_tokens,decision_output_tokens,action_output_tokens,total_seconds)
        if any(v<=0 for v in limits):raise ValueError('Positive finite budgets required')
        if any(not math.isfinite(v) for v in limits):raise ValueError('Finite budgets required')
        if any(not isinstance(v,int) for v in limits[:3]):raise ValueError('Integer token budgets required')
        self.backend=backend;self.arm=arm;self.total_tokens=total_output_tokens
        self.d_cap=decision_output_tokens;self.a_cap=action_output_tokens
        self.seconds=total_seconds;self.seed=seed

    def generate(self,obs):
        if not isinstance(obs.get('state_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',obs['state_sha256']):
            raise ValueError('A current native state digest is required')
        messages=copy.deepcopy(obs['messages']);images=[im.copy() for im in obs['images']]
        if len(images)!=5:raise ValueError('Exactly five actual views required')
        start=time.monotonic();used=0;draft=None;last=None
        result={'arm':self.arm,'state_sha256':obs['state_sha256'],'stages':[],
                'generation_attempts':0,'output_tokens':0,'input_tokens':0,'tool_calls':[],
                'raw':'','think':None,'full_scene_success':None,'paid_api_calls':0}
        def fail(status):
            result['status']=status
            raise PhaseOutcomeError(result)
        phases=['action'] if self.arm=='separate_direct_v3' else ['decision','action']
        effective_arm='separate_phase_v3' if self.arm=='separate_direct_v3' else self.arm
        try:
            for stage in phases:
                seconds=self.seconds-(time.monotonic()-start)
                cap=min(self.total_tokens-used,self.d_cap if stage=='decision' else self.a_cap)
                if seconds<=0 or cap<1:
                    fail('budget_exhausted_before_'+stage)
                visible_draft=draft if stage=='action' and self.arm=='paired_phase_v3' else None
                receipt={'stage':stage,'decision_visible_to_action':visible_draft is not None,
                         'output_budget':cap,'status':'started'}
                result['stages'].append(receipt);result['generation_attempts']+=1
                tick=time.monotonic()
                try:
                    generation=self.backend.generate(messages,images,arm=effective_arm,stage=stage,
                        decision=visible_draft,token_limit=cap,seconds_limit=seconds,
                        seed=self.seed+(0 if stage=='decision' else 1))
                finally:receipt['elapsed_seconds']=time.monotonic()-tick
                receipt.update(generation);ids=generation['output_ids'];used+=len(ids)
                result['output_tokens']=used;result['input_tokens']+=generation['input_tokens']
                if len(ids)>cap:fail('backend_exceeded_output_budget')
                if time.monotonic()-start>self.seconds:fail('time_budget_exhausted_after_'+stage)
                classified=classify_stage(self.backend.processor.tokenizer,ids,stage=stage,eos_ids=self.backend.eos_ids)
                receipt['classification']=classified;receipt['status']='returned'
                if classified['kind']=='invalid':
                    fail('invalid_'+stage)
                if stage=='decision':draft=classified['raw']
                else:last=classified
            result['raw']=last['raw'];result['status']=last['kind']
            if last['kind']=='tool_proposed':result['tool_calls']=[last['call']]
            result['host_contract']='Validate actual state/schema before dispatch; EOS proposes termination, never room success.'
            return result
        except PhaseOutcomeError:
            raise
        except Exception as exc:
            result.update(status='backend_error',error=type(exc).__name__+': '+str(exc),
                          output_token_accounting_complete=False)
            raise PhaseOutcomeError(result) from exc
        finally:
            result['elapsed_seconds']=time.monotonic()-start
            result['generated_decision_for_audit_only']=draft
