"""Already-loaded-model backend for the common native phase/batch contract.

No model/service constructor or training path. Merely importing this module
does not import Torch or initialize CUDA. The caller supplies model, processor,
normalizer, serialization lock and separately admitted generation limits.
"""
import copy
import hashlib
import time


def encode_native_phase_inputs(processor,normalize_messages,messages,*,assistant_prefix,max_context_tokens):
    """Shared CPU-testable path, also used by the already-loaded-model backend.

    No model access, truncation, device transfer, cached logits or target data.
    The normalizer preserves native image order and reports original byte hashes.
    """
    normalized,images,image_evidence=normalize_messages(messages)
    text=processor.apply_chat_template(normalized,tokenize=False,add_generation_prompt=True)+assistant_prefix
    inputs=processor(text=[text],images=images or None,return_tensors='pt')
    count=int(inputs['input_ids'].shape[1])
    if type(max_context_tokens) is not int or max_context_tokens<=0 or count>=max_context_tokens:
        raise ValueError('No combined model context capacity; no prompt truncation')
    return inputs,dict(input_tokens=count,images=copy.deepcopy(image_evidence),
                       prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       image_decode_and_processor_only=True)


class NativePhaseTorchBackend:
    execution_kind='actual_model_generation'

    def __init__(self,model,processor,*,normalize_messages,lock,max_context_tokens,
                 seed,sampling=None,max_model_generate_requests,max_output_tokens):
        if model.training:
            raise ValueError('Caller must supply an already loaded evaluation model')
        config=model.generation_config
        if (getattr(config,'forced_eos_token_id',None) is not None or
                getattr(config,'num_beams',1)!=1 or getattr(config,'num_return_sequences',1)!=1):
            raise ValueError('Forced EOS, beams or multiple sequences change the contract')
        limits=(max_context_tokens,max_model_generate_requests,max_output_tokens)
        if any(type(v) is not int or v<=0 for v in limits):
            raise ValueError('Explicit finite cumulative/model context limits required')
        self.model=model;self.processor=processor;self.tokenizer=processor.tokenizer
        self.normalize_messages=normalize_messages;self.lock=lock
        self.context_cap=max_context_tokens;self.call_cap=max_model_generate_requests
        self.token_cap=max_output_tokens;self.calls=0;self.output_tokens=0;self.seed=seed
        self.sampling=dict(sampling or {'do_sample':False})
        if set(self.sampling)-{'do_sample','temperature','top_p','top_k'}:
            raise ValueError('Uncontrolled generation option')
        eos=config.eos_token_id
        self.eos_ids=eos if isinstance(eos,list) else [eos]
        if not self.eos_ids or any(type(i) is not int or i<0 for i in self.eos_ids):
            raise ValueError('Original model EOS IDs required')
        self.close_id=self.tokenizer.convert_tokens_to_ids('</think>')
        if self.tokenizer.encode('</think>',add_special_tokens=False)!=[self.close_id]:
            raise ValueError('Decision stop must be one native token')
        self.last_attempt=None;self.closed_after_failure=False

    def generate(self,messages,*,assistant_prefix,phase,token_limit,seconds_limit):
        import torch
        tick=time.monotonic()
        with self.lock:
            self.last_attempt=dict(status='started',phase=phase,generation_started=False,
                                   cumulative_calls_before=self.calls,output_tokens_before=self.output_tokens)
            try:
                if self.closed_after_failure or self.calls>=self.call_cap:
                    raise RuntimeError('Explicit cumulative model budget closed')
                if phase not in ('joint','decision','action') or type(token_limit) is not int or token_limit<=0:
                    raise ValueError('Invalid phase or finite generation cap')
                if not 0<seconds_limit<float('inf'):
                    raise ValueError('Finite per-request soft time limit required')
                inputs,encoded=encode_native_phase_inputs(self.processor,self.normalize_messages,
                    messages,assistant_prefix=assistant_prefix,max_context_tokens=self.context_cap)
                input_count=encoded['input_tokens']
                cap=min(token_limit,self.context_cap-input_count,self.token_cap-self.output_tokens)
                if cap<1:
                    raise ValueError('No context/output capacity; no truncation')
                device=self.model.device
                inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
                stop=list(dict.fromkeys([self.close_id]+self.eos_ids)) if phase=='decision' else self.eos_ids
                self.calls+=1;self.last_attempt['generation_started']=True
                cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
                # Local inference seeding must not change the geometry/tool RNG
                # state owned by another component in this same process.
                with torch.random.fork_rng(devices=cuda_devices):
                    # torch.manual_seed also seeds ALL visible CUDA generators;
                    # the fork above deliberately saves only this model device.
                    # Seed only the generators whose state is actually restored.
                    torch.random.default_generator.manual_seed(self.seed+self.calls-1)
                    if cuda_devices:
                        with torch.cuda.device(cuda_devices[0]):
                            torch.cuda.manual_seed(self.seed+self.calls-1)
                    with torch.inference_mode():
                        output=self.model.generate(**inputs,max_new_tokens=cap,max_time=seconds_limit,
                            eos_token_id=stop,pad_token_id=self.tokenizer.pad_token_id
                            if self.tokenizer.pad_token_id is not None else self.eos_ids[0],
                            use_cache=True,return_dict_in_generate=False,**self.sampling)
                if output.ndim!=2 or output.shape[0]!=1 or not torch.equal(output[:,:input_count],inputs['input_ids']):
                    raise ValueError('Unexpected sequence shape or changed native prefix')
                ids=output[0,input_count:].detach().cpu().tolist()
                self.output_tokens+=len(ids)
                if len(ids)>cap:
                    raise ValueError('Model exceeded admitted output cap')
                self.last_attempt.update(status='returned',input_tokens=input_count,
                                         output_tokens=len(ids),output_ids=ids)
                return dict(output_ids=ids,input_tokens=input_count,
                            execution_kind=self.execution_kind,native_stop_ids=stop,
                            output_token_cap=cap,images=encoded['images'],
                            prompt_sha256=encoded['prompt_sha256'],
                            cumulative_model_generate_requests=self.calls,
                            cumulative_output_tokens=self.output_tokens,
                            time_limit_is_soft=True,seconds=time.monotonic()-tick)
            except BaseException as exc:
                self.closed_after_failure=True
                self.last_attempt.update(status='failed_no_retry',error=type(exc).__name__+': '+str(exc),
                                         seconds=time.monotonic()-tick)
                raise
