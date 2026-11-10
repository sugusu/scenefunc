"""Explicit in-process replacement for newly constructed native VLMService.

Uses an already loaded model; never imports an API client or loads weights.
Only VLMService.create_completion is covered. Agents SDK, image generators,
retrieval servers and other clients still require separate runtime review.
Tool analysis runs with an experimental adapter disabled, identical across
policy arms. A local replacement is NOT equivalent to the original provider.
"""
import base64
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time
import types


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def require_reversible_adapter(model):
    """Reject merged/mixed adapter state instead of mutating shared weights.

    A reversible context alone is insufficient: PEFT can unmerge on forward
    when disabled. This gate qualifies state, not the identity of base weights.
    """
    if not callable(getattr(model,'disable_adapter',None)) or not callable(getattr(model,'get_model_status',None)):
        raise ValueError('Reversible adapter context and actual model status required')
    status=model.get_model_status()
    if status.enabled not in (True,False) or status.merged_adapters:
        raise ValueError('Merged or irregular adapter state cannot be a common tool base')
    if status.active_adapters=='irregular':
        raise ValueError('Inconsistent active adapters cannot be a common tool base')


def prepare_tool_messages(messages, response_format=None, *, max_images=8,
                          max_image_bytes=16*1024**2, max_image_pixels=12000000):
    """Convert embedded native image messages without network/file lookups.

    Preserve message and image order. Unknown content is rejected, not dropped.
    The caller closes returned PIL images after processor encoding.
    """
    from PIL import Image
    if not messages or not isinstance(messages,list):raise ValueError('Messages required')
    if response_format not in (None,{'type':'json_object'}):
        raise ValueError('Only native text or json_object responses supported')
    converted=[];images=[];image_records=[]
    try:
        for message in messages:
            if set(message)!={'role','content'} or message['role'] not in ('system','user','assistant'):
                raise ValueError('Unsupported native message envelope')
            content=message['content']
            if isinstance(content,str):
                converted.append(copy.deepcopy(message));continue
            if not isinstance(content,list):raise ValueError('Unsupported content type')
            parts=[]
            for part in content:
                if part.get('type')=='text' and set(part)=={'type','text'} and isinstance(part['text'],str):
                    parts.append(copy.deepcopy(part));continue
                if part.get('type')!='image_url' or set(part)!={'type','image_url'}:
                    raise ValueError('Unsupported content item; no silent loss')
                spec=part['image_url']
                if not isinstance(spec,dict) or not set(spec)<={'url','detail'}:
                    raise ValueError('Unsupported image specification')
                url=spec.get('url')
                if not isinstance(url,str) or ',' not in url:raise ValueError('Embedded data image required')
                header,data=url.split(',',1)
                if header not in ('data:image/png;base64','data:image/jpeg;base64','data:image/webp;base64'):
                    raise ValueError('External/file URLs and unrecognized image MIME are not accepted')
                if len(images)>=max_images or len(data)>4*((max_image_bytes+2)//3):
                    raise ValueError('Image budget exceeded')
                raw=base64.b64decode(data,validate=True)
                if len(raw)>max_image_bytes:raise ValueError('Image byte budget exceeded')
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width*im.height>max_image_pixels:raise ValueError('Image pixel budget exceeded')
                    expected={'PNG':'data:image/png;base64','JPEG':'data:image/jpeg;base64','WEBP':'data:image/webp;base64'}
                    if expected.get(im.format)!=header:raise ValueError('Declared image format differs from content')
                    rgb=im.convert('RGB');rgb.load()
                record={'slot':len(images),'sha256':hashlib.sha256(raw).hexdigest(),
                    'size':list(rgb.size),'requested_detail':spec.get('detail','auto')}
                image_records.append(record);images.append(rgb)
                parts.append({'type':'image','image':'embedded_image_'+str(record['slot'])})
            converted.append({'role':message['role'],'content':parts})
        if response_format:
            if converted[-1]['role']!='user':raise ValueError('JSON request must end with user input')
            suffix='\n\nReturn one valid JSON object only, without Markdown or explanatory text.'
            if isinstance(converted[-1]['content'],str):converted[-1]['content']+=suffix
            else:converted[-1]['content'].append({'type':'text','text':suffix})
        return converted,images,image_records
    except BaseException:
        for im in images:im.close()
        raise


def encode_tool_request(processor,messages,response_format,*,max_context_tokens):
    converted,images,records=prepare_tool_messages(messages,response_format)
    try:
        text=processor.apply_chat_template(converted,tokenize=False,add_generation_prompt=True)
        batch=processor(text=[text],images=images or None,return_tensors='pt',padding=False)
        count=int(batch['input_ids'].shape[1])
        if count>=max_context_tokens:raise ValueError('Tool input exceeds context; no silent truncation')
        return batch,{'input_tokens':count,'images':records,'converted_messages':converted,
                      'chat_template_text_sha256':hashlib.sha256(text.encode()).hexdigest()}
    finally:
        for im in images:im.close()


class LocalToolVLMBackend:
    def __init__(self,model,processor,*,ledger_path,effective_model_id,
                 allowed_requested_models,has_adapter=False,max_context_tokens=16384,
                 max_new_tokens=1024,max_seconds=120,max_calls=32,
                 total_output_tokens=8192,total_seconds=600):
        if model.training:raise ValueError('Tool model must be in eval mode')
        if not effective_model_id:raise ValueError('Explicit effective local model identity required')
        if not allowed_requested_models:raise ValueError('Explicit provider model replacement map required')
        if any(type(v)is not int or v<=0 for v in (max_context_tokens,max_new_tokens,max_calls,total_output_tokens)):
            raise ValueError('Positive integer token budgets required')
        if not math.isfinite(max_seconds) or max_seconds<=0:raise ValueError('Invalid time budget')
        if not math.isfinite(total_seconds) or total_seconds<=0:raise ValueError('Invalid cumulative time budget')
        if has_adapter:require_reversible_adapter(model)
        self.model=model;self.processor=processor;self.has_adapter=has_adapter
        self.effective_model_id=effective_model_id;self.allowed=set(allowed_requested_models)
        self.context=max_context_tokens;self.cap=max_new_tokens;self.seconds=max_seconds
        self.max_calls=max_calls;self.total_tokens=total_output_tokens;self.total_seconds=total_seconds
        self.lock=threading.RLock();self.rows=[];self.path=Path(ledger_path)
        if self.path.exists():raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True,exist_ok=True);self.save()

    def save(self):
        d={'effective_model_id':self.effective_model_id,'records':self.rows,
           'budget':{'model_calls':self.max_calls,'output_tokens':self.total_tokens,'serial_processing_seconds':self.total_seconds},
           'calls_started':sum(r['generation_started'] for r in self.rows),
           'known_input_tokens':sum(r.get('input_tokens',0) for r in self.rows),
           'known_output_tokens':sum(r.get('output_tokens') or 0 for r in self.rows),
           'output_accounting_complete':all(not r['generation_started'] or r.get('output_tokens') is not None for r in self.rows),
           'scope':'In-process native VLMService replacement only. Model quality and other tool clients are not certified. max_time is soft; outer process owns hard timeout.'}
        tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(self.path)

    def create_completion(self,*,model,messages,reasoning_effort,verbosity,
                          response_format=None,vision_detail='auto',service_tier=None):
        import torch
        queued=time.monotonic()
        with self.lock:
            previous=list(self.rows)
            row={'requested_model':model,'effective_model':self.effective_model_id,
                'status':'started','generation_started':False,'output_tokens':None,
                'request_sha256':hashlib.sha256(canonical(messages).encode()).hexdigest(),
                'response_format':response_format,'adapter_enabled':False,'queue_seconds':time.monotonic()-queued,
                'provider_options_not_emulated':{'reasoning_effort':reasoning_effort,
                    'verbosity':verbosity,'vision_detail':vision_detail,'service_tier':service_tier}}
            self.rows.append(row);tick=time.monotonic();self.save()
            try:
                if model not in self.allowed:raise ValueError('Undeclared provider-to-local model replacement')
                if any(r['generation_started'] and r.get('output_tokens') is None for r in previous):
                    raise RuntimeError('Previous tool generation has incomplete accounting; no further inference')
                if sum(r['generation_started'] for r in previous)>=self.max_calls:
                    raise RuntimeError('Cumulative tool model-call budget exhausted')
                tokens_left=self.total_tokens-sum(r.get('output_tokens') or 0 for r in previous)
                seconds_left=self.total_seconds-sum(r.get('wall_seconds',0) for r in previous)
                if tokens_left<=0 or seconds_left<=0:raise RuntimeError('Cumulative tool budget exhausted')
                if getattr(self.model,'_scene_generation_meter_active',False):
                    raise RuntimeError('Tool request overlapped policy generation')
                if self.has_adapter:require_reversible_adapter(self.model)
                if getattr(self.model.generation_config,'forced_eos_token_id',None) is not None:
                    raise ValueError('Forced EOS would hide incomplete tool output')
                batch,info=encode_tool_request(self.processor,messages,response_format,max_context_tokens=self.context)
                row.update(info);cap=min(self.cap,self.context-info['input_tokens'],tokens_left)
                batch=batch.to(self.model.device)
                seconds_left-=time.monotonic()-tick
                if seconds_left<=0:raise RuntimeError('Tool encoding exhausted cumulative time budget')
                scope=self.model.disable_adapter() if self.has_adapter else nullcontext()
                row['generation_started']=True;row['max_new_tokens']=cap;self.save()
                with scope,torch.inference_mode():
                    output=self.model.generate(**batch,max_new_tokens=cap,max_time=min(self.seconds,seconds_left),
                        do_sample=False,num_beams=1,num_return_sequences=1,return_dict_in_generate=True,
                        output_scores=False,output_logits=False)
                seq=output.sequences
                if seq.ndim!=2 or seq.shape[0]!=1 or seq.shape[1]<info['input_tokens']:
                    raise ValueError('Unexpected generation shape')
                if not torch.equal(seq[:,:info['input_tokens']],batch['input_ids']):
                    raise ValueError('Tool generation changed the input prefix')
                ids=seq[0,info['input_tokens']:].detach().cpu().tolist()
                row['output_tokens']=len(ids);row['output_ids']=ids
                if time.monotonic()-tick>self.total_seconds-sum(r.get('wall_seconds',0) for r in previous):
                    raise RuntimeError('Tool processing exceeded cumulative time budget')
                eos=self.model.generation_config.eos_token_id
                eos=eos if isinstance(eos,list) else [eos]
                if not ids or ids[-1] not in eos or len(ids)>cap:
                    raise ValueError('Incomplete or over-budget tool response; no free repair')
                text=self.processor.tokenizer.decode(ids,skip_special_tokens=True)
                row['response_text']=text
                if not text.strip():raise ValueError('Empty tool response')
                if response_format:
                    from phase_policy import strict_json
                    if not isinstance(strict_json(text),dict):raise ValueError('Tool response is not a JSON object')
                row['status']='returned'
                return text
            except BaseException as exc:
                row.update(status='raised',error=type(exc).__name__+': '+str(exc));raise
            finally:
                row['wall_seconds']=time.monotonic()-tick;self.save()


class LocalToolVLMBinding:
    """Defer binding until native main constructs its ONE shared policy model.

    New service instances receive an instance-local completion method. They
    remain unusable after scope exit rather than reverting to remote requests.
    Pre-existing service objects and unrelated SDK clients are NOT intercepted.
    """
    def __init__(self,vlm_class,**settings):
        self.vlm_class=vlm_class;self.settings=settings;self.backend=None;self.active=False

    def bind(self,model,processor,has_adapter):
        if not self.active or self.backend is not None:raise RuntimeError('Unexpected local tool model binding')
        self.backend=LocalToolVLMBackend(model,processor,has_adapter=has_adapter,**self.settings)

    @contextmanager
    def scope(self):
        if self.active:raise RuntimeError('Nested local tool service scope')
        cls=self.vlm_class;original=cls.__init__;self.active=True
        binding=self
        def init(instance,service_tier=None):
            instance.service_tier=service_tier
            def completion(obj,model,messages,reasoning_effort,verbosity,response_format=None,vision_detail='auto'):
                if not binding.active:raise RuntimeError('Local tool session already closed')
                if binding.backend is None:raise RuntimeError('Tool requested before shared model is ready')
                return binding.backend.create_completion(model=model,messages=messages,
                    reasoning_effort=reasoning_effort,verbosity=verbosity,
                    response_format=response_format,vision_detail=vision_detail,service_tier=obj.service_tier)
            instance.create_completion=types.MethodType(completion,instance)
        cls.__init__=init
        try:yield self
        finally:
            cls.__init__=original;self.active=False
            if self.backend is not None:self.backend.save()
            self.backend=None  # release the shared weight reference on close
