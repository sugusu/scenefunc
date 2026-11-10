"""Actual CPU image/token encoding plus explicit model/service integration doubles.

The five images are retained native rendered observations. All generated output
is scripted, not inference; no weights, physical labels or scene results arise.
"""
import argparse
import ast
import base64
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import resource
import time
from types import SimpleNamespace
from typing import Any

from full_generation_bridge import GenerationLedger,shared_model_driver
from local_tool_vlm import LocalToolVLMBinding,encode_tool_request,prepare_tool_messages
from phase_host import StageBudget
from routed_multimodal import tensor_digest


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main(a):
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(str(a.processor),local_files_only=True)
    spec=importlib.util.spec_from_file_location('native_image_budget',a.observation_source)
    obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)
    obs.configure_image_budget(processor)
    bindings=json.loads(a.bindings.read_text());source=bindings[0]
    images=source['images'];assert len(images)==5
    before={str(p):sha(p) for p in [a.bindings,a.observation_source,a.vlm_source]}
    content=[{'type':'text','text':'Describe the visible furniture using a JSON object. This is an input-encoding check.'}]
    for r in images:
        p=Path(r['path']);assert sha(p)==r['sha256'];before[str(p)]=sha(p)
        content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()}})
    request=[{'role':'user','content':content}]
    batch,record=encode_tool_request(processor,request,{'type':'json_object'},max_context_tokens=16384)
    # Independent direct-file image input must equal embedded-byte conversion.
    converted,decoded,_=prepare_tool_messages(request,{'type':'json_object'})
    for im in decoded:im.close()
    pil=[]
    try:
        for r in images:
            with Image.open(r['path']) as im:pil.append(im.convert('RGB'))
        text=processor.apply_chat_template(converted,tokenize=False,add_generation_prompt=True)
        reference=processor(text=[text],images=pil,return_tensors='pt',padding=False)
        for key in ('input_ids','pixel_values','image_grid_thw'):
            assert torch.equal(batch[key],reference[key]),key
    finally:
        for im in pil:im.close()
    encoding={'source_unit':source['unit_id'],'source_state_sha256':source['source_state_sha256'],
        'historical_pixel_equivalence':source['historical_pixel_equivalence'],
        'native_view_files':images,'input_tokens':record['input_tokens'],
        'tensor_hashes':{k:tensor_digest(batch[k]) for k in ('input_ids','pixel_values','image_grid_thw')},
        'embedded_matches_direct_file_encoding':True}
    write(a.out/'encoding.json',encoding)
    del batch,reference

    # Compile the unchanged upstream class, not its OpenAI import/client.
    tree=ast.parse(a.vlm_source.read_text())
    cls=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VLMService'];assert len(cls)==1
    constructors=[]
    def forbidden_client():
        constructors.append('attempted');raise AssertionError('Remote client construction prohibited in this check')
    ns={'Any':Any,'OpenAI':forbidden_client,'console_logger':logging.getLogger('native_vlm_fixture')}
    exec(compile(ast.Module(body=cls,type_ignores=[]),str(a.vlm_source),'exec'),ns)
    VLM=ns['VLMService'];original_init=VLM.__init__
    created=[];calls=[]
    class ScriptedModel:
        training=False;device=torch.device('cpu')
        generation_config=SimpleNamespace(eos_token_id=processor.tokenizer.eos_token_id,forced_eos_token_id=None)
        def __init__(self):self.adapter_enabled=True;self.text='{"fixture_only": true}'
        def get_model_status(self):
            return SimpleNamespace(enabled=self.adapter_enabled,merged_adapters=[],active_adapters=['default'])
        @contextmanager
        def disable_adapter(self):
            prior=self.adapter_enabled;self.adapter_enabled=False
            try:yield
            finally:self.adapter_enabled=prior
        def generate(self,**kw):
            assert self.adapter_enabled is False
            calls.append({'input_tokens':int(kw['input_ids'].shape[1]),'fixture_only':True})
            ids=processor.tokenizer.encode(self.text,add_special_tokens=False)+[processor.tokenizer.eos_token_id]
            assert len(ids)<=kw['max_new_tokens']
            return SimpleNamespace(sequences=torch.cat((kw['input_ids'],torch.tensor([ids])),dim=1))
    class Policy:
        def __init__(self):self.model=ScriptedModel();self.processor=processor;created.append(self)
    async def driver(**kw):raise AssertionError('No native stage executes in this binding check')
    def session(*args,**kw):raise AssertionError('No service starts in this binding check')
    native=SimpleNamespace(ModelPolicy=Policy,driver_loop=driver,RolloutSession=session)
    binding=LocalToolVLMBinding(VLM,ledger_path=a.out/'fixture_tool_calls.json',
        effective_model_id='SCRIPTED_OUTPUT_WITH_REAL_PROCESSOR_NOT_LANGUAGE_INFERENCE',
        allowed_requested_models=['gpt-5.2'],max_new_tokens=128,max_seconds=60,max_calls=3)
    kwargs=dict(model='gpt-5.2',reasoning_effort='medium',verbosity='low',response_format={'type':'json_object'})
    checks=[]
    with shared_model_driver(native,phase_settings={'arm':'separate_direct_v3'},
            stage_budget=StageBudget(1,1,1,128,60),allowed_tools=('move_furniture_tool',),
            load_images=lambda _:[],acceptance=lambda _:None,evidence_format='canonical_v2',
            ledger=GenerationLedger(a.out/'fixture_policy_calls.json'),
            adapter_loader=lambda model:model,tool_vlm_binding=binding):
        service=VLM(service_tier='default')
        try:service.create_completion(messages=[{'role':'user','content':'JSON please'}],**kwargs)
        except RuntimeError as e:assert 'before shared model' in str(e);checks.append('early_call_rejected_without_remote_client')
        else:raise AssertionError('Unbound service accepted')
        policy=native.ModelPolicy()
        assert binding.backend.model is policy.model and len(created)==1
        assert json.loads(service.create_completion(messages=request,**kwargs))=={'fixture_only':True}
        assert json.loads(service.create_completion(messages=[{'role':'user','content':'Return a JSON object.'}],**kwargs))=={'fixture_only':True}
        checks.append('one_model_shared_by_policy_and_tool_with_experiment_adapter_disabled')
        assert policy.model.adapter_enabled is True
        count=len(calls)
        bad=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'https://invalid.example/image.png'}}]}]
        try:service.create_completion(messages=bad,**kwargs)
        except ValueError:checks.append('external_image_rejected_before_generation')
        else:raise AssertionError('External image accepted')
        assert len(calls)==count
        try:service.create_completion(messages=[{'role':'user','content':'JSON'}],**dict(kwargs,model='undeclared'))
        except ValueError:checks.append('undeclared_model_substitution_rejected')
        else:raise AssertionError('Unmapped model accepted')
        assert len(calls)==count
        policy.model.text='{"incomplete":'
        try:service.create_completion(messages=[{'role':'user','content':'JSON'}],**kwargs)
        except ValueError:checks.append('malformed_scripted_output_counted_without_repair')
        else:raise AssertionError('Invalid JSON accepted')
        assert policy.model.adapter_enabled is True
        count=len(calls)
        try:service.create_completion(messages=[{'role':'user','content':'JSON'}],**kwargs)
        except RuntimeError as e:assert 'model-call budget' in str(e);checks.append('failed_outputs_consume_cumulative_call_budget')
        else:raise AssertionError('Budget exceeded')
        assert len(calls)==count
    assert VLM.__init__ is original_init and native.ModelPolicy is Policy
    assert binding.backend is None
    try:service.create_completion(messages=[{'role':'user','content':'JSON'}],**kwargs)
    except RuntimeError as e:assert 'already closed' in str(e);checks.append('closed_instance_cannot_fall_back_to_remote')
    else:raise AssertionError('Closed service accepted')
    assert not constructors
    tool=json.loads((a.out/'fixture_tool_calls.json').read_text())
    assert tool['calls_started']==len(calls)==3 and tool['output_accounting_complete']
    assert all(sha(p)==h for p,h in before.items())
    write(a.out/'source_hashes.json',before)
    result={'status':'passed','checks':checks,'actual_input_encoding':True,
        'image_count':len(images),'image_input_tokens':encoding['input_tokens'],
        'fixture_generate_calls':len(calls),'language_model_inference_calls':0,
        'remote_clients_constructed':0,'new_complete_scenes':0,'training_updates':0,
        'shared_model_constructions':len(created),'old_sources_unchanged':True,
        'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cuda_initialized':torch.cuda.is_initialized(),
        'scope':'Actual native-view CPU encoding and native VLM class constructor override. Explicit scripted model output; no model-quality/physics/full-generation evidence.'}
    write(a.out/'summary.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('processor','observation-source','bindings','vlm-source','out'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
