"""CPU-only native SFT dataset encoding of a replayed ordinary-teacher batch.

Executes the original dataset __getitem__ body, not an approximate mask check.
Loads the local processor only; does not instantiate a neural model or optimize.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
from types import SimpleNamespace
from typing import Any

W = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
SFT = W/'autodl_continuation_20261110/training_source_review_20261110/upstream/src/sft_dataset.py'


def read(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--variant',choices=['full','catalog_projection'],default='full')
    a = ap.parse_args(); root = a.root.resolve()
    output=root/('encoding_result.json' if a.variant=='full' else 'encoding_catalog_projection.json')
    if output.exists():
        raise FileExistsError('No implicit processor rerun')
    start = time.monotonic()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    def offline(*_,**__):
        raise RuntimeError('No network allowed during local encoding')
    socket.socket.connect=offline;socket.create_connection=offline
    import torch
    from PIL import Image
    sys.path.insert(0,str(W/'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine
    engine = LocalQwenEngine(W/'critic_public_runtime_20261104/checkpoint',
        Path('/dev/shm/native_solver_teacher_processor_20261111'),
        max_pixels=16777216,max_input_tokens=65536,max_new_tokens=1)
    processor = engine.processor
    text = ast.parse(SFT.read_text())
    build = next(x for x in text.body if isinstance(x,ast.FunctionDef) and x.name=='_build_messages')
    klass = next(x for x in text.body if isinstance(x,ast.ClassDef) and x.name=='StepwiseSFTDataset')
    getitem = next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=='__getitem__')
    env = dict(Any=Any,StepSample=SimpleNamespace,Image=Image,torch=torch)
    exec(compile(ast.Module(body=[build,getitem],type_ignores=[]),str(SFT),'exec'),env)
    full = read(root/'sft_full_messages.json');prompt=read(root/'sft_prompt_messages.json')
    task,memory=prompt[1]['content'][-1]['text'][len('## Task\n'):].split('\n\n## Memory\n',1)
    if a.variant=='catalog_projection':
        original_memory=json.loads(memory);projected=copy.deepcopy(original_memory)
        # Deterministic catalog presentation only. Initial room state, all16
        # catalog IDs, semantic descriptions, dimensions and scale are retained.
        # File paths and neutral template bookkeeping remain in the executor.
        kept=['object_id','object_type','name','description','bbox_min','bbox_max','scale_factor','immutable']
        rows=[]
        for item in projected['asset_catalog']:
            t=item['template']
            assert t['support_surfaces']==[] and t['placement_info'] is None
            assert t['transform']==dict(translation=[0.,0.,0.],rotation_wxyz=[1.,0.,0.,0.])
            rows.append(dict(template={k:t[k] for k in kept},binding=item['binding'],limitations=item['limitations']))
        projected['asset_catalog']=rows
        assert projected['state']==original_memory['state']
        memory=json.dumps(projected,ensure_ascii=False,allow_nan=False)
        user_text='## Task\n'+task+'\n\n## Memory\n'+memory
        prompt[1]['content'][-1]['text']=user_text
        full[1]['content'][-1]['text']=user_text
        (root/'sft_catalog_projection_full_messages.json').write_text(json.dumps(full,ensure_ascii=False,indent=2)+'\n')
        (root/'sft_catalog_projection_prompt_messages.json').write_text(json.dumps(prompt,ensure_ascii=False,indent=2)+'\n')
        (root/'catalog_projection_contract.json').write_text(json.dumps(dict(
            method='Ordinary deterministic catalog field projection, not a learned/new compression algorithm',
            original_catalog_count=16,retained_catalog_count=len(rows),retained_template_fields=kept,
            unchanged=['Task','Complete initial room state','Five original PNGs and image order','All asset IDs','Asset bindings','Limitations','Teacher actions'],
            omitted_catalog_fields=['Identity template transform','Empty support surfaces','Null placement','geometry_path','sdf_path','image_path','metadata'],
            geometry_access='Executor retains the exact original registry and71resource files',
            semantic_losslessness_for_arbitrary_tasks=False,
            quality_or_model_comparison_claim=False),ensure_ascii=False,indent=2)+'\n')
    sample=SimpleNamespace(stage='furniture',system=prompt[0]['content'],task_context=task,
        memory_str=memory,scene_images=[v['image'] for v in prompt[1]['content'] if v['type']=='image'],
        image_path=None,assistant_content='',tool_calls=[dict(name=t['function']['name'],
        arguments=t['function']['arguments']) for t in full[-1]['tool_calls']],
        source_file=str(root/'sft_full_messages.json'),step_idx=0)
    assert env['_build_messages'](sample,no_thinking=True)==(full,prompt)
    dataset=SimpleNamespace(samples=[sample],no_thinking=True,processor=processor,max_length=65536)
    encoded=env['__getitem__'](dataset,0)
    prompt_text=processor.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
    full_text=processor.apply_chat_template(full,tokenize=False,add_generation_prompt=False)
    imgs=[Image.open(p).convert('RGB') for p in sample.scene_images]
    prefix=processor(text=[prompt_text],images=imgs,return_tensors='pt',padding=False)['input_ids'][0]
    P=len(prefix);N=len(encoded['input_ids'])
    assert torch.equal(encoded['input_ids'][:P],prefix)
    assert (encoded['labels'][:P]==-100).all()
    assert torch.equal(encoded['labels'][P:],encoded['input_ids'][P:])
    assert int((encoded['labels']!=-100).sum())==N-P>0
    assert len(encoded['image_grid_thw'])==5
    decoded=processor.tokenizer.decode(encoded['labels'][P:].tolist(),skip_special_tokens=False)
    assert decoded.count('<tool_call>')==2 and '<think>' not in decoded
    assert engine._model is None
    result=dict(status='native_dataset_encoding_completed',input_variant=a.variant,
        native_sft_source=str(SFT),source_sha256=hashlib.sha256(SFT.read_bytes()).hexdigest(),
        original_function_lines=dict(build=[build.lineno,build.end_lineno],getitem=[getitem.lineno,getitem.end_lineno]),
        prompt_tokens=P,full_tokens=N,supervised_tokens=N-P,
        fits_original_dataset_default_8192=N<=8192,fits_recent_policy_budget_24576=N<=24576,
        measurement_max_length=65536,no_truncation=True,image_count=5,
        image_grid_thw=encoded['image_grid_thw'].tolist(),
        tensor_shapes={k:list(v.shape) for k,v in encoded.items()},
        entire_prompt_loss_masked=True,exact_assistant_span_supervised=True,
        decoded_supervised_span=decoded,
        serialized_full_sha256=hashlib.sha256(full_text.encode()).hexdigest(),
        model_weights_loaded=False,model_forwards=0,training_updates=0,paid_API_calls=0,
        wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='One exposed ordinary-teacher batch. Encoding compatibility is not evidence of learnability, generalization, native full Actor parity or human design quality.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['decoded_supervised_span','tensor_shapes']}),flush=True)


if __name__=='__main__':
    main()
