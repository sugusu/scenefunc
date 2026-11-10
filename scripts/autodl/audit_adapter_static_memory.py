"""Read existing safetensors headers only; no tensors/model/GPU are loaded."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def main(checkpoint, output):
    if output.exists():raise FileExistsError(output)
    index_path=checkpoint/'model.safetensors.index.json'
    index=json.loads(index_path.read_text());tensors={};headers={}
    for name in sorted(set(index['weight_map'].values())):
        p=checkpoint/name
        with p.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0]
            if length>2**20:raise ValueError('Unexpected large safetensors header')
            payload=f.read(length)
        header=json.loads(payload)
        headers[name]=dict(file_bytes=p.stat().st_size,header_bytes=length,
                          header_sha256=hashlib.sha256(payload).hexdigest())
        for key,row in header.items():
            if key=='__metadata__':continue
            if key in tensors or index['weight_map'].get(key)!=name:raise ValueError('Duplicate/unindexed tensor')
            tensors[key]=row
    assert set(tensors)==set(index['weight_map'])
    count=sum(math.prod(v['shape']) for v in tensors.values())
    assert count==index['metadata']['total_parameters']==8767123696
    pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight')
    selected=[]
    for key,row in sorted(tensors.items()):
        if not pattern.fullmatch(key):continue
        assert len(row['shape'])==2
        selected.append(dict(weight_name=key,module_name=key.removesuffix('.weight'),shape=row['shape']))
    assert len(selected)==36*7
    plans=[]
    for rank in [8,16]:
        params=sum(rank*sum(v['shape']) for v in selected)
        # Explicit parameter-state convention: FP32 adapter parameters and
        # gradients + two FP32 Adam moments, no redundant master parameter.
        adapter_bytes=params*16
        plans.append(dict(rank=rank,modules=len(selected),trainable_parameters=params,
                          FP32_parameter_gradient_and_two_Adam_moments_bytes=adapter_bytes,
                          one_FP16_base_plus_adapter_static_bytes=count*2+adapter_bytes,
                          sharing_frozen_reference_requires_verified_adapter_disable=True))
    result=dict(status='existing_checkpoint_headers_read_no_model',checkpoint=str(checkpoint),
        parameter_count=count,checkpoint_metadata=index['metadata'],
        config_sha256=hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        shard_headers=headers,selected_language_modules=selected,plans=plans,
        one_FP16_base_weight_bytes=count*2,two_FP16_base_weight_bytes=count*4,
        excluded_from_estimate=['activations and backward temporaries','KV/attention workspaces',
            'vision encoder activations','native meshes/renderers and CPU model-load buffers',
            'allocator fragmentation','optional representation-alignment encoder/head/cache'],
        hardware_boundary='Config stores BF16; no BF16 kernel support or training fit on V100 inferred. Existing inference runtime casts toFP16. AMP scaling, activation checkpointing and actual memory/finite gradients still require a separately admitted run.',
        readiness='Static storage arithmetic only; not a training-fit result or authorization.',
        model_loads=0,VLM_forwards=0,training_updates=0,paid_API_calls=0)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','parameter_count','plans','two_FP16_base_weight_bytes']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.checkpoint.resolve(),a.output.resolve())
