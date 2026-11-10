"""Actual consumer processor encoding; NEVER load model weights or generate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main(stage):
    stage=stage.resolve();out=stage/'consumer_encoding.json'
    if out.exists():raise FileExistsError(out)
    sys.path.insert(0,'/root/autodl-tmp/agent_method_preparation_20261027/work/paris_local_run_results_20261104')
    from local_vlm_sdpa import LocalQwenEngine
    import torch
    tick=time.monotonic();source=stage/'consumer_observation.json';value=json.loads(source.read_text())
    engine=LocalQwenEngine('/root/autodl-tmp/agent_method_preparation_20261027/work/critic_public_runtime_20261104/checkpoint',
        stage/'processor_receipts',max_pixels=65536,max_input_tokens=16384,max_new_tokens=1024)
    inputs,receipt=engine.encode(value['messages'])
    assert engine._model is None
    assert all(torch.isfinite(x).all() for x in inputs.values())
    receipt.update(status='actual_furniture_consumer_input_encoded',model_weights_loaded=False,model_forwards=0,
        paid_API_calls=0,training_steps=0,seconds=time.monotonic()-tick,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tensor_sha256={k:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,t in inputs.items()},
        future_policy_output_cap=1024,whole_runtime_ready=False)
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    main(p.parse_args().stage)
