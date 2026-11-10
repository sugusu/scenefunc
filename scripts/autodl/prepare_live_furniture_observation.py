"""Use the original whole-module message builder on persisted actual views.

No scene, model or service is constructed. This replaces temporary image paths
with byte-identical persisted paths through the author's real builder.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def main(stage):
    stage=stage.resolve();out=stage/'consumer_observation.json'
    if out.exists():raise FileExistsError(out)
    prior=stage/'services_actual_v2'
    result=json.loads((prior/'result.json').read_text())
    text=json.loads((prior/'native_observation_text.json').read_text())
    root=stage/'runtime';meta=json.loads((stage/'runtime_overlay_manifest.json').read_text())
    for p in (root,root/'src'):sys.path.insert(0,str(p))
    os.environ['SCENESMITH_ROOT']=meta['native_root']
    os.environ['OPENAI_AGENTS_DISABLE_TRACING']='1'
    def audit(event,args):
        if event in ('socket.connect','subprocess.Popen'):raise RuntimeError('No network or services in message reconstruction')
    sys.addaudithook(audit)
    module=importlib.import_module('src.agent_rollout_decoupled')
    images=[]
    for row in result['images']:
        p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];images.append(p)
    messages=module.build_messages(text['system'],text['room_prompt'],text['memory_str'],images)
    value=dict(messages=messages,images=result['images'],actual_room_state_sha256=
        hashlib.sha256((prior/'actual_room_state.json').read_bytes()).hexdigest(),
        source_builder=module.build_messages.__module__,new_model_calls=0,
        source='Actual empty architecture from original model-generated floor, native furniture observation; no injected objects/answers')
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='actual_native_consumer_input_prepared',images=len(images),bytes=out.stat().st_size)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    main(p.parse_args().stage)
