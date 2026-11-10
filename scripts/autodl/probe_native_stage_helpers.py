"""Import exact author helper modules; summarize the actual generated floor.

No fake agent/modules, AST-isolated replacement, model, training or new scene.
Furniture serialization is imported, NOT claimed to have executed on furniture.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(stage):
    stage=stage.resolve();out=stage/'next_stage_helper_probe.json'
    if out.exists():raise FileExistsError(out)
    source=stage/'next_stage_sources'
    manifest=json.loads((source/'source_manifest.json').read_text())
    for r in manifest['files']:assert sha(r['path'])==r['sha256']
    runtime=Path(json.loads((stage/'materialization.json').read_text())['runtime_root'])
    for p in (runtime,runtime/'src',source):sys.path.insert(0,str(p))
    os.environ['SCENESMITH_ROOT']=str(runtime/'native_source')
    os.environ['OPENAI_AGENTS_DISABLE_TRACING']='1'
    blocked=[]
    def audit(event,args):
        if event in ('socket.connect','subprocess.Popen'):
            blocked.append(event);raise RuntimeError('No services/inference in helper check')
    sys.addaudithook(audit)
    started=time.monotonic()
    summary=importlib.import_module('room_summary')
    grpo=importlib.import_module('grpo_rollout')
    actual=stage/'actual_floor_geometry_v1/layout_with_geometry.json'
    text=summary.from_house_layout_dict(json.loads(actual.read_text()),'bedroom')
    assert text and '4.8m wide × 4m deep' in text and 'x∈[-2.4, +2.4]' in text
    assert callable(grpo._flush_furniture_json) and callable(grpo.tree_furniture_rollout)
    assert not blocked and 'transformers' not in sys.modules
    result=dict(status='whole_original_helper_modules_imported_actual_floor_summarized',
        task_dimensions_taken_from_actual_placed_room=True,actual_summary=text,
        helper_modules=[str(summary.__file__),str(grpo.__file__)],
        input=str(actual),input_sha256=sha(actual),seconds=time.monotonic()-started,
        model_calls=0,API_calls=0,training_steps=0,new_scenes=0,
        furniture_flush_on_real_furniture_executed=False,
        closed_runtime_not_modified=True,whole_runtime_ready=False)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    main(p.parse_args().stage)
