"""One real local feedback turn from a saved native-kernel checkpoint.

Uses actual current five-view geometry and native ordinary state-table memory.
This bounded furniture-stage adapter does not certify a full native rollout.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

P=Path(__file__).resolve().parent
W=P.parent
sys.path.insert(0,str(W/'native_local_inference_20261104'))
from local_vlm import LocalQwenEngine


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['A','B'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--views',type=Path,required=True)
    parser.add_argument('--audit',type=Path,help='Optional actual mesh-check output bound to this same exported scene')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--attention',choices=['eager','sdpa'],default='eager')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();network=[]
    def reject(*a,**kw):network.append('connection');raise RuntimeError('Network disabled for local policy')
    socket.socket.connect=reject;socket.create_connection=reject
    receipt={'arm':args.arm,'phase':'one_real_feedback_turn_after_common_retrieval_correction',
             'paid_api_calls':0,'training_runs':0,'complete_room_success':None,
             'original_first_turn_changed':False,'native_full_rollout':False,
             'status':'started','model_forward':False}
    receipt['attention']=args.attention
    try:
        document=json.loads(args.checkpoint.read_text());state=document['payload']
        canonical=json.dumps(state,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        assert hashlib.sha256(canonical.encode()).hexdigest()==document['payload_sha256']
        for path,h in state['geometry_hashes'].items():assert sha(path)==h
        views=json.loads(args.views.read_text());assert views['status']=='completed'
        # Bind observations to exactly the serialized source scene, not an old
        # or reference layout. Empty A/B shell exports may be byte-identical.
        scene=args.checkpoint.parent/'partial_scene.glb'
        assert sha(scene)==views['source_sha256']
        images=views['views']
        assert len(images)==5
        for image in images:assert sha(image['path'])==image['sha256']
        m=module('ordinary_native_memory',W/'professional_data_ingest_20261104/native_src/memory.py')
        memory=m.empty_memory('furniture')
        for event in state['tool_history']:
            call=event['call'];m.update_furniture(memory,call['name'],call['arguments'],event['response'])
        memory_text=m.render_memory('furniture',memory)
        (out/'ordinary_memory.json').write_text(json.dumps(memory,ensure_ascii=False,indent=2)+'\n')
        (out/'ordinary_memory.txt').write_text(memory_text)
        original=W/'paris_A_inference_20261104'/f'request_{args.arm}.json'
        request=json.loads(original.read_text());system=request[0]['content']
        system+='''\n\n## Actual current feedback-turn observation (overrides earlier schematic and annotation descriptions)
The five attached views are of the actual checkpoint after the preceding tool
calls, ordered topdown, south, east, west, north. They use a declared ceiling
cutaway and the original camera renderer; there are no native object-ID or arrow
overlays. Use the object IDs and actual dimensions in memory and returned asset
metadata. The original shell and task are unchanged. Do not infer physical or
functional success from an available asset or a successful tool call.
The common retrieval adapter now first resolves furniture category and ranks
matching cached real meshes by dimension mismatch. It does not resize assets.
Unavailable categories explicitly fail; do not substitute a different category
silently. Names are operation handles; actual_name/categories describe the mesh.
Currently executable tools: generate_assets, add_furniture_to_scene_tool,
move_furniture_tool, remove_furniture_tool. snap_to_object_tool,
rescale_furniture_tool and check_physics remain unsupported in this bounded
adapter and will return explicit failures. Report unverified requirements;
never claim the room collision-free or functionally complete without evidence.
Continue the actual furniture design from this state, preserving the task.
'''
        task=request[1]['content'][-1]['text'].split('## Memory')[0].strip()
        asset_info=[{k:r[k] for k in ['object_id','actual_name','categories','width','depth','height','rescaled','semantic_match']} for r in state['asset_records']]
        missing=[r for e in state['tool_history'] for r in e['response'].get('missing',[])]
        content=[{'type':'image','image':image['path']} for image in images]
        content.append({'type':'text','text':task+'\n\n## Current ordinary memory\n'+memory_text+'\n\n## Actual asset evidence\n'+json.dumps(asset_info,ensure_ascii=False)+'\n\n## Missing asset requests\n'+json.dumps(missing,ensure_ascii=False)})
        if args.audit:
            audit=json.loads(args.audit.read_text())
            assert audit['scene_sha256']==sha(scene)
            content.append({'type':'text','text':'''## Actual post-execution mesh checks
The following is a common external mesh-check observation, not a desired layout
or a human preference label. Use it with the current views and task to decide
whether and what to edit. FCL reports mesh-surface intersections, which do not
by themselves establish every functional violation. In particular, the stored
floor slab top is 2 cm above the native placement reference; floor intersections
are separately flagged and are not resolved by pretending XY edits fix this
vertical adapter discrepancy. Do not delete needed functions to improve a count.
No correct replacement coordinates, object ordering or design preference are
provided. Preserve the original task and state what remains unverified.
'''+json.dumps(audit,ensure_ascii=False)})
            receipt.update(mesh_feedback_path=str(args.audit.resolve()),mesh_feedback_sha256=sha(args.audit))
        messages=[{'role':'system','content':system},{'role':'user','content':content}]
        (out/'request.json').write_text(json.dumps(messages,ensure_ascii=False,indent=2)+'\n')
        receipt.update(input_checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha(args.checkpoint),
                       source_views_receipt=str(args.views.resolve()),source_views_sha256=sha(args.views),
                       request_sha256=sha(out/'request.json'))
        receipt['execution_sources']={str(path):sha(path) for path in [Path(__file__),P/'dispatch_corrected.py',P/'typed_retrieval.py',W/'native_local_inference_20261104/local_vlm.py']}
        if args.attention=='sdpa' and args.arm=='A' and args.audit:
            previous=P/'A_mesh_feedback/request.json'
            assert sha(previous)==sha(out/'request.json')
            receipt['same_request_as_failed_eager_attempt']=True
        import torch
        assert torch.cuda.is_available()
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        receipt['gpu_name']=torch.cuda.get_device_name()
        Engine=LocalQwenEngine
        if args.attention=='sdpa':
            from local_vlm_sdpa import LocalQwenEngine as Engine
            receipt['execution_sources'][str(P/'local_vlm_sdpa.py')]=sha(P/'local_vlm_sdpa.py')
        engine=Engine(W/'critic_public_runtime_20261104/checkpoint',out/'calls',max_pixels=65536,max_input_tokens=8192,max_new_tokens=2048)
        result=engine.complete(messages,purpose=f'paris_{args.arm}_actual_feedback_turn')
        (out/'response_raw.txt').write_text(result['raw']);(out/'response_text.txt').write_text(result['text'])
        run_module=module('first_turn_parser',W/'paris_A_inference_20261104/run.py')
        _,calls=run_module.parser()(result['raw'])
        (out/'parsed_tool_calls.json').write_text(json.dumps(calls,ensure_ascii=False,indent=2)+'\n')
        receipt.update(model_forward=True,model_receipt_dir=result['receipt_dir'],parsed_tool_calls=len(calls))
        # Release model memory before running mesh tools and the next renderer.
        del engine
        import gc
        gc.collect();torch.cuda.empty_cache()
        python='@LEGACY@/envs/hsm_cpu/bin/python'
        command=[python,str(P/'dispatch_corrected.py'),'--calls',str(out/'parsed_tool_calls.json'),
                 '--resume',str(args.checkpoint.resolve()),'--output',str(out/'tools')]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 PYTHONPATH='@GEOMETRY_DEPS@')
        proc=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
        (out/'tools.stdout').write_text(proc.stdout);(out/'tools.stderr').write_text(proc.stderr)
        receipt.update(tool_exit=proc.returncode,status='feedback_turn_saved',full_generation=False)
        assert proc.returncode==0,proc.stderr
    except BaseException as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-start,
                       peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       network_connection_attempts=len(network))
        (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
