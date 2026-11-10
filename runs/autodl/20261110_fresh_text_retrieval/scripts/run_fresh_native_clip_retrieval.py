"""Fresh text CLIP inference and native HSSD HTTP/mesh qualification.

The unchanged native embedding function uses a model cache loaded from the exact
previously authorized local checkpoint. No archived vector answers fresh queries.
Known mapping-offset suffix and unavailable assets are explicitly excluded from
the common candidate pool. This is backend qualification, not a new method.
"""
import argparse
import gc
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
from threading import Thread
import time
from types import ModuleType

SHA = 'c07a17b547d461c60a3cce5062b26bf8545b13de602c4c59d8490361eb716033'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def dump(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',type=Path,required=True)
    a = ap.parse_args()
    stage = a.stage.resolve()
    out = stage/'fresh_v1'
    out.mkdir(exist_ok=False)
    protocol = read(stage/'protocol.json')
    if len(protocol['fresh_descriptions']) != 3 or len(set(protocol['fresh_descriptions'])) != 3:
        raise ValueError('Frozen three-query budget changed')
    restoration = read(stage/protocol['restoration_receipt'])
    if restoration['status'] != 'official_original_hash_verified' or restoration['actual_sha256'] != SHA:
        raise RuntimeError('Exact authorized checkpoint was not verified')
    weights = Path(restoration['weight_path'])
    if not weights.is_file() or weights.stat().st_size != restoration['expected_bytes']:
        raise RuntimeError('Temporary checkpoint changed or is no longer available')
    root = Path(__file__).resolve().parents[2]
    source_stage = root/'runs/autodl/20261110_live_manager_request'
    source = source_stage/'source'
    manifest = read(source_stage/'source_manifest.json')
    if any(sha(source/r['path']) != r['actual_sha256'] for r in manifest['files']):
        raise RuntimeError('Qualified native source changed')
    native_protocol = read(source_stage/'protocol.json')
    inventory_path = Path(native_protocol['inventory'])
    archive = Path(native_protocol['embedding_archive'])
    frozen_inputs = [inventory_path,archive/'text_embeddings.npy',archive/'clip_retrieval_result.json',
                     Path(native_protocol['preprocessed'])/'clip_hssd_embeddings_index.yaml',
                     stage/'protocol.json',Path(__file__).resolve(),
                     Path(__file__).with_name('native_clip_runtime.py'),
                     Path('/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/model_cache/open_clip_config.json')]
    input_hashes = {str(p):sha(p) for p in frozen_inputs}
    started = time.monotonic()
    network = []
    allowed_ports = set()
    def audit(event,args):
        if event == 'socket.connect':
            address = args[1]
            if not isinstance(address,tuple) or address[0] not in ('127.0.0.1','::1') or address[1] not in allowed_ports:
                network.append({'event':'blocked_connection','address':str(address)})
                raise RuntimeError('External network or unowned service is forbidden')
        if event == 'socket.getaddrinfo' and args[0] not in ('127.0.0.1','localhost','::1'):
            network.append({'event':'blocked_dns'})
            raise RuntimeError('External DNS forbidden during inference')
    sys.addaudithook(audit)
    def expired(signum,frame):
        raise TimeoutError('Bounded fresh retrieval qualification deadline')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(protocol['max_execution_seconds'])
    for package in ('scenesmith','scenesmith.agent_utils','scenesmith.agent_utils.hssd_retrieval',
                    'scenesmith.agent_utils.hssd_retrieval_server'):
        module = ModuleType(package)
        module.__path__ = [str(source/Path(*package.split('.')))]
        sys.modules[package] = module
    import numpy as np
    import torch
    import trimesh
    from werkzeug.serving import make_server
    torch.set_num_threads(protocol['cpu_threads'])
    logging.basicConfig(filename=out/'native.log',level=logging.INFO)
    core = importlib.import_module('scenesmith.agent_utils.clip_embeddings')
    similarity = importlib.import_module('scenesmith.agent_utils.hssd_retrieval.clip_similarity')
    app_module = importlib.import_module('scenesmith.agent_utils.hssd_retrieval_server.server_app')
    client_module = importlib.import_module('scenesmith.agent_utils.hssd_retrieval_server.client')
    dto = importlib.import_module('scenesmith.agent_utils.hssd_retrieval_server.dataclasses')
    from native_clip_runtime import bind_original_clip
    binding = bind_original_clip(core,weights,
        '/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/model_cache/open_clip_config.json')
    # Explicit local-cache binding: native functions are not replaced, and no
    # fresh query can be served by the historical embedding archive.
    dump(out/'native_model_binding.json',binding)
    load_seconds = binding['load_and_verify_seconds']
    calls = []
    original_embedding = similarity.get_text_embedding
    def measured_embedding(text,device=None):
        tick = time.monotonic()
        result = original_embedding(text,device='cpu')
        if result.shape != (1024,) or not np.isfinite(result).all() or not np.isclose(np.linalg.norm(result),1,atol=1e-5):
            raise ValueError('Invalid actual CLIP output')
        calls.append({'text':text,'fresh_encoder_forward':True,'archive_returned':False,
                      'seconds':time.monotonic()-tick,'sha256':hashlib.sha256(result.tobytes()).hexdigest()})
        np.save(out/f'embedding_{len(calls):02d}.npy',result,allow_pickle=False)
        dump(out/'embedding_calls.json',calls)
        return result
    similarity.get_text_embedding = measured_embedding
    archive_info = read(archive/'clip_retrieval_result.json')
    archive_vectors = np.load(archive/'text_embeddings.npy',allow_pickle=False)
    if len(archive_info['fixtures']) != 3 or archive_vectors.shape != (3,1024):
        raise ValueError('Historical parity budget/schema changed')
    if set(protocol['fresh_descriptions']) & {x['text'] for x in archive_info['fixtures']}:
        raise ValueError('A fresh test description is in the old fixture archive')
    parity = []
    for i,fixture in enumerate(archive_info['fixtures']):
        fresh = measured_embedding(fixture['text'])
        error = float(np.max(np.abs(fresh-archive_vectors[i])))
        cosine = float(fresh@archive_vectors[i]/np.linalg.norm(archive_vectors[i]))
        parity.append({'query':fixture['text'],'max_absolute_error':error,'cosine':cosine,
                       'archive_used_only_after_fresh_forward_for_comparison':True})
    dump(out/'historical_parity.json',parity)
    if any(p['max_absolute_error']>protocol['historical_parity_max_abs_error'] for p in parity):
        raise ValueError('Fresh encoder differs from frozen historical model vectors')
    inventory = read(inventory_path)
    available = {x['mesh_id']:x for x in inventory['local_assets'] if Path(x['path']).is_file()}
    original_filter = similarity.filter_meshes_by_category
    pool_log = []
    def supported_available_pool(data,category):
        indices = original_filter(data,category)
        retained = [i for i in indices if i < protocol['excluded_mapping_suffix_starts_at_row']
                    and data.embedding_index[i] in available]
        pool_log.append({'category':category,'native_category_candidates':len(indices),'common_retained':len(retained),
                         'excluded_after_known_comment_boundary':True})
        return retained
    similarity.filter_meshes_by_category = supported_available_pool
    app = app_module.HssdRetrievalApp(preload_retriever=True,
        hssd_data_path=native_protocol['data'],hssd_preprocessed_path=native_protocol['preprocessed'],
        hssd_top_k=5,clip_device='cpu')
    server = make_server('127.0.0.1',0,app,threaded=True)
    allowed_ports.add(server.server_port)
    worker = Thread(target=server.serve_forever,daemon=True)
    worker.start();app.start_processing()
    client = client_module.HssdRetrievalClient(port=server.server_port)
    client.session.trust_env = False
    results = []
    counters = None
    try:
        if not client.health_check():raise RuntimeError('Native HSSD health check failed')
        for text in protocol['fresh_descriptions']:
            tick = time.monotonic()
            row = {'text':text,'status':'started','not_in_historical_fixture_queries':all(text!=x['text'] for x in archive_info['fixtures'])}
            try:
                with tempfile.TemporaryDirectory(prefix='fresh_native_mesh_',dir='/dev/shm') as temporary:
                    request = dto.HssdRetrievalServerRequest(object_description=text,object_type='FURNITURE',
                        output_dir=temporary,num_candidates=1,scene_id='fresh_clip_qualification')
                    received = list(client.retrieve_objects([request],max_retries=1,timeout_s=90))
                    if len(received)!=1 or received[0][0]!=0 or len(received[0][1].results)!=1:
                        raise ValueError('Missing or duplicate response')
                    item = received[0][1].results[0]
                    original = available[item.hssd_id]
                    if sha(original['path'])!=original['sha256']:
                        raise ValueError('Original selected asset changed')
                    mesh_path = Path(item.mesh_path)
                    if mesh_path.parent!=Path(temporary) or not mesh_path.is_file():
                        raise ValueError('Wrong native mesh-export location')
                    mesh = trimesh.load(mesh_path,force='mesh')
                    if not isinstance(mesh,trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
                        raise ValueError('Invalid native mesh result')
                    if not np.allclose(mesh.extents,item.size,atol=1e-5):
                        raise ValueError('Cold-loaded geometry differs from response')
                    row.update(status='fresh_native_mesh_returned_and_cold_loaded',hssd_id=item.hssd_id,
                        original_glb_sha256=original['sha256'],export_sha256=sha(mesh_path),
                        export_bytes=mesh_path.stat().st_size,vertices=len(mesh.vertices),faces=len(mesh.faces),
                        extent=mesh.extents.tolist(),original_metadata=original['metadata'],
                        embedding_row=app._retriever.preprocessed_data.get_embedding_index(item.hssd_id),
                        semantic_correctness_verified=False,material_verified=False,physical_properties_verified=False)
                    del mesh
            except Exception as exc:
                row.update(status='retrieval_failed_no_retry',error_type=type(exc).__name__,error=str(exc)[:300])
            row['seconds'] = time.monotonic()-tick
            results.append(row);dump(out/'requests.json',results)
            print(text,row['status'],flush=True)
            gc.collect()
        counters = {'total':app._total_requests,'completed':app._completed_requests,'failed':app._failed_requests}
    finally:
        client.session.close();server.shutdown();server.server_close();worker.join(timeout=5)
        app.stop_processing()
        similarity.get_text_embedding = original_embedding
        similarity.filter_meshes_by_category = original_filter
        signal.alarm(0)
    threads_stopped = not worker.is_alive() and not app._processing_thread.is_alive()
    if not threads_stopped:raise RuntimeError('Native worker did not terminate')
    if any(sha(p)!=v for p,v in input_hashes.items()):raise ValueError('Frozen inputs changed')
    module_hashes = {str(Path(m.__file__)):sha(m.__file__) for name,m in list(sys.modules.items())
                     if name.startswith('scenesmith.') and getattr(m,'__file__',None)}
    summary = {'status':'fresh_text_native_retrieval_qualification_completed','fresh_requests':len(results),
        'actual_text_encoder_forwards':len(calls),'historical_parity_forwards':len(parity),
        'fresh_mesh_responses':sum(r['status']=='fresh_native_mesh_returned_and_cold_loaded' for r in results),
        'server_counters':counters,'available_inventory_assets':len(available),'candidate_pool_log':pool_log,
        'model_load_seconds':load_seconds,'execution_seconds':time.monotonic()-started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'model_device':'cpu','used_gpu':False,'new_8B_VLM_calls':0,'training_updates':0,'paid_API_calls':0,
        'new_physical_scene_objects':0,'new_complete_rooms':0,'full_scene_success':None,
        'full_runtime_ready':False,'old_inputs_unchanged':True,'threads_stopped':threads_stopped,
        'source_module_hashes':module_hashes,'input_hashes':input_hashes,
        'network_blocks':network,'original_weight_sha256':SHA,
        'native_loading_boundary':'OpenCLIP local-file constructor populated the native module cache. The native _get_clip_model cache hit and get_text_embedding code are unchanged; not a test of native automatic HF downloading.',
        'mapping_boundary':'Original index untouched; only available assets before the known comment boundary are considered. This excludes the known offset suffix but does not independently certify every prefix image-ID pairing.',
        'weights_persistence':'Full original checkpoint temporarily in RAM; no weights placed on data disk. Repeatability requires this RAM file or a new verified restore.',
        'contribution_boundary':'Common backend qualification, not a learned-policy comparison, semantic retrieval benchmark or novel algorithm.'}
    dump(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_module_hashes','input_hashes','candidate_pool_log','network_blocks']}),flush=True)


if __name__ == '__main__':
    main()
