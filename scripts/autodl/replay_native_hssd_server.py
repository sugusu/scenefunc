"""Run pinned HSSD HTTP retrieval with strict historical query-vector replay.

Executes native loader, scoring, size ranking, mesh alignment/export, queue,
Flask routes and HTTP client. Replaces ONLY text embedding by an exact-query
archive. A second explicit acquisition policy filters unavailable IDs before
top-K. Neither is unrestricted fresh text inference or a physical SceneObject.
"""
import argparse
import ast
import gc
import hashlib
import importlib
import json
import logging
from pathlib import Path
import resource
import shutil
import sys
import tempfile
from threading import Thread
import time
from types import ModuleType


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'common-source', 'client-source', 'archive', 'data', 'preprocessed', 'inventory', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); sources = {}
    logging.basicConfig(filename=a.out/'native.log', level=logging.INFO)
    # Only localhost HTTP is allowed; no model/asset download fallback.
    network = []
    def audit(event, args):
        if event == 'socket.connect':
            address = args[1]; network.append(str(address))
            if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1'):
                raise RuntimeError('External network forbidden in replay')
        if event == 'socket.getaddrinfo' and args[0] not in ('127.0.0.1', '::1', 'localhost'):
            raise RuntimeError('External DNS forbidden in replay')
    sys.addaudithook(audit)
    roots = [a.source.resolve(), a.common_source.resolve(), a.client_source.resolve()]
    for package in ('scenesmith', 'scenesmith.agent_utils', 'scenesmith.agent_utils.hssd_retrieval',
                    'scenesmith.agent_utils.hssd_retrieval_server'):
        module = ModuleType(package)
        module.__path__ = [str(r/Path(*package.split('.'))) for r in roots]
        sys.modules[package] = module
    import numpy as np
    import trimesh
    from werkzeug.serving import make_server
    # Text vectors are archived, so importing torch/open_clip serves no purpose
    # here. Execute only the two exact native functions that this replay uses.
    # No generated similarity scores or surrogate model replace the native math.
    clip_path = a.common_source.resolve()/'scenesmith/agent_utils/clip_embeddings.py'
    core = ModuleType('scenesmith.agent_utils.clip_embeddings'); core.__file__ = str(clip_path)
    core.__dict__.update(np=np)
    def no_model(*args, **kwargs): raise RuntimeError('Fresh CLIP inference is outside this archive-only execution')
    core._get_clip_model = no_model
    definitions = [n for n in ast.parse(clip_path.read_text()).body if isinstance(n,ast.FunctionDef)
                   and n.name in ('compute_clip_similarities','get_text_embedding')]
    if len(definitions)!=2: raise ValueError('Native CLIP function selection changed')
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(clip_path),'exec'),core.__dict__)
    sys.modules[core.__name__] = core
    similarity = importlib.import_module('scenesmith.agent_utils.hssd_retrieval.clip_similarity')
    loader = importlib.import_module('scenesmith.agent_utils.hssd_retrieval.data_loader')
    app_module = importlib.import_module('scenesmith.agent_utils.hssd_retrieval_server.server_app')
    client_module = importlib.import_module('scenesmith.agent_utils.hssd_retrieval_server.client')
    dto = importlib.import_module('scenesmith.agent_utils.hssd_retrieval_server.dataclasses')
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('scenesmith.') and path:
            sources[path] = sha(path)
    manifest = json.loads((a.archive/'executed_source_manifest.json').read_text())
    historical_script = a.archive/'clip_retrieval.py'
    assert next(x['sha256'] for x in manifest['files'] if x['path']==str(historical_script)) == sha(historical_script)
    archive = json.loads((a.archive/'clip_retrieval_result.json').read_text())
    if archive['status'] != 'completed' or not archive['model_loaded']:
        raise ValueError('Archive does not record completed CLIP inference')
    queries = archive['fixtures']
    vectors = np.load(a.archive/'text_embeddings.npy', allow_pickle=False)
    if vectors.shape != (len(queries), 1024) or not np.isfinite(vectors).all():
        raise ValueError('Invalid archived vector shape')
    if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
        raise ValueError('Archived vectors are not normalized')
    cache = {q['text']:vectors[i].copy() for i,q in enumerate(queries)}
    if len(cache) != len(queries): raise ValueError('Duplicate cached descriptions')
    calls = []
    def exact_query(text, device=None):
        calls.append(dict(query=text, scope='archived_embedding_replay', found=text in cache))
        if text not in cache: raise ValueError('No archived vector for this exact query; no inference fallback')
        return cache[text].copy()
    original_embedding = similarity.get_text_embedding
    original_filter = similarity.filter_meshes_by_category
    similarity.get_text_embedding = exact_query
    inventory = json.loads(a.inventory.read_text())
    candidates = inventory['local_assets'] if 'local_assets' in inventory else None
    if candidates is None:
        # Explicit schema failure, never substitute an unaudited path scan.
        raise ValueError('Expected audited local_assets inventory')
    allowed = {x['mesh_id']:x for x in candidates}
    inputs = {str(p):sha(p) for p in (a.archive/'clip_retrieval_result.json', a.archive/'text_embeddings.npy',
              historical_script, a.inventory)}
    inputs.update(inventory['input_hashes'])
    for p,h in inputs.items():
        if sha(p)!=h:raise ValueError('Frozen retrieval input changed: '+p)
    mode = 'native_topk_then_load'; pool_log=[]
    def available_filter(data, category):
        indices = original_filter(data, category)
        retained = [i for i in indices if data.embedding_index[i] in allowed]
        pool_log.append(dict(category=category, original=len(indices), retained=len(retained)))
        return retained
    protocol = dict(queries=queries, modes=['native_topk_then_load','available_before_topk'],
        top_k=5, candidates_returned=1, object_type='FURNITURE',
        new_query_negative='query_not_present_in_the_frozen_archive',
        interpretation='Acquisition policy check, not an Agent-quality experiment; identical query vectors and native scoring. No index remapping.')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    app = app_module.HssdRetrievalApp(preload_retriever=True, hssd_data_path=str(a.data.resolve()),
        hssd_preprocessed_path=str(a.preprocessed.resolve()),hssd_top_k=5,clip_device='cpu')
    http = make_server('127.0.0.1',0,app,threaded=True)
    worker = Thread(target=http.serve_forever,daemon=True); worker.start(); app.start_processing()
    client = client_module.HssdRetrievalClient(port=http.server_port); client.session.trust_env=False
    outputs=[]; health=False
    try:
        health=client.health_check()
        if not health:raise RuntimeError('Actual native health request failed')
        for mode in protocol['modes']:
            similarity.filter_meshes_by_category = original_filter if mode=='native_topk_then_load' else available_filter
            for q in queries:
                tick=time.monotonic();row=dict(mode=mode,query=q['text'],label=q['label'],
                    desired_dimensions_yup=q['dimensions'],status='started')
                # Exported duplicates are temporary; original GLBs stay immutable.
                with tempfile.TemporaryDirectory(prefix='native_hssd_',dir='/dev/shm') as tmp:
                    request=dto.HssdRetrievalServerRequest(object_description=q['text'],object_type='FURNITURE',
                        desired_dimensions=tuple(q['dimensions']),output_dir=tmp,num_candidates=1,scene_id='fixed_archive_replay')
                    try:
                        result=list(client.retrieve_objects([request],max_retries=1,timeout_s=120))
                        if len(result)!=1 or result[0][0]!=0:raise ValueError('Missing/duplicate streamed response')
                        response=result[0][1]
                        if len(response.results)!=1:raise ValueError('Expected exactly one real mesh result')
                        item=response.results[0]; path=Path(item.mesh_path)
                        if path.parent!=Path(tmp) or not path.is_file():raise ValueError('Wrong output path')
                        original=allowed[item.hssd_id]
                        if sha(original['path'])!=original['sha256']:raise ValueError('Original GLB changed')
                        mesh=trimesh.load(path,force='mesh')
                        if not isinstance(mesh,trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
                            raise ValueError('Invalid exported mesh')
                        if not np.allclose(mesh.extents,item.size,atol=1e-5):raise ValueError('Response/cold mesh extents differ')
                        idx=app._retriever.preprocessed_data.get_embedding_index(item.hssd_id)
                        row.update(status='mesh_returned_and_cold_loaded',response=response.to_dict(),
                            mesh_id=item.hssd_id,metadata=original['metadata'],source_glb=original['path'],
                            source_sha256=original['sha256'],export_sha256=sha(path),export_bytes=path.stat().st_size,
                            vertices=len(mesh.vertices),faces=len(mesh.faces),bounds=mesh.bounds.tolist(),
                            index_row=idx,index_identity_status='unverified_original_mapping',
                            metadata_orientation_present=bool(original['metadata']['up'] and original['metadata']['front']),
                            temporary_export_removed_after_verification=True)
                    except RuntimeError as exc:
                        row.update(status='retrieval_failed',error=str(exc))
                row['seconds']=time.monotonic()-tick;outputs.append(row)
                (a.out/'completed_requests.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
                print(mode,q['label'],row['status'],flush=True)
                if 'mesh' in locals(): del mesh
                gc.collect()
        with tempfile.TemporaryDirectory(prefix='native_hssd_negative_',dir='/dev/shm') as tmp:
            bad=dto.HssdRetrievalServerRequest(object_description=protocol['new_query_negative'],
                object_type='FURNITURE',output_dir=tmp)
            try:list(client.retrieve_objects([bad],max_retries=1,timeout_s=30))
            except RuntimeError as exc:negative=dict(status='rejected',error=str(exc))
            else:raise AssertionError('Uncached query silently accepted')
        counters=dict(total=app._total_requests,completed=app._completed_requests,failed=app._failed_requests)
    finally:
        client.session.close();http.shutdown();http.server_close();worker.join(timeout=5)
        app.stop_processing()
        similarity.get_text_embedding=original_embedding;similarity.filter_meshes_by_category=original_filter
    if worker.is_alive() or app._processing_thread.is_alive():raise RuntimeError('Native worker did not stop')
    if any(sha(p)!=h for p,h in sources.items()) or any(sha(p)!=h for p,h in inputs.items()):
        raise ValueError('Source/input mutated')
    summary=dict(status='actual_native_http_and_mesh_replay_completed',health=health,rows=outputs,
        negative=negative,counters=counters,embedding_replays=calls,available_pool_counts=pool_log,
        source_sha256=sources,input_sha256=inputs,localhost_connections=network,
        actual_clip_inference_calls=0,new_weights=0,paid_api_calls=0,new_complete_scenes=0,
        physical_scene_objects_created=0,unrestricted_text_retrieval_ready=False,full_runtime_ready=False,
        threads_stopped=True,old_sources_unchanged=True,seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        clip_source_execution='Exact compute_clip_similarities/get_text_embedding definitions only; unused torch/open_clip imports excluded. Native retrieval/HTTP modules fully imported under package namespace.',
        limitation='Real server/client/native mesh processing using archived text vectors only. No VLM physics, GLTF/Drake SDF conversion, collision decomposition or semantic-groundtruth claim. Available-first policy is an explicit common catalogue adaptation; not native-global retrieval equivalence.')
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','source_sha256','input_sha256','embedding_replays','localhost_connections','available_pool_counts')}))


if __name__=='__main__':main()
