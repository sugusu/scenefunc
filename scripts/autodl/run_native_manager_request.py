"""One real normal AssetManager request through actual local services.

Text CLIP embedding is a strict historical-vector replay for one fixed query.
Native retrieval/scoring/mesh export, VLM RPC, Blender, collision, SDF and
registry are live. No AST-isolated manager, manual property answer or repair.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
from threading import Thread
import time


def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True)
    a=ap.parse_args();stage=a.stage.resolve();out=stage/'actual_v1';out.mkdir(exist_ok=False)
    protocol=read(stage/'protocol.json');repo=Path(__file__).resolve().parents[2]
    assert shutil.disk_usage(out).free>5*1024**3+20*1024**2
    assert int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text())>=24*1024**3
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    assert int(gpu.strip())>=26*1024
    sys.path.insert(0,str(stage/'source'))
    sources=read(stage/'source_manifest.json')
    assert all(sha(stage/'source'/r['path'])==r['actual_sha256'] for r in sources['files'])
    network=[];allowed_ports=set()
    def audit(event,args):
        if event=='socket.connect':
            address=args[1]
            if (not isinstance(address,tuple) or address[0] not in ('127.0.0.1','::1')
                    or address[1] not in allowed_ports):
                network.append(str(address));raise RuntimeError('External networking prohibited')
        if event=='socket.getaddrinfo' and args[0] not in ('127.0.0.1','localhost','::1'):
            network.append('dns');raise RuntimeError('External DNS prohibited')
    sys.addaudithook(audit)
    import numpy as np
    import requests
    from werkzeug.serving import make_server
    from omegaconf import OmegaConf,open_dict
    from scenesmith.agent_utils.asset_manager import AssetManager,AssetGenerationRequest
    from scenesmith.agent_utils.room import AgentType,ObjectType
    from scenesmith.agent_utils.blender import BlenderServer
    from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionServer
    from scenesmith.agent_utils.hssd_retrieval_server.server_app import HssdRetrievalApp
    from scenesmith.agent_utils.hssd_retrieval import clip_similarity
    from scenesmith.utils.logging import ConsoleLogger
    from local_vlm_rpc import LocalVLMClient
    from retrieval_image_binding import bind_retrieval_without_images
    from audit_native_services import compose_native
    logging.basicConfig(filename=out/'native.log',level=logging.INFO)
    # Normal native modules remain imported. Only unavailable fresh text
    # encoding is explicitly replaced by this exact-query archive.
    archive=Path(protocol['embedding_archive']);fixtures=read(archive/'clip_retrieval_result.json')['fixtures']
    vectors=np.load(archive/'text_embeddings.npy',allow_pickle=False)
    idx=next(i for i,r in enumerate(fixtures) if r['text']==protocol['description'])
    original_embedding=clip_similarity.get_text_embedding
    replay=[]
    def exact_query(text,device=None):
        replay.append({'text':text,'archived':True,'matched':text==protocol['description']})
        if text!=protocol['description']:raise ValueError('No fresh CLIP model or uncached-query fallback')
        return vectors[idx].copy()
    clip_similarity.get_text_embedding=exact_query
    inventory=read(protocol['inventory']);allowed={r['mesh_id'] for r in inventory['local_assets']}
    original_filter=clip_similarity.filter_meshes_by_category;pool=[]
    def available_filter(data,category):
        indices=original_filter(data,category)
        retained=[i for i in indices if data.embedding_index[i] in allowed]
        pool.append({'category':category,'original':len(indices),'available':len(retained)})
        return retained
    clip_similarity.filter_meshes_by_category=available_filter
    actor=None;actor_log=None;http=None;worker=None;app=None;manager=None;vlm=None
    blender=BlenderServer(port_range=(18400,18410),gpu_id=None,log_file=out/'blender.log')
    collision=ConvexDecompositionServer(port_range=(18420,18430),omp_threads=1,log_file=out/'collision.log')
    result={'status':'started','full_runtime_ready':False,'whole_scene_success':None,
            'paid_api_calls':0,'training_updates':0,'new_complete_scenes':0}
    started=time.monotonic()
    def deadline(signum,frame):raise TimeoutError('300-second single live-manager qualification cap')
    old_alarm=signal.signal(signal.SIGALRM,deadline);signal.alarm(300)
    try:
        port=None
        for candidate in range(18440,18450):
            with socket.socket() as sock:
                try:sock.bind(('127.0.0.1',candidate));port=candidate;break
                except OSError:continue
        assert port
        allowed_ports.add(port)
        child_env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        child_env.pop('PYTHONPATH',None);child_env.pop('CUDA_VISIBLE_DEVICES',None)
        actor_log=(out/'model_actor.log').open('w')
        actor=subprocess.Popen([protocol['model_python'],'-B',str(repo/'scripts/autodl/local_vlm_rpc.py'),
            '--port',str(port),'--out',str(out/'model_actor'),'--maximum-calls','1'],
            stdout=actor_log,stderr=subprocess.STDOUT,env=child_env,start_new_session=True)
        dump(out/'actor_process.json',{'pid':actor.pid,'port':port})
        vlm=LocalVLMClient(port,timeout=180)
        t=time.monotonic()
        while True:
            assert actor.poll() is None
            try:
                if vlm.health()['status']=='ready':break
            except requests.RequestException:pass
            if time.monotonic()-t>20:raise TimeoutError('local VLM actor readiness')
            time.sleep(.1)
        app=HssdRetrievalApp(preload_retriever=True,hssd_data_path=protocol['data'],
            hssd_preprocessed_path=protocol['preprocessed'],hssd_top_k=5,clip_device='cpu')
        http=make_server('127.0.0.1',0,app,threaded=True)
        allowed_ports.add(http.server_port)
        worker=Thread(target=http.serve_forever,daemon=True);worker.start();app.start_processing()
        blender.start();allowed_ports.add(blender._actual_port);blender.wait_until_ready()
        collision.start();allowed_ports.add(collision._actual_port);collision.wait_until_ready()
        dump(out/'service_processes.json',{'blender_pid':blender._server_process.pid,
            'collision_pid':collision._server_process.pid,'hssd_http_port':http.server_port})
        cfg=compose_native(stage/'source',repo/'runs/autodl/20261110_native_service_contract/inputs/integration').furniture_agent
        cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        with open_dict(cfg):
            cfg.asset_manager.general_asset_source='hssd';cfg.asset_manager.router.enabled=False
            cfg.asset_manager.router.strategies.thin_covering.generator.enabled=False
            cfg.context_image_generation.enabled=False
        with bind_retrieval_without_images(AssetManager,receipt=out/'image_binding.json'):
            manager=AssetManager(logger=ConsoleLogger(output_dir=out/'manager'),vlm_service=vlm,
                blender_server=blender,collision_client=collision.get_client(),cfg=cfg,
                agent_type=AgentType.FURNITURE,hssd_server_port=http.server_port)
            request=AssetGenerationRequest(object_descriptions=[protocol['description']],
                short_names=['nightstand'],object_type=ObjectType.FURNITURE,
                desired_dimensions=[protocol['desired_dimensions']],scene_id='single_native_manager_qualification')
            t=time.monotonic();generated=manager.generate_assets(request)
            result.update(manager_request_seconds=time.monotonic()-t,
                successful_assets=len(generated.successful_assets),failed_assets=[r.__dict__ for r in generated.failed_assets],
                native_all_succeeded_property=generated.all_succeeded,
                actual_registry_size=manager.registry.size(),AST_isolated_native_methods=False)
            dump(out/'native_asset_result.json',{'objects':[r.to_dict() for r in generated.successful_assets],
                'failed':result['failed_assets']})
            if len(generated.successful_assets)!=1 or generated.failed_assets:
                raise RuntimeError('The fixed mandatory asset was not completed; no retry')
            obj=generated.successful_assets[0]
            from pydrake.multibody.plant import MultibodyPlant
            from pydrake.geometry import SceneGraph
            from pydrake.multibody.parsing import Parser
            plant=MultibodyPlant(.001);sg=SceneGraph();plant.RegisterAsSourceForSceneGraph(sg)
            models=Parser(plant).AddModels(str(obj.sdf_path));plant.Finalize()
            body=plant.GetBodyByName('base_link',models[0])
            result.update(status='normal_native_manager_completed_real_asset_request',asset_id=str(obj.object_id),
                source_hssd_id=obj.metadata['hssd_mesh_id'],sdf_path=str(obj.sdf_path),
                drake_models=len(models),drake_collision_geometries=plant.num_collision_geometries(),
                drake_visual_geometries=len(plant.GetVisualGeometriesForBody(body)),
                estimated_mass_kg=body.default_mass(),inertia_valid=body.default_spatial_inertia().IsPhysicallyValid(),
                physical_properties_measured=False,functional_front_verified=False)
        assert len(replay)==1 and app._total_requests==1
        result['local_vlm_actor_state']=vlm.health()
    except BaseException as exc:
        result.update(status='live_manager_failure_no_retry',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm)
        if vlm:vlm.close()
        blender.stop();collision.stop()
        if http:http.shutdown();http.server_close()
        if worker:worker.join(timeout=5)
        if app:app.stop_processing()
        if actor:
            if actor.poll() is None:
                os.killpg(actor.pid,signal.SIGTERM)
                try:actor.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(actor.pid,signal.SIGKILL);actor.wait()
            result['actor_terminal']=actor.poll() is not None
        if actor_log:actor_log.close()
        clip_similarity.get_text_embedding=original_embedding;clip_similarity.filter_meshes_by_category=original_filter
        result.update(seconds=time.monotonic()-started,parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            archived_embedding_replays=replay,fresh_CLIP_calls=0,available_pool=pool,
            external_network_attempts=network,source_unchanged=all(sha(stage/'source'/r['path'])==r['actual_sha256'] for r in sources['files']),
            no_general_fresh_text_retrieval_claim=True)
        dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
