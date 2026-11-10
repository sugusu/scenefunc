"""Owned original HSSD HTTP service using the qualified original CLIP cache.

Ordinary local-service integration, not retrieval/search innovation. No image
API, generated-asset service or suffix remapping. Native embeddings and HTTP
retrieval are retained; unavailable/misaligned suffix assets share one filter.
"""
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
from threading import Thread
import time


def dump(path,v):
    Path(path).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


@contextmanager
def owned_hssd_backend(native, *, receipt, weights, config, inventory, max_cpu_threads=2,
                       allowed_ports=None):
    import torch
    from werkzeug.serving import make_server
    from scenesmith.agent_utils import clip_embeddings
    from scenesmith.agent_utils.hssd_retrieval import clip_similarity
    from scenesmith.agent_utils.hssd_retrieval_server.server_app import HssdRetrievalApp
    from native_clip_runtime import bind_original_clip
    original_class=native.BackendServers
    original_filter=clip_similarity.filter_meshes_by_category
    original_embedding=clip_similarity.get_text_embedding
    if clip_embeddings._cached_model is not None:raise RuntimeError('Unexpected existing native CLIP ownership')
    allowed={r['mesh_id'] for r in json.loads(Path(inventory).read_text())['local_assets'] if Path(r['path']).is_file()}
    pools=[];encodings=[];instances=[]
    torch.set_num_threads(max_cpu_threads)
    started=time.monotonic()
    binding=bind_original_clip(clip_embeddings,weights,config,device='cpu')
    def save():
        dump(receipt,dict(binding=binding,pools=pools,encoder_requests=encodings,
            services=[dict(port=s.port,requests=s.app._total_requests,stopped=s.stopped) for s in instances],
            fixed_excluded_suffix_starts_at=12619,archive_embeddings_used=False,
            paid_API_calls=0,scope='Original HSSD/CLIP local service and common availability filter only'))
    def available(data,category):
        candidates=original_filter(data,category)
        retained=[i for i in candidates if i<12619 and data.embedding_index[i] in allowed]
        pools.append(dict(category=category,original=len(candidates),retained=len(retained)));save()
        return retained
    def encode(text,device=None):
        tick=time.monotonic();value=original_embedding(text,device='cpu')
        encodings.append(dict(text=text,seconds=time.monotonic()-tick,
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),fresh=True));save();return value
    class Backend(original_class):
        def _start_hssd(self):
            for stage in ('furniture','wall','ceiling','manipuland'):
                cfg=self.cfg[stage+'_agent']
                assert cfg.asset_manager.general_asset_source=='hssd'
                assert cfg.asset_manager.router.enabled is False
                assert cfg.asset_manager.router.strategies.articulated.enabled is False
            settings=self.cfg.experiment.hssd_retrieval_server
            assert settings.host=='127.0.0.1'
            cfg=self.cfg.furniture_agent.asset_manager.hssd
            self.app=HssdRetrievalApp(preload_retriever=True,hssd_data_path=str(cfg.data_path),
                hssd_preprocessed_path=str(cfg.preprocessed_path),hssd_top_k=int(cfg.use_top_k),clip_device='cpu')
            self.http=make_server('127.0.0.1',int(settings.port),self.app,threaded=True)
            self.port=self.http.server_port;self.stopped=False
            if allowed_ports is not None:allowed_ports.add(self.port)
            self.worker=Thread(target=self.http.serve_forever,daemon=True);self.worker.start()
            self.app.start_processing();instances.append(self);save()
        def stop_all(self):
            if hasattr(self,'http') and not self.stopped:
                self.http.shutdown();self.http.server_close();self.worker.join(timeout=5)
                self.app.stop_processing();self.stopped=True;save()
            super().stop_all()
    clip_similarity.filter_meshes_by_category=available
    clip_similarity.get_text_embedding=encode
    native.BackendServers=Backend
    try:
        save();yield dict(binding=binding,instances=instances,encoder_requests=encodings)
    finally:
        for service in instances:service.stop_all()
        native.BackendServers=original_class
        clip_similarity.filter_meshes_by_category=original_filter
        clip_similarity.get_text_embedding=original_embedding
        for attr in ('_cached_model','_cached_tokenizer','_cached_preprocess','_device'):
            setattr(clip_embeddings,attr,None)
        gc.collect();save()


@contextmanager
def bounded_native_collision_threads(*,receipt,max_threads=2):
    """Native source derives threads from host count, not the container quota.

    Explicit common resource cap at the real constructor; no fake cpu_count.
    """
    from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionServer
    original=ConvexDecompositionServer.__init__;rows=[]
    def initialize(self,*args,**kwargs):
        if args:raise ValueError('Revalidate positional collision constructor before adapting')
        requested=kwargs.get('omp_threads')
        if requested is not None:kwargs['omp_threads']=min(int(requested),max_threads)
        else:kwargs['omp_threads']=max_threads
        rows.append(dict(requested=requested,actual=kwargs['omp_threads']))
        dump(receipt,rows);return original(self,**kwargs)
    ConvexDecompositionServer.__init__=initialize
    try:yield rows
    finally:ConvexDecompositionServer.__init__=original
