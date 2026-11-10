"""Import/construct the real full native AssetManager, with no AST substitutes.

Every network connection and subprocess launch is prohibited. Retrieval's
unused image constructor is explicitly suppressed by the common adapter; no
real acquisition, VLM response or whole scene is fabricated by this command.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import traceback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--construct',action='store_true')
    a=ap.parse_args();stage=a.stage.resolve();out=a.out.resolve();out.mkdir(exist_ok=False)
    root=Path(__file__).resolve().parents[2];source=stage/'source'
    manifest=json.loads((stage/'source_manifest.json').read_text())
    for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
    attempted=[]
    def prohibit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto','subprocess.Popen'):
            attempted.append(event)
            raise RuntimeError('Network/process launch prohibited in full-module qualification: '+event)
    sys.addaudithook(prohibit)
    sys.path.insert(0,str(source))
    tick=time.monotonic();result={'status':'started','model_calls':0,'paid_api_calls':0,
        'training_updates':0,'new_complete_scenes':0,'full_runtime_ready':False}
    try:
        from scenesmith.agent_utils.asset_manager import AssetManager
        from scenesmith.agent_utils.room import AgentType
        from scenesmith.utils.logging import ConsoleLogger
        result.update(status='native_full_module_imported',class_module=AssetManager.__module__,
            class_source_sha256=sha(source/'scenesmith/agent_utils/asset_manager.py'),
            AST_isolated_classes=False)
        if a.construct:
            from audit_native_services import compose_native
            from omegaconf import OmegaConf,open_dict
            from retrieval_image_binding import bind_retrieval_without_images
            cfg=compose_native(source,root/'runs/autodl/20261110_native_service_contract/inputs/integration').furniture_agent
            cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
            with open_dict(cfg):
                cfg.asset_manager.general_asset_source='hssd'
                cfg.asset_manager.router.strategies.thin_covering.generator.enabled=False
                cfg.context_image_generation.enabled=False
            # Native text/image service object exists, but the processor/model
            # is deliberately not initialized and no completion is requested.
            work=Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
            sys.path.insert(0,str(work/'paris_local_run_results_20261104'))
            from local_vlm_sdpa import LocalQwenEngine,LocalVLMService
            engine=LocalQwenEngine(work/'critic_public_runtime_20261104/checkpoint',out/'unused_model_calls')
            service=LocalVLMService(engine)
            logger=ConsoleLogger(output_dir=out/'native_manager')
            original=AssetManager.__init__
            with bind_retrieval_without_images(AssetManager,receipt=out/'image_binding.json'):
                manager=AssetManager(logger=logger,vlm_service=service,blender_server=None,
                    collision_client=None,cfg=cfg,agent_type=AgentType.FURNITURE)
                assert manager.image_generator is None
                registry_path=root/'runs/autodl/20261110_native_physics_qualification/asset_v2/registry.json'
                before=sha(registry_path)
                manager.registry.load_from_file(registry_path)
                objects=manager.list_available_assets()
                assert len(objects)==1
                assert manager.get_asset_by_id(objects[0].object_id) is objects[0]
                manager.registry.save_to_file(out/'restored_registry.json')
                assert sha(registry_path)==before and engine._model is None and engine._processor is None
                result.update(status='native_full_module_constructed_and_asset_restored',
                    registry_objects=len(objects),restored_object_id=str(objects[0].object_id),
                    real_hssd_client_type=type(manager.hssd_client).__module__+'.'+type(manager.hssd_client).__name__,
                    real_router_type=type(manager.router).__module__+'.'+type(manager.router).__name__,
                    articulated_client_constructed=manager.articulated_client is not None,
                    thin_covering_materials_client_constructed=manager.materials_client is not None,
                    no_new_estimate=True,source_registry_unchanged=True,model_loaded=False,
                    scope='Full normal import and real constructor using explicit image absence; one already generated asset restored. No retrieval/server/renderer/SDK Runner or new scene executed.')
                for client in (manager.hssd_client,manager.articulated_client,manager.materials_client):
                    if client is not None and getattr(client,'session',None) is not None:client.session.close()
            assert AssetManager.__init__ is original
            result['image_binding_restored']=True
        assert not attempted
    except BaseException as exc:
        result.update(status='native_import_or_constructor_failure',error=type(exc).__name__+': '+str(exc))
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        modules=[]
        for name,mod in list(sys.modules.items()):
            path=getattr(mod,'__file__',None)
            if name.startswith('scenesmith') and path:
                p=Path(path).resolve();modules.append({'name':name,'path':str(p),'sha256':sha(p)})
        result.update(seconds=time.monotonic()-tick,network_or_subprocess_attempts=attempted,
            native_modules_loaded=len(modules),source_unchanged=all(sha(source/r['path'])==r['sha256'] for r in manifest['files']),
            parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        dump(out/'loaded_native_modules.json',modules);dump(out/'result.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
