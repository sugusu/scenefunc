"""Exercise exact native constructors with a real local HTTP client object.

Definitions are loaded without the full module's unused server/model imports.
No retrieval response, geometry, model decision or successful scene is faked.
This establishes construction/restoration only, not an operational asset server.
"""
import argparse
import ast
import enum
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import resource
import sys
import time
from types import ModuleType, SimpleNamespace

from omegaconf import OmegaConf, open_dict

from audit_native_services import compose_native
from retrieval_image_binding import bind_retrieval_without_images


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes, type_ignores=[])),
                 str(path), 'exec'), namespace)


def main(a):
    start = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=False)
    old = [a.manager, a.registry, a.room, a.router, a.factory]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in old}
    network = []
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            network.append(event)
            raise RuntimeError('Network prohibited during constructor qualification')
    sys.addaudithook(audit)
    package = ModuleType('qualified_hssd')
    package.__path__ = [str(a.client.parent.resolve())]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location('qualified_hssd.client', a.client)
    client = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = client
    spec.loader.exec_module(client)
    module = ModuleType('qualified_native_asset_definitions')
    sys.modules[module.__name__] = module
    g = module.__dict__
    g.update(Path=Path, Enum=enum.Enum, console_logger=logging.getLogger('native_ctor'),
             HssdRetrievalClient=client.HssdRetrievalClient)
    definitions(a.room, ['ObjectType', 'AgentType'], g)
    definitions(a.registry, ['AssetRegistry'], g)
    definitions(a.router, ['AssetRouter'], g)
    definitions(a.factory, ['create_image_generator'], g)
    definitions(a.manager, ['AssetManager'], g)
    cls = g['AssetManager']
    original_init, original_factory = cls.__init__, g['create_image_generator']
    full = compose_native(a.configs, a.integration)
    cfg = OmegaConf.create(OmegaConf.to_container(full.furniture_agent, resolve=True))
    with open_dict(cfg):
        cfg.asset_manager.general_asset_source = 'hssd'
        cfg.asset_manager.router.strategies.thin_covering.generator.enabled = False
        # These services are not constructed in this probe. No claim that
        # disabling them is an admitted final study configuration.
        cfg.asset_manager.router.strategies.thin_covering.enabled = False
        cfg.asset_manager.router.strategies.articulated.enabled = False
    results = []
    def build(config, directory):
        return cls(logger=SimpleNamespace(output_dir=directory), vlm_service=None,
            blender_server=None, collision_client=None, cfg=config,
            agent_type=g['AgentType'].FURNITURE)
    with bind_retrieval_without_images(cls, receipt=a.out/'binding.json') as rows:
        for enabled in (True, False):
            config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            config.asset_manager.router.enabled = enabled
            instance = build(config, a.out/('router_on' if enabled else 'router_off'))
            assert instance.image_generator is None
            assert type(instance.hssd_client) is client.HssdRetrievalClient
            assert type(instance.registry) is g['AssetRegistry'] and instance.registry.size() == 0
            assert (instance.router is not None) == enabled
            assert g['create_image_generator'] is original_factory
            results.append(dict(router_enabled=enabled, image_generator=None,
                client_type=type(instance.hssd_client).__name__, client_url=instance.hssd_client.base_url,
                registry_size=instance.registry.size(), directories_created=all(p.is_dir() for p in
                    (instance.images_dir, instance.geometry_dir, instance.sdf_dir, instance.debug_dir))))
            # No server response or asset is supplied. Missing real collision
            # service must still prevent acquisition, not count as success.
            if not enabled:
                try:
                    instance._retrieve_hssd_assets(None)
                except RuntimeError as exc:
                    assert 'Collision client not available' in str(exc)
                    results[-1]['physical_dependency_still_required'] = True
                else:
                    raise AssertionError('Native asset processing bypassed')
            instance.hssd_client.session.close()
        for kind in ('generated_source', 'texture_fallback'):
            invalid = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            if kind == 'generated_source':
                invalid.asset_manager.general_asset_source = 'generated'
            else:
                invalid.asset_manager.router.strategies.thin_covering.generator.enabled = True
            try:
                build(invalid, a.out/kind)
            except ValueError as exc:
                results.append(dict(rejected=kind, error=str(exc)))
            else:
                raise AssertionError('Unavailable image capability admitted')
        bad_path = a.out/'not_a_directory'
        bad_path.write_text('Intentional constructor filesystem failure\n')
        try:
            build(cfg, bad_path)
        except NotADirectoryError:
            assert g['create_image_generator'] is original_factory
            results.append(dict(native_failure='NotADirectoryError', factory_restored=True))
        else:
            raise AssertionError('Native filesystem error not propagated')
    assert cls.__init__ is original_init and g['create_image_generator'] is original_factory
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    assert not network
    summary = dict(status='passed', actual_native_constructor_returns=2,
        exact_definitions=['AssetManager', 'AssetRegistry', 'AssetRouter', 'ObjectType', 'AgentType',
                           'create_image_generator'], actual_hssd_client_objects=2,
        model_calls=0, paid_api_calls=0, retrieval_requests=0, assets_created=0,
        new_complete_scenes=0, old_sources_unchanged=True, source_hashes=hashes,
        tests=results, network_attempts=network, full_runtime_ready=False,
        seconds=time.perf_counter()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitation='Exact native class definitions loaded via AST; real client objects, '
        'no full-module import/server/render/geometry. Articulated and thin-covering '
        'services disabled in this constructor probe only. Shared VLM was not invoked.')
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('manager', 'registry', 'room', 'router', 'factory', 'client', 'configs', 'integration', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    main(ap.parse_args())
