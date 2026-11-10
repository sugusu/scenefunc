"""Suppress an unused image constructor in explicit HSSD-only acquisition.

This is an engineering adaptation, not an image backend. It leaves the native
AssetManager constructor and all retrieval/router/physics methods unchanged.
An actual image capability is represented as absent (None). No runtime or scene
quality admission is granted here; the full-generation service guard remains.
"""
from contextlib import contextmanager
import functools
import inspect
import json
from pathlib import Path
import threading
import time

from native_service_contract import selected


_binding_lock = threading.Lock()


def require_retrieval_without_images(cfg):
    if selected(cfg, 'asset_manager.general_asset_source') != 'hssd':
        raise ValueError('Explicit HSSD acquisition required; no text-to-3D substitution')
    if selected(cfg, 'asset_manager.router.strategies.thin_covering.generator.enabled') is not False:
        raise ValueError('Generated-texture fallback must be explicitly disabled')
    if selected(cfg, 'context_image_generation.enabled', False) is not False:
        raise ValueError('Context image generation is not covered by this adapter')
    if selected(cfg, 'asset_manager.image_generation') is None:
        raise ValueError('Native image configuration missing; refusing unknown constructor')
    return dict(asset_source='hssd', generated_texture=False,
                router_enabled=selected(cfg, 'asset_manager.router.enabled'),
                articulated_enabled=selected(cfg, 'asset_manager.router.strategies.articulated.enabled'),
                thin_covering_enabled=selected(cfg, 'asset_manager.router.strategies.thin_covering.enabled'))


@contextmanager
def bind_retrieval_without_images(asset_manager_class, *, receipt):
    """Temporarily wrap one real native class, restoring it even on failure.

Global native factory mutation exists only within a serialized constructor.
Other threads reaching that factory fail explicitly rather than obtaining an
incorrect image object or falling through to a provider. This is not a general
network sandbox and must not be used concurrently with unrelated native runs.
"""
    receipt = Path(receipt)
    if receipt.exists():
        raise FileExistsError(receipt)
    original = asset_manager_class.__init__
    namespace = original.__globals__
    factory = namespace.get('create_image_generator')
    signature = inspect.signature(original)
    if not callable(factory) or 'cfg' not in signature.parameters:
        raise ValueError('Unrecognized native AssetManager constructor')
    if not _binding_lock.acquire(blocking=False):
        raise RuntimeError('Concurrent/nested retrieval binding is unsupported')
    rows = []
    constructor_lock = threading.Lock()
    def save():
        receipt.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(constructors=rows, image_service_implemented=False,
            full_runtime_ready=False, scope='Unused image construction suppression only'), indent=2)+'\n')
        tmp.replace(receipt)
    try:
        @functools.wraps(original)
        def construct(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            cfg = bound.arguments['cfg']
            row = dict(status='started', image_factory_calls=0)
            rows.append(row)
            save()
            tick = time.monotonic()
            try:
                row['configuration'] = require_retrieval_without_images(cfg)
                image_cfg = cfg.asset_manager.image_generation
                owner = threading.get_ident()
                def absent_image_generator(*, backend, config):
                    if threading.get_ident() != owner or config is not image_cfg:
                        raise RuntimeError('Unexpected concurrent image construction')
                    row['image_factory_calls'] += 1
                    if row['image_factory_calls'] != 1:
                        raise RuntimeError('Native image constructor call contract changed')
                    if backend != image_cfg.backend:
                        raise RuntimeError('Native image backend changed during construction')
                    return None
                with constructor_lock:
                    if namespace['create_image_generator'] is not factory:
                        raise RuntimeError('Native image factory changed by another adapter')
                    namespace['create_image_generator'] = absent_image_generator
                    try:
                        original(self, *args, **kwargs)
                    finally:
                        namespace['create_image_generator'] = factory
                if row['image_factory_calls'] != 1 or self.image_generator is not None:
                    raise RuntimeError('Native constructor did not retain explicit absence')
                row['status'] = 'constructed_without_image_service'
            except BaseException as exc:
                row.update(status='raised', error=type(exc).__name__+': '+str(exc))
                raise
            finally:
                row['seconds'] = time.monotonic()-tick
                save()
        asset_manager_class.__init__ = construct
        save()
        yield rows
    finally:
        asset_manager_class.__init__ = original
        if callable(factory):
            namespace['create_image_generator'] = factory
        _binding_lock.release()
