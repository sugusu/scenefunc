"""Explicit local-file binding for the original native CLIP module cache.

This helper does not select assets, return archived embeddings, alter native
embedding functions, download anything, or authorize other inference stages.
"""
import hashlib
import json
from pathlib import Path
import time

MODEL = 'ViT-H-14-378-quickgelu'
WEIGHT_SHA = 'c07a17b547d461c60a3cce5062b26bf8545b13de602c4c59d8490361eb716033'
CONFIG_SHA = 'e043bd1129b97d3afc77eef2e8bfda940af588dabf2cbaacc21e956345402f67'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def bind_original_clip(native_module, weights, config, device='cpu'):
    """Bind the same original parameters to native get_text/image_embedding.

    A new process must explicitly choose this local binding. It cannot silently
    reuse an existing different cache or fall back to remote initialization.
    """
    started = time.monotonic()
    weights, config = Path(weights), Path(config)
    if sha(weights) != WEIGHT_SHA or sha(config) != CONFIG_SHA:
        raise ValueError('Original authorized checkpoint/config hash differs')
    if native_module._cached_model is not None:
        raise ValueError('Existing native CLIP cache must be audited separately')
    import open_clip
    cfg = json.loads(config.read_text())
    registered = open_clip.get_model_config(MODEL)
    if any(registered.get(k) != v for k,v in cfg['model_cfg'].items()):
        raise ValueError('Installed OpenCLIP architecture differs from original config')
    pre = cfg['preprocess_cfg']
    model,_,preprocess = open_clip.create_model_and_transforms(
        MODEL,pretrained=str(weights),device=device,
        image_mean=pre['mean'],image_std=pre['std'],
        image_interpolation=pre['interpolation'],image_resize_mode=pre['resize_mode'])
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL)
    native_module._cached_model = model
    native_module._cached_tokenizer = tokenizer
    native_module._cached_preprocess = preprocess
    native_module._device = device
    return {
        'model':MODEL,'open_clip_version':open_clip.__version__,
        'weight_sha256':WEIGHT_SHA,'config_sha256':CONFIG_SHA,'device':device,
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'source_module_path':str(Path(native_module.__file__).resolve()),
        'source_module_sha256':sha(native_module.__file__),
        'native_functions_replaced':False,'archive_embedding_provider':False,
        'explicit_local_model_cache_binding':True,'official_preprocess_cfg':pre,
        'load_and_verify_seconds':time.monotonic()-started,
    }
