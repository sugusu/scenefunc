"""Inspect native configuration without constructing any model or service.

This detects known external image paths; it is NOT a complete egress sandbox,
asset qualification, or a proof that a scene task can be completed. Text VLM
replacement does not implement an image-generation backend.
"""
import json

STAGES = ('floor_plan', 'furniture', 'wall', 'ceiling', 'manipuland')


def selected(cfg, path, default=None):
    node = cfg
    for key in path.split('.'):
        if not hasattr(node, 'get'):
            return default
        node = node.get(key, default)
        if node is None:
            return default
    return node


def inspect_config(cfg):
    """Only allowlisted non-secret fields are returned; never dump environment."""
    rows = []
    for stage in STAGES:
        section = stage + '_agent'
        if selected(cfg, section) is None:
            raise ValueError('Missing native agent section: ' + section)
        am = section + '.asset_manager.'
        row = dict(stage=stage,
            session_summarization=selected(cfg, section+'.session_memory.enable_summarization'),
            context_images=selected(cfg, section+'.context_image_generation.enabled', False))
        if stage == 'floor_plan':
            row['materials_retrieval'] = selected(cfg, section+'.materials.use_retrieval_server')
        else:
            row.update(asset_source=selected(cfg, am+'general_asset_source'),
                geometry_backend=selected(cfg, am+'backend'),
                image_backend=selected(cfg, am+'image_generation.backend'),
                router=selected(cfg, am+'router.enabled', False),
                articulated=selected(cfg, am+'router.strategies.articulated.enabled', False),
                thin_covering=selected(cfg, am+'router.strategies.thin_covering.enabled', False),
                generated_texture=selected(cfg, am+'router.strategies.thin_covering.generator.enabled', False))
        rows.append(row)
    return rows


def reject_known_external_images(cfg, *, active_stages):
    """Fail before service startup, including retrieval's eager image constructor.

    The pinned AssetManager constructs the image backend even if it is never
    used. Therefore retrieval-only and disabled texture fallback do not bypass
    this check. Unknown backends also fail; a future local backend needs an
    actual implementation and separate qualification, not a renamed setting.
    """
    if not active_stages or len(set(active_stages)) != len(active_stages):
        raise ValueError('Explicit nonempty unique stage sequence required')
    if set(active_stages) - set(STAGES):
        raise ValueError('Unknown native stage')
    rows = inspect_config(cfg)
    blocked = [dict(stage=r['stage'], backend=r['image_backend'],
                    asset_source=r['asset_source'])
               for r in rows if r['stage'] in active_stages and r['stage'] != 'floor_plan']
    if blocked:
        raise ValueError('Native image service is not locally qualified; eager '
                         'construction is outside the local VLM binding: '
                         + json.dumps(blocked, sort_keys=True))
    return rows


def preflight_native_services(native_module, args):
    """Use the native composition function before main_async creates services.

    Check all five configured stages, including the common manipuland callback.
    Config composition itself loads no model and starts no service in the
    audited native source. A missing composer is not treated as qualification.
    """
    composer = getattr(native_module, 'compose_full_cfg', None)
    if not callable(composer):
        raise ValueError('Native config composer unavailable before startup')
    cfg, _ = composer(args.config, args.results_dir)
    return reject_known_external_images(cfg, active_stages=STAGES)
