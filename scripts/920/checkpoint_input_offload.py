"""CPU storage for language-layer checkpoint inputs; no model/data changes.

Install after Trainer enables checkpointing, before the first forward. The
checkpoint's inner saved-tensor hook continues to handle layer intermediates;
the surrounding save_on_cpu scope handles tensors saved at its entry only.
"""
from functools import wraps
import re

import torch


def install_checkpoint_input_offload(model, expected_layers):
    layers = [(n, m) for n, m in model.named_modules()
              if re.search(r'(?:^|\.)language_model\.layers\.\d+$', n)]
    if len(layers) != expected_layers:
        raise ValueError('Unexpected number of language decoder layers')
    # Validate the whole set before replacing any callable.
    for name, layer in layers:
        fn = getattr(layer, '_gradient_checkpointing_func', None)
        if (not getattr(layer, 'gradient_checkpointing', False) or fn is None
                or getattr(fn, '_checkpoint_input_cpu', False)
                or getattr(fn, 'keywords', {}).get('use_reentrant') is not False):
            raise ValueError('Expected unwrapped non-reentrant checkpoint: ' + name)
    stats = {'installed_layers': len(layers), 'checkpoint_forward_calls': 0,
             'input_bytes_seen': 0, 'pin_memory': False,
             'input_bytes_are_not_measured_memory_savings': True}

    def wrap(fn):
        @wraps(fn)
        def call(function, *args, **kwargs):
            stats['checkpoint_forward_calls'] += 1
            stats['input_bytes_seen'] += sum(x.numel() * x.element_size()
                                            for x in args if isinstance(x, torch.Tensor))
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                return fn(function, *args, **kwargs)
        call._checkpoint_input_cpu = True
        return call

    for _, layer in layers:
        layer._gradient_checkpointing_func = wrap(layer._gradient_checkpointing_func)
    return stats
