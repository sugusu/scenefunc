"""Native Qwen3-VL fixed-response scoring inputs; no model/training entrypoint.

An ordinary multimodal SDPO-style transport port. The caller must separately
authorize data/optimization and freeze the teacher. Nothing here certifies
feedback correctness or implements the original distributed SDPO trainer.
"""
from types import SimpleNamespace


def make_scoring_inputs(processor_inputs, response, *, config, max_sequence_tokens=16384):
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    if response.ndim != 2 or response.shape[0] != 1 or response.shape[1] == 0:
        raise ValueError('Expected one nonempty exact response sequence')
    prefix = processor_inputs['input_ids']
    if prefix.ndim != 2 or prefix.shape[0] != 1:
        raise ValueError('Only unpadded, single-example CPU input port is qualified')
    if not processor_inputs['attention_mask'].bool().all():
        raise ValueError('Padding requires separate position/normalization qualification')
    full_ids = torch.cat([prefix, response], dim=1)
    if full_ids.shape[1] > max_sequence_tokens:
        raise ValueError('Combined teacher/student input exceeds cap; no truncation')
    full_mask = torch.cat([processor_inputs['attention_mask'], torch.ones_like(response)], dim=1)
    kwargs = dict(processor_inputs, input_ids=full_ids, attention_mask=full_mask)
    # Unmodified library method needs only config/tensors, not model weights.
    owner = SimpleNamespace(config=config)
    position_ids, deltas = Qwen3VLModel.get_rope_index(
        owner, input_ids=full_ids, image_grid_thw=kwargs.get('image_grid_thw'),
        video_grid_thw=kwargs.get('video_grid_thw'), attention_mask=full_mask)
    prefix_positions, _ = Qwen3VLModel.get_rope_index(
        owner, input_ids=prefix, image_grid_thw=kwargs.get('image_grid_thw'),
        video_grid_thw=kwargs.get('video_grid_thw'), attention_mask=processor_inputs['attention_mask'])
    assert torch.equal(position_ids[:, :, :prefix.shape[1]], prefix_positions)
    assert position_ids.shape == (3, 1, full_ids.shape[1])
    if 'image_grid_thw' in kwargs:
        merge = config.vision_config.spatial_merge_size
        image_tokens = int((kwargs['image_grid_thw'].prod(-1) // merge**2).sum())
        if int((full_ids == config.image_token_id).sum()) != image_tokens:
            raise ValueError('Image placeholder/feature-grid count mismatch')
    target_pos = position_ids[:, :, prefix.shape[1]:]
    if not (torch.equal(target_pos[0], target_pos[1]) and torch.equal(target_pos[1], target_pos[2])):
        raise ValueError('This text/native-call response port does not admit generated vision tokens')
    if target_pos.shape[-1] > 1 and not ((target_pos[..., 1:] - target_pos[..., :-1]) == 1).all():
        raise ValueError('Response rotary positions must advance one per text token')
    kwargs['position_ids'] = position_ids
    kwargs['logits_to_keep'] = torch.arange(prefix.shape[1]-1, full_ids.shape[1]-1)
    kwargs['use_cache'] = False
    return kwargs, dict(prompt_tokens=prefix.shape[1], response_tokens=response.shape[1],
                        rope_deltas=deltas, image_grid_thw=kwargs.get('image_grid_thw'),
                        model_forward_executed=False, training_admitted=False)
