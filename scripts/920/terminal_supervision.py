"""Explicit prospective <done> serialization; historical OTA is unchanged.

The local upstream loader reads observed terminal turns, but its OTA serializer
wraps content in <think>. This adapter deliberately emits the deployed <done>
control format. Terminal tokens belong to the control/action objective, never
to the professional-decision objective. It does not authorize training.
"""
import json


def terminal_text(target):
    if set(target) != {'requested_mode', 'measured_scope', 'unmet_in_measured_scope',
                       'unknowns', 'whole_room_success'}:
        raise ValueError('Unexpected terminal target fields')
    if (not target['requested_mode'] or not target['measured_scope'] or
            target['unmet_in_measured_scope'] != [] or not target['unknowns'] or
            target['whole_room_success'] != 'unknown'):
        raise ValueError('Terminal target must delimit its satisfied scope and unknowns')
    content = json.dumps(target, ensure_ascii=False, separators=(',', ':'))
    if any(tag in content for tag in ('<tool_call>', '<done>', '</done>', '<think>')):
        raise ValueError('Control delimiter in terminal payload')
    return '<done>' + content + '</done>'


def encode_terminal(request, sample, target, processor, max_length=16384):
    from PIL import Image
    import torch
    text = terminal_text(target)
    if sample.tool_calls or sample.assistant_content != text:
        raise ValueError('Official sample differs from terminal candidate')
    paths = [part['image'] for message in request if isinstance(message['content'], list)
             for part in message['content'] if part['type'] == 'image']
    if len(paths) != 5 or paths != list(map(str, sample.scene_images)):
        raise ValueError('Terminal observation mismatch')
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert('RGB'))
    messages = [*request, {'role': 'assistant', 'content': text}]
    def encoded(messages, generation):
        return processor(text=[processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=generation)], images=images, return_tensors='pt', padding=False)
    full, prompt = encoded(messages, False), encoded(request, True)
    boundary = prompt['input_ids'].shape[1]
    if not boundary < full['input_ids'].shape[1] <= max_length:
        raise ValueError('Terminal sequence outside frozen length budget')
    if not torch.equal(prompt['input_ids'], full['input_ids'][:, :boundary]):
        raise ValueError('Terminal prompt differs from deployed generation prefix')
    out = {k: v[0] if k in ('input_ids', 'attention_mask', 'mm_token_type_ids') else v
           for k, v in full.items()}
    out['labels'] = out['input_ids'].clone()
    out['labels'][:boundary] = -100
    control = out['labels'] != -100
    decision = torch.zeros_like(control)
    decoded = processor.tokenizer.decode(out['labels'][control], skip_special_tokens=False)
    if text not in decoded or '<tool_call>' in decoded or '<think>' in decoded:
        raise ValueError('Terminal target serialization changed')
    return out, control, decision, boundary
