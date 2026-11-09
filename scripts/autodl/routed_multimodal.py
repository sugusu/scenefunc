"""CPU material encoder for task-routed D and A objectives; no trainer/model.

The same encoder constructs training and prospective deployment prefixes.
Training targets are the frozen token arrays, not newly synthesized answers.
"""
import copy
import hashlib
from pathlib import Path

from audit_material_routing import tagged_request, action_prefix


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def resolve_messages(messages, image_refs):
    """Preserve every message and view order, rewriting only exact image paths."""
    output = copy.deepcopy(messages)
    observed = []
    for message in output:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get('type') != 'image':
                continue
            key = 'image' if 'image' in item else 'path'
            source = item[key]
            slot = len(observed)
            if slot >= len(image_refs) or source != image_refs[slot]['path']:
                raise ValueError('Corpus and request image order differ')
            match = image_refs[slot].get('match')
            if not match:
                raise FileNotFoundError('Missing original image: ' + source)
            target = Path(match['local_path'])
            if hashlib.sha256(target.read_bytes()).hexdigest() != image_refs[slot]['sha256']:
                raise ValueError('Image SHA256 changed: ' + str(target))
            item[key] = str(target)
            observed.append(str(target))
    if len(observed) != 5 or len(image_refs) != 5:
        raise ValueError('Five source views are required, without substitution')
    return output, observed


def encode_prefix(processor, messages, images, assistant_prefix=''):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text + assistant_prefix], images=images,
                     return_tensors='pt', padding=False)


def encode_component(processor, component, messages, images, max_length=16384):
    import torch
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = prompt_text + component['assistant_prefix']
    full = processor(text=[prefix + component['target_text']], images=images,
                     return_tensors='pt', padding=False)
    prompt = processor(text=[prefix], images=images, return_tensors='pt', padding=False)
    boundary = prompt['input_ids'].shape[1]
    if full['input_ids'].shape[1] > max_length:
        raise ValueError('Sequence exceeds the frozen limit; no truncation allowed')
    if not torch.equal(full['input_ids'][:, :boundary], prompt['input_ids']):
        raise ValueError('Multimodal prompt/target boundary mismatch')
    expected = torch.tensor(component['target_token_ids'], dtype=torch.long)
    if not torch.equal(full['input_ids'][0, boundary:], expected):
        raise ValueError('Actual multimodal supervised target differs from source')
    for name in ('pixel_values', 'image_grid_thw'):
        if not torch.equal(full[name], prompt[name]):
            raise ValueError('Target changed image encoding')
    labels = full['input_ids'].clone()
    labels[:, :boundary] = -100
    full['labels'] = labels
    for name in ('decision', 'action'):
        full[name + '_mask'] = (labels != -100) if component['component'] == name else torch.zeros_like(labels, dtype=torch.bool)
    # Includes image tokens, system tags, and (for paired A) the D prefill.
    if bool((labels[:, :boundary] != -100).any()):
        raise ValueError('Prompt unexpectedly supervised')
    if not torch.equal(full['action_mask'] | full['decision_mask'], labels != -100):
        raise ValueError('Objective masks do not partition the targets')
    return full, boundary


def deployment_prefix(processor, raw_messages, images, arm, stage, recorded_decision=None):
    """No corpus/gold argument; the caller supplies an actually recorded draft.

    This builds a model input only. It does not assert that a supplied draft is
    correct or that a second model response will have the desired format.
    """
    if stage not in ('decision', 'action'):
        raise ValueError('Unknown stage')
    if arm not in ('separate_tagged', 'paired_tagged'):
        raise ValueError('Unknown arm')
    prefix = ''
    if stage == 'action':
        prefix = action_prefix(arm, model_decision=recorded_decision, evaluation=True)
        if arm == 'paired_tagged':
            prefix = frame_completed_decision(prefix)
    return encode_prefix(processor, tagged_request(raw_messages, stage), images, prefix)


def frame_completed_decision(text):
    """Add the serializer's one separator newline after host stop at </think>.

    This never repairs semantic text or accepts an incomplete decision. The
    newline is explicit host framing, not a newly generated rationale token.
    """
    if (not text.startswith('<think>') or text.count('<think>') != 1
            or text.count('</think>') != 1 or not text.rstrip().endswith('</think>')
            or '<tool_call>' in text or '<|im_end|>' in text):
        raise ValueError('Malformed/missing recorded decision: no free repair or gold fallback')
    return text.rstrip() + '\n'


def require_complete_material(records, expected_components=64):
    """Prevent a partial CPU smoke check from becoming a silently reduced study."""
    if len(records) != expected_components or any(r.get('status') != 'encoded' for r in records):
        raise ValueError('Full fixed corpus is unavailable or failed encoding; do not train a reduced subset')
