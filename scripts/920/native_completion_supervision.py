"""Prospective native OTA completion; closed experiments remain unchanged.

The shared request transformer changes only the completion protocol sentence.
Terminal scope/unknowns use the official assistant template, and all terminal
tokens belong to control supervision, not sourced professional decisions.
"""
import copy
import json


OLD_PROTOCOL = ('Then emit exactly one <tool_call>{"name":"...","arguments":{...}}</tool_call>, '
                'or <done> with unmet/unknown requirements and no call. ')
NATIVE_PROTOCOL = ('Then emit exactly one <tool_call>{"name":"...","arguments":{...}}</tool_call> '
                   'if an edit is needed. Otherwise finish the assistant response without any tool call, '
                   'stating the measured scope and remaining unmet or unknown requirements. ')


def native_request(request):
    """Target-free conversion also used by the prospective deployment builder."""
    if not request or request[0].get('role') != 'system':
        raise ValueError('Expected shared system request')
    system = request[0].get('content')
    if not isinstance(system, str) or system.count(OLD_PROTOCOL) != 1:
        raise ValueError('Unknown or already converted completion protocol')
    result = copy.deepcopy(request)
    result[0]['content'] = system.replace(OLD_PROTOCOL, NATIVE_PROTOCOL, 1)
    return result


def build_native_request(**inputs):
    from executed_edit_request import build_executed_request
    request, accounting = build_executed_request(**inputs)
    return native_request(request), accounting


def completion_content(target):
    # Retain historical field validation without retaining its control markers.
    from terminal_supervision import terminal_text
    terminal_text(target)
    return json.dumps(target, ensure_ascii=False, separators=(',', ':'))


def encode_native_completion(request, sample, target, processor, official, max_length):
    from PIL import Image
    import torch
    content = completion_content(target)
    if sample.tool_calls or sample.assistant_content != content:
        raise ValueError('Official sample differs from native completion record')
    official_full, _ = official._build_messages(sample, no_thinking=False)
    assistant = official_full[-1]
    if assistant.get('tool_calls') or assistant['content'] != f'<think>\n{content}\n</think>':
        raise ValueError('Pinned official completion serialization changed')
    paths = [p['image'] for m in request if isinstance(m['content'], list)
             for p in m['content'] if p['type'] == 'image']
    if len(paths) != 5 or paths != list(map(str, sample.scene_images)):
        raise ValueError('Completion observation mismatch')
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert('RGB'))
    def encode(messages, generation):
        return processor(text=[processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=generation)], images=images, return_tensors='pt', padding=False)
    full, prompt = encode([*request, assistant], False), encode(request, True)
    boundary = prompt['input_ids'].shape[1]
    if not boundary < full['input_ids'].shape[1] <= max_length:
        raise ValueError('Completion sequence exceeds frozen length budget')
    if not torch.equal(prompt['input_ids'], full['input_ids'][:, :boundary]):
        raise ValueError('Training prefix differs from deployment')
    ex = {k: v[0] if k in ('input_ids', 'attention_mask', 'mm_token_type_ids') else v
          for k, v in full.items()}
    ex['labels'] = ex['input_ids'].clone()
    ex['labels'][:boundary] = -100
    control = ex['labels'] != -100
    decision = torch.zeros_like(control)
    target_ids = ex['labels'][control].tolist()
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    newline = processor.tokenizer.encode('\n', add_special_tokens=False)
    if target_ids.count(eos) != 1 or target_ids[-(1 + len(newline)):] != [eos, *newline]:
        raise ValueError('Completion EOS missing or not supervised')
    decoded = processor.tokenizer.decode(target_ids, skip_special_tokens=False)
    if '<done>' in decoded or '<tool_call>' in decoded or content not in decoded:
        raise ValueError('Native completion target changed')
    return ex, control, decision, boundary
