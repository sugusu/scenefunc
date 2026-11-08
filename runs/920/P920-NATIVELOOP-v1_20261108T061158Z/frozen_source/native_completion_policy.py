"""Native OTA response classification and matched-input live reconstruction.

No target, reference endpoint, or aggregate review enters the request. Structural
completion is deliberately separate from measured activity/room acceptance.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

from edit_request import normalize_context
from native_checkpoint import canonical_hash
from native_completion_supervision import build_native_request
from shared_action_contract_v2 import strict_loads, schemas_from_observation


def classify_response(raw, record):
    """Return tool/completion/rejection; never infer completion from no calls."""
    def reject(reason):
        return dict(kind='rejection', reason=reason)
    if record.get('status') != 'completed' or record.get('output_rejected_reason'):
        return reject('Generation failed or was truncated')
    if record.get('raw_output_sha256') != hashlib.sha256(raw.encode()).hexdigest():
        return reject('Raw output does not match generation receipt')
    text = raw.strip()
    eos = next((tag for tag in ('<|im_end|>', '<|endoftext|>') if text.endswith(tag)), None)
    if eos:
        if record.get('ended_with_eos') is not True:
            return reject('Unverified EOS')
        text = text[:-len(eos)].rstrip()
    elif record.get('ended_with_eos'):
        return reject('EOS absent from raw output')
    if any(tag in text for tag in ('<|', '<done', '</done')):
        return reject('Unexpected role or legacy completion delimiter')
    thinking = ''
    if text.startswith('<think>'):
        end = text.find('</think>')
        if end < 0:
            return reject('Unclosed reasoning block')
        thinking, text = text[len('<think>'):end], text[end+len('</think>'):].strip()
        if '<think' in thinking or '</think' in thinking or '<tool_call' in thinking or '</tool_call' in thinking:
            return reject('Nested control block')
    if '<think' in text or '</think' in text:
        return reject('Misplaced reasoning delimiter')
    if '<tool_call' in text or '</tool_call' in text:
        match = re.fullmatch(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if not match or text.count('<tool_call>') != 1 or text.count('</tool_call>') != 1:
            return reject('Expected one complete tool block and no trailing content')
        if not eos and record.get('ended_at_complete_single_action') is not True:
            return reject('Tool block lacks a verified generation boundary')
        try:
            call = strict_loads(match.group(1))
        except ValueError as exc:
            return reject(str(exc))
        if (not isinstance(call, dict) or set(call) != {'name', 'arguments'}
                or not isinstance(call['name'], str) or not isinstance(call['arguments'], dict)):
            return reject('Malformed tool call')
        return dict(kind='tool', call=call, raw_json=match.group(1))
    if not eos or not (thinking.strip() or text.strip()):
        return reject('Empty or unterminated completion')
    return dict(kind='completion', content=(thinking+'\n'+text).strip(),
                whole_room_success='unknown', semantic_success_not_inferred=True)


def build_matched_session_request(session, observation, plan, rounds, audit):
    reference = plan['matched_input']
    path = Path(reference['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Matched input template changed')
    inputs = json.loads(path.read_text())
    state = session.scene.to_state_dict()
    state_hash = canonical_hash(state)
    if audit['state_sha256'] != state_hash:
        raise ValueError('Measurements are not bound to current state')
    context = inputs['context']
    if audit['review']['requested_mode'] != context['current_requested_mode']:
        raise ValueError('Current activity differs from the frozen request')
    available = {t['name']: t for t in schemas_from_observation(observation)}
    for tool in inputs['tools']:
        if tool['name'] not in available or tool['parameters'] != available[tool['name']]['parameters']:
            raise ValueError('Candidate/native advertised tool schemas differ')
    # Keep precisely the observed fields used by training. No raw geometry path
    # or unobserved property is added, and no old pose survives a new observation.
    current = normalize_context({'objects': state['objects']})['objects']
    original = context['objects']
    templates = original if isinstance(original, list) else list(original.values())
    if {o['object_id'] for o in templates} != set(current):
        raise ValueError('Existing-object task identities changed unexpectedly')
    projected = []
    for obj in templates:
        native = current[obj['object_id']]
        if not set(obj) <= set(native):
            raise ValueError('Current state missing a required observed field')
        projected.append({k: copy.deepcopy(native[k]) for k in obj})
    context['objects'] = projected if isinstance(original, list) else {o['object_id']:o for o in projected}
    context['available_measurement'] = copy.deepcopy(audit['measurements'])
    history = inputs['history']
    initial_attempt = inputs['next_attempt']
    if [r['execution']['attempt'] for r in rounds] != list(range(1, len(rounds)+1)):
        raise ValueError('Incomplete live execution history')
    for row in rounds:
        entry = copy.deepcopy(row['execution'])
        entry['attempt'] += initial_attempt - 1
        history.append(entry)
    if history and history[-1]['after_sha256'] != state_hash:
        raise ValueError('History is not bound to the restored/current state')
    inputs['next_attempt'] = initial_attempt + len(rounds)
    inputs['images'] = observation['image_paths']
    request, accounting = build_native_request(**inputs)
    accounting.update(current_native_state_sha256=state_hash,
        inherited_attempts=initial_attempt-1, live_attempts=len(rounds),
        matched_template_sha256=reference['sha256'])
    return request, accounting
