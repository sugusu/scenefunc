"""Shared, observation-only evidence envelope for training and live deployment.

This module has no teacher target, policy, renderer, model or geometry execution.
The caller must supply actual BEFORE-action state and actual receipts only.
It does not make different scene tasks or source prompts distribution-identical.
"""
import copy
import json

HOST_MARKER = 'Current host evidence (not task completion):\n'
TOOLS_MARKER = 'Available executable tools:\n'
STATE_MARKER = 'CURRENT NATIVE STATE:\n'


def text_slots(messages):
    for message in messages:
        if isinstance(message.get('content'), str):
            yield message, 'content'
        elif isinstance(message.get('content'), list):
            for part in message['content']:
                if part.get('type') == 'text':
                    yield part, 'text'


def _validate(messages, schemas, history):
    if any(HOST_MARKER in slot[key] for slot, key in text_slots(messages)):
        raise ValueError('Host evidence already attached; do not duplicate it')
    if not isinstance(schemas, dict) or not schemas:
        raise ValueError('Actual nonempty tool schemas required')
    if not isinstance(history, list):
        raise ValueError('Host history must be a receipt list')


def visible_scene_state(state):
    source = state['objects']
    objects = list(source.values()) if isinstance(source, dict) else source
    fields = ('object_id', 'object_type', 'name', 'description', 'transform',
              'bbox_min', 'bbox_max', 'immutable', 'scale_factor', 'placement_info')
    projected = [{k: copy.deepcopy(o[k]) for k in fields if k in o} for o in objects]
    if len({o['object_id'] for o in projected}) != len(projected):
        raise ValueError('Duplicate live object identity')
    return {'objects': projected, 'bounds_convention': 'local asset metres',
            'pose_convention': 'native world transform; tool yaw follows native advertised units'}


def refresh_visible_evidence(evidence, before_state):
    """Version a measured observation without misbinding archived context.

    Runtime deserialization may change object order. Source extras and dict
    keys belong to object IDs, not to their list positions. Retained context
    is still archived task/asset metadata, not newly measured runtime metadata.
    Call this on a valid old envelope, never one already partially overwritten.
    """
    out = copy.deepcopy(evidence)
    old_ids = [o['object_id'] for o in out['actual_scene_state']['objects']]
    new = visible_scene_state(before_state)
    new_ids = [o['object_id'] for o in new['objects']]
    if len(set(old_ids)) != len(old_ids) or set(old_ids) != set(new_ids):
        raise ValueError('Observation refresh must preserve unique object identities')
    context = out.get('source_context')
    if context is not None:
        extras = context['object_extra_fields']
        if len(extras) != len(old_ids):
            raise ValueError('Archived extras are not aligned with source objects')
        index = {uid: i for i, uid in enumerate(old_ids)}
        context['object_extra_fields'] = [extras[index[uid]] for uid in new_ids]
        layout = context['object_layout']
        if layout['kind'] == 'dict':
            keys = layout['keys']
            if len(keys) != len(old_ids) or len(set(keys)) != len(keys):
                raise ValueError('Invalid archived object-key mapping')
            layout['keys'] = [keys[index[uid]] for uid in new_ids]
        elif layout['kind'] != 'list':
            raise ValueError('Unknown archived object layout')
    out['actual_scene_state'] = new
    return out


def append_host_evidence(messages, *, schemas, before_state, host_history):
    """Preserve base prompts, all views and source memory; append the shared block.

    host_history means events executed by THIS host episode, not all historical
    actions. Legacy/source memory remains in the original messages. On isolated
    archived examples it is empty; that is not a claim of no past tool actions.
    Source task prompts, tool availability and history coverage still need a
    separate audit before claiming complete train/deploy distribution matching.
    """
    _validate(messages, schemas, host_history)
    evidence = {'executable_tools': copy.deepcopy(schemas),
                'actual_scene_state': visible_scene_state(before_state),
                'host_execution_history': copy.deepcopy(host_history)}
    text = json.dumps(evidence, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)
    out = copy.deepcopy(messages)
    out.append({'role': 'user', 'content': [{'type':'text','text':HOST_MARKER+text}]})
    return out


def _relocate_json(messages, marker, location):
    """Only a declared JSON block is moved; all surrounding text stays exact."""
    found = [(slot, key) for slot, key in text_slots(messages) if marker in slot[key]]
    if not found:
        return None
    if len(found) != 1 or found[0][0][found[0][1]].count(marker) != 1:
        raise ValueError('Ambiguous structured evidence block: ' + marker)
    slot, key = found[0]
    before, after = slot[key].split(marker)
    value, end = json.JSONDecoder().raw_decode(after)
    slot[key] = before + marker + '[See ' + location + ' in host evidence.]' + after[end:]
    return value


def canonical_host_evidence(messages, *, schemas, before_state, host_history):
    """Single-copy structured observation, with lossless legacy context transfer.

    Native state projection and history are the same as v1. When an archived
    request already exposes these fields, verify equality before relocating
    them. Preserve knowledge, measurements, task/asset provenance, tool
    descriptions, extra object fields and surrounding prose. This is format
    normalization, not retrieval, summarization or a new memory algorithm.
    """
    _validate(messages, schemas, host_history)
    out = copy.deepcopy(messages)
    tools = _relocate_json(out, TOOLS_MARKER, 'executable_tools and source_tool_metadata')
    source = _relocate_json(out, STATE_MARKER, 'actual_scene_state and source_context')
    visible = visible_scene_state(before_state)
    evidence = {'executable_tools': copy.deepcopy(schemas),
                'actual_scene_state': visible,
                'host_execution_history': copy.deepcopy(host_history)}
    if tools is not None:
        if (not isinstance(tools, list) or len({t['name'] for t in tools}) != len(tools)
                or {t['name']: t['parameters'] for t in tools} != schemas):
            raise ValueError('Embedded tool definitions disagree with live schemas')
        metadata = [{k: v for k, v in t.items() if k != 'parameters'} for t in tools]
        assert [dict(t, parameters=schemas[t['name']]) for t in metadata] == tools
        evidence['source_tool_metadata'] = metadata
    if source is not None:
        if visible_scene_state(source) != visible:
            raise ValueError('Embedded state disagrees with actual before-action state')
        raw_objects = source['objects']
        values = list(raw_objects.values()) if isinstance(raw_objects, dict) else raw_objects
        extras = [{k: v for k, v in obj.items() if k not in projected}
                  for obj, projected in zip(values, visible['objects'])]
        context = {k: v for k, v in source.items() if k != 'objects'}
        layout = {'kind': 'dict', 'keys': list(raw_objects)} if isinstance(raw_objects, dict) else {'kind': 'list'}
        evidence['source_context'] = {'non_object_fields': context,
                                      'object_layout': layout, 'object_extra_fields': extras}
        # Recover every original JSON value, including representation labels.
        restored = [dict(p, **e) for p, e in zip(visible['objects'], extras)]
        if layout['kind'] == 'dict':
            restored = dict(zip(layout['keys'], restored))
        assert dict(context, objects=restored) == source
    text = json.dumps(evidence, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)
    out.append({'role': 'user', 'content': [{'type': 'text', 'text': HOST_MARKER + text}]})
    return out
