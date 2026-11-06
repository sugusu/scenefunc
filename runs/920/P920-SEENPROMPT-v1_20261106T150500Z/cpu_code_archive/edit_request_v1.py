"""Candidate common input boundary; historical runs never import this module.

Only pre-action facts enter build_request. Targets stay with the caller. Native
schema validation checks syntax/identity, not task compliance or design quality.
"""
import copy
import json
import re

from shared_action_contract import digest, schemas_from_observation, strict_loads


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def normalize_context(context):
    result = copy.deepcopy(context)
    objects = result['objects']
    if isinstance(objects, list):
        ids = [obj['object_id'] for obj in objects]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate object IDs')
        objects = {obj['object_id']: obj for obj in objects}
    if not isinstance(objects, dict):
        raise ValueError('Expected an object list or table')
    for oid, obj in objects.items():
        if obj['object_id'] != oid:
            raise ValueError('Object table key/ID mismatch')
    result['objects'] = objects
    catalog = result.get('asset_catalog', [])
    ids = [asset['asset_id'] for asset in catalog]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate callable asset IDs')
    # Do not infer a catalog from source IDs, target actions, or asset bindings.
    digest(result)  # Also rejects nonfinite numbers.
    return result


def tools_and_scope(system, native_tools):
    marker = 'Available executable tools:\n'
    if marker in system:
        before, tail = system.split(marker, 1)
        tools, end = json.JSONDecoder().raw_decode(tail.lstrip())
        scope = before + tail.lstrip()[end:]
    else:
        # Existing legacy episodes explicitly advertise only this signature.
        # Reject unknown text interfaces; never infer tools from gold actions.
        if not re.search(r'move_furniture_tool\(object_id,\s*x,\s*y,\s*yaw\)', system):
            raise ValueError('Unsupported legacy tool interface')
        tools = [copy.deepcopy(next(t for t in native_tools if t['name'] == 'move_furniture_tool'))]
        scope = system
    return tools, scope.strip()


def build_request(*, context, tools, scope, images, additional_text=''):
    """The same function is used for candidate training and recorded deployment."""
    from jsonschema import Draft202012Validator
    context = normalize_context(context)
    if len(images) != 5:
        raise ValueError('Exactly five ordered views required')
    names = [tool['name'] for tool in tools]
    if len(set(names)) != len(names) or not names:
        raise ValueError('Missing or duplicate tools')
    for tool in tools:
        Draft202012Validator.check_schema(tool['parameters'])
    identifiers = dict(
        callable_asset_ids=[a['asset_id'] for a in context.get('asset_catalog', [])],
        rule='For add asset_id, use only callable_asset_ids. source_asset_id and asset_bindings.asset_id describe provenance; they are not callable IDs unless also explicitly listed here.',
        catalog_status='provided' if 'asset_catalog' in context else 'not_provided_no_addition_advertised')
    if 'asset_catalog' not in context and 'add_furniture_to_scene_tool' in names:
        raise ValueError('Addition advertised without a catalog')
    system = ('Revise the current room under the original task constraints. Current objects exist even if action history is empty or unavailable. '
              'Use exact object IDs and callable catalog IDs. Units are metres; yaw is degrees about +Z. '
              'You may produce a short <think> decision describing conditions, applicable source knowledge, intended changes and expected costs/unknowns. '
              'Then emit exactly one <tool_call>{"name":"...","arguments":{...}}</tool_call>, or <done> with unmet/unknown requirements and no call. '
              'Do not claim unmeasured success.\nAvailable executable tools:\n' + dumps(tools))
    text = ('## Original task and operation constraints\n' + scope +
            '\n## Current room (not an action-history snapshot)\nCURRENT NATIVE STATE:\n' + dumps(context) +
            '\n## Identifier namespaces\n' + dumps(identifiers) +
            '\n## Additional recorded task, history and feedback\n' + (additional_text or '(none supplied; this does not mean the room is empty)'))
    return [dict(role='system', content=system), dict(role='user', content=[
        *[dict(type='image', image=str(path)) for path in images], dict(type='text', text=text)])]


def from_episode_inputs(document, native_tools):
    """Never reads assistant content, tool_calls, evaluation_only or provenance."""
    if len(document['turns']) != 1:
        raise ValueError('Only single observed steps are supported')
    tools, scope = tools_and_scope(document['system'], native_tools)
    return build_request(context=strict_loads(document['task_context']['scene_description']),
                         tools=tools, scope=scope, images=document['turns'][0]['images'])


def from_recorded_request(messages):
    if len(messages) != 2 or [m['role'] for m in messages] != ['system', 'user']:
        raise ValueError('Unexpected historical message structure')
    content = messages[1]['content']
    if not isinstance(content, list) or any(x['type'] not in ('text', 'image') for x in content):
        raise ValueError('Unexpected multimodal content')
    images = [x['image'] for x in content if x['type'] == 'image']
    text = '\n'.join(x['text'] for x in content if x['type'] == 'text')
    if text.count('CURRENT NATIVE STATE:') != 1:
        raise ValueError('Expected one current-state block')
    before, rest = text.split('CURRENT NATIVE STATE:', 1)
    rest = rest.lstrip()
    context, end = json.JSONDecoder().raw_decode(rest)
    tools, scope = tools_and_scope(messages[0]['content'], [])
    return build_request(context=context, tools=tools, scope=scope,
                         images=images, additional_text=before + rest[end:])


def validate_action(call, request):
    """Validate declared schema and IDs only; does not claim task/geometry success."""
    from jsonschema import Draft202012Validator
    from shared_action_contract import context_from_request
    call = strict_loads(dumps(call))
    if set(call) != {'name', 'arguments'}:
        raise ValueError('Unexpected action keys')
    tools = schemas_from_observation(dict(system=request[0]['content']))
    matches = [t for t in tools if t['name'] == call['name']]
    if len(matches) != 1:
        raise ValueError('Tool not advertised')
    Draft202012Validator(matches[0]['parameters']).validate(call['arguments'])
    ctx = context_from_request(request)
    for field, value in call['arguments'].items():
        if field == 'asset_id' and value not in [a['asset_id'] for a in ctx.get('asset_catalog', [])]:
            raise ValueError('Asset ID is not callable')
        if field in ('object_id', 'object_a_id', 'object_b_id'):
            if value not in ctx['objects'] or ctx['objects'][value]['object_type'] != 'furniture':
                raise ValueError('Object ID is not current furniture')
            if call['name'] in ('move_furniture_tool', 'remove_furniture_tool') and ctx['objects'][value]['immutable']:
                raise ValueError('Immutable object edit')
    return True
