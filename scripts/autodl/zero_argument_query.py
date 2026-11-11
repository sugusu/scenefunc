"""Opt-in ordinary input normalization for the advertised read-only scene query.

Does not dispatch tools, repair edits, select furniture, override object IDs,
change observations or authorize additional attempts. Callers retain raw output
and must apply their normal fresh-context validation after this function.
"""
import copy


def canonicalize_scene_query(call, tools):
    """Drop semantically undefined arguments only for the zero-argument query.

    The exact tool and empty argument schema must be advertised. All other
    calls, including malformed outer envelopes, are left unchanged.
    """
    original = copy.deepcopy(call)
    if not isinstance(call, dict) or set(call) != {'name', 'arguments'}:
        return original, None
    if call['name'] != 'get_current_scene_state':
        return original, None
    matches = [t for t in tools if t.get('name') == call['name']]
    if len(matches) != 1:
        return original, None
    schema = matches[0].get('parameters', {})
    if (schema.get('type') != 'object' or
            schema.get('properties') != {} or
            schema.get('required', []) != [] or
            schema.get('additionalProperties') is not False):
        return original, None
    if not isinstance(call['arguments'], dict) or not call['arguments']:
        return original, None
    normalized = dict(name=call['name'], arguments={})
    return normalized, dict(kind='ordinary_zero_argument_query_normalization',
                            dropped_argument_names=sorted(call['arguments']),
                            scene_edit=False,
                            final_validation_required=True)
