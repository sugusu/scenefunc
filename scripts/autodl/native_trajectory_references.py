"""Common literal reference binding for a native whole-plan baseline.

References use only earlier successful tool outputs, never evaluation labels,
future observations or invented IDs. This ordinary adapter is shared by all
policy arms; it is not SceneOrchestra's original SceneWeaver implementation.
"""
import copy


def resolve_arguments(value, previous_outputs):
    if isinstance(value, dict) and '$result' in value:
        if set(value) != {'$result'}:
            raise ValueError('A reference cannot also supply literal fields')
        ref = value['$result']
        if not isinstance(ref, dict) or set(ref) != {'call', 'path'}:
            raise ValueError('Reference requires call and path')
        index, path = ref['call'], ref['path']
        if type(index) is not int or index < 0 or index >= len(previous_outputs):
            raise ValueError('Only already observed output indices are allowed')
        if not isinstance(path, list):
            raise ValueError('Reference path must be a literal list')
        result = previous_outputs[index]
        if not isinstance(result, dict) or result.get('success') is not True:
            raise ValueError('Reference source has no native success certificate')
        for key in path:
            if isinstance(result, list):
                if type(key) is not int or not 0 <= key < len(result):
                    raise ValueError('Invalid literal list index')
            elif isinstance(result, dict):
                if not isinstance(key, str) or key not in result:
                    raise ValueError('Invalid literal object key')
            else:
                raise ValueError('Cannot traverse a scalar output')
            result = result[key]
        return copy.deepcopy(result)
    if isinstance(value, dict):
        return {key: resolve_arguments(v, previous_outputs) for key, v in value.items()}
    if isinstance(value, list):
        return [resolve_arguments(v, previous_outputs) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError('Unsupported non-JSON trajectory argument')


def observed_id_paths(output):
    """Read IDs explicitly returned by a successful call, without deriving labels."""
    if not isinstance(output, dict) or output.get('success') is not True:
        return []
    found = []
    def walk(value, path):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ('object_id', 'asset_id') and isinstance(v, str) and v:
                    found.append((v, path + [k]))
                else:
                    walk(v, path + [k])
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, path + [i])
    walk(output, [])
    return found


def abstract_observed_ids(arguments, known_refs):
    if isinstance(arguments, str):
        return copy.deepcopy(known_refs.get(arguments, arguments))
    if isinstance(arguments, dict):
        return {k: abstract_observed_ids(v, known_refs) for k, v in arguments.items()}
    if isinstance(arguments, list):
        return [abstract_observed_ids(v, known_refs) for v in arguments]
    return copy.deepcopy(arguments)
