"""Existing-information controls for design-record conditioning.

No action target, scene ID, corpus lookup, model call or semantic completion is
available to this router. All modes keep the full decision supervision outside
this function; they differ only in what the action phase can read.
"""
import json
import re


MODES = ('full_record', 'without_principle_field', 'plan_field_only')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def route_decision_text(text, mode):
    """Route an already generated record; never infer missing fields.

Malformed, plain-text and other-schema records pass through identically for
ALL modes, with status recorded. This is not proof of model format compliance.
The caller must log actual generated D at evaluation, not supply a gold lookup.
"""
    if mode not in MODES or not isinstance(text, str):
        raise ValueError('Unknown route or non-string decision')
    if not text:
        return text, dict(status='no_decision_common', changed=False, removed_fields=[])
    match = re.fullmatch(r'\s*<think>\s*(?P<body>\{.*\})\s*</think>(?P<tail>\s*)', text, re.S)
    if not match:
        return text, dict(status='nonrecord_common_passthrough', changed=False, removed_fields=[])
    try:
        value = json.loads(match['body'], object_pairs_hook=_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError):
        return text, dict(status='invalid_JSON_common_passthrough', changed=False, removed_fields=[])
    required = {'current_condition', 'principle_applicability', 'proposed_decision', 'expected_cost_and_unknowns'}
    if (not isinstance(value, dict) or set(value) != required
            or not isinstance(value['principle_applicability'], dict)
            or not isinstance(value['proposed_decision'], dict)):
        return text, dict(status='other_schema_common_passthrough', changed=False, removed_fields=[])
    if mode == 'full_record':
        return text, dict(status='structured_record_routed', changed=False, removed_fields=[])
    if mode == 'without_principle_field':
        selected = {k:v for k,v in value.items() if k != 'principle_applicability'}
    else:
        selected = {'proposed_decision': value['proposed_decision']}
    routed = '<think>\n' + json.dumps(selected, ensure_ascii=False) + '\n</think>' + match['tail']
    return routed, dict(status='structured_record_routed', changed=routed != text,
                        removed_fields=sorted(set(value)-set(selected)))
