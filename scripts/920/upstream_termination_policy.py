"""Separate upstream no-tool termination from syntax and room acceptance.

The pinned parser is used verbatim. This existing-object executor still admits
one validated tool per turn, as advertised; it is not the full upstream rollout.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

from native_completion_policy import classify_response


def load_parser(reference):
    path = Path(reference['path'])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('Pinned upstream parser source changed')
    selected = []
    for node in ast.parse(data.decode()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id in {'_TOOL_CALL_RE', '_THINK_RE'} for t in node.targets):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == 'parse_assistant_text':
            selected.append(node)
    if len(selected) != 3:
        raise ValueError('Unexpected pinned parser structure')
    namespace = {'re': re, 'json': json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['parse_assistant_text']


def classify_for_execution(raw, record, parser):
    if record.get('status') != 'completed':
        raise RuntimeError('Generation did not complete; no policy retry')
    if hashlib.sha256(raw.encode()).hexdigest() != record.get('raw_output_sha256'):
        raise RuntimeError('Generation receipt/raw output mismatch')
    strict = classify_response(raw, record)
    _, calls = parser(raw)
    result = dict(format_classification=strict, upstream_parsed_calls=calls,
                  semantic_success_not_inferred=True, whole_room_success='unknown')
    if not calls:
        # No edits and no fabricated rejection receipt. Even a malformed answer
        # ends this trajectory; syntax and activity remain separately scored.
        return dict(result, kind='completion', stop_reason='no_tool_call',
                    format_valid=strict['kind'] == 'completion')
    if record.get('output_rejected_reason'):
        return dict(result, kind='completion', stop_reason='generation_rejected', format_valid=False)
    if len(calls) != 1:
        return dict(result, kind='rejection', reason='This executor advertises one tool call per turn')
    call = calls[0]
    if call.get('_parse_error'):
        return dict(result, kind='rejection', reason='Upstream parser reported malformed tool JSON')
    return dict(result, kind='tool', call=call, raw_json=json.dumps(call, ensure_ascii=False),
                format_valid=strict['kind'] == 'tool')
