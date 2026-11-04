"""Read-only audit of the frozen P920-DD-v1 prompt and actual dispatcher.

Does not load the model, execute tools, or modify the historical run.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dispatched_tools(source):
    """Read the explicit `name == ...` / `name in (...)` dispatcher branches."""
    result = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != 'name':
            continue
        right = node.comparators[0]
        if isinstance(node.ops[0], ast.Eq) and isinstance(right, ast.Constant):
            if isinstance(right.value, str):
                result.add(right.value)
        if isinstance(node.ops[0], ast.In) and isinstance(right, (ast.Tuple, ast.List)):
            for item in right.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    result.add(item.value)
    if not result:
        raise ValueError('No explicit tool dispatcher branches found')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run = args.root.resolve() / 'runtime/P920-DD-v1_20261104T083214Z'
    dispatcher = run / 'work/paris_local_run_results_20261104/dispatch_corrected.py'
    supported = dispatched_tools(dispatcher.read_text(encoding='utf-8'))
    records = []
    for label in ('A_G', 'A_U', 'B_G', 'B_U'):
        request = run / 'comparison_inputs' / label / 'request.json'
        messages = json.loads(request.read_text(encoding='utf-8'))
        system = messages[0]['content']
        advertised = set(re.findall(r'^- ([A-Za-z_][A-Za-z_0-9]*)\(', system, re.M))
        guard = re.search(r'Executable tools are (.*?)\. Other tools', system, re.S)
        if not advertised or not guard:
            raise ValueError(f'{label}: frozen prompt structure changed')
        allowed = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', guard.group(1))) - {'and'}
        if allowed != supported:
            raise ValueError(f'{label}: late guard differs from dispatcher')
        calls_path = run / 'model_runs' / ('formal_' + label) / 'parsed_calls.json'
        calls = json.loads(calls_path.read_text(encoding='utf-8'))
        responses_path = run / 'revisions' / label / 'tools/tool_responses.json'
        responses = json.loads(responses_path.read_text(encoding='utf-8'))
        if len(calls) != len(responses):
            raise ValueError(f'{label}: calls and responses do not align')
        unsupported = []
        other_failures = []
        for index, (call, event) in enumerate(zip(calls, responses)):
            if event['call'] != call:
                raise ValueError(f'{label}: tool event differs from model call')
            if call['name'] not in supported:
                if event['response'].get('success') is not False:
                    raise ValueError('Unsupported operation reported as successful')
                unsupported.append({'index': index, 'name': call['name'],
                                    'response': event['response']})
            elif event['response'].get('success') is not True:
                other_failures.append({'index': index, 'call': call,
                                       'response': event['response']})
        records.append({
            'label': label, 'request_sha256': digest(request),
            'system_prompt_sha256': hashlib.sha256(system.encode('utf-8')).hexdigest(),
            'advertised_tools': sorted(advertised),
            'late_guard_tools': sorted(allowed),
            'advertised_but_unavailable': sorted(advertised - supported),
            'parsed_calls_sha256': digest(calls_path),
            'tool_responses_sha256': digest(responses_path),
            'total_calls': len(calls), 'unsupported_calls': unsupported,
            'other_failures': other_failures,
        })
    report = {
        'source_run_id': 'P920-DD-v1_20261104T083214Z',
        'audit_kind': 'static_prompt_dispatch_and_saved_response_audit',
        'model_calls': 0, 'tool_executions': 0,
        'dispatcher_sha256': digest(dispatcher),
        'dispatched_tools': sorted(supported), 'records': records,
        'affected_arms': [r['label'] for r in records if r['unsupported_calls']],
        'unsupported_calls': sum(len(r['unsupported_calls']) for r in records),
        'total_calls': sum(r['total_calls'] for r in records),
        'finding': 'The inherited system prompt advertises tools that a later '
                   'paragraph disallows and the dispatcher does not implement.',
        'inference_limit': 'This is a verified engineering confound, not proof '
                           'that correcting it improves design or functional use.',
        'next_step': 'Freeze a new development protocol with one consistent '
                     'executable tool contract shared by G/U; preserve v1 unchanged.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'affected_arms': report['affected_arms'],
                      'unsupported_calls': report['unsupported_calls'],
                      'total_calls': report['total_calls'],
                      'out': str(args.out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
