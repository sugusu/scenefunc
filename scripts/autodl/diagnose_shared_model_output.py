"""Decode the actual returned tokens and compare exact native/phase parsers.

No model, output repair, relabeling, policy dispatch or changed acceptance rule.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import socket


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=Path, required=True)
    args = p.parse_args()
    target = args.stage / 'token_parser_diagnostic.json'
    if target.exists():
        raise FileExistsError(target)
    from phase_policy import classify_stage
    from transformers import AutoTokenizer
    protocol = json.loads((args.stage / 'protocol.json').read_text())
    def reject(*a, **k):
        raise RuntimeError('Local-only diagnostic; no network')
    socket.socket.connect = reject
    socket.create_connection = reject
    tokenizer = AutoTokenizer.from_pretrained(protocol['checkpoint'], local_files_only=True)
    work = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
    source = work / 'native_execution_inventory_20261104/source/Designer/agent_rollout.py'
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name == 'parse_assistant_text') or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id in ('_TOOL_CALL_RE', '_THINK_RE') for t in n.targets))]
    assert len(nodes) == 3
    namespace = {'re': re, 'json': json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    rows = []
    clients = json.loads((args.stage / 'actual_v1/client_results.json').read_text())
    for client in clients:
        if 'result' not in client:
            continue
        result = client['result']
        for stage in result['stages']:
            ids = stage.get('output_ids')
            if ids is None:
                continue
            raw = tokenizer.decode(ids, skip_special_tokens=False)
            assert raw == stage['classification']['raw']
            replay = classify_stage(tokenizer, ids, stage=stage['stage'],
                                    eos_ids=stage['generation_eos_ids'])
            assert replay == stage['classification']
            think, tools = namespace['parse_assistant_text'](raw)
            rows.append({'condition': client['condition'], 'stage': stage['stage'],
                'output_tokens': len(ids), 'cap': stage['allowed_output_tokens'],
                'ended_at_cap': len(ids) >= stage['allowed_output_tokens'],
                'last_id': ids[-1], 'last_decoded_token': tokenizer.decode([ids[-1]]),
                'closing_think_count': ids.count(tokenizer.convert_tokens_to_ids('</think>')),
                'closing_tool_count': ids.count(tokenizer.convert_tokens_to_ids('</tool_call>')),
                'phase_classification': replay, 'native_think': think,
                'native_tool_calls': tools, 'native_parser_is_permissive': True,
                'action_dispatched': False,
                'finding': 'Returned token sequence contains tool-closing control tokens but '
                    'no decision-closing token. Strict phase rejection is reproducible; '
                    'native regex can still extract the tool substring. Neither proves task success.'})
    mapping = {s: {'id': tokenizer.convert_tokens_to_ids(s),
                   'roundtrip': tokenizer.decode([tokenizer.convert_tokens_to_ids(s)])}
               for s in ('<think>', '</think>', '<tool_call>', '</tool_call>', '<|im_end|>')}
    assert len({v['id'] for v in mapping.values()}) == len(mapping)
    assert all(v['roundtrip'] == s for s, v in mapping.items())
    assert len(rows) == 2
    assert all(not r['ended_at_cap'] and r['closing_think_count'] == 0 for r in rows)
    record = {'status': 'actual_token_and_parser_replay_completed', 'rows': rows,
        'token_mapping': mapping,
        'native_parser_source': str(source), 'native_parser_source_sha256': sha(source),
        'no_model_forwards': True, 'changed_thresholds_or_parser': False,
        'scope': 'One archived observation; current frozen base has not been fine-tuned for '
            'the prospective split contract. This is not a trained-policy comparison or '
            'proof that learning the proposed contract is impossible.'}
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'token_mapping': mapping, 'new_model_forwards': 0}))


if __name__ == '__main__':
    main()
