"""Offline parser sensitivity on recorded outputs; never rerun a trajectory."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    args = ap.parse_args()
    source = args.upstream.read_text()
    checkout = args.upstream.parent.parent
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    committed = subprocess.check_output(['git', '-C', str(checkout), 'show',
        'HEAD:' + str(args.upstream.relative_to(checkout))])
    if committed != args.upstream.read_bytes():
        raise ValueError('Upstream parser differs from recorded commit')
    tree = ast.parse(source)
    # Execute only the two literal regex assignments and the standalone parser,
    # not upstream imports, runtime initialization, model loading or scene code.
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id in {'_TOOL_CALL_RE', '_THINK_RE'} for t in node.targets):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == 'parse_assistant_text':
            selected.append(node)
    if len(selected) != 3:
        raise ValueError('Unexpected upstream parser structure')
    namespace = {'re': re, 'json': json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.upstream), 'exec'), namespace)
    parser = namespace['parse_assistant_text']
    batch = json.loads((args.run / 'BATCH_RESULT.json').read_text())
    if batch['status'] == 'running':
        raise ValueError('Wait for the existing batch to end')
    rows = []
    for cell in batch['cells']:
        path = args.run / 'cells' / cell['cell_id'] / 'ROLLOUT_RESULT.json'
        result = json.loads(path.read_text())
        divergence = None
        turns = []
        for turn in result['rounds']:
            raw = turn['raw_output']
            if hashlib.sha256(raw.encode()).hexdigest() != turn['model']['raw_output_sha256']:
                raise ValueError('Raw output hash mismatch')
            think, calls = parser(raw)
            official_stop = not calls
            strict = turn['response_classification']
            record = dict(turn=turn['turn'], raw_output_sha256=turn['model']['raw_output_sha256'],
                upstream_parsed_calls=calls, upstream_has_closed_think=think is not None,
                upstream_no_tool_stop=official_stop, strict_classification=strict,
                observed_only_after_prior_parser_divergence=divergence is not None,
                ended_with_eos=turn['model'].get('ended_with_eos'),
                generation_rejected=turn['model'].get('output_rejected_reason'))
            differs = ((official_stop and strict['kind'] != 'completion') or
                (bool(calls) and (strict['kind'] != 'tool' or calls != [strict.get('call')])))
            record['parser_control_flow_differs'] = differs
            record['difference_kind'] = ('upstream_no_tool_stop' if official_stop else
                'upstream_extracts_calls_rejected_or_differently_parsed_by_strict_gate') if differs else None
            if divergence is None and differs:
                divergence = turn['turn']
            turns.append(record)
        first = turns[0] if turns else None
        rows.append(dict(cell=cell['cell_id'], result_sha256=digest(path),
            first_parser_divergence_turn=divergence,
            first_output_upstream_stop=first['upstream_no_tool_stop'] if first else None,
            initial_measured_status=result.get('transition_score', {}).get('initial_status'),
            actual_final_outcome=result.get('transition_score', {}).get('outcome'),
            actual_valid_completion=result.get('valid_completion'), turns=turns))
    output = dict(kind='posthoc_offline_parser_sensitivity', training_updates=0, model_calls=0,
        upstream_source=str(args.upstream), upstream_source_sha256=digest(args.upstream),
        upstream_revision=revision, upstream_parser_matches_committed_source=True,
        upstream_parser_lines=[selected[-1].lineno, selected[-1].end_lineno],
        stop_rule='rollout_client.py: if not tool_calls -> stop_reason=no_tool_call, break',
        rows=rows, trajectories_with_parser_divergence=sum(r['first_parser_divergence_turn'] is not None for r in rows),
        recorded_outputs_with_no_tool_stop_but_strict_rejection=sum(t['upstream_no_tool_stop'] and
            t['strict_classification']['kind']=='rejection' for r in rows for t in r['turns']),
        interpretation=[
            'The frozen strict primary scoring and executed trajectories are unchanged.',
            'Official no-tool termination does not certify valid syntax, fulfilled needs or professional quality.',
            'After the first parser divergence, subsequent recorded outputs are conditional on strict feedback; they are not outputs of an official-parser trajectory.',
            'An upstream stop on the first output would preserve the observed initial state; this is a control-flow implication, not a newly executed scene or alternative benchmark score.',
            'Reject/retry may mediate later edits; observational evidence does not establish that changing the parser improves design.'
        ])
    (args.run / 'PARSER_SENSITIVITY.json').write_text(json.dumps(output, indent=2) + '\n')
    snippets = '\n\n'.join(ast.get_source_segment(source, node) for node in selected)
    (args.run / 'UPSTREAM_PARSER_EXCERPT.txt').write_text(snippets + '\n')
    print(json.dumps({k: output[k] for k in ['kind', 'trajectories_with_parser_divergence']}))


if __name__ == '__main__':
    main()
