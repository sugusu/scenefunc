"""Qualify sourced knowledge against pinned interfaces, without scene execution.

This is an interface audit and candidate export, not a performance test or an
exhaustive claim about the capabilities of SceneSmith/Blender.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'begin source-scope interface audit', flush=True)
    plan = json.loads(a.plan.read_text())
    for item in plan['inputs'].values():
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError(f"Input changed: {item['path']}")
    paths = {k: Path(v['path']) for k, v in plan['inputs'].items()}
    schemas = runpy.run_path(str(paths['schemas']))['STAGE_SCHEMAS']
    parameters = {stage: {name: sorted(set(spec.get('required', {})) |
                                     set(spec.get('optional', {})))
                          for name, spec in tools.items()}
                  for stage, tools in schemas.items()}
    tree = ast.parse(paths['furniture_tools'].read_text())
    move = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            and n.name == 'move_furniture_tool']
    if len(move) != 1:
        raise ValueError('Expected one inspected move wrapper')
    calls = [n for n in ast.walk(move[0]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == '_move_furniture_impl']
    zvalues = [ast.literal_eval(k.value) for c in calls for k in c.keywords if k.arg == 'z']
    records = json.loads(paths['knowledge'].read_text())['records']
    checks = {
        'two_unique_source_decisions': len(records) == len({r['decision_id'] for r in records}) == 2,
        'two_unique_source_families': len({r['family_id'] for r in records}) == 2,
        'furniture_move_has_no_public_z': 'z' not in parameters['furniture']['move_furniture_tool'],
        'native_move_wrapper_forces_floor_z': zvalues == [0.0],
        'no_explicit_spectral_parameters_in_inspected_schemas': not any(
            any(s in key.lower() for s in ('spectrum', 'spectral', 'cri', 'spd'))
            for tools in parameters.values() for keys in tools.values() for key in keys
            if key not in ('description', 'object_descriptions')),
        'sources_not_promoted_to_executed_pairs': all(not r['maturity']['source_grounded_action_replayed'] for r in records),
        'training_remains_unadmitted': all(not r['training_eligible'] and not r['training_authorized'] for r in records),
        'families_allocated_before_derivation': all(r['split_fixed_at'] < r['derived_knowledge_at'] for r in records),
    }
    audit = {'checked_at': datetime.now(timezone.utc).isoformat(), 'inputs': plan['inputs'],
             'parameters': parameters, 'move_wrapper': {'line': move[0].lineno, 'z_values': zvalues},
             'checks': checks, 'scope': 'Pinned public schemas and move wrapper only; no exhaustive engine capability claim.',
             'manual_measurement_review': plan['manual_measurement_review'],
             'new_scene_actions': 0, 'model_calls': 0, 'training_updates': 0}
    write(a.out / 'TOOL_SCOPE_AUDIT.json', audit)
    for r in records:
        answer = {k: r[k] for k in ('condition_summary', 'rationale_summary', 'decision_scope')}
        answer['source_url'] = r['professional_evidence']['primary_crosscheck']['url']
        answer['unknowns'] = r['unknowns']
        episode = {
            'stage': 'qa',
            'system': 'Explain attributed project knowledge. Distinguish the architect account from reconstruction and unverified outcomes.',
            'provenance': {'kind': 'researcher_formulated_attributed_knowledge_QA', 'family_id': r['family_id'],
                           'decision_id': r['decision_id'], 'allocation': 'train_source_candidate',
                           'professional_review': False, 'training_eligible': False, 'training_authorized': False,
                           'actual_loader_checked': False, 'images': 0, 'actions': 0},
            'turns': [
                {'role': 'user', 'content': 'Explain the documented conditions, design decision, rationale and limits for ' + r['decision_id'] + '. Do not invent original tool operations or measured benefits.'},
                {'role': 'assistant', 'content': json.dumps(answer, ensure_ascii=False, sort_keys=True)}]}
        write(a.out / 'episodes' / r['decision_id'] / 'episode.json', episode)
    if not all(checks.values()):
        raise ValueError('Scope audit failed; inspect retained report')
    print(json.dumps({'checks_passed': sum(checks.values()), 'knowledge_QA_candidates': len(records),
                      'actual_loader_checked': False, 'new_actions': 0, 'training_updates': 0}), flush=True)


if __name__ == '__main__':
    main()
