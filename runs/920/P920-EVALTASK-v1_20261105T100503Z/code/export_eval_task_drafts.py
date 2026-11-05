"""Export two draft tasks without passing evaluator/source records to Builder.

This field boundary is not a semantic leakage proof or evaluation admission.
It does not construct scenes, call models, or score any output.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'export two research task drafts', flush=True)
    root = args.run.resolve()
    data = json.loads((root / 'TASK_DRAFTS.json').read_text())
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    records = data['tasks']
    if len(records) != 2 or {x['family_id'] for x in records} != set(protocol['families']):
        raise ValueError('Unexpected source selection')
    fields = {'task_id', 'version', 'task_kind', 'raw_resident_brief',
              'source_condition_basis', 'researcher_authored_brief',
              'geometry_assumptions', 'required_activities', 'allowed_solutions',
              'uncertainties', 'execution_state'}
    outputs = []
    # Validate every record before writing any output.
    for row in records:
        task = row['builder_task']
        if set(task) != fields or task['raw_resident_brief'] is not None:
            raise ValueError('Unreviewed builder fields or purported raw brief')
        if row['evaluation_only']['formal_evaluation_ready']:
            raise ValueError('Draft cannot be promoted by export')
        if task['task_id'] not in ('whole_room_transfer_01', 'whole_room_transfer_02'):
            raise ValueError('Unreviewed task identity')
        text = json.dumps(task, ensure_ascii=False, indent=2) + '\n'
        for token in ('excluded_hindsight', 'family_id', 'evaluation_only',
                      'primary_url', 'target_action', 'reference_furniture'):
            if token in text:
                raise ValueError('Forbidden source/evaluator field in builder task')
        outputs.append((task['task_id'] + '.json', text))
    out = root / 'builder_inputs'
    out.mkdir(exist_ok=False)
    manifest = []
    for name, text in outputs:
        payload = text.encode('utf-8')
        (out / name).write_bytes(payload)
        manifest.append({'path': 'builder_inputs/' + name,
                         'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
    result = {'status': 'draft_exports_completed', 'outputs': manifest,
              'model_calls': 0, 'native_calls': 0, 'training_updates': 0,
              'formal_evaluation_ready': False, 'semantic_leakage_certified': False,
              'scope': 'Allowlisted data export only; manual source attribution review remains separate.'}
    (root / 'EXPORT_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
