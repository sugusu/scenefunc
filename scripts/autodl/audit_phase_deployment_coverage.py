"""Quantify partial rationale supervision versus unconditional D/A deployment.

This is a material/runtime scope audit, not an empirical prediction of failure.
No labels, target coefficients, examples or inference routing are changed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--components', type=Path, required=True)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = [json.loads(line) for line in args.components.read_text().splitlines()]
    paired = [r for r in data if r['arm'] == 'paired_phase_v3']
    decision = {r['unit_id'] for r in paired if r['stage'] == 'decision'}
    rows = []
    for r in paired:
        if r['stage'] != 'action':
            continue
        has_prefix = bool(r['assistant_prefix_token_ids'])
        assert has_prefix == (r['unit_id'] in decision)
        rows.append({'unit_id': r['unit_id'], 'family_id': r['family_id'],
                     'target_kind': r['target_kind'], 'has_D_supervision': r['unit_id'] in decision,
                     'training_action_conditioned_on_D': has_prefix,
                     'paired_runtime_always_generates_D': True,
                     'runtime_rationale_from_model_not_teacher': True})
    source = args.policy.read_text()
    # Verify the exact current route rather than inferring it from the name.
    route = "phases=['action'] if self.arm=='separate_direct_v3' else ['decision','action']"
    assert route in source
    assert "visible_draft=draft if stage=='action' and self.arm=='paired_phase_v3' else None" in source
    assert len(rows) == len({r['unit_id'] for r in rows}) == 24
    with (args.out/'units.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    result = {
        'units': len(rows), 'D_supervised_units': len(decision),
        'paired_training_actions_with_D': sum(r['training_action_conditioned_on_D'] for r in rows),
        'paired_training_actions_without_D': sum(not r['training_action_conditioned_on_D'] for r in rows),
        'paired_deployment_actions_always_with_generated_D': len(rows),
        'new_native_prompt_alignment_verified': False,
        'interpretation': 'Partial-rationale training with always-reasoning deployment; not uniformly reason-conditioned supervision on all 24 units.',
        'not_implied': ['training is invalid', 'model must fail', 'new empirical advantage', 'test-time teacher rationale is available'],
        'required_reporting': [
            'Keep the 14 no-D training units in the denominator and state their direct-action supervision.',
            'Do not route test examples using whether their teacher rationale exists.',
            'Evaluate self-generated D followed by actual execution, including unsupported/no-edit states.',
            'Do not claim native prompt alignment from previous archived-prefix equality checks.',
            'Construct training/deployment inputs with the same shared native formatter before any new matched-prompt trial.'
        ],
        'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.components, args.policy)},
        'training_updates': 0, 'model_calls': 0}
    (args.out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if isinstance(v, (int, bool))}))


if __name__ == '__main__':
    main()
