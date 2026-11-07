"""Compare full declared requirement vectors, never inferred expert preference.

Accepts actual per-state reviews; no target pose or preferred action is scored.
The paired audits must use the same requested condition and requirement keys.
"""
import argparse
import json
from pathlib import Path

from build_activity_completion_reviews import aggregate


def checked_review(review):
    rows = review['checks']
    keys = [x['requirement'] for x in rows]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Missing or duplicate requirement keys')
    status = aggregate(rows)
    if status != review['measured_requirements_status']:
        raise ValueError('Stored verdict disagrees with requirement vector')
    return {x['requirement']: x['pass'] for x in rows}


def score_transition(initial, final, *, attempts, native_calls, state_changes,
                     generated_tokens=None, gpu_seconds=None):
    before, after = checked_review(initial), checked_review(final)
    if initial['requested_mode'] != final['requested_mode'] or before.keys() != after.keys():
        raise ValueError('Condition or requirement coverage changed across evaluation')
    for v in (attempts, native_calls, state_changes):
        if type(v) is not int or v < 0:
            raise ValueError('Invalid execution costs')
    if not state_changes <= native_calls <= attempts:
        raise ValueError('Inconsistent execution accounting')
    if generated_tokens is not None and (type(generated_tokens) is not int or generated_tokens < 0):
        raise ValueError('Invalid token accounting')
    if gpu_seconds is not None:
        import math
        if isinstance(gpu_seconds, bool) or not isinstance(gpu_seconds, (int, float)) or not math.isfinite(gpu_seconds) or gpu_seconds < 0:
            raise ValueError('Invalid GPU accounting')
    newly_failed = [k for k in before if before[k] is True and after[k] is False]
    newly_unknown = [k for k in before if before[k] is True and after[k] is None]
    repaired = [k for k in before if before[k] is False and after[k] is True]
    final_status = final['measured_requirements_status']
    initial_status = initial['measured_requirements_status']
    if final_status == 'pass':
        outcome = 'resolved_declared_deficit' if initial_status == 'fail' else (
            'maintained_declared_requirements' if initial_status == 'pass' else 'resolved_unknown_checks')
    elif newly_failed:
        outcome = 'introduced_declared_failure'
    elif final_status == 'unknown' or newly_unknown:
        outcome = 'unresolved_evidence'
    else:
        outcome = 'remaining_declared_deficit'
    # No editing is rewarded as successful only when the supplied current
    # evidence passes. Conversely pass->pass moves receive no improvement credit;
    # they may still have an unmeasured benefit, which this scorer cannot decide.
    return dict(outcome=outcome, initial_status=initial_status, final_status=final_status,
        requested_mode=initial['requested_mode'], requirement_count=len(before),
        newly_failed=newly_failed, newly_unknown=newly_unknown, repaired=repaired,
        final_failed=[k for k,v in after.items() if v is False],
        final_unknown=[k for k,v in after.items() if v is None],
        measured_success=final_status == 'pass',
        measured_incremental_gain=initial_status == 'fail' and final_status == 'pass',
        editing_without_declared_deficit=initial_status == 'pass' and state_changes > 0,
        costs=dict(attempts=attempts,native_calls=native_calls,state_changes=state_changes,
                   validation_rejections=attempts-native_calls,generated_tokens=generated_tokens,gpu_seconds=gpu_seconds),
        whole_room_success='unknown', professional_preference='not_assessed',
        limitation='Declared geometric/activity checks only; pass->pass movement is not improvement or evidence of waste. Costs break ties only after requirements; no scalar weighted reward.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-review',type=Path,required=True)
    p.add_argument('--final-review',type=Path,required=True)
    p.add_argument('--costs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    r=score_transition(json.loads(a.initial_review.read_text()),json.loads(a.final_review.read_text()),
                       **json.loads(a.costs.read_text()))
    with a.out.open('x') as f: f.write(json.dumps(r,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
