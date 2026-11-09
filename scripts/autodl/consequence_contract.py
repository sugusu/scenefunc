"""Shared scoped consequence objective/selector, not a trained predictor.

The exact checker and any future predictor use identical requirement keys.
Unknown does not mean failure; unknown hard requirements prevent certification.
No scalar reward weights or human preference labels are invented here.
"""
import math


def masked_binary_loss(probabilities, targets):
    """Ordinary BCE on measured labels, retaining the missing-label count.

    This is conditional-on-observation risk, not an unbiased estimator of all
    candidates' risk when early geometry rejection censors functional checks.
    """
    if probabilities.keys() != targets.keys():
        raise ValueError('Prediction and target keys differ')
    losses = []
    for key, target in targets.items():
        if target is None:
            continue
        if type(target) is not bool:
            raise ValueError('Only measured bool or unknown None targets accepted')
        probability = probabilities[key]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid probability')
        likelihood = probability if target else 1 - probability
        losses.append(float('inf') if likelihood == 0 else -math.log(likelihood))
    return {'mean_known_bce': sum(losses) / len(losses) if losses else None,
            'known': len(losses), 'unknown': len(targets) - len(losses)}


def exact_admission(values, required_keys):
    if not required_keys or len(set(required_keys)) != len(required_keys):
        raise ValueError('An explicit nonempty unique requirement set is required')
    missing = [k for k in required_keys if values.get(k) is None]
    failed = [k for k in required_keys if values.get(k) is False]
    invalid = [k for k in required_keys if values.get(k) is not None and type(values[k]) is not bool]
    if invalid:
        raise ValueError('Non-boolean requirement values: ' + str(invalid))
    return {'scoped_status': 'fail' if failed else ('unknown' if missing else 'pass'),
            'failed': failed, 'unknown': missing, 'certified_keys': list(required_keys),
            'whole_room_success': None}


def interval_dominates(candidate, reference, keys, hard_lower_bounds=None):
    """Sufficient dominance under caller-supplied simultaneous valid intervals.

    The function cannot establish calibration. Passing means an algebraic
    certificate conditional on interval validity, never a safety guarantee.
    """
    def interval(row, key):
        value = row.get(key)
        if value is None:
            return None
        if len(value) != 2 or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in value) or value[0] > value[1]:
            raise ValueError('Invalid finite interval')
        return value
    comparisons = []
    for key in keys:
        a, b = interval(candidate, key), interval(reference, key)
        if a is None or b is None:
            return {'dominates': None, 'reason': 'unknown interval', 'key': key, 'calibration_verified': False}
        comparisons.append((a[0] >= b[1], a[0] > b[1]))
    for key, threshold in (hard_lower_bounds or {}).items():
        a = interval(candidate, key)
        if a is None or a[0] < threshold:
            return {'dominates': False, 'reason': 'hard lower bound not certified', 'key': key, 'calibration_verified': False}
    return {'dominates': bool(comparisons) and all(c[0] for c in comparisons) and any(c[1] for c in comparisons),
            'reason': 'conditional interval comparison', 'calibration_verified': False}
