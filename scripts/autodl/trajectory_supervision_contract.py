"""Separate executed-call records, surrogate preferences and verified full tasks.

This is a common data-admission rule, not a new learning loss. A native success
reply is not a successful room. Unknown observations never become false facts
or positive terminal targets. The task's original clause IDs are immutable.
"""
from dataclasses import dataclass
from typing import Literal

Truth = Literal['true', 'false', 'unknown']


@dataclass(frozen=True)
class ClauseEvidence:
    clause_id: str
    value: Truth
    evidence_ref: str
    independently_verified: bool

    def __post_init__(self):
        if self.value not in ('true', 'false', 'unknown'):
            raise ValueError('Invalid three-valued observation')
        if not self.clause_id or not self.evidence_ref:
            raise ValueError('Clause identity and evidence reference required')


def terminal_admission(required_ids, evidence, *, terminal_reason,
                       execution_errors, disallowed_change=False):
    """Accept only an explicitly completed task with complete true evidence.

    This is a positive-label admission predicate, NOT an estimator of human
    quality or a declaration that unknown tasks are physically impossible.
    """
    required_ids = tuple(required_ids)
    if not required_ids or len(set(required_ids)) != len(required_ids):
        raise ValueError('Nonempty unique original requirements required')
    observations = {}
    for row in evidence:
        if row.clause_id in observations:
            raise ValueError('Duplicate/conflicting clause observations')
        observations[row.clause_id] = row
    if set(observations) - set(required_ids):
        raise ValueError('Evidence cannot change the original task denominator')
    missing = sorted(set(required_ids) - set(observations))
    unknown = sorted(k for k, v in observations.items()
                     if v.value == 'unknown' or not v.independently_verified)
    failed = sorted(k for k, v in observations.items()
                    if v.value == 'false' and v.independently_verified)
    accepted = (terminal_reason == 'task_complete' and not execution_errors
                and not disallowed_change and not missing and not unknown
                and not failed)
    return dict(eligible_complete_positive=accepted,
                terminal_reason=terminal_reason, execution_errors=execution_errors,
                missing_clause_ids=missing, unknown_or_unverified_ids=unknown,
                verified_false_ids=failed, disallowed_change=disallowed_change,
                evaluation_failure=int(not accepted),
                note='Unverified is an unsuccessful evaluated attempt, not a verified negative quality label.')


def outcome_interval(required_ids, evidence):
    """Bounds on the all-original-clauses indicator; excludes runtime status.

    A trusted false clause certifies failure. All trusted true clauses certify
    success. Otherwise [0,1] records lack of identification, not probability.
    """
    admission = terminal_admission(required_ids, evidence,
                                   terminal_reason='task_complete', execution_errors=0)
    if admission['verified_false_ids']:
        return (0, 0)
    if admission['eligible_complete_positive']:
        return (1, 1)
    return (0, 1)


def certified_preference(required_ids, first, second):
    """Only strict outcome separation licenses a terminal-success preference.

    Surrogate composition scores remain reportable under their own objective;
    this routine does not relabel them as independently verified preferences.
    """
    a, b = outcome_interval(required_ids, first), outcome_interval(required_ids, second)
    if a[0] > b[1]:
        return 'first'
    if b[0] > a[1]:
        return 'second'
    if a[0] == a[1] == b[0] == b[1]:
        return 'tie_on_terminal_indicator'
    return 'not_identified'
