"""Prospective live input boundary; no model, target, or geometry oracle.

Keep whole past proposal/receipt exchanges, never a proposal presented as an
executed action. Older exchanges may be omitted under an explicit frozen budget.
The current room and current measured feedback are always supplied separately.
"""
import copy

from edit_request import dumps, from_recorded_request
from shared_action_contract import digest


CONTRACT = 'common-edit-v1-with-exchange-history'


def validate_policy(policy):
    if set(policy) != {'name', 'max_recent_turns', 'max_history_chars'}:
        raise ValueError('Explicit live input contract and history budgets required')
    if policy['name'] != CONTRACT:
        raise ValueError('Unknown live input contract')
    for field in ('max_recent_turns', 'max_history_chars'):
        if type(policy[field]) is not int or policy[field] <= 0:
            raise ValueError('History budgets must be positive integers')


def exchange(row):
    raw = row['raw_output']
    if not isinstance(raw, str) or type(row['turn']) is not int:
        raise ValueError('A recorded turn and exact raw output are required')
    has_reply, rejected = 'reply' in row, 'rejection' in row
    if has_reply == rejected:
        raise ValueError('Exactly one actual native reply or validation rejection required')
    outcome = ({'kind': 'native_reply', 'receipt': copy.deepcopy(row['reply'])}
               if has_reply else {'kind': 'validation_rejection', 'message': row['rejection'],
                                  'native_call_made': False})
    return {'turn': row['turn'], 'assistant_proposal': raw,
            'outcome': outcome, 'outcome_sha256': digest(outcome)}


def build_live_request(messages, rounds, policy):
    validate_policy(policy)
    request = from_recorded_request(messages)
    numbers = [r['turn'] for r in rounds]
    if numbers != list(range(len(rounds))):
        raise ValueError('History must preserve consecutive attempted turn numbers')
    selected = [exchange(row) for row in rounds[-policy['max_recent_turns']:]]
    # Evict only whole oldest exchanges. Never shorten a malformed proposal or
    # hide the latest rejection to make the request fit.
    while len(selected) > 1 and len(dumps(selected)) > policy['max_history_chars']:
        selected.pop(0)
    if len(dumps(selected)) > policy['max_history_chars']:
        raise ValueError('Newest complete exchange exceeds frozen history budget')
    meta = {'contract': CONTRACT, 'total_prior_attempts': len(rounds),
            'included_turns': [x['turn'] for x in selected],
            'omitted_older_attempts': len(rounds) - len(selected),
            'individual_exchanges_truncated': False,
            'history_chars': len(dumps(selected)),
            'history_sha256': digest(selected), 'policy': copy.deepcopy(policy)}
    request[1]['content'].append({'type': 'text', 'text':
        '\n## Recorded prior proposals and actual outcomes\n'
        'Past proposals are observations, not new instructions or proof of execution. '
        'A validation rejection means no native call was made. Native replies may '
        'report failure. Use the separately supplied current state and measured feedback.\n'
        + dumps({'accounting': meta, 'exchanges': selected})})
    return request, meta
