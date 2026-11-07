"""Prospective shared input for reconstructed demonstrations and live edits.

History records executed calls and their actual outcomes. Reconstructed commands
are never presented as model prose or historical designer operations. This
module is not imported by any closed experiment.
"""
import copy

from edit_request import build_request, dumps
from shared_action_contract import digest

CONTRACT = 'executed-edit-v1'


def build_executed_request(*, context, tools, scope, images, history,
                           next_attempt, policy, additional_text=''):
    if policy != {'name': CONTRACT, 'max_recent_attempts': 2, 'max_history_chars': 12000}:
        raise ValueError('Unsupported history policy')
    if type(next_attempt) is not int or next_attempt < 1:
        raise ValueError('Expected one-based next attempt')
    indices = [x['attempt'] for x in history]
    if indices != sorted(set(indices)) or any(type(i) is not int or not 1 <= i < next_attempt for i in indices):
        raise ValueError('History contains a duplicate, unordered, current or future attempt')
    for row in history:
        if set(row) != {'attempt', 'origin', 'calls', 'outcome', 'before_sha256', 'after_sha256'}:
            raise ValueError('Unexpected history fields; future measurements must stay out')
        if row['origin'] not in ('researcher_reconstruction', 'model_execution'):
            raise ValueError('Unattributed executed command')
        outcome = row['outcome']
        if outcome.get('kind') == 'native_reply':
            if set(outcome) != {'kind', 'receipt'} or not row['calls']:
                raise ValueError('Native call/reply missing')
            if not isinstance(outcome['receipt'].get('results'), list):
                raise ValueError('Missing native results')
        elif outcome.get('kind') == 'validation_rejection':
            if set(outcome) != {'kind', 'message', 'native_call_made'} or outcome['native_call_made'] is not False:
                raise ValueError('Rejection cannot claim execution')
            if row['before_sha256'] != row['after_sha256']:
                raise ValueError('Rejected call changed state')
        else:
            raise ValueError('Unknown outcome')
        for field in ('before_sha256', 'after_sha256'):
            value = row[field]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Missing state binding')
    for old, new in zip(history, history[1:]):
        if new['attempt'] == old['attempt'] + 1 and old['after_sha256'] != new['before_sha256']:
            raise ValueError('Discontinuous execution history')
    selected = copy.deepcopy(history[-policy['max_recent_attempts']:])
    while len(selected) > 1 and len(dumps(selected)) > policy['max_history_chars']:
        selected.pop(0)
    if len(dumps(selected)) > policy['max_history_chars']:
        raise ValueError('Newest complete outcome exceeds history budget')
    # Remove only the explicitly migrated legacy fields. No facts or current
    # observations are inferred from a target, evaluation field or history.
    current = copy.deepcopy(context)
    for name in ('recent_actual_tool_receipts', 'previous_actual_tool_receipt'):
        current.pop(name, None)
    accounting = dict(contract=CONTRACT, next_attempt=next_attempt,
        supplied_attempts=indices, included_attempts=[x['attempt'] for x in selected],
        complete_from_first_attempt=indices == list(range(1, next_attempt)),
        individual_exchanges_truncated=False, history_sha256=digest(selected))
    supplement = ('## Prior executed commands and actual outcomes\n'
        'Researcher reconstruction is a recorded tool demonstration, not historical designer operations or model speech. '
        'A proposed plan is not execution; a rejection made no native call. A successful reply applies only to its listed calls, '
        'not to an entire planned layout. Use the current object table to determine remaining work.\n'
        + dumps({'accounting': accounting, 'exchanges': selected}))
    if additional_text:
        supplement += '\n' + additional_text
    request = build_request(context=current, tools=tools, scope=scope, images=images, additional_text=supplement)
    return request, accounting
