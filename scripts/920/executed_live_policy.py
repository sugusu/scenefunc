"""Live wiring for the already encoded executed-edit-v1 input contract.

Only current observations and recorded outcomes enter the request. This does
not approve a training or rollout plan, or infer a successful room from <done>.
"""
import copy
from pathlib import Path

from compact_catalog_context import pack_context
from edit_request import dumps, tools_and_scope
from executed_edit_request import CONTRACT, build_executed_request
from native_checkpoint import canonical_hash
from native_project import sha256
from shared_action_contract import schemas_from_observation


POLICY = dict(name=CONTRACT, max_recent_attempts=2, max_history_chars=12000)


def validate_policy(policy):
    if policy != POLICY:
        raise ValueError('Use the encoded executed-edit-v1 policy without implicit defaults')


def bind_outcome(attempt, before, after, *, call=None, reply=None,
                 proposed=None, rejection=None):
    """A rejected proposal is explicitly distinct from a native execution."""
    if (reply is None) == (rejection is None):
        raise ValueError('Exactly one reply or rejection is required')
    if rejection is not None:
        if before != after or call is not None:
            raise ValueError('A rejected proposal cannot change the native state')
        calls = [] if proposed is None else [copy.deepcopy(proposed)]
        outcome = dict(kind='validation_rejection', message=rejection, native_call_made=False)
    else:
        if call is None or proposed is not None:
            raise ValueError('Native outcome requires its actual validated call')
        calls = [copy.deepcopy(call)]
        outcome = dict(kind='native_reply', receipt=copy.deepcopy(reply))
    return dict(attempt=attempt, origin='model_execution', calls=calls,
                outcome=outcome, before_sha256=before, after_sha256=after)


def build_session_request(session, observation, plan, rounds, feedback, remaining):
    validate_policy(plan['input_contract'])
    if type(remaining) is not int or remaining < 0:
        raise ValueError('Remaining attempts must be nonnegative')
    history = [copy.deepcopy(row['execution']) for row in rounds]
    if [row['attempt'] for row in history] != list(range(1, len(history)+1)):
        raise ValueError('Incomplete current-run history')
    state_hash = canonical_hash(session.scene.to_state_dict())
    if history and history[-1]['after_sha256'] != state_hash:
        raise ValueError('History is not bound to the actual current native state')
    # Compact only repeated current catalog facts. Do not also supply a second,
    # differently truncated legacy history from the checkpoint.
    context, _ = pack_context(session.current_context())
    event = plan.get('current_activity_request')
    if event is not None:
        if set(event) != {'request', 'authorship', 'scope'} or event['authorship'] != 'researcher_specified_pre_action_activity':
            raise ValueError('Undeclared current activity request')
        context['current_activity_request'] = copy.deepcopy(event)
    tools, native_scope = tools_and_scope(observation['system'], schemas_from_observation(observation))
    scope = session.project.task['design_request'] + '\n' + native_scope
    item = plan['prompt_supplement']
    if sha256(Path(item['path'])) != item['sha256']:
        raise ValueError('Frozen task/knowledge supplement changed')
    extra = (f'Remaining tool attempts including rejections: {remaining}.\n'
             'History covers this rollout only; inherited checkpoint history is not replayed. '
             'Current objects still exist. No scaling, articulation, wall or service edits. '
             'A rejection lists proposed calls only; none was executed. '
             'Stopping does not establish whole-room success.\n'
             '## Common task/knowledge supplement\n' + Path(item['path']).read_text() +
             '\nCURRENT MEASURED FEEDBACK (declared proxies; no reference solution):\n' + dumps(feedback))
    request, accounting = build_executed_request(context=context, tools=tools, scope=scope,
        images=observation['image_paths'], history=history, next_attempt=len(history)+1,
        policy=plan['input_contract'], additional_text=extra)
    accounting['current_native_state_sha256'] = state_hash
    return request, accounting
