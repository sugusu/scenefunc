"""CPU regression using the real failed observation and context, no model calls."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from jsonschema.exceptions import ValidationError
from shared_action_contract import ActionContract as OldContract
from shared_action_contract_v2 import ActionContract, digest, schemas_from_observation
from resume_interrupted_edit import restore_interrupted_round
from run_edit_policy_executed_v2 import compact_rejection_message
from executed_live_policy import bind_outcome, build_session_request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    root = Path(plan['interrupted_rollout']['directory'])
    obs = json.loads((root/'final_observation.json').read_text())
    ctx = json.loads((root/'house/session_initial_context.json').read_text())
    state = json.loads((root/'final_state.json').read_text())
    tools = schemas_from_observation(obs)
    checks = []

    def rejects(name, fn):
        try:
            fn()
        except (ValueError, ValidationError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)

    rejects('original_failure_reproduced', lambda: OldContract(tools, ctx))
    original_hash = digest(ctx)
    contract = ActionContract(tools, ctx, backend='existing')
    assert digest(ctx) == original_hash and 'asset_catalog' not in ctx
    checks.append('real_existing_context_unmodified_without_catalog')
    call = dict(name='move_furniture_tool', arguments=dict(
        object_id='recon_mobile_table_0', x=-1.75, y=-1.1, yaw=0))
    assert contract.validate(json.dumps(call), ctx)['kind'] == 'tool'
    checks.append('valid_real_object_native_schema_accepted')
    bad = copy.deepcopy(call); bad['arguments']['object_id'] = 'nonexistent'
    rejects('unknown_id', lambda: contract.validate(json.dumps(bad), ctx))
    rejects('unknown_tool', lambda: contract.validate('{"name":"invented","arguments":{}}', ctx))
    rejects('duplicate_tool', lambda: ActionContract(tools+[tools[0]], ctx, backend='existing'))
    rejects('missing_tool', lambda: ActionContract(tools[:-1], ctx, backend='existing'))
    rejects('wrong_backend', lambda: ActionContract(tools, ctx, backend='catalog'))
    rejects('nonempty_existing_catalog', lambda: ActionContract(tools, {**ctx, 'asset_catalog':[{'asset_id':'fake'}]}, backend='existing'))
    rejects('stale_state', lambda: contract.validate(json.dumps(call), {**ctx, 'total_tool_batches':999}))
    rejects('duplicate_json_keys', lambda: contract.validate('{"name":"__done__","name":"__done__","arguments":{"unmet_or_unknown":"unknown"}}', ctx))
    rejects('nonfinite', lambda: contract.validate(json.dumps(call).replace('-1.75','NaN'), ctx))
    bad = copy.deepcopy(call);bad['arguments']['extra'] = True
    rejects('extra_argument', lambda: contract.validate(json.dumps(bad), ctx))
    # Exercise the real recovery function with actual saved model output; the
    # session is a read-only state stand-in, not a claimed native execution.
    session = SimpleNamespace(scene=SimpleNamespace(to_state_dict=lambda:state), current_context=lambda:ctx,
        project=SimpleNamespace(task=json.loads((root/'final_checkpoint/task.json').read_text())))
    with tempfile.TemporaryDirectory() as tmp:
        report = {}
        rejection = restore_interrupted_round(plan, session, obs, report, Path(tmp),
            compact_rejection_message, bind_outcome)
        assert 'Expected one complete tool block' in rejection
        assert report['model_calls'] == report['tool_attempts'] == 1 and report['native_calls'] == 0
        assert report['gpu_subprocess_seconds'] == 24.04812056897208
        assert not report['rounds'][0]['new_model_call']
        checks.append('actual_bad_output_rejected_with_original_counts_and_cost')
        audit = json.loads((root/'final_audit.json').read_text())
        feedback = {**audit['feedback'], 'unknown':audit['limits'], 'previous_rejection':rejection}
        request, metadata = build_session_request(session, obs, plan, report['rounds'], feedback, 5)
        text = json.dumps(request)
        assert 'Expected one complete tool block' in text and metadata['included_attempts'] == [1]
        assert 'Remaining tool attempts including rejections: 5.' in text
        assert 'recovered_from' not in text
        checks.append('next_real_request_retains_rejection_and_remaining_five_attempts')
        changed = copy.deepcopy(plan);changed['interrupted_rollout']['sha256']['RESULT.json'] = '0'*64
        rejects('changed_interruption_evidence', lambda: restore_interrupted_round(
            changed, session, obs, {}, Path(tmp), compact_rejection_message, bind_outcome))
        wrong = SimpleNamespace(scene=SimpleNamespace(to_state_dict=lambda:{'changed':True}), current_context=lambda:ctx)
        rejects('restored_state_mismatch', lambda: restore_interrupted_round(
            plan, wrong, obs, {}, Path(tmp), compact_rejection_message, bind_outcome))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(status='passed', checks=checks, count=len(checks),
        new_model_calls=0, new_training_updates=0, native_edits=0,
        scope='Actual captured schemas/context/output; read-only stand-in for restoration. Native restore checked separately.'),indent=2)+'\n')
    print(a.out.read_text())


if __name__ == '__main__':
    main()
