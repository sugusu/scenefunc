"""Recover only the observed STOPDATA pre-dispatch failure, without a new draw.

This is evaluation bookkeeping, not a general checkpoint retry mechanism.
All source evidence is hash-bound by the revised plan. A different failure or
state must fail closed before any model subprocess is started.
"""
import copy
import json
from pathlib import Path
import re
import shutil

from native_checkpoint import canonical_hash
from native_project import sha256
from shared_action_contract_v2 import ActionContract, schemas_from_observation


def restore_interrupted_round(plan, session, observation, result, out,
                              rejection_message, bind_outcome):
    source = plan['interrupted_rollout']
    root = Path(source['directory'])
    required = {'RESULT.json', 'final_state.json', 'final_observation.json',
                'round_000/request.json', 'round_000/model/raw_output.txt',
                'round_000/model/RESULT.json', 'final_checkpoint/manifest.json'}
    if not required.issubset(source['sha256']):
        raise ValueError('Incomplete interrupted evidence binding')
    for name, digest in source['sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError('Interrupted evidence changed: ' + name)
    old = json.loads((root/'RESULT.json').read_text())
    if (old['status'] != 'failed' or old['stop_reason'] != 'infrastructure_error'
            or old['error'] != 'ValueError: Expected the five catalog tools exactly once'
            or old['model_calls'] != 1 or old['tool_attempts'] != 1
            or old['native_calls'] != 0 or len(old['rounds']) != 1
            or old['cell_id'] != plan['cell_id']):
        raise ValueError('Not the approved pre-dispatch interruption')
    before = canonical_hash(session.scene.to_state_dict())
    if before != canonical_hash(json.loads((root/'final_state.json').read_text())):
        raise ValueError('Restored interrupted state differs; no generation allowed')
    original_tools = schemas_from_observation(json.loads((root/'final_observation.json').read_text()))
    if schemas_from_observation(observation) != original_tools:
        raise ValueError('Interrupted tool schemas changed')
    ActionContract(original_tools, session.current_context(), backend=plan['session_backend'])
    row = copy.deepcopy(old['rounds'][0])
    raw = (root/'round_000/model/raw_output.txt').read_text()
    if raw != row['raw_output'] or row.get('execution'):
        raise ValueError('Inconsistent raw output or prior dispatch')
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', raw, re.DOTALL)
    # Only this malformed observed output is replayed. Never turn a valid action
    # into a rejection, repair the model text, or silently execute a saved call.
    if not (row['model'].get('output_rejected_reason') or len(blocks) != 1
            or raw.count('<tool_call>') != 1):
        raise ValueError('Saved output is not the known format rejection')
    exc = ValueError('Expected one complete tool block')
    rejection = rejection_message(exc)
    row.update(rejection=rejection, full_validation_error=f'ValueError: {exc}',
               execution=bind_outcome(1, before, before, proposed=None, rejection=rejection),
               recovered_from=str(root/'round_000'), new_model_call=False)
    rd = Path(out)/'round_000'
    rd.mkdir(exist_ok=False)
    for name in ('request.json', 'model/raw_output.txt', 'model/RESULT.json'):
        target = rd/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/'round_000'/name, target)
    (rd/'validation_error.json').write_text(json.dumps(dict(
        feedback=rejection, full_error=row['full_validation_error'],
        replayed_after_interface_fix=True), indent=2)+'\n')
    result.update(rounds=[row], model_calls=1, tool_attempts=1, native_calls=0,
                  generated_tokens=old['generated_tokens'],
                  gpu_subprocess_seconds=old['gpu_subprocess_seconds'],
                  carried_gpu_subprocess_seconds=old['gpu_subprocess_seconds'],
                  interrupted_result_sha256=source['sha256']['RESULT.json'])
    return rejection
