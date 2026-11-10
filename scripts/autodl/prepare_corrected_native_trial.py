"""One corrected setup protocol, same task/weights/pool/budget; no inference."""
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    repo=Path(__file__).resolve().parents[2];stage=repo/'runs/autodl/20261110_complete_native_runtime'
    target=stage/'trial_protocol_v2.json'
    if target.exists():raise FileExistsError('Corrected protocol already frozen')
    first=json.loads((stage/'trial_actual_v1/result.json').read_text())
    abort=json.loads((stage/'trial_actual_v1/operator_abort.json').read_text())
    probe=json.loads((stage/'resume_binding_actual_v2/result.json').read_text())
    assert first['status']=='failed_no_restart' and abort['not_a_policy_quality_result']
    assert probe['doors_after']==1 and probe['doors_after_rollback']==0 and probe['restore_exact_state_equal']
    protocol=json.loads((stage/'trial_protocol.json').read_text())
    source=repo/'scripts/autodl/run_complete_native_trial.py'
    protocol['code_sha256'][str(source)]=sha(source)
    helper=repo/'scripts/autodl/restore_native_floor_binding.py'
    protocol['code_sha256'][str(helper)]=sha(helper)
    protocol['setup_recovery']=dict(reason=abort['reason'],first_attempt_model_forwards=len(first['model_ledger']['forwards']),
        first_attempt_output_tokens=first['model_ledger']['output_tokens'],first_attempt_is_not_quality_evidence=True,
        first_attempt_original_code_snapshot='source_trial_v1',
        actual_native_replay_probe_sha256=sha(stage/'resume_binding_actual_v2/result.json'),
        task_assets_weights_budgets_unchanged=True,unused_first_allowance_not_carried=True,
        new_attempt_inference_not_started=True)
    for path,h in protocol['code_sha256'].items():assert sha(path)==h,path
    target.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');print(target)


if __name__=='__main__':main()
