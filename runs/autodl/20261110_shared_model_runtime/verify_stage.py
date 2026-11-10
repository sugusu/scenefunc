"""Verify the real inference/data lineage, not a learned method or room score."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    protocol = read(ROOT / 'protocol.json')
    preflight = read(ROOT / 'preflight.json')
    result = read(ROOT / 'actual_v1/result.json')
    ledger = read(ROOT / 'actual_v1/actor/shared_model_ledger.json')
    diagnostic = read(ROOT / 'token_parser_diagnostic.json')
    cache = read(ROOT / 'prefix_cache_audit.json')
    assert preflight['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert preflight['model_loaded'] is False and preflight['model_forwards'] == 0
    assert result['status'] == 'bounded_runtime_probe_completed'
    assert result['actor_process_terminal'] and result['actor_exit_code'] == -15
    assert not result['blocked_parent_network'] and not ledger['blocked_network']
    assert result['training_steps'] == result['new_complete_scenes'] == result['paid_API_calls'] == 0
    assert not result['policy_action_dispatch'] and not result['whole_runtime_ready']
    forwards = ledger['forwards']
    assert len(forwards) == result['actual_model_forwards'] == 3 <= protocol['max_model_forwards']
    assert [r['role'] for r in forwards] == protocol['fixed_order']
    assert len(ledger['model_loads']) == 1 and ledger['model_loads'][0]['all_parameters_frozen']
    assert len({r['model_object_id'] for r in forwards}) == 1
    assert len({r['first_parameter_data_ptr'] for r in forwards}) == 1
    assert all(r['status'] == 'returned' and r['generation_started'] for r in forwards)
    assert not ledger['incomplete_forward_accounting']
    assert sum(len(r['output_ids']) for r in forwards) == ledger['total_output_tokens'] == 464
    assert ledger['total_output_tokens'] <= protocol['max_total_output_tokens']
    integrity = read(ROOT / 'actual_v1/input_integrity.json')
    for p, expected in integrity.items():
        assert sha(p) == expected, p
    # Full shard hashes were actually checked during preflight; avoid a second
    # 17.5GB read merely for report regeneration. Current stat identity recorded.
    for r in protocol['weight_sha256_verified']:
        st = Path(r['path']).stat()
        assert (st.st_size, st.st_mtime_ns, st.st_ino) == (r['bytes'], r['mtime_ns'], r['inode'])
    obs = read(protocol['phase_input'])
    sys.path.insert(0, str(REPO / 'scripts/autodl'))
    from phase_objective import PHASE_MODE
    component = next(json.loads(l) for l in Path(protocol['source_material']).read_text().splitlines()
        if (lambda d: d['unit_id'] == protocol['fixed_source_unit'] and
            d['arm'] == 'separate_phase_v3' and d['stage'] == 'action')(json.loads(l)))
    messages = copy.deepcopy(component['messages'])
    assert messages[0]['content'].endswith(PHASE_MODE['action'])
    messages[0]['content'] = messages[0]['content'][:-len(PHASE_MODE['action'])]
    assert obs['messages'] == messages
    assert set(obs) == {'messages', 'images', 'state_sha256'}
    state = read(protocol['state_file'])
    assert hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == obs['state_sha256']
    for row in obs['images']:
        assert sha(row['path']) == row['sha256']
    request_dir = ROOT / 'actual_v1/actor'
    for i in (1, 3):
        payload = read(request_dir / f'request_{i:02d}.json')
        assert payload['messages'] == obs['messages'] and payload['images'] == obs['images']
        assert set(payload) == set(obs) | {'arm'}
        receipt = read(request_dir / f'request_{i:02d}_receipt.json')
        assert receipt['request_sha256'] == sha(request_dir / f'request_{i:02d}.json')
        assert receipt['status'] == 'returned'
    assert read(request_dir / 'request_02.json') == read(protocol['tool_input'])
    assert len(result['client_conditions'][2]['result']['stages']) == 1
    assert result['client_conditions'][0]['policy_status'] == 'invalid_action'
    assert result['client_conditions'][2]['policy_status'] == 'invalid_decision'
    added = {r['content']: r['id'] for r in read(Path(protocol['checkpoint']) / 'tokenizer.json')['added_tokens']}
    for text, row in diagnostic['token_mapping'].items():
        assert row['id'] == added[text] and row['roundtrip'] == text
    assert sha(diagnostic['native_parser_source']) == diagnostic['native_parser_source_sha256']
    for row, fwd in zip(diagnostic['rows'], [forwards[0], forwards[2]]):
        assert row['output_tokens'] == len(fwd['output_ids'])
        assert row['closing_think_count'] == fwd['output_ids'].count(added['</think>']) == 0
        assert row['closing_tool_count'] == fwd['output_ids'].count(added['</tool_call>'])
        assert not row['ended_at_cap'] and not row['action_dispatched']
        assert row['native_think'] is None
    assert cache['longest_common_prefix_tokens'] == 143
    assert not cache['model_loaded'] and cache['new_model_forwards'] == 0
    assert not cache['teacher_or_synthetic_D_inserted'] and not cache['paired_action_actually_executed']
    assert cache['pixel_inputs_identical'] and not cache['whole_prefixes_identical']
    assert cache['input_tensor_sha256']['action'] == result['client_conditions'][0]['result']['stages'][0]['input_tensor_sha256']
    assert cache['input_tensor_sha256']['decision'] == result['client_conditions'][2]['result']['stages'][0]['input_tensor_sha256']
    closed = read(ROOT / 'CLOSED.json')
    assert closed['status'] == 'closed_after_terminal_run' and not closed['allowance_carries_forward']
    assert not closed['automatic_retry_allowed']
    manifest = read(ROOT / 'artifact_manifest.json')
    for row in manifest:
        p = REPO / row['path']
        assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    with zipfile.ZipFile(ROOT / 'evidence.zip') as z:
        assert z.testzip() is None
        for row in manifest:
            assert hashlib.sha256(z.read(row['path'])).hexdigest() == row['sha256']
        archive_count = len(z.namelist())
    receipt = {'status': 'passed', 'manifest_files_verified': len(manifest),
        'archive_files_verified': archive_count, 'actual_model_forwards_verified': 3,
        'single_frozen_model_load': True, 'state_and_input_lineage_verified': True,
        'strict_contract_failures_retained': 2, 'whole_room_or_learning_success_verified': False,
        'weight_payload_rehashed_in_this_verifier': False,
        'weight_payload_hashes_verified_in_preflight': True}
    (ROOT / 'delivery_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
