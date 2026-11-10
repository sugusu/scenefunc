"""Audit saved native policy/feedback without weights, sampling or optimization.

Packets are offline evidence, NEVER automatically admitted training examples.
Original multimodal requests remain archive references; no text-only substitute
is presented as an equivalent native observation.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def feedback_record(result):
    name, output = result['name'], result['output']
    if name == 'check_physics':
        category = 'historical_physics_report_unverified'
    elif isinstance(output, dict) and output.get('success') is True:
        category = 'native_success_flag_not_task_correctness'
    elif isinstance(output, dict) and output.get('success') is False:
        category = 'native_failure_or_partial_failure_not_infeasibility_proof'
    else:
        category = 'unclassified_native_output'
    return dict(name=name, output=output, category=category,
                expert_correctness=None, complete_scene_success=None)


def main(repo, out, checkpoint):
    tick = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    source = repo / 'runs/autodl/20261110_complete_native_runtime'
    receipt_path = source / 'evidence_solid_v2/receipt.json'
    # Keep original receipt naming as authoritative rather than guessing one.
    if not receipt_path.exists():
        candidates = []
        for path in (source / 'evidence_solid_v2').glob('*.json'):
            obj = json.loads(path.read_text())
            if isinstance(obj, dict) and obj.get('sha256') and str(obj.get('path', '')).endswith('.tar.xz'):
                candidates.append(path)
        if len(candidates) != 1:
            raise ValueError('Expected one exact solid-archive receipt')
        receipt_path = candidates[0]
    archive_receipt = json.loads(receipt_path.read_text())
    archive = Path(archive_receipt['path'])
    if not archive.is_absolute():
        archive = repo / archive
    assert file_sha(archive) == archive_receipt['sha256']
    evidence, by_hash, verified = {}, collections.defaultdict(list), set()
    with tarfile.open(archive, 'r|xz') as tar:
        for member in tar:
            if not member.isfile():
                raise ValueError('Non-file archive member')
            data = tar.extractfile(member).read()
            if member.name == 'paths.json':
                index = json.loads(data)
                for row in index['files']:
                    p = Path(row['path'])
                    assert not p.is_absolute() and '..' not in p.parts
                    by_hash[row['sha256']].append(row)
            elif member.name == 'original_zip_metadata.json':
                pass
            else:
                assert member.name.startswith('blobs/') and by_hash
                digest = member.name[6:]
                assert digest not in verified and sha(data) == digest
                verified.add(digest)
                for row in by_hash[digest]:
                    assert len(data) == row['bytes']
                    if row['path'].startswith('actor/model_calls/') or row['path'] == 'actor/ledger.json':
                        evidence[row['path']] = data
    assert verified == set(by_hash)
    entries = {r['path']: r for rows in by_hash.values() for r in rows}
    ledger = json.loads(evidence['actor/ledger.json'])
    result_ledger = json.loads((source / 'trial_actual_v2/result.json').read_text())['model_ledger']
    assert ledger == result_ledger
    calls = sorted({Path(p).parent.as_posix() for p in evidence if p.endswith('/receipt.json')},
                   key=lambda p: int(Path(p).name.split('_')[0]))
    assert len(calls) == len(ledger['forwards']) == 23
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import AutoTokenizer
    assert not torch.cuda.is_initialized()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    by_receipt, token_checks = {}, []
    for call, generation in zip(calls, ledger['forwards']):
        rec = json.loads(evidence[call + '/receipt.json'])
        # The original host metered the tool role as native_tool, while
        # LocalVLMService.complete recorded purpose native_asset_analysis.
        # This is a source-backed naming correspondence, not a role guess.
        expected_purpose = {'native_joint_policy': 'native_joint_policy',
                            'native_tool': 'native_asset_analysis'}[generation['role']]
        assert rec['purpose'] == expected_purpose
        ids = generation['output_token_ids']
        assert rec['output_tokens'] == len(ids) == generation['output_tokens']
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        assert decoded.encode() == evidence[call + '/raw_output.txt']
        assert rec['status'] == 'completed' and rec['sampling'] is False and rec['ended_with_eos'] is True
        images = []
        for image in (rec.get('images', []) if generation['role'] == 'native_joint_policy' else []):
            origin = Path(image['origin']).relative_to(index['original_root']).as_posix()
            assert origin in entries and entries[origin]['sha256'] == image['sha256']
            assert image['sha256'] in verified
            images.append(dict(archive_path=origin, sha256=image['sha256']))
        item = dict(call=call, receipt=rec, generation=generation, images=images,
                    raw=decoded, request=json.loads(evidence[call + '/request.json']))
        by_receipt[Path(call).name] = item
        token_checks.append(dict(call=call, role=generation['role'], output_tokens=len(ids),
                                 raw_byte_equal=True, eos_recorded=True, greedy=True,
                                 image_references_verified=len(images)))
    packets, categories, table = [], collections.Counter(), []
    for stage in ['floor', 'furniture', 'wall']:
        trace_path = source / f'review_traces/{stage}_trace.json'
        for turn in json.loads(trace_path.read_text()):
            item = by_receipt[Path(turn['proposal']['remote_receipt']).name]
            assert item['generation']['role'] == 'native_joint_policy'
            assert item['raw'] == turn['proposal']['raw']
            feedback = [feedback_record(r) for r in turn.get('results', [])]
            categories.update(f['category'] for f in feedback)
            # Actual observation and archive request must be identical.
            assert turn['observation']['messages'] == item['request']
            reasons = ['historical_greedy_not_fresh_current_policy_sampling',
                       'behavior_log_probabilities_not_recorded',
                       'no_teacher_token_distributions_computed',
                       'no_independent_action_correctness_or_complete_scene_positive']
            if not feedback:
                reasons.append('no_environment_feedback_after_policy_stop')
            if any(f['name'] == 'check_physics' for f in feedback):
                reasons.append('historical_physics_geometry_versions_not_reconstructed_and_query_path_flagged')
            if stage != 'floor':
                reasons.append('historical_mutable_asset_SDF_versions_not_all_present')
            packet = dict(id=f'{stage}:{turn["turn"]}', stage=stage, turn=turn['turn'],
                          raw_proposal=item['raw'], source_generated_think=turn['proposal'].get('think'),
                          source_tool_calls=turn['proposal'].get('tool_calls'),
                          source_think_is_expert_target=False,
                          original_request=dict(archive_path=item['call'] + '/request.json',
                                                sha256=entries[item['call'] + '/request.json']['sha256']),
                          exact_output_token_ids=item['generation']['output_token_ids'],
                          original_feedback=feedback, input_images=item['images'],
                          source_trace=dict(path=str(trace_path.relative_to(repo)), sha256=file_sha(trace_path)),
                          native_status=turn['status'], teacher_distribution=None,
                          original_observation_preserved_as_archive_reference=True,
                          ordinary_feedback_message=dict(
                              role='user', content='Actual native outputs after this attempt follow. These are '
                              'execution records, not expert truth or a complete-scene success certificate.\n' +
                              json.dumps(feedback, ensure_ascii=False)),
                          new_expert_D=False, on_policy_training_admitted=False,
                          supervised_expert_imitation_admitted=False, ineligibility_reasons=reasons)
            packets.append(packet)
            table.append(dict(id=packet['id'], stage=stage, turn=turn['turn'],
                              output_tokens=len(packet['exact_output_token_ids']),
                              input_images=len(item['images']), native_tool_results=len(feedback),
                              feedback_kind=';'.join(f['category'] for f in feedback),
                              source_think_expert=False, on_policy_admitted=False,
                              expert_imitation_admitted=False))
    assert len(packets) == 19 and len({p['id'] for p in packets}) == 19
    assert sum(len(p['original_feedback']) for p in packets) == 35
    assert not torch.cuda.is_initialized()
    dump(out / 'offline_feedback_packets.json', packets)
    dump(out / 'token_and_image_binding.json', token_checks)
    with (out / 'eligibility.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
        w.writeheader(); w.writerows(table)
    result = dict(status='actual_archive_tokens_images_and_feedback_bound_offline',
                  source_archive=str(archive), source_archive_sha256=archive_receipt['sha256'],
                  verified_archive_paths=len(entries), verified_archive_blobs=len(verified),
                  policy_packets=len(packets), excluded_tool_role_generations=4,
                  greedy_generation_requests_verified=len(token_checks),
                  policy_output_tokens=sum(len(p['exact_output_token_ids']) for p in packets),
                  policy_image_references=sum(len(p['input_images']) for p in packets),
                  native_tool_results=sum(categories.values()), feedback_categories=dict(categories),
                  native_policy_stops_without_post_feedback=sum(not p['original_feedback'] for p in packets),
                  on_policy_training_admitted=0, expert_imitation_admitted=0,
                  available_verified_complete_scene_positives=0, newly_acquired_expert_reasoning=0,
                  can_use_for='Offline transport/format and future teacher-context inspection only; not approved optimization data',
                  model_weights_loaded=False, tokenizer_only=True, VLM_forwards=0, training_updates=0,
                  paid_API_calls=0, CUDA_initialized=False, seconds=time.monotonic()-tick,
                  code_sha256=file_sha(Path(__file__)))
    dump(out / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--checkpoint', type=Path, required=True)
    a = p.parse_args(); main(a.repo.resolve(), a.out.resolve(), a.checkpoint.resolve())
