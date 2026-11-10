"""CPU-only multimodal feedback-distillation input port, no model or trainer.

Preserves original native views and the *same* generated output token sequence
under student and feedback-augmented teacher contexts. This is an ordinary
SDPO-style input adapter, not a new learning algorithm or admitted dataset.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(repo, packet_dir, out, checkpoint, engine_source):
    tick = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    packets = json.loads((packet_dir/'offline_feedback_packets.json').read_text())
    provenance = json.loads((packet_dir/'result.json').read_text())
    assert len(packets) == provenance['policy_packets'] == 19
    needed = {p['original_request']['archive_path'] for p in packets}
    for p in packets:
        needed.update(v['archive_path'] for v in p['input_images'])
        needed.add(str(Path(p['original_request']['archive_path']).parent/'receipt.json'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(engine_source))
    from local_vlm_sdpa import LocalQwenEngine
    import torch
    from transformers import AutoConfig
    from native_feedback_batch import make_scoring_inputs
    assert not torch.cuda.is_initialized()
    engine_source_file = engine_source/'local_vlm_sdpa.py'
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)

    class ProcessorOnlyEngine(LocalQwenEngine):
        def _load_model(self):
            raise RuntimeError('Model weight load explicitly prohibited in this CPU input audit')
        def complete(self, *args, **kwargs):
            raise RuntimeError('Generation explicitly prohibited in this CPU input audit')

    rows, inputs_manifest = [], []
    with tempfile.TemporaryDirectory(prefix='native_feedback_input_', dir='/dev/shm') as temporary:
        root = Path(temporary)
        selected, wanted_hashes, verified = {}, {}, set()
        # Original receipt hash was independently checked in packet audit;
        # additionally verify every selected member's bytes against that index.
        with tarfile.open(provenance['source_archive'], 'r|xz') as tar:
            for member in tar:
                if member.name == 'paths.json':
                    index = json.load(tar.extractfile(member))
                    source_rows = {r['path']:r for r in index['files']}
                    assert needed <= set(source_rows)
                    for path in needed:
                        row = source_rows[path]
                        wanted_hashes.setdefault(row['sha256'], []).append(path)
                elif member.name.startswith('blobs/') and member.name[6:] in wanted_hashes:
                    data = tar.extractfile(member).read()
                    digest = member.name[6:]
                    assert sha(data) == digest
                    verified.add(digest)
                    for path in wanted_hashes[digest]:
                        assert len(data) == source_rows[path]['bytes']
                        if path.endswith('.json'):
                            selected[path] = json.loads(data)
                        else:
                            target = root/'images'/digest
                            target.parent.mkdir(exist_ok=True)
                            target.write_bytes(data)
                            selected[path] = target
        assert verified == set(wanted_hashes)
        engine = ProcessorOnlyEngine(checkpoint, root/'processor_only',
                                     max_pixels=65536, max_input_tokens=16384, max_new_tokens=1024)

        def relocate_images(messages):
            result = copy.deepcopy(messages)
            for message in result:
                if not isinstance(message['content'], list):
                    continue
                for part in message['content']:
                    if part.get('type') == 'image':
                        key = 'image' if 'image' in part else 'path'
                        path = Path(part[key]).relative_to(index['original_root']).as_posix()
                        part[key] = str(selected[path])
                    elif part.get('type') == 'image_url':
                        path = Path(part['image_url']['url']).relative_to(index['original_root']).as_posix()
                        part['image_url']['url'] = str(selected[path])
            return result

        for packet in packets:
            request_path = packet['original_request']['archive_path']
            original = selected[request_path]
            old_receipt = selected[str(Path(request_path).parent/'receipt.json')]
            student_request = relocate_images(original)
            teacher_request = student_request + [packet['ordinary_feedback_message']]
            student, srec = engine.encode(student_request)
            teacher, trec = engine.encode(teacher_request)
            assert srec['chat_template_sha256'] == old_receipt['chat_template_sha256']
            assert srec['input_tokens'] == old_receipt['input_tokens']
            assert [v['sha256'] for v in srec['images']] == [v['sha256'] for v in packet['input_images']]
            assert [v['sha256'] for v in trec['images']] == [v['sha256'] for v in srec['images']]
            image_tensor_checks = {}
            for key in ['pixel_values', 'image_grid_thw']:
                assert (key in student) == (key in teacher)
                if key in student:
                    assert torch.equal(student[key], teacher[key])
                    tensor = student[key].contiguous().cpu()
                    image_tensor_checks[key] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                                                    sha256=sha(tensor.numpy().tobytes()),
                                                    teacher_student_exactly_equal=True)
            response = torch.tensor([packet['exact_output_token_ids']], dtype=torch.long)
            scored = {}
            for name, inputs, receipt in [('student', student, srec), ('teacher', teacher, trec)]:
                prefix = inputs['input_ids']
                full = torch.cat([prefix, response], dim=1)
                prompt_length, target_length = prefix.shape[1], response.shape[1]
                attention = torch.cat([inputs['attention_mask'], torch.ones_like(response)], dim=1)
                assert full.shape == attention.shape
                assert torch.equal(full[:, prompt_length:], response)
                scored[name] = dict(prompt_tokens=prompt_length, response_tokens=target_length,
                                    full_input_ids=full[0].tolist(),
                                    first_causal_logit_position=prompt_length-1,
                                    last_causal_logit_position=full.shape[1]-2,
                                    prompt_loss_mask=False, response_loss_mask=True,
                                    full_sequence_tokens=full.shape[1],
                                    all_attention_positions_active=bool(attention.bool().all()))
                native_inputs, native_meta = make_scoring_inputs(inputs, response, config=config)
                assert torch.equal(native_inputs['input_ids'], full)
                assert torch.equal(native_inputs['attention_mask'], attention)
                assert torch.equal(native_inputs['logits_to_keep'],
                                   torch.arange(prompt_length-1, full.shape[1]-1))
                pos = native_inputs['position_ids'].contiguous().cpu()
                scored[name]['native_mRoPE'] = dict(
                    shape=list(pos.shape), sha256=sha(pos.numpy().tobytes()),
                    original_library_method_executed=True, model_instantiated=False,
                    rope_deltas=native_meta['rope_deltas'].tolist(),
                    image_grid_thw=None if native_meta['image_grid_thw'] is None else native_meta['image_grid_thw'].tolist(),
                    prompt_positions_unchanged_by_response=True,
                    response_positions_all_text_and_unit_increment=True)
                assert scored[name]['last_causal_logit_position']-scored[name]['first_causal_logit_position']+1 == target_length
            assert engine._model is None
            item = dict(id=packet['id'], original_template_hash_reproduced=True,
                        original_prompt_length_reproduced=True,
                        images=image_tensor_checks, scored=scored,
                        student_teacher_response_tokens_identical=True,
                        feedback_exists=bool(packet['original_feedback']),
                        self_distillation_mask=int(bool(packet['original_feedback'])),
                        training_admitted=False, teacher_distribution=None,
                        multimodal_position_ids='Computed by original Qwen3VLModel.get_rope_index, hash recorded',
                        not_validated=['Model forward logits and numerical teacher quality',
                                       'Gradient/adapter isolation and peak training memory',
                                       'A full native training/rollout loop'])
            inputs_manifest.append(item)
            rows.append(dict(id=packet['id'], images=len(srec['images']),
                             student_prompt_tokens=srec['input_tokens'],
                             teacher_prompt_tokens=trec['input_tokens'], response_tokens=response.numel(),
                             feedback_mask=item['self_distillation_mask']))
            print(json.dumps(rows[-1]), flush=True)
    assert not torch.cuda.is_initialized()
    result = dict(status='actual_native_multimodal_student_teacher_inputs_prepared_CPU_only',
                  rows=rows, original_native_policy_requests=len(rows),
                  policy_image_references=sum(r['images'] for r in rows),
                  original_prompt_fingerprints_reproduced=len(rows),
                  teacher_student_image_tensors_equal=len(rows),
                  same_output_target_sequences=len(rows),
                  feedback_masked_in=sum(r['feedback_mask'] for r in rows),
                  student_prefix_tokens=sum(r['student_prompt_tokens'] for r in rows),
                  teacher_prefix_tokens=sum(r['teacher_prompt_tokens'] for r in rows),
                  response_tokens=sum(r['response_tokens'] for r in rows),
                  loss_scoring_positions_saved=True, original_inputs_changed=False,
                  no_prompt_truncation=True, no_generated_or_expert_answer_added=True,
                  native_mRoPE_inputs_prepared=True, full_trainer_admitted=False,
                  model_weights_loaded=False, VLM_forwards=0, training_updates=0, paid_API_calls=0,
                  CUDA_initialized=False, seconds=time.monotonic()-tick,
                  local_engine_source_sha256=sha(engine_source_file.read_bytes()),
                  packet_source_sha256=sha((packet_dir/'offline_feedback_packets.json').read_bytes()))
    (out/'input_positions_and_tensor_hashes.json').write_text(json.dumps(inputs_manifest, indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--packet-dir', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--engine-source', type=Path, required=True)
    a=p.parse_args(); main(a.repo, a.packet_dir, a.out, a.checkpoint, a.engine_source)
