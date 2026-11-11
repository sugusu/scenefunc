"""Encode one held-out design demonstration with the actual local processor.

This is a new pair of scoring-format examples, not a replay of old neural
outputs or a training run. Pine remains excluded from the training split.
Source-bound reference views are shared by both arms; any difference from the
historical deployment images is reported rather than silently equated.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace
from typing import Any


PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
OFFICIAL_SHA = '9e4d18f67e746af16f830d7e8d0da714964a43bfa320ef091a0092b3e9bb2569'
CASE = 'runs/920/P920-EDITSTUDY-v1_20261106T074752Z/evaluation/Pine_action_only_17/recorded/round_000/'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False)+'\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_digest(tensor):
    return digest(tensor.detach().cpu().contiguous().numpy().tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--official-source', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('No overwrite of example preparation evidence')
    a.out.mkdir(parents=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    source_receipts = []

    def blob(path):
        raw = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show', PIN+':'+path])
        source_receipts.append(dict(path=path, bytes=len(raw), sha256=digest(raw)))
        return raw

    try:
        unit_path = Path('runs/autodl/20261111_pine_decision_supervision/actual_v1/one_decision_example.json')
        unit = json.loads(unit_path.read_text())
        assert unit['training_admitted'] is False
        original_messages = json.loads(blob(CASE+'request.json'))
        old_encoding = json.loads(blob(CASE+'model/RESULT.json'))['encoding']
        messages = copy.deepcopy(original_messages)
        image_paths = []
        comparisons = []
        for index, image in enumerate(unit['input']['five_view_references']):
            raw = blob(image['git_path'])
            assert digest(raw) == image['sha256']
            path = a.scratch/Path(image['git_path']).name
            path.write_bytes(raw)
            image_paths.append(str(path.resolve()))
            comparisons.append(dict(slot=index, reference_sha256=digest(raw),
                old_deployed_sha256=old_encoding['images'][index]['sha256'],
                same_bytes=digest(raw)==old_encoding['images'][index]['sha256'],
                pixel_equality_not_inferred_from_different_file_hash=True))
        index = 0
        for message in messages:
            if not isinstance(message.get('content'), list):
                continue
            for part in message['content']:
                if part.get('type') == 'image':
                    part['image'] = image_paths[index]
                    index += 1
        assert index == 5
        # The original context-sensitive contract is shared with deployment.
        contract_raw = blob('scripts/920/shared_action_contract.py')
        contract_path = a.scratch/'shared_action_contract.py'
        contract_path.write_bytes(contract_raw)
        contract_module = load_module('pine_original_contract', contract_path)
        context = contract_module.context_from_request(messages)
        system = next(m['content'] for m in messages if m['role']=='system')
        tools = contract_module.schemas_from_observation(dict(system=system))
        action = unit['target']['first_native_action']
        contract_module.ActionContract(tools, context).validate(json.dumps(action), context)

        raw = a.official_source.read_bytes()
        assert digest(raw) == OFFICIAL_SHA
        tree = ast.parse(raw.decode())
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                  and x.name == '_build_messages')
        namespace = dict(StepSample=SimpleNamespace, Any=Any)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(a.official_source), 'exec'), namespace)

        # Existing encoder has a lazy processor and never creates a model here.
        engine_path = Path('paris/src/native_local_inference_20261104/local_vlm.py')
        engine_module = load_module('pine_existing_local_encoder', engine_path)
        engine = engine_module.LocalQwenEngine(a.checkpoint, a.scratch/'unused_model_logs',
            max_pixels=65536, max_input_tokens=18432, max_new_tokens=1024)
        import torch
        torch.set_num_threads(1)
        prompt, prompt_receipt = engine.encode(messages)
        normalized, images, _ = engine.normalize_messages(messages)
        # Remapping local image filenames must not alter the rendered text.
        old_placeholder_messages = copy.deepcopy(original_messages)
        old_image_index = 0
        for message in old_placeholder_messages:
            if isinstance(message.get('content'), list):
                for part in message['content']:
                    if part.get('type') == 'image':
                        part['image'] = 'local_image_'+str(old_image_index)
                        old_image_index += 1
        old_template_text = engine.processor.apply_chat_template(
            old_placeholder_messages, tokenize=False, add_generation_prompt=True)
        prompt_text = engine.processor.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True)
        assert old_template_text == prompt_text
        write(a.out/'deployment_prefix.json', messages)
        write(a.out/'image_identity_comparison.json', dict(
            views=comparisons,
            common_to_both_new_examples=True,
            reference_views_bound_to_initial_native_state=True,
            exact_historical_five_image_replay=all(x['same_bytes'] for x in comparisons),
            historical_input_token_count=old_encoding['input_tokens'],
            new_prompt_token_count=prompt_receipt['input_tokens'],
            scope='Same old task/system text and reference-state views; do not '
                  'claim four unequal file hashes prove unequal decoded pixels.'))

        records = []
        action_ids = {}
        for arm, decision in [('action_only', ''), ('decision_action', unit['target']['decision'])]:
            sample = SimpleNamespace(stage='furniture', system=system,
                task_context='', memory_str='', scene_images=image_paths,
                image_path=None, assistant_content=decision, tool_calls=[action])
            official_full, _ = namespace['_build_messages'](sample, no_thinking=not bool(decision))
            assistant = official_full[-1]
            full_messages = normalized+[assistant]
            text = engine.processor.apply_chat_template(full_messages,
                tokenize=False, add_generation_prompt=False)
            full = engine.processor(text=[text], images=images,
                return_tensors='pt', padding=False)
            boundary = prompt['input_ids'].shape[1]
            assert torch.equal(prompt['input_ids'], full['input_ids'][:, :boundary])
            assert torch.equal(prompt['pixel_values'], full['pixel_values'])
            assert torch.equal(prompt['image_grid_thw'], full['image_grid_thw'])
            labels = full['input_ids'].clone()
            labels[:, :boundary] = -100
            marker = engine.processor.tokenizer.encode('<tool_call>', add_special_tokens=False)
            target_ids = full['input_ids'][0, boundary:].tolist()
            positions = [i for i in range(len(target_ids)-len(marker)+1)
                         if target_ids[i:i+len(marker)] == marker]
            assert len(positions) == 1
            action_start = boundary+positions[0]
            action_mask = torch.zeros_like(labels, dtype=torch.bool)
            action_mask[:, action_start:] = True
            decision_mask = (labels != -100) & ~action_mask
            assert not bool((labels[:, :boundary] != -100).any())
            if not decision:
                assert not bool(decision_mask.any())
            assert int((labels != -100).sum()) == int(action_mask.sum()+decision_mask.sum())
            target = a.scratch/(arm+'.pt')
            torch.save(dict(inputs=full, labels=labels,
                action_mask=action_mask, decision_mask=decision_mask), target)
            action_ids[arm] = full['input_ids'][action_mask]
            record = dict(arm=arm, original_function_lines=[fn.lineno, fn.end_lineno],
                prompt_tokens=boundary, full_tokens=full['input_ids'].shape[1],
                action_tokens=int(action_mask.sum()), decision_tokens=int(decision_mask.sum()),
                prompt_ids_sha256=tensor_digest(prompt['input_ids']),
                pixel_sha256=tensor_digest(full['pixel_values']),
                target_action_text=engine.processor.tokenizer.decode(action_ids[arm].tolist(), skip_special_tokens=False),
                decision_text=engine.processor.tokenizer.decode(full['input_ids'][decision_mask].tolist(), skip_special_tokens=False),
                tensor_path=str(target.resolve()), tensor_file_sha256=digest(target.read_bytes()),
                RAM_only=True, training_admitted=False)
            records.append(record)
            write(a.out/(arm+'_assistant_target.json'), assistant)
            del full, labels, action_mask, decision_mask
        assert torch.equal(action_ids['action_only'], action_ids['decision_action'])
        assert engine._model is None and not torch.cuda.is_initialized()
        write(a.out/'encoding.json', dict(records=records, prompt=prompt_receipt,
            same_original_deployment_text=True, same_new_images_and_prompt=True,
            same_action_target_tokens=True, target_not_in_generation_prefix=True,
            original_assistant_serialization=True,
            whole_original_dataset_loader_run=False,
            optimizer_or_neural_forward_run=False,
            Pine_training_exclusion_unchanged=True))
        write(a.out/'source_receipts.json', dict(pin=PIN, files=source_receipts,
            official_pin='a0565e50ed7d32c0e09132b9311c0b9b91165405',
            official_source_sha256=OFFICIAL_SHA,
            encoder_path=str(engine_path), encoder_sha256=digest(engine_path.read_bytes()),
            prior_unit_sha256=digest(unit_path.read_bytes())))
        write(a.out/'result.json', dict(status='Two matched real-processor examples encoded',
            unique_positive_demonstrations=1, source_families=1,
            encodings=2, old_22_action_loader_not_rerun=True,
            new_model_calls=0, new_neural_forwards=0, new_training=0,
            weights_loaded=False, cuda_initialized=False,
            training_admitted=False, full_scene_success_not_inferred=True,
            elapsed_seconds=time.monotonic()-start,
            process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        print(json.dumps(dict(status='encoded',records=records,
            new_model_calls=0,weights_loaded=False),ensure_ascii=False))
    except BaseException as exc:
        write(a.out/'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
              elapsed_seconds=time.monotonic()-start, new_model_calls=0,
              new_training=0))
        raise


if __name__ == '__main__':
    main()
