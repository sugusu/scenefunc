"""Validate real target text and fail-closed material views; CPU only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoTokenizer
    from conditional_material_dataset import EqualMaterialExamples, component_batch
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest = json.loads(args.manifest.read_text())
    records = manifest['records']
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    tool = tokenizer.convert_tokens_to_ids('<tool_call>')
    texts, checked = [], []
    for row in records:
        batch = component_batch(row)
        target = batch['labels'][batch['labels'] != -100].tolist()
        text = tokenizer.decode(target, skip_special_tokens=False)
        if row['component'] == 'decision':
            assert eos not in target and tool not in target
            assert text.startswith('<think>') and text.rstrip().endswith('</think>')
            assert int(batch['labels'][0, -1]) != -100
            assert int(batch['labels'][0, row['source_prompt_tokens']]) != -100
        else:
            assert target.count(eos) == 1
            assert (tool in target) == (row['target_kind'] == 'tool_action')
        texts.append(dict(id=row['id'], arm=row['arm'], target=text))
    checked.append('all64_actual_component_targets_preserve_native_markers_and_correct_EOS')
    def reject(name, fn):
        try:
            fn()
        except ValueError:
            checked.append(name)
        else:
            raise AssertionError('Did not reject: ' + name)
    reject('empty_dataset', lambda: EqualMaterialExamples([]))
    reject('duplicate_component', lambda: EqualMaterialExamples([records[0], records[0]]))
    mixed = [copy.deepcopy(records[0]), copy.deepcopy(records[1])]
    mixed[1]['id'] = 'distinct_id'
    reject('mixed_training_arms', lambda: EqualMaterialExamples(mixed))
    bad = copy.deepcopy(records[0]); bad['source_tensor']['sha256'] = '0' * 64
    reject('source_tensor_hash_corruption', lambda: component_batch(bad))
    bad = copy.deepcopy(records[0]); bad['transformed_sha256']['labels'] = '0' * 64
    reject('transformed_label_corruption', lambda: component_batch(bad))
    bad = copy.deepcopy(records[0]); bad['component'] = 'unrecognized'
    reject('unknown_component', lambda: component_batch(bad))
    bad = copy.deepcopy(records[0]); bad['decision_tokens'] += 1
    reject('declared_target_count_corruption', lambda: EqualMaterialExamples([bad])[0])
    # A source without D must never be repurposed into an invented reason.
    bad = copy.deepcopy(next(r for r in records if r['arm']=='separate' and r['component']=='action'))
    bad['component'] = 'decision'
    reject('absent_reason_cannot_be_invented', lambda: component_batch(bad))
    assert not torch.cuda.is_initialized()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out:
        json.dump(dict(status='passed', checks=checked, target_texts=texts,
            manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            tokenizer_source=str(args.model), model_loaded=False, model_calls=0,
            optimizer_updates=0, cuda_initialized=False), out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps(dict(status='passed', groups=len(checked), actual_components=len(texts))))


if __name__ == '__main__':
    main()
