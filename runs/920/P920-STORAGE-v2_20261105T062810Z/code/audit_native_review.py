"""Export one objective native-scene review and audit the pinned QA loader.

No professional preference labels, model inference, optimizer or training.
Presentation reversal is a loader check, not an independent review example.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from decision_state import canonical, sha
from native_checkpoint import canonical_hash
from review_samples import checked_ref, decode_review_answer, validate_judgment


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def observation(root, candidate):
    state = json.loads(checked_ref(root, candidate['state']).read_text())
    obs = json.loads(checked_ref(root, candidate['observation']).read_text())
    binding_path = checked_ref(root, candidate['binding'])
    if Path(obs['state_binding']).resolve() != binding_path:
        raise ValueError('Observation binding mismatch')
    binding = json.loads(binding_path.read_text())
    if binding['state_sha256'] != canonical_hash(state) or canonical_hash(binding['native_state']) != canonical_hash(state):
        raise ValueError('Image binding belongs to another state')
    context = json.loads(checked_ref(root, candidate['context']).read_text())
    if binding['current_context_sha256'] != canonical_hash(context):
        raise ValueError('Observation context mismatch')
    if len(binding['images']) != 5 or len(obs['image_paths']) != 5:
        raise ValueError('Missing native view')
    paths = []
    for view, image, actual in zip(('topdown', 'south', 'east', 'west', 'north'), binding['images'], obs['image_paths']):
        path = Path(image['path']).resolve()
        if path != Path(actual).resolve() or not path.is_relative_to(root) or not path.name.endswith('.'+view+'.png'):
            raise ValueError('Wrong view path/order')
        checked_ref(root, {'path': str(path.relative_to(root)), 'sha256': image['sha256']})
        paths.append(str(path.relative_to(root)))
    objects = [{'object_id': oid, 'name': obj['name'], 'transform': obj['transform'],
                'representation': obj.get('metadata', {}).get('asset_source', 'structure')}
               for oid, obj in state['objects'].items()]
    return {'images': paths, 'objects': objects, 'state_sha256': canonical_hash(state)}


def export(record, root, order):
    candidates = {c['candidate_id']: c for c in record['candidates']}
    if len(candidates) != len(record['candidates']) or sorted(order) != sorted(candidates):
        raise ValueError('Invalid candidate order')
    observations = {cid: observation(root, candidates[cid]) for cid in order}
    validate_judgment(record['judgment'], list(candidates))
    for evidence in record['judgment']['evidence']:
        checked_ref(root, evidence['artifact'])
    for claim in record['judgment']['claims']:
        ids = {o['object_id'] for o in observations[claim['candidate_id']]['objects']}
        if not set(claim['object_ids']) <= ids:
            raise ValueError('Unknown object in claim')
    images, layout = [], []
    slots = {cid: 'ABCD'[i] for i, cid in enumerate(order)}
    for cid in order:
        obs = observations[cid]
        layout.append({'candidate_id': cid, 'slot': slots[cid],
                       'image_indices_zero_based': list(range(len(images), len(images)+5)),
                       'view_order': ['topdown', 'south', 'east', 'west', 'north'],
                       'objects': obs['objects'], 'state_sha256': obs['state_sha256']})
        images.extend(obs['images'])
    answer = {k: record['judgment'][k] for k in ('verdict', 'ranking_groups', 'claims', 'unknowns')}
    prompt = (record['brief']+'\n\n'+record['question']+'\n\nSource scope:\n'+canonical(record['source_context'])+
              '\n\nDeclared assumptions:\n'+canonical(record['assumptions'])+
              '\n\nActual measurement tool results (available input, not human labels):\n'+canonical(record['measurement_input'])+
              '\n\nCandidate views and objects:\n'+canonical(layout))
    return {'stage': 'qa', 'system': 'Use only supplied source, scene and measurement evidence. Do not infer professional preference or physical usability from proxy access counts.',
            'provenance': {**record['provenance'], 'display_order': order, 'candidate_slots': slots},
            'turns': [{'role': 'user', 'content': prompt, 'image': images},
                      {'role': 'assistant', 'content': '', 'answer': canonical(answer)}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); plan = json.loads(a.plan.read_text()); root = Path(plan['root']).resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(), 'native objective QA audit; no model/training', flush=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    for name, digest in plan['scripts'].items():
        if sha(Path(__file__).parent/name) != digest:
            raise ValueError('Audit script changed: '+name)
    source = Path(plan['source'])
    for row in json.loads(checked_ref(root, plan['source_lock']).read_text())['files']:
        if sha(source/row['source_path']) != row['sha256']:
            raise ValueError('Official source changed')
    record = json.loads(checked_ref(root, plan['record']).read_text())
    orders = [list(c['candidate_id'] for c in record['candidates'])]
    orders.append(list(reversed(orders[0])))
    # Upstream QA can silently drop missing images; validate all references first.
    episodes = [export(record, root, order) for order in orders]
    bad = copy.deepcopy(record); bad['candidates'][0]['state'] = bad['candidates'][1]['state']
    try:
        export(bad, root, orders[0])
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched image/state was accepted')
    sys.path.insert(0, str(source/'src'))
    import sft_dataset as upstream
    from scene_observation import configure_image_budget
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(plan['processor'], local_files_only=True)
    configure_image_budget(processor, plan['budget']['image_max_pixels'])
    metrics, tensors = [], []
    for i, episode in enumerate(episodes):
        directory = a.out/('forward' if i == 0 else 'reversed')
        save(directory/'episode.json', episode)
        ds = upstream.StepwiseSFTDataset(directory, root, processor, max_length=plan['budget']['max_tokens'])
        assert len(ds) == 1
        sample, example = ds.samples[0], ds[0]
        assert len(sample.qa_images) == 10
        _, prompt = upstream._build_messages(sample)
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        images = []
        for path in sample.qa_images:
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        try:
            prefix = processor(text=[text], images=images, return_tensors='pt')['input_ids'][0]
        finally:
            for im in images: im.close()
        n = len(prefix); labels = example['labels']
        assert torch.equal(example['input_ids'][:n], prefix)
        assert bool((labels[:n] == -100).all())
        assert torch.equal(labels[n:], example['input_ids'][n:])
        assert 0 < len(labels)-n and len(labels) < plan['budget']['max_tokens']
        target = processor.tokenizer.decode(labels[n:], skip_special_tokens=False)
        decoded = decode_review_answer(target, episode['provenance']['candidate_slots'])
        expected = json.loads(episode['turns'][-1]['answer'])
        assert decoded == expected and decoded['verdict'] == 'insufficient_evidence' and decoded['ranking_groups'] == []
        assert expected['claims'][0]['assessment'] not in text
        metrics.append({'variant': directory.name, 'images': len(sample.qa_images), 'masked_input_tokens': n,
                        'supervised_tokens': len(labels)-n, 'total_tokens': len(labels), 'decoded_answer': decoded,
                        'episode_sha256': sha(directory/'episode.json'),
                        'labels_sha256': hashlib.sha256(labels.numpy().tobytes()).hexdigest()})
        tensors.append(example)
        print(directory.name, n, 'masked input tokens;', len(labels)-n, 'supervised tokens', flush=True)
    assert not torch.equal(tensors[0]['pixel_values'], tensors[1]['pixel_values'])
    result = {'status': 'native_review_official_QA_loader_audited', 'ended_at': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': sha(a.plan), 'unique_objective_reviews': 1, 'presentation_variants': 2,
              'professional_reviews': 0, 'training_eligible': False, 'model_calls': 0, 'training_updates': 0,
              'checks': {'native_state_context_and_ten_image_hashes': True, 'swapped_state_rejected': True,
                         'all_prompt_and_image_tokens_masked': True, 'assistant_target_round_trip': True,
                         'reversal_changes_actual_pixels': True, 'unknown_does_not_become_preference': True},
              'variants': metrics, 'limit': 'Data-loading and objective-label audit; no learned critic performance or professional preference evidence.'}
    save(a.out/'RESULT.json', result)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
