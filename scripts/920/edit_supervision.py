"""Candidate future SFT adapter: no hidden, unsupervised assistant prefix.

Historical encoders and completed experiment tensors are deliberately unchanged.
This module does not authorize training or fix heterogeneous scene representations.
"""
import json
from pathlib import Path
from decision_pairing import mask_role


def encode_single_episode(episode, repo_root, processor, official, max_length=16384):
    episode=Path(episode)
    doc=json.loads(episode.read_text())
    if len(doc['turns'])!=1 or doc['turns'][0]['role']!='assistant':
        raise ValueError('Expected a single observed assistant action')
    has_decision=bool(doc['turns'][0].get('content','').strip())
    dataset=official.StepwiseSFTDataset(episode.parent,repo_root,processor,
        max_length=max_length,no_thinking=not has_decision)
    if len(dataset)!=1:
        raise ValueError('Expected exactly one official sample')
    example=dataset[0]
    action=mask_role(example,processor.tokenizer,'action')
    action_mask=action['labels']!=-100
    decision_mask=(example['labels']!=-100)&~action_mask
    if not has_decision and bool(decision_mask.any()):
        raise ValueError('Action-only serialization unexpectedly contains a decision target')
    # Every assistant token is a target, including the first freely generated
    # token. Inputs and actual receipts remain masked by the official loader.
    if not bool(((action_mask|decision_mask)==(example['labels']!=-100)).all()):
        raise ValueError('Unsupervised assistant gap')
    return example,action_mask,decision_mask,dataset.samples[0]


def encode_request_targets(request, sample, processor, official, max_length=16384):
    """Use the deployment request verbatim, with official assistant serialization.

    Only the official target message is retained; its legacy Task/Memory prompt
    is replaced by the shared pre-action request. No global monkeypatching.
    """
    from PIL import Image
    import torch
    has_decision = bool((sample.assistant_content or '').strip())
    official_full, _ = official._build_messages(sample, no_thinking=not has_decision)
    messages = [*request, official_full[-1]]
    paths = [p['image'] for m in request if isinstance(m['content'], list)
             for p in m['content'] if p['type'] == 'image']
    if len(paths) != 5 or list(map(str, sample.scene_images)) != paths:
        raise ValueError('Request and target observation views differ')
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    full = processor(text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)],
                     images=images, return_tensors='pt', padding=False)
    prompt = processor(text=[processor.apply_chat_template(request, tokenize=False, add_generation_prompt=True)],
                       images=images, return_tensors='pt', padding=False)
    boundary = prompt['input_ids'].shape[1]
    if not boundary < full['input_ids'].shape[1] <= max_length:
        raise ValueError(f'Invalid token lengths: prompt={boundary}, full={full["input_ids"].shape[1]}, limit={max_length}')
    if not torch.equal(prompt['input_ids'], full['input_ids'][:, :boundary]):
        raise ValueError('Training prompt differs from generation prefix')
    out = {key: value[0] if key in ('input_ids', 'attention_mask', 'mm_token_type_ids') else value
           for key, value in full.items()}
    out['labels'] = out['input_ids'].clone()
    out['labels'][:boundary] = -100
    action = mask_role(out, processor.tokenizer, 'action')['labels'] != -100
    decision = (out['labels'] != -100) & ~action
    if not has_decision and bool(decision.any()):
        raise ValueError('Unexpected decision targets in action-only episode')
    return out, action, decision, boundary
