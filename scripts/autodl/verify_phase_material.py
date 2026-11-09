"""Actual original-view encoding plus loss-shift/weight checks, CPU only.

The probability probe uses constructed logits, never language-model weights.
It validates the implemented objective, not any learned behavior.
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import resource
import time

from audit_material_routing import digest, token_hash, write
from phase_objective import attach_objective_weights, weighted_causal_nll
from routed_multimodal import resolve_messages, encode_component, tensor_digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--material', type=Path, required=True)
    ap.add_argument('--availability', type=Path, required=True)
    ap.add_argument('--processor', type=Path, required=True)
    ap.add_argument('--scene-observation', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep previous execution receipts')
    start = time.perf_counter()
    material = [json.loads(s) for s in args.material.read_text().splitlines()]
    availability = json.loads(args.availability.read_text())
    views = {r['unit_id']: r for r in availability['rows']}
    old_rows = json.loads(Path('runs/autodl/20261110_objective_source_review/material_manifest.json').read_text())['records']
    old = {(r['unit_id'], r['component']): r for r in old_rows if r['arm'] == 'separate'}
    if digest(args.scene_observation.read_bytes()) != '1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3':
        raise ValueError('Wrong upstream image budget source')
    spec = importlib.util.spec_from_file_location('original_visual_contract', args.scene_observation)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True, trust_remote_code=False)
    official.configure_image_budget(processor, 65536)
    encoded = []
    for c in material:
        row = {k: c[k] for k in ('unit_id', 'arm', 'stage', 'target_kind', 'target_sha256')}
        view = views[c['unit_id']]
        if not view['complete_five_views']:
            row['status'] = 'missing_original_views'
            encoded.append(row)
            continue
        messages, paths = resolve_messages(c['messages'], view['images'])
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        adapted = dict(c, component=c['stage'])
        batch, boundary = encode_component(processor, adapted, messages, images)
        attach_objective_weights(batch, c)
        visual = {k: tensor_digest(batch[k]) for k in ('pixel_values', 'image_grid_thw')}
        expected = old[c['unit_id'], 'action']['transformed_sha256']
        assert all(visual[k] == expected[k] for k in visual)
        assert not batch['token_objective_weights'][:, :boundary].any()
        assert batch['token_objective_weights'][batch['labels'] != -100].tolist() == c['token_objective_weights']
        assert token_hash(batch['labels'][batch['labels'] != -100].tolist()) == c['target_sha256']
        row.update(status='encoded', sequence_tokens=batch['input_ids'].shape[1], boundary=boundary,
                   labels_sha256=tensor_digest(batch['labels']), image_sha256=visual,
                   weights_sha256=tensor_digest(batch['token_objective_weights']),
                   objective_mass=float(batch['token_objective_weights'].sum()))
        encoded.append(row)
        print(c['unit_id'], c['arm'], c['stage'], 'encoded', flush=True)

    # CE probability and gradient oracle, distinct from implementation internals.
    # Position 1 predicts label at 2; position 2 predicts label at 3. All other
    # positions (including final position) must contribute zero gradient.
    probs = torch.tensor([[[.3,.3,.4], [.1,.7,.2], [.6,.3,.1], [.2,.3,.5], [.3,.4,.3]]], dtype=torch.float64)
    logits = probs.log().detach().requires_grad_(True)
    labels = torch.tensor([[-100,-100,1,2,-100]], dtype=torch.long)
    weights = torch.tensor([[0.,0.,.2,.5,0.]], dtype=torch.float64)
    loss = weighted_causal_nll(logits, labels, weights)
    expected_loss = -.2 * torch.log(torch.tensor(.7, dtype=torch.float64)) - .5 * torch.log(torch.tensor(.1, dtype=torch.float64))
    assert torch.allclose(loss, expected_loss, atol=1e-12, rtol=0)
    gradient = torch.autograd.grad(loss, logits)[0]
    expected_gradient = torch.zeros_like(logits)
    expected_gradient[0,1] = .2 * (probs[0,1] - torch.tensor([0.,1.,0.]))
    expected_gradient[0,2] = .5 * (probs[0,2] - torch.tensor([0.,0.,1.]))
    assert torch.allclose(gradient, expected_gradient, atol=1e-12, rtol=0)
    # Independent microbatch sums reproduce the combined objective and gradient.
    split_losses = []
    for index in (2,3):
        lab = torch.full_like(labels, -100); lab[0,index] = labels[0,index]
        w = torch.zeros_like(weights); w[0,index] = weights[0,index]
        split_losses.append(weighted_causal_nll(logits, lab, w))
    assert torch.allclose(sum(split_losses), loss, atol=1e-12, rtol=0)
    shifted_wrong = -.2 * probs[0,2,1].log() - .5 * probs[0,3,2].log()
    assert not torch.allclose(shifted_wrong, loss)
    rejected = []
    for name, bad in [('prompt_weight', torch.tensor([[.1,0.,.2,.5,0.]], dtype=torch.float64)),
                      ('missing_target_weight', torch.tensor([[0.,0.,0.,.5,0.]], dtype=torch.float64))]:
        try:
            weighted_causal_nll(logits, labels, bad)
        except ValueError:
            rejected.append(name)
    assert len(rejected) == 2
    counts = Counter(r['status'] for r in encoded)
    assert len(encoded) == 68 and counts['encoded'] == 16 and counts['missing_original_views'] == 52
    result = {'status':'completed_partial_encoding', 'records':encoded, 'counts':dict(counts),
              'full_corpus_components':68, 'full_corpus_training_ready':False,
              'unavailable_components_retained':counts['missing_original_views'],
              'loss_probe':{'actual':float(loss.detach()),'independent_expected':float(expected_loss),
                            'unshifted_wrong':float(shifted_wrong), 'gradient_exact':True,
                            'microbatch_sum_exact':True,'invalid_weight_cases_rejected':rejected},
              'material_sha256':digest(args.material.read_bytes()),
              'availability_sha256':digest(args.availability.read_bytes()),
              'source_sha256':{str(p):digest(p.read_bytes()) for p in
                   (Path(__file__), Path(__file__).with_name('phase_objective.py'),
                    Path(__file__).with_name('routed_multimodal.py'), args.scene_observation)},
              'wall_seconds':time.perf_counter()-start,
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'model_weights_loaded':False,'model_calls':0,'optimizer_updates':0,
              'cuda_initialized':torch.cuda.is_initialized(),
              'scope':'New phase prompts with original five-view tensors and masks. Constructed-logit gradient oracle only; no trained policy, generated room or full Trainer integration.'}
    write(args.output, result)
    print(json.dumps({k:result[k] for k in ('counts','loss_probe','wall_seconds','peak_rss_kib','cuda_initialized')}))


if __name__ == '__main__':
    main()
