"""Real Trainer/Accelerate backward with a fixed scalar probability oracle.

Uses every original target coefficient in both arms. Targets are projected to
two classes ONLY for this numerical test. No language model, images, optimizer,
parameter update or scene is involved; this is not a training/effectiveness run.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import resource
import time

import torch
import accelerate
import transformers
from transformers import TrainingArguments

from audit_material_routing import digest, write
from phase_trainer import OriginalMassTrainer, require_full_corpus_update


class ProbabilityOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(.37, dtype=torch.float64))

    def forward(self, input_ids, attention_mask, logits_to_keep, use_cache=False):
        assert not use_cache
        f = (logits_to_keep.to(torch.float64).remainder(7)-3)/5
        value = self.theta*f
        return {'logits': torch.stack((value, -value), dim=-1).unsqueeze(0)}


class UncompensatedProbe(OriginalMassTrainer):
    def _training_multiplier(self):
        return 1


def batch(component):
    prefix = 3
    ids = component['target_token_ids']
    labels = [-100]*prefix + [x % 2 for x in ids]
    return {'input_ids': torch.arange(len(labels)).unsqueeze(0),
            'attention_mask': torch.ones((1, len(labels)), dtype=torch.long),
            'labels': torch.tensor([labels], dtype=torch.long),
            'token_objective_weights': torch.tensor([[0.]*prefix + component['token_objective_weights']], dtype=torch.float64)}


def analytic(b):
    nll, grad = 0., 0.
    for pos, (label, w) in enumerate(zip(b['labels'][0].tolist(), b['token_objective_weights'][0].tolist())):
        if label == -100:
            assert w == 0
            continue
        f = ((pos-1) % 7-3)/5
        p0 = 1/(1+math.exp(-2*.37*f))
        nll -= w*math.log(p0 if label == 0 else 1-p0)
        grad += w*2*f*(p0-(1 if label == 0 else 0))
    return nll, grad


def exercise(components, directory, window, compensate):
    model = ProbabilityOracle()
    before = digest(model.theta.detach().numpy().tobytes())
    batches = [batch(c) for c in components]
    args = TrainingArguments(output_dir=str(directory), use_cpu=True, report_to=[],
        per_device_train_batch_size=1, gradient_accumulation_steps=window,
        remove_unused_columns=False, disable_tqdm=True, dataloader_pin_memory=False)
    cls = OriginalMassTrainer if compensate else UncompensatedProbe
    trainer = cls(model=model, args=args, train_dataset=batches,
                  data_collator=lambda values: values[0])
    if window == len(components):
        require_full_corpus_update(trainer, len(components))
    iterator = iter(batches)
    windows, observed_losses = [], []
    expected_loss = expected_gradient = 0.
    for offset in range(0, len(batches), window):
        current, count = trainer.get_batch_samples(iterator, min(window, len(batches)-offset), args.device)
        trainer.current_gradient_accumulation_steps = len(current)
        windows.append(len(current))
        for b in current:
            nll, grad = analytic(b)
            scale = 1. if compensate else 1./len(current)
            expected_loss += nll*scale
            expected_gradient += grad*scale
            # Actual inherited Trainer.training_step -> actual Accelerator.backward.
            # No monkey-patched backward and no call to Trainer.train/optimizer.step.
            observed_losses.append(float(trainer.training_step(model, b, count)))
    actual_gradient = float(model.theta.grad)
    loss_error = abs(sum(observed_losses)-expected_loss)
    gradient_error = abs(actual_gradient-expected_gradient)
    assert loss_error < 1e-12 and gradient_error < 1e-12
    assert trainer.optimizer is None and before == digest(model.theta.detach().numpy().tobytes())
    # compute_loss outside a training step must return the raw weighted objective.
    evaluation = float(trainer.compute_loss(model, batches[-1]).detach())
    assert abs(evaluation-analytic(batches[-1])[0]) < 1e-12
    return {'compensated': compensate, 'components': len(components), 'accumulation_windows': windows,
            'observed_loss_sum': sum(observed_losses), 'analytic_loss_sum': expected_loss,
            'actual_gradient': actual_gradient, 'analytic_gradient': expected_gradient,
            'loss_absolute_error': loss_error, 'gradient_absolute_error': gradient_error,
            'accelerate_accumulation_steps': trainer.accelerator.gradient_accumulation_steps,
            'real_backward_calls': len(observed_losses), 'optimizer_created': trainer.optimizer is not None,
            'parameters_unchanged': True, 'nontraining_loss_is_unscaled': True}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--material', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); torch.set_num_threads(1)
    result = {'status': 'started', 'language_model_weights_loaded': False,
              'model_inference_calls': 0, 'optimizer_updates': 0, 'scene_edits': 0}
    try:
        material = [json.loads(line) for line in args.material.read_text().splitlines()]
        runs = []
        expected_arms = ('separate_phase_v3', 'paired_phase_v3')
        if {c['arm'] for c in material} != set(expected_arms):
            raise ValueError('Expected the frozen phase-v3 material arms')
        for arm in expected_arms:
            components = [c for c in material if c['arm'] == arm]
            assert len(components) == 34
            assert sum(len(c['target_token_ids']) for c in components) == 2975
            for window, compensated in ((34, False), (34, True), (7, True)):
                name = f'{arm}_k{window}_comp{int(compensated)}'
                runs.append(dict(arm=arm, **exercise(components, args.output/name, window, compensated)))
        for arm in expected_arms:
            group = [r for r in runs if r['arm'] == arm]
            assert math.isclose(group[0]['actual_gradient']*34, group[1]['actual_gradient'], abs_tol=1e-12)
            assert math.isclose(group[1]['actual_gradient'], group[2]['actual_gradient'], abs_tol=1e-12)
        files = [Path(inspect.getfile(transformers.Trainer)), Path(inspect.getfile(accelerate.Accelerator)),
                 Path(__file__), Path(__file__).with_name('phase_trainer.py'), args.material]
        result.update(status='passed', cases=runs,
            source_sha256={str(p): digest(p.read_bytes()) for p in files},
            versions={'torch': torch.__version__, 'transformers': transformers.__version__, 'accelerate': accelerate.__version__},
            key_result='With 34 components, uncompensated original coefficients are divided by 34. The adapter restores the declared sum, including a short final window.',
            scope='Selected-logit numerical oracle through real Trainer/Accelerate backward. Not full multimodal/model/optimizer training; no equality of real neural gradients between arms is asserted.',
            old_920_scope='The inspected old EditExamples already multiplies coefficients by dataset length; this test does not establish that old runs were misnormalized.')
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        result.update(wall_seconds=time.perf_counter()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      cuda_initialized=torch.cuda.is_initialized())
        write(args.output/'receipt.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
