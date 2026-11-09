"""Trainer adapter for immutable original-token objective coefficients.

This implements ordinary weighted causal cross entropy, not a new objective.
The supported execution path is one process, batch size one, Transformers
4.57.1 / Accelerate 1.10.1. No CLI, model download or automatic training.
"""
import torch
import torch.nn.functional as F
import transformers
import accelerate
from transformers import Trainer


class OriginalMassTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if transformers.__version__ != '4.57.1' or accelerate.__version__ != '1.10.1':
            raise ValueError('Revalidate accumulation semantics before changing library versions')
        if kwargs.get('compute_loss_func') is not None:
            raise ValueError('A second custom-loss hook would change accumulation semantics')
        super().__init__(*args, **kwargs)
        if self.args.remove_unused_columns or self.label_smoother is not None:
            raise ValueError('Preserve weight columns and disable label smoothing')
        self.model_accepts_loss_kwargs = False
        self._mass_training_step = False
        self.mass_micro_log = []
        self._check_execution_scope()

    def _check_execution_scope(self):
        if (self.accelerator.num_processes != 1 or self.args.n_gpu > 1
                or self.accelerator.gradient_accumulation_steps != 1
                or self.is_deepspeed_enabled or self.is_fsdp_enabled
                or self.use_apex or self.args.per_device_train_batch_size != 1):
            raise ValueError('Only the validated single-process accumulation path is supported')

    def _training_multiplier(self):
        # Base Trainer.training_step divides by this exact current-window count.
        # Accelerate.backward must independently use GAS=1 (checked above).
        return self.current_gradient_accumulation_steps

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._check_execution_scope()
        k = getattr(self, 'current_gradient_accumulation_steps', None)
        if not isinstance(k, int) or k < 1 or self._mass_training_step:
            raise ValueError('Missing/current accumulation window or nested training step')
        self._mass_training_step = True
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        finally:
            self._mass_training_step = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        x = dict(inputs)
        labels = x.pop('labels')
        weights = x.pop('token_objective_weights')
        if labels.ndim != 2 or labels.shape[0] != 1 or labels.shape != weights.shape:
            raise ValueError('Expected one source component with aligned token weights')
        active = labels != -100
        if (active[:, 0].any() or not active.any() or not weights.is_floating_point()
                or not torch.isfinite(weights).all() or (weights[~active] != 0).any()
                or (weights[active] <= 0).any()):
            raise ValueError('Invalid original-token mask/coefficients')
        if 'action_mask' in x or 'decision_mask' in x:
            action = x.pop('action_mask'); decision = x.pop('decision_mask')
            if not torch.equal(action | decision, active) or (action & decision).any():
                raise ValueError('Component masks do not partition supervised targets')
        positions = active[0].nonzero().flatten()
        # The Qwen native path supports selected logits. Avoid allocating a
        # full context-length x vocabulary tensor solely to score a short target.
        output = model(**x, logits_to_keep=positions-1, use_cache=False)
        logits = output['logits'] if isinstance(output, dict) else output.logits
        if logits.ndim != 3 or logits.shape[:2] != (1, len(positions)):
            raise ValueError('Model did not return exactly the requested causal positions')
        numerical_logits = logits[0] if logits.dtype == torch.float64 else logits[0].float()
        nll = F.cross_entropy(numerical_logits, labels[0, positions], reduction='none')
        if not torch.isfinite(nll).all():
            raise FloatingPointError('Nonfinite target likelihood')
        raw = (nll * weights[0, positions]).sum()
        factor = self._training_multiplier() if self._mass_training_step else 1
        self.mass_micro_log.append({'training_path': self._mass_training_step,
            'target_tokens': len(positions), 'original_mass': float(weights.sum()),
            'raw_weighted_nll': float(raw.detach()), 'trainer_compensation': factor})
        loss = raw * factor
        return (loss, output) if return_outputs else loss


def require_full_corpus_update(trainer, expected_components):
    """Guard the intended one-whole-corpus-per-update experiment schedule.

    A subset/padded/repeated corpus needs a new protocol. This count check is
    additional to immutable material/image hashes, never a replacement for them.
    """
    if (trainer.train_dataset is None or len(trainer.train_dataset) != expected_components
            or trainer.args.gradient_accumulation_steps != expected_components
            or trainer.args.dataloader_drop_last):
        raise ValueError('The intended full-corpus update is not configured')
