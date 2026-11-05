"""Four CPU backwards on two random tiny five-image models, zero updates."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from checkpoint_input_offload import install_checkpoint_input_offload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'CPU random-model offload equivalence; no optimizer', flush=True)
    torch.set_num_threads(4)
    results = []
    for length in (32, 48):
        torch.manual_seed(42)
        cfg = Qwen3VLConfig(
            text_config=dict(vocab_size=256, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=4, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16,
                             rope_scaling={'rope_type': 'default', 'mrope_section': [3, 3, 2]},
                             attention_dropout=0., use_cache=False),
            vision_config=dict(depth=3, hidden_size=32, intermediate_size=64,
                               num_heads=4, patch_size=2, temporal_patch_size=2,
                               spatial_merge_size=2, out_hidden_size=64,
                               num_position_embeddings=16, deepstack_visual_indexes=[0, 1, 2]),
            image_token_id=241, video_token_id=243, vision_start_token_id=240,
            vision_end_token_id=242)
        cfg._attn_implementation = 'sdpa'
        model = Qwen3VLForConditionalGeneration(cfg)
        targets = [f'model.language_model.layers.{i}.self_attn.{p}'
                   for i in range(4) for p in ('q_proj', 'v_proj')]
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                    bias='none', task_type=TaskType.CAUSAL_LM, target_modules=targets))
        # Nonzero synthetic B exposes gradients through both LoRA matrices.
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'lora_B' in n:
                    p.normal_(0., .01)
        model.train()
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        inputs = torch.randint(3, 200, (1, length))
        inputs[0, :15] = torch.tensor([240, 241, 242] * 5)
        pixels = torch.randn(20, 24)
        grid = torch.tensor([[1, 2, 2]] * 5)
        positions = torch.arange(length - 5, length)
        labels = inputs[0, positions].clone()
        kwargs = dict(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                      pixel_values=pixels, image_grid_thw=grid,
                      logits_to_keep=positions - 1, use_cache=False)
        answers = []
        stats = None
        for offload in (False, True):
            model.zero_grad(set_to_none=True)
            if offload:
                stats = install_checkpoint_input_offload(model, 4)
            out = model(**kwargs)
            loss = torch.nn.functional.cross_entropy(out.logits[0].float(), labels)
            loss.backward()
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError('Frozen model acquired gradients')
            if any(not torch.isfinite(g).all() for g in grads.values()):
                raise ValueError('Nonfinite adapter gradient')
            answers.append((out.logits.detach().clone(), loss.detach().clone(), grads))
            del out, loss
        logits_error = (answers[0][0] - answers[1][0]).abs().max().item()
        loss_error = abs(answers[0][1].item() - answers[1][1].item())
        grad_error = max((answers[0][2][n] - answers[1][2][n]).abs().max().item()
                         for n in answers[0][2])
        if max(logits_error, loss_error, grad_error) > 1e-7:
            raise ValueError('Offload changed CPU fixture result')
        if any(not torch.equal(original[n], p.detach()) for n, p in model.named_parameters()):
            raise ValueError('Parameters changed without an optimizer')
        if stats['checkpoint_forward_calls'] != 4 or torch.cuda.is_initialized():
            raise ValueError('Unexpected checkpoint or CUDA execution')
        results.append(dict(input_tokens=length, images=5, logits_max_error=logits_error,
                            loss_error=loss_error, gradient_max_error=grad_error,
                            gradient_tensors=len(answers[0][2]), stats=copy.deepcopy(stats)))
    report = dict(status='passed', cases=results, cpu_backwards=4, optimizer_updates=0,
                  real_model_loaded=False, cuda_initialized=torch.cuda.is_initialized(),
                  gpu_transfer_verified=False, real_model_memory_fit_verified=False,
                  scope='Tiny random float32 Qwen3-VL/LoRA with five synthetic images, actual installed checkpoint implementation.')
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
