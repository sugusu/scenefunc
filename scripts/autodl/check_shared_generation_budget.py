"""Exercise the actual two backend methods with recording-only token callbacks.

There is no pretrained model, neural module, processor weight, network, native
SDK, training or CUDA use. Synthetic token fixtures test accounting, not policy
quality or scene generation. No output here is counted as a model experiment.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from shared_generation_budget import (SharedGenerationBudget,
                                      GenerationBudgetExceeded,
                                      budgeted_native_services)


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class RecordingRoles:
    """Mode/lock fixture, NOT a PEFT controller or learned model."""
    def __init__(self, model):
        self.model = model
        self.lock = threading.RLock()
        self.active = None

    @contextmanager
    def role(self, name):
        with self.lock:
            if self.active is not None:
                raise RuntimeError("Fixture does not allow nested roles")
            self.active = name
            try:
                yield self.model
            finally:
                self.active = None


class FixtureTokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, text):
        assert text == "</think>"
        return 11

    def encode(self, text, add_special_tokens=False):
        assert text == "</think>" and not add_special_tokens
        return [11]

    def decode(self, ids, skip_special_tokens=True):
        # A decoder-invalid reply is intentionally not silently repaired.
        return "not JSON" if ids[0] == 99 else '{"fixture_only": true}'


class FixtureProcessor:
    tokenizer = FixtureTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, sort_keys=True)

    def __call__(self, **kwargs):
        import torch
        class Batch(dict):
            def to(self, device):
                return Batch({k: v.to(device) for k, v in self.items()})
        # This is a documented fixed-shape fixture, not an actual VLM encoding.
        return Batch(input_ids=torch.tensor([[1, 2, 3, 4]]),
                     attention_mask=torch.ones((1, 4), dtype=torch.long))


class ScriptedTokenCallback:
    """No nn.Module: .generate only returns declared tiny token fixtures."""
    def __init__(self, outputs, clock):
        import torch
        self.training = False
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace(eos_token_id=[12],
            forced_eos_token_id=None, num_beams=1, num_return_sequences=1)
        self.outputs = list(outputs)
        self.clock = clock
        self.invocations = []

    def generate(self, **kwargs):
        import torch
        self.invocations.append({"output_cap": kwargs["max_new_tokens"],
                                 "soft_seconds": kwargs["max_time"]})
        value = self.outputs.pop(0)
        self.clock.now += 0.1
        if isinstance(value, BaseException):
            raise value
        sequence = torch.cat([kwargs["input_ids"], torch.tensor([value])], dim=1)
        return SimpleNamespace(sequences=sequence) if kwargs.get("return_dict_in_generate") else sequence


def run(out):
    import torch
    assert not torch.cuda.is_initialized()
    out.mkdir(parents=True, exist_ok=False)
    checks = []
    processor = FixtureProcessor()
    normalizer = lambda messages: (messages, [], [])

    def make(name, outputs, *, requests=3, inputs=40, tokens=20, seconds=10):
        clock = ManualClock()
        model = ScriptedTokenCallback(outputs, clock)
        roles = RecordingRoles(model)
        meter = SharedGenerationBudget(ledger_path=out / (name + "_shared.json"),
            max_generate_requests=requests, max_input_tokens=inputs,
            max_output_tokens=tokens, max_wall_seconds=seconds, clock=clock)
        settings = dict(ledger_path=out / (name + "_tool.json"),
            effective_model_id="SCRIPTED_TOKEN_CALLBACK_NO_NEURAL_MODEL",
            allowed_requested_models=["fixture"], max_context_tokens=64,
            max_new_tokens=5, max_seconds=2, max_calls=8,
            total_output_tokens=100, total_seconds=10)
        scope = budgeted_native_services(roles, processor, generation_budget=meter,
            normalize_messages=normalizer,
            phase_settings=dict(max_context_tokens=64, seed=0,
                                max_model_generate_requests=8, max_output_tokens=100),
            tool_settings=settings)
        return clock, model, roles, meter, scope

    def policy(service):
        return service.generate([{"role": "user", "content": "fixture"}],
            assistant_prefix="", phase="joint", token_limit=5, seconds_limit=2)

    def tool(service):
        return service.create_completion("fixture", [{"role": "user", "content": "fixture"}],
            "none", "low", response_format={"type": "json_object"})

    clock, model, roles, meter, scope = make("cross_role_call_cap", [[8, 9, 12]] * 4)
    original = model.generate
    with scope as service:
        policy(service)
        tool(service)
        policy(service)
        try:
            tool(service)
        except GenerationBudgetExceeded:
            pass
        else:
            raise AssertionError("Fourth request crossed shared request cap")
        assert meter.calls == len(model.invocations) == 3
        assert meter.input_tokens == 12 and meter.output_tokens == 9
        assert [x["role"] for x in meter.rows if x["admitted"]] == ["policy_eval", "tool_eval", "policy_eval"]
    assert model.generate == original and "generate" not in vars(model)
    assert roles.active is None and not hasattr(model, "_shared_generation_budget_bound")
    checks.append("Actual phase/tool backend methods share one call cap and restore class method")

    _, model, _, meter, scope = make("decoder_retry_charged", [[99, 12], [8, 9, 12]])
    with scope as service:
        try:
            tool(service)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid fixture JSON passed original decoder")
        assert meter.calls == 1 and meter.output_tokens == 2
        tool(service)
        assert meter.calls == 2 and meter.output_tokens == 5
    checks.append("Original decoder rejection remains charged and next retry consumes another request")

    _, model, _, meter, scope = make("unknown_failure", [RuntimeError("intentional callback failure")])
    with scope as service:
        try:
            policy(service)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Callback failure missing")
        try:
            tool(service)
        except GenerationBudgetExceeded:
            pass
        else:
            raise AssertionError("Tool inference admitted after unknown-output policy failure")
        assert len(model.invocations) == 1 and not meter.snapshot()["output_accounting_complete"]
    checks.append("Unknown output after a callback exception closes both inference paths")

    for name, limits, first in [("input_cap", dict(inputs=7), True),
                                 ("output_reservation", dict(tokens=7), True),
                                 ("elapsed_before_call", dict(seconds=0.05), False)]:
        clock, model, _, meter, scope = make(name, [[8, 9, 12]] * 2, **limits)
        with scope as service:
            if first:
                policy(service)
            else:
                clock.now = 0.1  # Whole-scope idle/encoding time already spent.
            try:
                tool(service)
            except GenerationBudgetExceeded:
                pass
            else:
                raise AssertionError("Shared cap failed: " + name)
            assert len(model.invocations) == int(first)
        checks.append("Shared admission: " + name)

    _, model, _, meter, scope = make("soft_overrun", [[8, 9, 12]], seconds=0.05)
    with scope as service:
        try:
            policy(service)
        except GenerationBudgetExceeded:
            pass
        else:
            raise AssertionError("Soft overrun not detected after callback return")
        assert meter.output_tokens == 3 and meter.closed_reason
        assert model.invocations[0]["soft_seconds"] == 0.05
    checks.append("Soft deadline is passed to generation; overrun charged and reported, not called a hard kill")

    # An existing instance override is restored exactly as well.
    clock = ManualClock()
    model = ScriptedTokenCallback([], clock)
    override = lambda **kwargs: None
    model.generate = override
    roles = RecordingRoles(model)
    meter = SharedGenerationBudget(ledger_path=out / "instance_restore_shared.json",
        max_generate_requests=1, max_input_tokens=4, max_output_tokens=5,
        max_wall_seconds=1, clock=clock)
    with meter.scope(roles):
        assert model.generate is not override
    assert model.generate is override
    try:
        with meter.scope(roles):
            raise AssertionError("Nonresumable scope restarted")
    except RuntimeError:
        pass
    checks.append("Existing instance method restored; completed meter cannot reopen")

    # Reject shared/tool temporary-file collisions before constructing backends.
    collision_meter = SharedGenerationBudget(ledger_path=out / "collision.json",
        max_generate_requests=1, max_input_tokens=4, max_output_tokens=5,
        max_wall_seconds=1, clock=clock)
    try:
        with budgeted_native_services(roles, processor, generation_budget=collision_meter,
                normalize_messages=normalizer, phase_settings={},
                tool_settings={"ledger_path": out / "collision.json.tmp"}):
            raise AssertionError("Colliding temporary ledger admitted")
    except ValueError:
        pass
    assert not collision_meter.path.exists()
    checks.append("Shared/tool ledger and temporary-file paths cannot overlap")

    result = dict(status="recording_callback_boundary_checks_passed", checks=checks,
                  fixture_checks=len(checks), actual_pretrained_model_generations=0,
                  neural_forwards=0, optimizer_updates=0, new_scene_trials=0,
                  CUDA_initialized=torch.cuda.is_initialized(),
                  scope="Actual original backend method execution with synthetic fixed-shape processor and token callbacks. "
                        "Not a pretrained VLM, full native service, multimodal/numerical or scene-quality qualification.")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.out)))
