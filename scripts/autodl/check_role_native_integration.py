"""Model-free dispatch checks; archived tokens and recording callbacks only.

Real imports of the existing policy/Trainer classes, not a Trainer training run.
No neural module is constructed; no model.forward/generate, backward, optimizer
step, SDK, renderer, API or CUDA operation is executed.
"""
import argparse
import ast
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
from types import SimpleNamespace
from unittest.mock import patch


class RecordingRoles:
    """Dispatch fixture, deliberately NOT a PEFT/SharedLearningRoles substitute."""
    def __init__(self):
        self.model = SimpleNamespace(training=False)
        self.params = {"lora.student.weight": object(), "lora.teacher.weight": object(), "base.weight": object()}
        self.model.named_parameters = lambda: list(self.params.items())
        self.student_names = {"lora.student.weight"}
        self.lock = threading.RLock()
        self.active = None
        self.events = []

    @contextmanager
    def role(self, name):
        with self.lock:
            if self.active is not None:
                raise RuntimeError("Recording fixture disallows nested role")
            self.active = name
            previous = self.model.training
            self.model.training = name == "student_train"
            self.events.append({"role": name, "event": "enter"})
            try:
                yield self.model
            finally:
                self.events.append({"role": name, "event": "exit"})
                self.model.training = previous
                self.active = None


def main(repo, checkpoint, out):
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    import accelerate
    from transformers import AutoTokenizer
    from native_multistage_phase import NativeMultiStagePolicy
    from replay_native_phase_batches import ArchivedBackend
    from role_bound_native_services import RoleBoundNativeServices, build_role_bound_native_services
    from role_isolated_mass_trainer import RoleIsolatedMassTrainer
    from phase_trainer import OriginalMassTrainer
    assert not torch.cuda.is_initialized()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    eos = json.loads((checkpoint / "generation_config.json").read_text())["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    source = repo / "runs/autodl/20261110_complete_native_runtime"
    parser_source = source / "runtime/Designer/agent_rollout.py"
    tree = ast.parse(parser_source.read_text())
    nodes = [n for n in tree.body if
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
             ("_TOOL_CALL_RE", "_THINK_RE") for t in n.targets) or
             isinstance(n, ast.FunctionDef) and n.name == "parse_assistant_text"]
    ns = dict(re=re, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(parser_source), "exec"), ns)
    packet_path = repo / "runs/autodl/20261111_closed_loop_data/feedback_material_v2/offline_feedback_packets.json"
    packets = {x["id"]: x for x in json.loads(packet_path.read_text())}
    roles = RecordingRoles()
    roles.model.generation_config = SimpleNamespace(eos_token_id=eos,
        forced_eos_token_id=None, num_beams=1, num_return_sequences=1)
    constructor_processor = SimpleNamespace(tokenizer=tokenizer)
    constructed = build_role_bound_native_services(roles, constructor_processor,
        normalize_messages=lambda *a, **k: (_ for _ in ()).throw(AssertionError("Encoder must not run")),
        phase_settings=dict(max_context_tokens=16384, seed=0,
            max_model_generate_requests=1, max_output_tokens=32),
        tool_settings=dict(ledger_path=out / "constructor_only_tool_ledger.json",
            effective_model_id="recording_metadata_object_not_a_neural_model",
            allowed_requested_models=["declared-recording-label"], max_context_tokens=16384,
            max_new_tokens=32, max_seconds=1, max_calls=1,
            total_output_tokens=32, total_seconds=1))
    assert constructed.tokenizer is tokenizer and constructed.eos_ids == eos
    assert constructed._phase.model is constructed._tool.model is roles.model
    assert constructed._phase.lock is constructed._tool.lock is roles.lock
    assert not constructed._tool.has_adapter and roles.active is None
    processor = object()
    callbacks = []

    def tool_record(**kwargs):
        assert roles.active == "tool_eval"
        callbacks.append({"operation": "tool_dispatch_callback", "role": roles.active,
                          "passed_fields": sorted(kwargs)})
        return "recording-only callback, not a model output"

    tool = SimpleNamespace(model=roles.model, processor=processor, lock=roles.lock,
                           has_adapter=False, create_completion=tool_record)
    replayed = []
    for short in ("floor", "furniture", "wall"):
        trace = json.loads((source / f"review_traces/{short}_trace.json").read_text())
        for row in trace:
            key = short + ":" + str(row["turn"])
            phase = ArchivedBackend(tokenizer, eos, packets[key])
            phase.model = roles.model
            phase.processor = processor
            phase.lock = roles.lock
            phase.last_attempt = None
            wrapped = RoleBoundNativeServices(roles, phase_backend=phase, tool_backend=tool)
            policy = NativeMultiStagePolicy(wrapped, native_parser=ns["parse_assistant_text"],
                condition="native_joint", total_output_tokens=1024, decision_tokens=512,
                action_tokens=512, seconds=20, max_calls_per_batch=64)
            proposal = policy.generate(row["observation"])
            assert proposal["tool_calls"] == row["proposal"]["tool_calls"]
            assert proposal["full_scene_success"] is None
            assert proposal["transport_receipt"]["actual_model_generate_requests"] == 0
            assert wrapped.events[0]["role"] == "policy_eval" and wrapped.events[0]["status"] == "returned"
            replayed.append({"source_id": key, "calls_preserved": len(proposal["tool_calls"]),
                             "execution_kind": "archived_sequence_replay"})
    assert len(replayed) == 19 and sum(x["calls_preserved"] for x in replayed) == 35
    # No original tool return/model text is fabricated as a new decision.
    wrapped.create_completion("declared-local-label", [], "none", "low")
    with roles.role("student_train"):
        try:
            wrapped.create_completion(model="declared-local-label", messages=[], reasoning_effort="none", verbosity="low")
        except RuntimeError:
            blocked_during_training = True
        else:
            raise AssertionError("Nested tool call admitted")
    assert len(callbacks) == 1
    first_inside = threading.Event()
    release_first = threading.Event()
    overlap_log = []
    def queued_tool_record(**kwargs):
        assert roles.active == "tool_eval"
        label = kwargs["messages"][0]["content"]
        overlap_log.append(label + ":enter")
        if label == "first":
            first_inside.set()
            assert release_first.wait(2), "Recording concurrency check timed out"
        overlap_log.append(label + ":exit")
        return "recording callback only"
    tool.create_completion = queued_tool_record
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(wrapped.create_completion, "declared-local-label",
                            [{"role": "user", "content": "first"}], "none", "low")
        assert first_inside.wait(2)
        second = pool.submit(wrapped.create_completion, "declared-local-label",
                             [{"role": "user", "content": "second"}], "none", "low")
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert overlap_log == ["first:enter", "first:exit", "second:enter", "second:exit"]
    tool.create_completion = tool_record

    def make_trainer():
        t = object.__new__(RoleIsolatedMassTrainer)
        t.learning_roles = roles
        t.model = roles.model
        t.args = SimpleNamespace(load_best_model_at_end=False, torch_compile=False,
                                 jit_mode_eval=False, auto_find_batch_size=False)
        t.model_init = None
        t._role_train_open = False
        t.optimizer = None
        return t

    hooks = []
    def record_create(t):
        assert roles.active == "student_train" and t._role_train_open
        t.optimizer = SimpleNamespace(param_groups=[{"params": [roles.params["lora.student.weight"]]}])
        hooks.append("create_optimizer_callback_without_optimizer_construction")
        return t.optimizer
    def record_training_step(t, model, inputs, num_items_in_batch=None):
        assert roles.active == "student_train" and t._role_train_open
        hooks.append("training_step_callback_without_forward_or_backward")
        return "recording-only training step"
    def record_train(t, **kwargs):
        assert roles.active == "student_train" and t._role_train_open
        t.create_optimizer()
        t.training_step(t.model, {})
        # This is only the optimizer-call LOCATION in a recording callback.
        # No optimizer object or .step is invoked.
        assert roles.active == "student_train"
        hooks.append("outer_optimizer_location_still_in_training_role")
        return "recording-only inherited-loop callback"
    with patch.object(OriginalMassTrainer, "create_optimizer", record_create), \
         patch.object(OriginalMassTrainer, "training_step", record_training_step), \
         patch.object(OriginalMassTrainer, "train", record_train):
        t = make_trainer()
        t.train()
        assert roles.active is None and not t._role_train_open
        t.optimizer.param_groups[0]["params"].append(roles.params["lora.teacher.weight"])
        try:
            t.train()
        except ValueError:
            contaminated_optimizer_rejected = True
        else:
            raise AssertionError("Teacher parameter ownership was accepted")
        assert roles.active is None and not t._role_train_open
    def failing_loop(t, **kwargs):
        assert roles.active == "student_train"
        raise RuntimeError("intentional recording-callback error")
    with patch.object(OriginalMassTrainer, "train", failing_loop):
        t = make_trainer()
        try:
            t.train()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Intentional failure missing")
        assert roles.active is None and not t._role_train_open
    old_base = roles.params["base.weight"]
    roles.params["base.weight"] = SimpleNamespace(grad=object())
    try:
        make_trainer().train()
    except ValueError:
        stale_nonstudent_gradients_rejected = True
    else:
        raise AssertionError("Stale base gradient entered Trainer clipping path")
    finally:
        roles.params["base.weight"] = old_base
    assert not torch.cuda.is_initialized()
    result = dict(scope="Archived token policy routing and recording Trainer hooks only",
        source_policy_turns_replayed=19, source_tool_calls_preserved=35,
        reused_original_parser=True, original_inherited_training_loop_executed=False,
        role_controller="Recording fixture only; no new PEFT correctness claim",
        tool_blocked_during_training=blocked_during_training,
        contaminated_optimizer_rejected=contaminated_optimizer_rejected,
        stale_nonstudent_gradients_rejected=stale_nonstudent_gradients_rejected,
        original_backend_constructors_executed_on_metadata_fixture=True,
        nontraining_tool_concurrency_serialized=overlap_log,
        normal_and_error_scope_exit_restored=True, hooks=hooks,
        actual_neural_models_constructed=0, model_forwards=0, backward_calls=0,
        optimizer_objects_constructed=0, optimizer_updates=0, new_generated_scenes=0,
        API_calls=0, SDK_calls=0, CUDA_initialized=False,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, accelerate=accelerate.__version__),
        files={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [repo / "scripts/autodl/role_bound_native_services.py",
                repo / "scripts/autodl/role_isolated_mass_trainer.py", packet_path, parser_source]})
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (out / "role_events.json").write_text(json.dumps(roles.events, indent=2) + "\n")
    (out / "archive_replay.json").write_text(json.dumps(replayed, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.repo.resolve(), a.checkpoint.resolve(), a.out.resolve())
