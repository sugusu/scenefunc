"""One serial inference ledger at the existing shared model.generate boundary.

No model loader, forward pass, optimizer, HTTP client, quota lookup or automatic
run. Caps are explicit caller inputs, not authorization to generate. The existing
backend caps stay in force. This is accounting infrastructure, not a new policy.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
from types import MethodType


class GenerationBudgetExceeded(RuntimeError):
    pass


class SharedGenerationBudget:
    """Single process, one model, one RLock, one nonresumable evaluation scope.

    Input/output lengths include all model.generate requests from the bound
    policy/tool backends, including output later rejected by their decoders.
    Unknown output length after an exception closes BOTH inference paths.
    Wall time includes idle/encoding/tools inside scope but is a SOFT limit;
    an outer process watchdog must still own any hard deadline.
    """
    def __init__(self, *, ledger_path, max_generate_requests, max_input_tokens,
                 max_output_tokens, max_wall_seconds, clock=time.monotonic):
        if any(type(v) is not int or v <= 0 for v in
               (max_generate_requests, max_input_tokens, max_output_tokens)):
            raise ValueError("Explicit positive cumulative integer caps required")
        if (isinstance(max_wall_seconds, bool) or
                not isinstance(max_wall_seconds, (int, float)) or
                not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
            raise ValueError("Explicit positive finite soft wall cap required")
        self.path = Path(ledger_path)
        if self.path.exists():
            raise FileExistsError("No implicit resume/reset of an existing ledger")
        self.caps = dict(generate_requests=max_generate_requests,
                         input_tokens=max_input_tokens, output_tokens=max_output_tokens,
                         wall_seconds=max_wall_seconds)
        self.clock = clock
        self.rows = []
        self.calls = self.input_tokens = self.output_tokens = 0
        self.active = self.used = self.in_call = False
        self.start = None
        self.closed_reason = None

    def snapshot(self):
        elapsed = self.clock() - self.start if self.start is not None else 0.0
        return dict(caps=self.caps, generate_requests_admitted=self.calls,
                    known_input_tokens=self.input_tokens,
                    known_output_tokens=self.output_tokens,
                    elapsed_wall_seconds=elapsed, active=self.active,
                    closed_reason=self.closed_reason,
                    output_accounting_complete=all(r.get("output_tokens") is not None
                                                  for r in self.rows if r["admitted"]),
                    records=self.rows,
                    model_identity_or_neural_execution_verified=False,
                    scope="Bound local shared-model generation only; not raw model.forward, training, "
                          "other model objects/processes/clients, FLOPs, or complete scene quality",
                    time_limit_is_soft=True)

    def _save(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), ensure_ascii=False,
                                  indent=2, allow_nan=False) + "\n")
        tmp.replace(self.path)

    def _reject(self, role, reason):
        self.rows.append(dict(role=role, admitted=False, status="rejected_before_generate",
                              reason=reason, output_tokens=None))
        self._save()
        raise GenerationBudgetExceeded(reason)

    def _generate(self, original, roles, args, kwargs):
        with roles.lock:
            role = roles.active
            if not self.active or self.closed_reason:
                self._reject(role, "Shared generation budget is closed")
            if self.in_call or role not in ("policy_eval", "tool_eval"):
                self._reject(role, "Only nonnested policy/tool evaluation generation is supported")
            if args:
                self._reject(role, "Explicit keyword input_ids contract required")
            shape = getattr(kwargs.get("input_ids"), "shape", ())
            if len(shape) != 2 or shape[0] != 1:
                self._reject(role, "Only one input sequence per admitted request")
            count = int(shape[1])
            cap = kwargs.get("max_new_tokens")
            seconds = kwargs.get("max_time")
            if count < 1 or type(cap) is not int or cap < 1:
                self._reject(role, "Explicit nonempty input and positive output cap required")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or
                    not math.isfinite(seconds) or seconds <= 0):
                self._reject(role, "Explicit finite per-request soft time limit required")
            if kwargs.get("num_beams", 1) != 1 or kwargs.get("num_return_sequences", 1) != 1:
                self._reject(role, "Beams/multiple sequences require different accounting")
            remaining = self.caps["wall_seconds"] - (self.clock() - self.start)
            if (self.calls >= self.caps["generate_requests"] or
                    self.input_tokens + count > self.caps["input_tokens"] or
                    self.output_tokens + cap > self.caps["output_tokens"] or remaining <= 0):
                # Reserve the ENTIRE requested upper bound. Do not silently
                # truncate a backend request or change its advertised token cap.
                self._reject(role, "Requested generation does not fit remaining shared caps")
            row = dict(role=role, admitted=True, status="admitted_at_call_boundary",
                       input_tokens=count, requested_output_cap=cap, output_tokens=None,
                       local_soft_seconds=seconds, admitted_soft_seconds=min(seconds, remaining))
            self.rows.append(row)
            self.calls += 1
            self.input_tokens += count
            self.in_call = True
            tick = self.clock()
            try:
                self._save()  # interrupted/unknown requests remain visible
                options = dict(kwargs, max_time=min(seconds, remaining))
                result = original(**options)
                sequences = getattr(result, "sequences", result)
                out_shape = getattr(sequences, "shape", ())
                if len(out_shape) != 2 or out_shape[0] != 1 or out_shape[1] < count:
                    raise ValueError("Output length cannot be accounted for")
                generated = int(out_shape[1]) - count
                row["output_tokens"] = generated
                self.output_tokens += generated
                if generated > cap:
                    raise GenerationBudgetExceeded("Backend exceeded reserved output length")
                if self.clock() - self.start > self.caps["wall_seconds"]:
                    raise GenerationBudgetExceeded("Soft shared wall limit exceeded")
                row["status"] = "returned_to_backend_for_validation"
                return result
            except BaseException as exc:
                self.closed_reason = "Generation failed or exceeded shared limits; no automatic retry"
                row.update(status="raised_closed_both_roles", error_type=type(exc).__name__)
                raise
            finally:
                self.in_call = False
                row["generation_boundary_seconds"] = self.clock() - tick
                self._save()

    @contextmanager
    def scope(self, roles):
        """Temporarily bind this meter to the already-created shared model.

        All model users must obey roles.lock. No protection is claimed for a
        cached reference to the old generate method, another process or client.
        """
        model = roles.model
        with roles.lock:
            if self.used or roles.active is not None:
                raise RuntimeError("One fresh scope, entered outside any active role, required")
            if self.path.exists():
                raise FileExistsError("A ledger appeared before scope entry; do not overwrite it")
            if hasattr(model, "_shared_generation_budget_bound"):
                raise RuntimeError("A shared budget is already bound to this model")
            original = model.generate
            if not callable(original):
                raise ValueError("Already-created model.generate method required")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.start = self.clock()
            self.active = self.used = True
            missing = object()
            previous_attribute = vars(model).get("generate", missing)

            def metered(_model, *args, **kwargs):
                return self._generate(original, roles, args, kwargs)

            bound = MethodType(metered, model)
            model.generate = bound
            model._shared_generation_budget_bound = True
            try:
                self._save()
            except BaseException:
                if previous_attribute is missing:
                    delattr(model, "generate")
                else:
                    model.generate = previous_attribute
                delattr(model, "_shared_generation_budget_bound")
                self.active = False
                self.closed_reason = "Ledger initialization failed"
                raise
        try:
            yield self
        finally:
            with roles.lock:
                unchanged = model.generate is bound
                if unchanged:
                    if previous_attribute is missing:
                        delattr(model, "generate")
                    else:
                        model.generate = previous_attribute
                delattr(model, "_shared_generation_budget_bound")
                self.active = False
                if not unchanged:
                    self.closed_reason = "Model.generate changed outside the shared scope"
                elif self.closed_reason is None:
                    self.closed_reason = "Evaluation scope ended"
                self._save()
                if not unchanged:
                    raise RuntimeError("Do not overwrite another owner's generate method")


@contextmanager
def budgeted_native_services(roles, processor, *, generation_budget,
                             normalize_messages, phase_settings, tool_settings):
    """Compose the existing two backends, role isolation and explicit ledger.

    No new runtime/model or HTTP wiring. Existing backend caps are unchanged.
    """
    from role_bound_native_services import build_role_bound_native_services
    shared_path = generation_budget.path.resolve()
    tool_path = Path(tool_settings["ledger_path"]).resolve()
    shared_files = {shared_path, shared_path.with_suffix(shared_path.suffix + ".tmp")}
    tool_files = {tool_path, tool_path.with_suffix(".tmp")}
    if shared_files & tool_files:
        raise ValueError("Shared and tool ledgers must be different new files")
    with generation_budget.scope(roles):
        yield build_role_bound_native_services(roles, processor,
            normalize_messages=normalize_messages, phase_settings=phase_settings,
            tool_settings=tool_settings)
