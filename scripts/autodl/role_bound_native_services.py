"""Connect already-created native backends to the common learning-role lock.

No model/processor loader, optimizer, HTTP server or automatic inference.
The existing per-backend limits are NOT a provider-wide aggregate budget.
This is mode routing, not a newly authorized generation protocol.
"""
from contextlib import contextmanager


class RoleBoundNativeServices:
    def __init__(self, roles, *, phase_backend, tool_backend):
        self.roles = roles
        self._phase = phase_backend
        self._tool = tool_backend
        if phase_backend.model is not roles.model or tool_backend.model is not roles.model:
            raise ValueError("All roles/backends must reference the same model object")
        if phase_backend.processor is not tool_backend.processor:
            raise ValueError("One explicit shared processor required")
        if phase_backend.lock is not roles.lock or tool_backend.lock is not roles.lock:
            raise ValueError("Backends must already use the one common RLock")
        # SharedLearningRoles owns adapter selection. The legacy backend must
        # not enter a second, potentially differently restored adapter scope.
        if tool_backend.has_adapter:
            raise ValueError("Tool backend must delegate adapter disabling to the outer role")
        self.events = []

    @contextmanager
    def _operation(self, role, operation):
        # In particular, a training callback must not wait for a model operation
        # on a second thread while the same training call holds the common lock.
        # Reject instead of entering an unqualified reentrant/nested lifecycle.
        if self.roles.active == "student_train":
            raise RuntimeError("No policy/tool operation during the serial training phase")
        # Other threads' nontraining requests serialize on roles.lock. The role
        # controller itself rejects same-thread nested scopes after acquiring it.
        event = dict(role=role, operation=operation, status="entered")
        self.events.append(event)
        try:
            with self.roles.role(role):
                yield
            event["status"] = "returned"
        except BaseException as exc:
            event.update(status="raised", error=type(exc).__name__ + ": " + str(exc))
            raise

    def generate(self, messages, *, assistant_prefix, phase, token_limit, seconds_limit):
        """Existing NativePhaseTorchBackend interface, using the student adapter."""
        with self._operation("policy_eval", "native_phase_generate"):
            return self._phase.generate(messages, assistant_prefix=assistant_prefix,
                phase=phase, token_limit=token_limit, seconds_limit=seconds_limit)

    @property
    def execution_kind(self):
        return self._phase.execution_kind

    @property
    def tokenizer(self):
        return self._phase.tokenizer

    @property
    def eos_ids(self):
        return self._phase.eos_ids

    @property
    def last_attempt(self):
        return self._phase.last_attempt

    def create_completion(self, model, messages, reasoning_effort, verbosity,
                          response_format=None, vision_detail="auto", *, service_tier=None):
        """Existing native VLM service interface, using adapters-disabled base.

        Delegates original schema, encoding, JSON, EOS and per-backend budget
        checks. Provider options retain the original documented limitations.
        """
        with self._operation("tool_eval", "native_tool_completion"):
            return self._tool.create_completion(model=model, messages=messages,
                reasoning_effort=reasoning_effort, verbosity=verbosity,
                response_format=response_format, vision_detail=vision_detail,
                service_tier=service_tier)


def build_role_bound_native_services(roles, processor, *, normalize_messages,
                                     phase_settings, tool_settings):
    """Create the existing backends around an ALREADY materialized PEFT model.

    Caller supplies every future budget explicitly. Nothing is loaded, generated
    or resumed. The tool ledger constructor may create its new local JSON file.
    """
    from native_phase_torch_backend import NativePhaseTorchBackend
    from local_tool_vlm import LocalToolVLMBackend
    if {"model", "processor", "lock", "normalize_messages"} & set(phase_settings):
        raise ValueError("Phase settings cannot replace shared objects")
    if {"model", "processor", "lock", "has_adapter"} & set(tool_settings):
        raise ValueError("Tool settings cannot replace shared objects/role ownership")
    required_phase = {"max_context_tokens", "seed", "max_model_generate_requests", "max_output_tokens"}
    required_tool = {"ledger_path", "effective_model_id", "allowed_requested_models",
                    "max_context_tokens", "max_new_tokens", "max_seconds", "max_calls",
                    "total_output_tokens", "total_seconds"}
    if not required_phase <= set(phase_settings) or not required_tool <= set(tool_settings):
        raise ValueError("Explicit existing per-backend budgets and identities required")
    with roles.role("policy_eval"):
        phase = NativePhaseTorchBackend(roles.model, processor,
            normalize_messages=normalize_messages, lock=roles.lock, **phase_settings)
        tool = LocalToolVLMBackend(roles.model, processor, has_adapter=False, **tool_settings)
        tool.lock = roles.lock  # newly constructed backend, before exposure
    return RoleBoundNativeServices(roles, phase_backend=phase, tool_backend=tool)
