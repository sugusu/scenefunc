"""Ordinary task-input routing for a caller-supplied frozen Critic ranker.

Not a learned algorithm, reward change, or model launcher. All main scene-method
comparisons should receive the same original task through this interface. The
old task-omitted prompt is a wiring diagnostic, not a strong research baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class OriginalTask:
    """Provenance supplied by the application, not an authenticity certificate."""

    request_id: str
    text: str

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("An explicit original request ID is required")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Do not silently substitute a generic room purpose")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


def with_original_task(base_prompt: str, task: OriginalTask) -> str:
    """Preserve the existing ranking format and append explicit task evidence.

    This does not add a reference ranking, target scene, future tool outcome,
    or generated rationale. Supplying text does not prove model use of it.
    """
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        raise ValueError("Preserve an explicit common ranking prompt")
    evidence = json.dumps(
        {"original_request_id": task.request_id,
         "original_request": task.text,
         "original_request_sha256": task.sha256},
        ensure_ascii=False,
    )
    return (
        base_prompt
        + "\n\nOriginal task evidence (the requested room and use, not a reference answer):\n"
        + evidence
        + "\nAssess functional suitability and requirement fidelity against this "
        "original task. Keep the same physical checks and output schema. Do not "
        "replace explicit requirements with a generic room template."
    )


def rank_with_original_task(
    ranker: Callable[..., Any],
    default_prompt_factory: Callable[[int], str],
    *,
    model: Any,
    processor: Any,
    image_paths: Sequence[Any],
    task: OriginalTask,
    **rank_kwargs: Any,
) -> Any:
    """Call an already-qualified base ranker with task-conditioned text.

    The caller owns model identity, roles, budgets and inference authorization.
    No model is loaded here. Pass the actual image subset after any common
    legality filtering, so its label/count prompt matches the images. This
    adapter wraps advantage_from_critic, not its filtering outer wrapper.
    Existing complete training factories are not automatically patched.
    """
    if "user_text" in rank_kwargs or "legal_flags" in rank_kwargs:
        raise ValueError("Apply common filtering first; this adapter owns user_text")
    paths = list(image_paths)
    if not 2 <= len(paths) <= 4:
        raise ValueError("The reviewed native ranker supports 2–4 actual candidates")
    prompt = with_original_task(default_prompt_factory(len(paths)), task)
    return ranker(model, processor, paths, user_text=prompt, **rank_kwargs)
