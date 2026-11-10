"""Ordinary original-task passthrough control for the pinned Critic factory.

No learned method, reward changes, source mutations or model calls live here.
The factory's function body and the legality routine are reused verbatim. A
private import namespace supplies a context-bound ranker so that task text is
attached AFTER the native legality subset has been selected. No sys.modules or
native module monkeypatch is needed. Use a fresh binding per training worker.
"""
from __future__ import annotations

import builtins
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import types


def _clone_function(fn, globals_dict):
    clone = types.FunctionType(fn.__code__, globals_dict, fn.__name__,
                               fn.__defaults__, fn.__closure__)
    clone.__kwdefaults__ = fn.__kwdefaults__
    clone.__annotations__ = dict(fn.__annotations__)
    return clone


def bind_factory_import(factory, critic_namespace):
    """Reuse the exact factory code with only its local Critic import redirected.

    All other imports use the original importer. This is dependency injection,
    not a copy or AST rewrite of the reward calculation. Factory globals are a
    shallow snapshot; bind again if upstream training flags subsequently change.
    """
    globals_copy = dict(factory.__globals__)
    original_builtins = globals_copy.get('__builtins__', builtins)
    private_builtins = dict(original_builtins if isinstance(original_builtins, dict)
                            else vars(original_builtins))
    original_import = private_builtins['__import__']

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'src_critic.grpo_advantage' and level == 0:
            if not fromlist:
                raise ValueError('Expected the native factory from-import')
            for item in fromlist:
                if not hasattr(critic_namespace, item):
                    raise ImportError(item)
            return critic_namespace
        return original_import(name, globals, locals, fromlist, level)

    private_builtins['__import__'] = private_import
    globals_copy['__builtins__'] = private_builtins
    return _clone_function(factory, globals_copy)


def task_ranking_text(native, count, original_task):
    if not isinstance(original_task, str) or not original_task.strip():
        raise ValueError('buffer.prompt must contain the original task')
    if count not in (2, 3, 4):
        raise ValueError('The pinned ranker supports 2--4 images')
    # Labels/count come from the FINAL image list, not the pre-gate branches.
    return (native.build_default_user_text(count)
            + '\n\nOriginal task shared by every candidate (JSON string):\n'
            + json.dumps(original_task, ensure_ascii=False)
            + '\nEvaluate arrangement quality with respect to this original task. '
              'The task is input evidence, not a preferred-scene label. '
              'Keep the ranking output format specified above.')


def make_task_conditioned_reward(factory, native, critic_model,
                                 critic_processor, *, is_policy=False,
                                 audit=None):
    """Return the unchanged native reward calculation with original-task input.

    `buffer.prompt` must be the pre-rollout original task, not a teacher answer,
    model-generated plan or post-hoc assessment. The caller owns that provenance.
    Audit entries describe transport, never correctness or full-scene success.
    """
    current_task = ContextVar('original_critic_task', default=None)
    original_ranker = native.advantage_from_critic

    @wraps(original_ranker)
    def rank(model, processor, image_paths, user_text=None, **kwargs):
        task = current_task.get()
        if task is None:
            raise RuntimeError('Critic call outside its original-task scope')
        if user_text is not None:
            raise ValueError('Refuse to silently overwrite a custom Critic prompt')
        labels = kwargs.get('labels')
        if labels is not None and list(labels) != list('ABCD'[:len(image_paths)]):
            raise ValueError('Native prompt assumes positional A.. labels')
        prompt = task_ranking_text(native, len(image_paths), task)
        if audit is not None:
            audit.append({'image_paths': list(map(str, image_paths)),
                          'labels': list('ABCD'[:len(image_paths)]),
                          'original_task_sha256': hashlib.sha256(task.encode()).hexdigest(),
                          'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
        return original_ranker(model, processor, image_paths, prompt, **kwargs)

    legality_globals = dict(native.advantage_from_critic_with_legality.__globals__)
    legality_globals['advantage_from_critic'] = rank
    gated_rank = _clone_function(native.advantage_from_critic_with_legality,
                                 legality_globals)
    namespace = types.SimpleNamespace(
        advantage_from_critic=rank,
        advantage_from_critic_with_legality=gated_rank,
        legality_flags_from_scenes=native.legality_flags_from_scenes)
    bound_factory = bind_factory_import(factory, namespace)
    native_reward = bound_factory(critic_model, critic_processor, is_policy=is_policy)

    @wraps(native_reward)
    def reward(buffer):
        task = getattr(buffer, 'prompt', None)
        if not isinstance(task, str) or not task.strip():
            raise ValueError('Missing original task; do not substitute model plans')
        token = current_task.set(task)
        try:
            return native_reward(buffer)
        finally:
            current_task.reset(token)

    # Exposed for isolated, no-model source-call inspection, not new operations.
    reward.native_reward = native_reward
    reward.context_namespace = namespace
    reward.task_scope = current_task
    return reward
