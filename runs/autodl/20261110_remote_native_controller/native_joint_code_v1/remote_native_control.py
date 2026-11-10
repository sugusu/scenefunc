"""Native joint policy and VLM interfaces without native-process model weights.

The actor must implement /native_policy and /completion under a separately
qualified finite protocol. This does not enable a provider fallback or authorize
a run. Old native source/parsers and previous model-host protocols stay intact.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import types
from pathlib import Path


_binding = ContextVar('remote_native_runtime', default=None)


class RemoteNativeClient:
    def __init__(self, port, *, checkpoint, timeout=180):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False
        self.base = f'http://127.0.0.1:{port}'
        self.checkpoint = Path(checkpoint).resolve()
        self.timeout = timeout
        self.lock = threading.RLock()

    def policy(self, messages, max_new_tokens):
        with self.lock:
            response = self.session.post(self.base + '/native_policy', json={
                'messages': messages, 'max_new_tokens': max_new_tokens}, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result.get('raw'), str) or not result['raw']:
                raise ValueError('No complete raw native policy response')
            return result

    def create_completion(self, model, messages, reasoning_effort, verbosity,
                          response_format=None, vision_detail='auto'):
        with self.lock:
            response = self.session.post(self.base + '/completion', json=dict(
                model=model, messages=messages, reasoning_effort=reasoning_effort,
                verbosity=verbosity, response_format=response_format, vision_detail=vision_detail),
                timeout=self.timeout)
            response.raise_for_status()
            return response.json()['text']

    def close(self):
        self.session.close()


class RemoteNativePolicy:
    """Drop-in native ModelPolicy.generate shape; exact native parser supplied.

    No .model/.processor facade: code needing in-process training must use an
    explicit training host. No retry, tag repair, or completion-success label.
    """
    def __init__(self, model_dir, max_new_tokens=512, *, parser):
        binding = _binding.get()
        if binding is None or not binding['active']:
            raise RuntimeError('Remote native policy must be constructed in an owned runtime scope')
        if Path(model_dir).resolve() != binding['client'].checkpoint:
            raise ValueError('Native model directory differs from declared remote checkpoint')
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('Explicit positive output cap required')
        self.binding = binding
        self.parser = parser
        self.max_new_tokens = max_new_tokens

    def generate(self, obs):
        if not self.binding['active']:
            raise RuntimeError('Remote native runtime already closed')
        reply = self.binding['client'].policy(obs['messages'], self.max_new_tokens)
        think, calls = self.parser(reply['raw'])
        return {'raw': reply['raw'], 'think': think, 'tool_calls': calls,
                'remote_receipt': reply.get('receipt_dir'), 'full_scene_success': None}


@contextmanager
def bind_remote_native_runtime(vlm_class, client):
    """Instance-local native VLM replacement and deferred policy client binding.

    Serialized construction scope only, not a global egress sandbox. Existing
    instances/SDK clients are not covered; explicit external-service guards stay.
    """
    if _binding.get() is not None:
        raise RuntimeError('Nested remote native runtime binding')
    original = vlm_class.__init__
    binding = {'client': client, 'active': True, 'vlm_instances': 0}
    token = _binding.set(binding)

    def initialize(instance, service_tier=None):
        instance.service_tier = service_tier
        binding['vlm_instances'] += 1
        def complete(obj, model, messages, reasoning_effort, verbosity,
                     response_format=None, vision_detail='auto'):
            if not binding['active']:
                raise RuntimeError('Remote native tool runtime already closed')
            return client.create_completion(model=model, messages=messages,
                reasoning_effort=reasoning_effort, verbosity=verbosity,
                response_format=response_format, vision_detail=vision_detail)
        instance.create_completion = types.MethodType(complete, instance)

    vlm_class.__init__ = initialize
    try:
        yield binding
    finally:
        vlm_class.__init__ = original
        binding['active'] = False
        _binding.reset(token)
