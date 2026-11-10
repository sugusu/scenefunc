"""One frozen local model host for native tools and prospective phase policies.

Cross-Python loopback transport. The host owns one model and one total forward
budget. No API, adapter training, source edit, output repair or automatic retry.
This provides a common execution backend, not an assertion of policy quality.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time


def dump(p, value):
    p = Path(p)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


class SharedNativeClient:
    def __init__(self, port, *, timeout=180):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False
        self.base = f'http://127.0.0.1:{port}'
        self.timeout = timeout

    def health(self):
        return self.session.get(self.base + '/health', timeout=1).json()

    def phase(self, *, arm, messages, images, state_sha256):
        response = self.session.post(self.base + '/phase', json={
            'arm': arm, 'messages': messages, 'images': images, 'state_sha256': state_sha256},
            timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def create_completion(self, model, messages, reasoning_effort, verbosity,
                          response_format=None, vision_detail='auto'):
        response = self.session.post(self.base + '/completion', json=dict(
            model=model, messages=messages, reasoning_effort=reasoning_effort,
            verbosity=verbosity, response_format=response_format, vision_detail=vision_detail),
            timeout=self.timeout)
        response.raise_for_status()
        return response.json()['text']

    def close(self):
        self.session.close()


class RemotePhasePolicy:
    """Existing native host/PhasePolicy generate(obs) interface, no local weights.

    Image paths and hashes cross the shared filesystem; callers must keep the
    current observation immutable until the request completes. No teacher D or
    training targets are accepted. Invalid responses remain failed attempts.
    """
    def __init__(self, client, *, arm):
        self.client, self.arm = client, arm

    def generate(self, obs):
        from phase_policy import PhaseOutcomeError
        paths = obs.get('image_paths')
        if not isinstance(paths, list) or len(paths) != 5:
            raise ValueError('Five current image paths required for remote phase policy')
        images = [{'path': str(p), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                  for p in paths]
        result = self.client.phase(arm=self.arm, messages=obs['messages'], images=images,
                                   state_sha256=obs['state_sha256'])
        if result['status'] not in ('tool_proposed', 'stop_proposed'):
            raise PhaseOutcomeError(result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((a.stage / 'protocol.json').read_text())
    assert protocol['max_model_forwards'] == 4
    assert protocol['max_total_output_tokens'] == 1536
    for name, expected in protocol['code_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    def hard_deadline(signum, frame):
        dump(a.out / 'actor_hard_deadline.json', {'status': 'terminal_hard_timeout',
             'seconds': 480, 'retry_allowed': False})
        os._exit(124)
    signal.signal(signal.SIGALRM, hard_deadline)
    signal.alarm(480)
    work = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
    sys.path.insert(0, str(work / 'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine, LocalVLMService
    from phase_policy import PhasePolicy, TorchPhaseBackend, PhaseOutcomeError
    from PIL import Image
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    rows = []
    role = {'name': None}
    loads = []
    blocked = []
    total_output = 0
    incomplete = False

    def save():
        dump(a.out / 'shared_model_ledger.json', {
            'model_loads': loads, 'forwards': rows, 'max_forwards': 4,
            'total_output_tokens': total_output, 'max_total_output_tokens': 1536,
            'incomplete_forward_accounting': incomplete, 'blocked_network': blocked,
            'paid_API_calls': 0, 'optimizer_updates': 0,
            'scope': 'Single frozen model inference only; no generation-quality certification'})

    class Engine(LocalQwenEngine):
        def _load_model(self):
            nonlocal total_output, incomplete
            absent = self._model is None
            super()._load_model()
            if not absent:
                return
            self._model.requires_grad_(False)
            original = self._model.generate
            model_id = id(self._model)
            ptr = next(self._model.parameters()).data_ptr()
            loads.append({'model_object_id': model_id, 'first_parameter_data_ptr': ptr,
                          'dtype': str(next(self._model.parameters()).dtype),
                          'all_parameters_frozen': all(not x.requires_grad for x in self._model.parameters()),
                          'adapter_loaded': False})
            def metered(*args, **kwargs):
                nonlocal total_output, incomplete
                if incomplete or len(rows) >= 4:
                    raise RuntimeError('Shared forward budget closed; no retry')
                cap = kwargs.get('max_new_tokens')
                if not isinstance(cap, int) or cap <= 0 or cap > 1536 - total_output:
                    raise ValueError('Shared output-token budget exceeded')
                input_ids = kwargs.get('input_ids')
                if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
                    raise ValueError('Expected exactly one serialized model request')
                record = {'role': role['name'], 'status': 'started', 'generation_started': True,
                          'input_tokens': int(input_ids.shape[1]), 'max_new_tokens': cap,
                          'model_object_id': model_id, 'first_parameter_data_ptr': ptr}
                rows.append(record); incomplete = True; save()
                tick = time.monotonic()
                kwargs['max_time'] = min(float(kwargs.get('max_time', 90)), 90.0)
                try:
                    output = original(*args, **kwargs)
                    seq = output if isinstance(output, torch.Tensor) else output.sequences
                    if seq.ndim != 2 or seq.shape[0] != 1 or not torch.equal(
                            seq[:, :input_ids.shape[1]], input_ids):
                        raise ValueError('Model changed the request prefix')
                    ids = seq[0, input_ids.shape[1]:].detach().cpu().tolist()
                    record.update(status='returned', output_tokens=len(ids), output_ids=ids)
                    total_output += len(ids)
                    if len(ids) > cap or total_output > 1536:
                        raise RuntimeError('Observed output budget violation')
                    incomplete = False
                    return output
                except BaseException as exc:
                    record.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
                    raise
                finally:
                    record['seconds'] = time.monotonic() - tick
                    record['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    record['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
                    save()
            self._model.generate = metered
            save()

    engine = Engine(Path(protocol['checkpoint']), a.out / 'tool_model_calls',
                    max_pixels=65536, max_input_tokens=16384, max_new_tokens=512)
    local = LocalVLMService(engine)

    def audit(event, args):
        if event == 'socket.connect':
            blocked.append('connect'); save()
            raise RuntimeError('Model host outbound connections prohibited')
        if event == 'socket.getaddrinfo' and args[0] not in ('127.0.0.1', 'localhost', '::1'):
            blocked.append('dns'); save()
            raise RuntimeError('Model host outbound DNS prohibited')
    sys.addaudithook(audit)
    request_index = 0
    save()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'error': 'not_found'})
            self.reply(200, {'status': 'ready', 'model_loads': len(loads), 'forwards': len(rows),
                             'model_loaded': engine._model is not None,
                             'incomplete_forward_accounting': incomplete})

        def do_POST(self):
            nonlocal request_index
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 8 * 1024**2:
                return self.reply(413, {'error': 'request_size'})
            raw = self.rfile.read(size)
            request_index += 1
            path = a.out / f'request_{request_index:02d}.json'
            path.write_bytes(raw)
            request_receipt = {'endpoint': self.path, 'request_sha256': hashlib.sha256(raw).hexdigest(),
                               'status': 'started', 'first_forward': len(rows)}
            tick = time.monotonic()
            try:
                payload = json.loads(raw)
                if self.path == '/completion':
                    role['name'] = 'native_tool'
                    if set(payload) != {'model', 'messages', 'reasoning_effort', 'verbosity',
                                        'response_format', 'vision_detail'}:
                        raise ValueError('Unexpected native tool fields')
                    result = {'text': local.create_completion(**payload)}
                elif self.path == '/phase':
                    if set(payload) != {'arm', 'messages', 'images', 'state_sha256'}:
                        raise ValueError('Only current observations accepted; no target/teacher D')
                    if payload['arm'] not in ('separate_direct_v3', 'paired_phase_v3'):
                        raise ValueError('Unadmitted policy condition')
                    if len(payload['images']) != 5:
                        raise ValueError('Five current views required')
                    images = []
                    try:
                        for r in payload['images']:
                            data = Path(r['path']).read_bytes()
                            if hashlib.sha256(data).hexdigest() != r['sha256']:
                                raise ValueError('Current observation image changed')
                            with Image.open(r['path']) as im:
                                images.append(im.convert('RGB'))
                        role['name'] = payload['arm']
                        engine._load_model()
                        backend = TorchPhaseBackend(engine._model, engine.processor, max_context_tokens=16384)
                        policy = PhasePolicy(backend, arm=payload['arm'], total_output_tokens=512,
                            decision_output_tokens=192, action_output_tokens=512, total_seconds=90, seed=0)
                        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                            try:
                                result = policy.generate(dict(messages=payload['messages'], images=images,
                                                              state_sha256=payload['state_sha256']))
                            except PhaseOutcomeError as exc:
                                result = exc.receipt
                    finally:
                        for im in images:
                            im.close()
                else:
                    raise ValueError('Unknown endpoint')
                request_receipt.update(status='returned', result=result)
                self.reply(200, result)
            except BaseException as exc:
                request_receipt.update(status='failed_no_retry', error=type(exc).__name__ + ': ' + str(exc))
                self.reply(500, {'error': request_receipt['error']})
            finally:
                role['name'] = None
                request_receipt.update(seconds=time.monotonic() - tick, last_forward=len(rows))
                dump(a.out / f'request_{request_index:02d}_receipt.json', request_receipt)
                save()

        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)

    class Server(HTTPServer):
        def server_bind(self):
            import socketserver
            socketserver.TCPServer.server_bind(self)
            self.server_name = 'localhost'; self.server_port = self.server_address[1]

    server = Server(('127.0.0.1', a.port), Handler)
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        signal.alarm(0)
        server.server_close(); save()


if __name__ == '__main__':
    main()
