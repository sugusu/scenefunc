"""Bounded shared host for an actual complete-scene continuation trial.

This separate qualification protocol does not reopen the closed D/A study.
No phase suffix, teacher prefix, tag repair, host retry, training or provider fallback.
Native caller feedback attempts share the global budget and are fully recorded.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import signal
import sys
import time


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text())
    assert protocol['max_model_forwards'] == 32
    assert protocol['max_total_output_tokens'] == 16384
    cap_total=protocol['max_total_output_tokens']
    call_total=protocol['max_model_forwards']
    output_so_far=0
    incomplete=False
    for path, expected in protocol['code_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    a.out.mkdir(parents=True, exist_ok=False)
    def deadline(signum, frame):
        dump(a.out / 'hard_deadline.json', {'status': 'terminal_timeout', 'retry': False})
        os._exit(124)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(protocol['actor_seconds'])
    work = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
    sys.path.insert(0, str(work / 'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine, LocalVLMService
    import torch
    forwards, loads, blocked = [], [], []
    role = {'name': None}
    def save():
        dump(a.out / 'ledger.json', dict(forwards=forwards, model_loads=loads,
            blocked_network=blocked, paid_API_calls=0, optimizer_updates=0,
            max_forwards=call_total, max_total_output_tokens=cap_total,
            output_tokens=output_so_far, incomplete_forward_accounting=incomplete))
    class Engine(LocalQwenEngine):
        def _load_model(self):
            absent = self._model is None
            super()._load_model()
            if not absent:
                return
            self._model.requires_grad_(False)
            loads.append(dict(model_object_id=id(self._model),
                first_parameter_ptr=next(self._model.parameters()).data_ptr(),
                all_parameters_frozen=all(not p.requires_grad for p in self._model.parameters())))
            original = self._model.generate
            def metered(*args, **kwargs):
                nonlocal output_so_far,incomplete
                if incomplete or len(forwards)>=call_total:
                    raise RuntimeError('Shared trial budget closed or previous forward accounting incomplete')
                cap = kwargs.get('max_new_tokens')
                if type(cap) is not int or not 0 < cap <= min(1024,cap_total-output_so_far):
                    raise ValueError('Unadmitted output cap')
                ids = kwargs['input_ids']
                row = dict(role=role['name'], status='started', input_tokens=int(ids.shape[1]),
                    max_output_tokens=cap, **loads[0])
                forwards.append(row);incomplete=True;save()
                tick = time.monotonic()
                try:
                    kwargs['max_time'] = 90.
                    output = original(*args, **kwargs)
                    seq = output if isinstance(output, torch.Tensor) else output.sequences
                    assert seq.ndim == 2 and seq.shape[0] == 1 and torch.equal(seq[:, :ids.shape[1]], ids)
                    generated = seq[0, ids.shape[1]:].detach().cpu().tolist()
                    assert len(generated) <= cap
                    row.update(status='returned', output_token_ids=generated, output_tokens=len(generated))
                    output_so_far+=len(generated);incomplete=False
                    eos=self._model.generation_config.eos_token_id
                    eos=eos if isinstance(eos,list) else [eos]
                    if not generated or generated[-1] not in eos:
                        raise ValueError('Incomplete natural output retained but not dispatched')
                    return output
                except BaseException as exc:
                    row.update(status='failed_no_retry', error=type(exc).__name__ + ': ' + str(exc))
                    raise
                finally:
                    row.update(seconds=time.monotonic()-tick,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                    save()
            self._model.generate = metered
            save()
    engine = Engine(Path(protocol['checkpoint']), a.out / 'model_calls',
        max_pixels=65536, max_input_tokens=16384, max_new_tokens=1024)
    local = LocalVLMService(engine)
    def audit(event, args):
        if event == 'socket.connect' or (event == 'socket.getaddrinfo' and
                args[0] not in ('127.0.0.1', 'localhost', '::1')):
            blocked.append(event); save()
            raise RuntimeError('Model host external connections prohibited')
    sys.addaudithook(audit)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self.reply(200 if self.path == '/health' else 404,
                dict(status='ready', model_forwards=len(forwards), model_loads=len(loads),
                    output_tokens=output_so_far,incomplete_forward_accounting=incomplete))
        def do_POST(self):
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16*1024**2 or len(requests)>=2*call_total or incomplete or len(forwards)>=call_total:
                return self.reply(409, {'error': 'Closed request budget or invalid size'})
            raw = self.rfile.read(size)
            index=len(requests)+1
            (a.out / f'request_{index:03d}.json').write_bytes(raw)
            record = dict(endpoint=self.path, request_sha256=hashlib.sha256(raw).hexdigest(), status='started')
            requests.append(record)
            try:
                v = json.loads(raw)
                if self.path == '/native_policy':
                    if set(v) != {'messages', 'max_new_tokens'}:
                        raise ValueError('Only native messages and output cap are accepted')
                    role['name'] = 'native_joint_policy'
                    value = engine.complete(v['messages'], purpose=role['name'],
                        max_new_tokens=v['max_new_tokens'])
                elif self.path == '/completion':
                    role['name'] = 'native_tool'
                    value = {'text': local.create_completion(**v)}
                else:
                    raise ValueError('Unknown endpoint')
                record.update(status='returned', result=value)
                self.reply(200, value)
            except BaseException as exc:
                record.update(status='failed_no_retry', error=type(exc).__name__ + ': ' + str(exc))
                self.reply(500, {'error': record['error']})
            finally:
                role['name'] = None
                dump(a.out / f'request_{index:03d}_receipt.json', record); save()
        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)
    class Server(HTTPServer):
        def server_bind(self):
            import socketserver
            socketserver.TCPServer.server_bind(self)
            self.server_name='localhost'; self.server_port=self.server_address[1]
    save()
    server = Server(('127.0.0.1', a.port), Handler)
    try:
        server.serve_forever(.1)
    finally:
        server.server_close(); save(); signal.alarm(0)


if __name__ == '__main__':
    main()
