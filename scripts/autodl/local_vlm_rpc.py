"""Loopback transport to the existing shared local VLM, with a hard call cap.

No provider fallback, remote-image download, JSON repair or retry. A successful
HTTP reply means only that the same local adapter returned; physics truth and
whole-scene success require their own evidence.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import sys
import time


class LocalVLMClient:
    def __init__(self,port,timeout=180):
        import requests
        self.session=requests.Session();self.session.trust_env=False
        self.url=f'http://127.0.0.1:{port}';self.timeout=timeout

    def health(self):
        return self.session.get(self.url+'/health',timeout=1).json()

    def create_completion(self,model,messages,reasoning_effort,verbosity,
                          response_format=None,vision_detail='auto'):
        response=self.session.post(self.url+'/completion',json=dict(model=model,
            messages=messages,reasoning_effort=reasoning_effort,verbosity=verbosity,
            response_format=response_format,vision_detail=vision_detail),timeout=self.timeout)
        response.raise_for_status()
        return response.json()['text']

    def close(self):self.session.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--maximum-calls',type=int,default=1)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert a.maximum_calls==1,'This qualification admits exactly one actual inference attempt'
    work=Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
    sys.path.insert(0,str(work/'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine,LocalVLMService
    engine=LocalQwenEngine(work/'critic_public_runtime_20261104/checkpoint',
        a.out/'model_calls',max_pixels=65536,max_input_tokens=8192,max_new_tokens=512)
    local=LocalVLMService(engine)
    counter=0
    def audit(event,args):
        if event=='socket.connect':
            raise RuntimeError('Inference server cannot open outbound sockets')
        if event=='socket.getaddrinfo' and args[0] not in ('127.0.0.1','localhost','::1'):
            raise RuntimeError('Inference server external DNS prohibited')
    sys.addaudithook(audit)

    class Handler(BaseHTTPRequestHandler):
        def reply(self,code,obj):
            raw=json.dumps(obj).encode();self.send_response(code)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)

        def do_GET(self):
            if self.path!='/health':return self.reply(404,{'error':'not_found'})
            self.reply(200,{'status':'ready','admitted_calls':counter,
                           'model_loaded':engine._model is not None,'maximum_calls':a.maximum_calls})

        def do_POST(self):
            nonlocal counter
            if self.path!='/completion':return self.reply(404,{'error':'not_found'})
            if counter>=a.maximum_calls:return self.reply(429,{'error':'call_budget_exhausted_no_retry'})
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=8*1024**2:return self.reply(413,{'error':'request_size'})
            raw=self.rfile.read(length);request=json.loads(raw)
            expected={'model','messages','reasoning_effort','verbosity','response_format','vision_detail'}
            if set(request)!=expected:return self.reply(400,{'error':'unsupported_request_fields'})
            counter+=1;tick=time.monotonic()
            row={'call':counter,'status':'admitted','request_sha256':hashlib.sha256(raw).hexdigest(),
                 'request_bytes':len(raw),'paid_api_calls':0,'training_updates':0}
            request_path=a.out/f'call_{counter:02d}_request_local_only.json';request_path.write_bytes(raw)
            receipt=a.out/f'call_{counter:02d}_receipt.json'
            receipt.write_text(json.dumps(row,indent=2)+'\n')
            try:
                returned=local.create_completion(**request)
                row.update(status='returned',response_sha256=hashlib.sha256(returned.encode()).hexdigest())
                self.reply(200,{'text':returned,'actual_backend':'existing_local_SceneBuilder8B'})
            except BaseException as exc:
                row.update(status='failed_no_retry',error=type(exc).__name__+': '+str(exc))
                self.reply(500,{'error':row['error']})
            finally:
                row['seconds']=time.monotonic()-tick
                receipt.write_text(json.dumps(row,indent=2)+'\n')

        def log_message(self,format,*args):
            print(format%args,flush=True)

    class LoopbackServer(HTTPServer):
        # Avoid HTTPServer's getfqdn lookup; no remote name resolution needed.
        def server_bind(self):
            import socketserver
            socketserver.TCPServer.server_bind(self)
            self.server_name='localhost';self.server_port=self.server_address[1]

    server=LoopbackServer(('127.0.0.1',a.port),Handler)
    try:server.serve_forever(poll_interval=.1)
    finally:server.server_close()


if __name__=='__main__':main()
