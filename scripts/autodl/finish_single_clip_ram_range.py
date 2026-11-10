"""One inspected dependency recovery after a terminal bounded transfer.

Requires a confirmed original-process termination receipt and exactly one
uncompleted range. No whole-file restart, model query, or recursive retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import urllib.request


def dump(p, obj):
    p.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',type=Path,required=True)
    a=ap.parse_args();stage=a.stage.resolve()
    out=stage/'restore_v2';out.mkdir(exist_ok=False)
    original=json.loads((stage/'restore_v1/transfer.json').read_text())
    terminal=json.loads((stage/'restore_v1/budget_termination.json').read_text())
    if Path('/proc',str(terminal['pid']),'cmdline').exists():
        cmd=Path('/proc',str(terminal['pid']),'cmdline').read_bytes()
        if b'restore_original_clip_in_ram.py' in cmd:
            raise RuntimeError('Original transfer still alive; do not overlap')
    size,chunk=original['expected_bytes'],original['chunk_bytes']
    done={r['start'] for r in original['completed']}
    missing=[(s,min(size,s+chunk)) for s in range(0,size,chunk) if s not in done]
    if len(missing)!=1 or original['failed']:
        raise ValueError('Recovery is limited to one inspected incomplete range')
    left,right=missing[0];path=Path(original['weight_path'])
    if not path.is_file() or path.stat().st_size!=size:
        raise ValueError('Original RAM file no longer present')
    r={'status':'started','original_transfer':'../restore_v1/transfer.json',
       'original_process_terminal_receipt':'../restore_v1/budget_termination.json',
       'expected_bytes':size,'weight_path':str(path),
       'original_official_sha256':original['original_official_sha256'],
       'range_start':left,'range_end_exclusive':right,'received':0,
       'one_manual_inspected_range_recovery':True,'automatic_further_retry':False,
       'max_range_seconds':60,'model_calls':0,'paid_API_calls':0,'training_updates':0,
       'conservative_total_weight_transfer_upper_bytes':size+(right-left)+2**20}
    dump(out/'transfer.json',r);start=time.monotonic();fd=None
    def timeout(signum,frame):raise TimeoutError('Single recovery range exceeded60seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req=urllib.request.Request(original['source']+f'?download=true&range_start={left}&recovery=20261110_once',
            headers={'Range':f'bytes={left}-{right-1}','Accept-Encoding':'identity'})
        fd=os.open(path,os.O_WRONLY)
        with opener.open(req,timeout=20) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {left}-{right-1}/{size}':
                raise ValueError('Recovery response has wrong byte range')
            while r['received']<right-left:
                block=response.read(min(2**20,right-left-r['received']))
                if not block:raise ValueError('Short recovery body')
                if os.pwrite(fd,block,left+r['received'])!=len(block):raise IOError('Partial recovery write')
                r['received']+=len(block);r['seconds']=time.monotonic()-start;dump(out/'transfer.json',r)
            if response.read(1):raise ValueError('Recovery body too long')
        os.fsync(fd);os.close(fd);fd=None
        signal.alarm(0)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
        r['actual_sha256']=h.hexdigest()
        if r['actual_sha256']!=r['original_official_sha256']:raise ValueError('Full official SHA mismatch')
        r['status']='official_original_hash_verified'
    except BaseException as exc:
        r['status']='recovery_failed_no_automatic_retry';r['error_type']=type(exc).__name__
        raise
    finally:
        if fd is not None:os.close(fd)
        signal.alarm(0);r['seconds']=time.monotonic()-start;dump(out/'transfer.json',r)
        print(json.dumps(r),flush=True)


if __name__=='__main__':main()
