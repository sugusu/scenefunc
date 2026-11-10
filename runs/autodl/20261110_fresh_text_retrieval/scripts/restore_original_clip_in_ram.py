"""Restore the previously authorized exact DFN checkpoint into temporary RAM.

No data-disk weights, new model choice, inference, training or paid API. The
mirror must return exact bounded ranges; the final original official SHA is
mandatory before any load. A failed transfer is terminal, not auto-restarted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import urllib.request

SIZE = 3947081637
SHA = 'c07a17b547d461c60a3cce5062b26bf8545b13de602c4c59d8490361eb716033'
REVISION = '01b771ed0d1395ca5ffdd279897d665ebe00dfd2'
REPO = 'apple/DFN5B-CLIP-ViT-H-14-378'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--deadline-seconds', type=int, default=600)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(a.out).free < 5*2**30:
        raise RuntimeError('Data reserve not satisfied')
    if shutil.disk_usage('/dev/shm').free < SIZE + 512*2**20:
        raise RuntimeError('Insufficient temporary RAM filesystem space')
    limit = int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text())
    if limit < 24*2**30:
        raise RuntimeError('This bounded restore/load plan requires the currently available 24GiB+ instance')
    old = Path('/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/official_states')
    plan = json.loads((old/'next_download_request.json').read_text())
    lfs = json.loads((old/'clip_official_lfs.json').read_text())
    row = next(x for x in plan['files'] if x['path'] == 'open_clip_pytorch_model.bin')
    if row['revision'] != REVISION or row['repo'] != REPO or row['bytes'] != SIZE or lfs['lfs']['oid'] != SHA:
        raise RuntimeError('Original authorized model provenance differs')
    tmp = Path(tempfile.mkdtemp(prefix='original_dfn_clip_20261110_', dir='/dev/shm'))
    target = tmp/'open_clip_pytorch_model.bin'
    fd = os.open(target, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    os.ftruncate(fd, SIZE)
    started = time.monotonic()
    chunk = 16*2**20
    spans = [(i, min(i+chunk, SIZE)) for i in range(0, SIZE, chunk)]
    record = {'status':'started', 'repo':REPO, 'revision':REVISION,
              'original_official_sha256':SHA, 'expected_bytes':SIZE,
              'weight_path':str(target), 'ephemeral_ram_storage':True,
              'data_disk_weight_bytes':0, 'workers':8, 'chunk_bytes':chunk,
              'max_transfer_seconds':a.deadline_seconds,
              'source':'https://hf-mirror.com/'+REPO+'/resolve/'+REVISION+'/open_clip_pytorch_model.bin',
              'completed':[], 'failed':[], 'received_bytes':0,
              'model_calls':0,'training_updates':0,'paid_API_calls':0,
              'authorization':'Same exact official model explicitly authorized earlier; original download cap subsequently removed by user. Not a new model.',
              'no_automatic_whole_transfer_retry':True}
    lock = threading.Lock()
    def journal():
        temp = a.out/'transfer.tmp'
        temp.write_text(json.dumps(record,indent=2)+'\n')
        temp.replace(a.out/'transfer.json')
    def fetch(span):
        left, right = span
        received = 0
        try:
            if time.monotonic()-started > a.deadline_seconds:
                raise TimeoutError('Transfer deadline reached before range start')
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(record['source']+f'?download=true&range_start={left}&stage=20261110',
                headers={'Range':f'bytes={left}-{right-1}','Accept-Encoding':'identity'})
            with opener.open(req,timeout=25) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {left}-{right-1}/{SIZE}':
                    raise ValueError('Server did not honor exact range')
                while received < right-left:
                    if time.monotonic()-started > a.deadline_seconds:
                        raise TimeoutError('Transfer deadline reached')
                    block = response.read(min(2**20, right-left-received))
                    if not block:
                        raise ValueError('Short range body')
                    wrote = os.pwrite(fd, block, left+received)
                    if wrote != len(block):
                        raise IOError('Partial RAM write')
                    received += len(block)
                if response.read(1):
                    raise ValueError('Oversized range body')
            result = {'start':left,'end_exclusive':right,'received':received}
            with lock:
                record['completed'].append(result)
                record['received_bytes'] += received
                record['elapsed_seconds'] = time.monotonic()-started
                journal()
                if len(record['completed'])%8 == 0 or len(record['completed']) == len(spans):
                    print(json.dumps({'completed_ranges':len(record['completed']),
                        'total_ranges':len(spans),'received_mib':record['received_bytes']/2**20,
                        'seconds':record['elapsed_seconds']}),flush=True)
        except Exception as exc:
            with lock:
                record['failed'].append({'start':left,'end_exclusive':right,'received':received,
                    'error_type':type(exc).__name__, 'http_code':getattr(exc,'code',None)})
                record['received_bytes'] += received
                journal()
    journal()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in as_completed([pool.submit(fetch, span) for span in spans]):f.result()
        os.fsync(fd)
        os.close(fd)
        fd = None
        if record['failed']:
            raise RuntimeError('Some bounded ranges failed; no automatic retry')
        h = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
        record['actual_sha256'] = h.hexdigest()
        if h.hexdigest() != SHA:
            raise ValueError('Official full-file SHA mismatch')
        record['status'] = 'official_original_hash_verified'
        record['ephemeral_weight_retained_for_same_stage'] = True
    except BaseException as exc:
        record['status'] = 'failed_terminal_no_automatic_retry'
        record['error_type'] = type(exc).__name__
        if fd is not None:os.close(fd)
        target.unlink(missing_ok=True)
        tmp.rmdir()
        record['partial_weight_removed'] = True
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        record['data_disk_free_bytes_after'] = shutil.disk_usage(a.out).free
        journal()
        print(json.dumps({k:v for k,v in record.items() if k not in ('completed','failed')}),flush=True)


if __name__ == '__main__':
    main()
