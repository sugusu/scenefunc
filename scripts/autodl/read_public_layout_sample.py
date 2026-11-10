"""Read a fixed small sample from the original IL3D layout ZIP using HTTP ranges.

No meshes, model weights, full archive or full dataset are downloaded. These are
static author layouts, NOT reasoning/action demonstrations or verified positives.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import zipfile


class RangeReader(io.RawIOBase):
    def __init__(self,url,size,*,traffic_cap=12000000,transport_retries=0):
        self.url=url;self.size=size;self.position=0;self.traffic=0
        self.traffic_cap=traffic_cap;self.receipts=[];self.reserved=0
        self.transport_retries=transport_retries
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=io.SEEK_SET):
        value=offset if whence==io.SEEK_SET else self.position+offset if whence==io.SEEK_CUR else self.size+offset
        if value<0:raise ValueError('Negative archive position')
        self.position=value;return value
    def read(self,n=-1):
        if n<0:n=self.size-self.position
        n=min(n,self.size-self.position)
        if n>4000000:raise ValueError('Single metadata read exceeds4MB')
        if n<=0:return b''
        pieces=[]
        while n:
            take=min(n,500000)
            start=self.position;end=start+take-1
            req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}',
                   'Accept-Encoding':'identity','User-Agent':'bounded-layout-schema-audit'})
            for attempt in range(self.transport_retries+1):
                if self.reserved+take>self.traffic_cap:raise ValueError('Bounded range traffic exhausted')
                self.reserved+=take
                tick=time.monotonic()
                try:
                    with urllib.request.urlopen(req,timeout=30) as f:
                        content_range=f.headers.get('Content-Range','')
                        if f.status!=206 or content_range!=f'bytes {start}-{end}/{self.size}':
                            raise ValueError('Server did not honor exact byte range; full download refused')
                        data=f.read(take+1)
                    break
                except (urllib.error.URLError,TimeoutError) as exc:
                    # Access or range denial is never bypassed. Retry only a bounded
                    # transport failure at the exact same immutable archive offset.
                    if isinstance(exc,urllib.error.HTTPError):raise
                    self.receipts.append(dict(start=start,end=end,attempt=attempt,
                          error=type(exc).__name__,reserved_bytes=take))
                    if attempt>=self.transport_retries:raise
            if len(data)!=take:raise ValueError('Unexpected range size')
            self.position+=take;self.traffic+=take;n-=take;pieces.append(data)
            self.receipts.append(dict(start=start,end=end,bytes=take,
                 sha256=hashlib.sha256(data).hexdigest(),seconds=time.monotonic()-tick))
        return b''.join(pieces)


def main(stage, resume_from=None):
    out=stage/('il3d_static_layout_sample_v2' if resume_from else 'il3d_static_layout_sample_v1')
    out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((stage/'public_data/il3d_minimal.json').read_text())
    tree=json.loads((stage/'public_data/il3d_tree.json').read_text())
    entry=next(x for x in tree if x['path']=='layout.zip')
    assert metadata['id']=='WenxuZhou/IL3D' and metadata['gated'] is False
    url=f'https://huggingface.co/datasets/WenxuZhou/IL3D/resolve/{metadata["sha"]}/layout.zip'
    previous=json.loads((resume_from/'result.json').read_text()) if resume_from else None
    prior_reserve=previous['network_bytes_read']+500000 if previous else 0
    if previous:assert previous['dataset_revision']==metadata['sha'] and previous['URL']==url
    reader=RangeReader(url,entry['size'],traffic_cap=12000000-prior_reserve,
                       transport_retries=1 if previous else 0)
    result=dict(status='started',
        dataset_revision=metadata['sha'],URL=url,
        source_project='https://wenxuzhou.github.io/project/IL3D/',
        original_archive_bytes=entry['size'],full_archive_downloaded=False,
        geometry_downloaded=False,static_layouts_not_teacher_trajectories=True,
        training_split_status='not_established_do_not_use_as_training',
        independently_verified_quality=False,model_forwards=0,training_updates=0,paid_API_calls=0)
    try:
        with zipfile.ZipFile(reader) as z:
            all_files=z.infolist()
            inventory=[dict(name=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,
                            CRC32=f'{x.CRC:08x}') for x in all_files]
            if not resume_from:
                (out/'archive_inventory.json').write_text(json.dumps(inventory,separators=(',',':'))+'\n')
            else:
                old_inventory=json.loads((resume_from/'archive_inventory.json').read_text())
                assert inventory==old_inventory
                result['inventory_reference']=str(resume_from/'archive_inventory.json')
            selected=sorted((x for x in all_files if x.filename.endswith('.json')
                    and not x.filename.startswith('__MACOSX/')),key=lambda x:x.filename)[:8]
            if len(selected)!=8:raise ValueError('Fewer than8 JSON entries; do not substitute other samples')
            # Selection is frozen BEFORE reading any layout payload. Static layouts
            # have not been established as the first training rows promised for an
            # original reasoning corpus; this is a separate schema-only probe.
            selection=dict(rule='First8 lexicographic non-MacOS JSON members, before payload inspection',
                           scope='Static author layout schema only; no inferred train split or quality label',
                           entries=[x.filename for x in selected],
                           max_expanded_bytes_per_sample=250000)
            if resume_from:assert selection==json.loads((resume_from/'SELECTION.json').read_text())
            (out/'SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n')
            samples=[]
            for i,info in enumerate(selected):
                if info.file_size>250000:raise ValueError('Selected sample too large; retain selection without replacement')
                old=next((x for x in previous.get('samples',[]) if x['archive_member']==info.filename),None) if previous else None
                if old:
                    data=(resume_from/old['local_file']).read_bytes()
                    assert hashlib.sha256(data).hexdigest()==old['sha256']
                else:data=z.read(info)
                value=json.loads(data)
                p=out/f'layout_{i:02d}.json';p.write_bytes(data)
                samples.append(dict(archive_member=info.filename,local_file=p.name,
                     bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                     CRC32=f'{info.CRC:08x}',root_type=type(value).__name__,
                     keys=list(value) if isinstance(value,dict) else None))
                result['samples']=samples
                (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            result['status']='selected8_static_layout_payloads_read_CRC_checked'
            result['archive_file_count']=len(all_files)
    except Exception as exc:
        result.update(status='bounded_probe_stopped_no_automatic_retry',
                      error=type(exc).__name__+': '+str(exc))
    result['network_bytes_read']=reader.traffic
    result['worst_case_reserved_network_bytes_including_prior']=reader.reserved+prior_reserve
    result['range_requests']=reader.receipts
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('range_requests','samples')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--resume-from',type=Path)
    a=p.parse_args();main(a.stage.resolve(),a.resume_from.resolve() if a.resume_from else None)
