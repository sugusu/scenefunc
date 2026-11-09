"""Recover exact missing views from other published runs, without rerendering.

Only SHA-locked corpus images are retained. Inspect existing local Git blobs
first, then a fixed list of relevant runs over bounded public HTTP reads. This
does not fetch remote Git objects or consume model/API inference credits.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote
from urllib.request import urlopen

from audit_material_routing import CORPUS, PIN, digest, write

RUN_PREFIXES = ('P920-MATCHEDROLL-', 'P920-ALTREPLAY-', 'P920-EVALFIX-',
                'P920-EVALTASK-', 'P920-EDITSTUDY-', 'P920-STOPDATA-', 'P920-KBASE-')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--network-bytes', type=int, default=96*1024**2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output/'cache'; cache.mkdir()
    tick = time.perf_counter()
    raw = subprocess.check_output(['git', 'show', PIN+':'+CORPUS])
    corpus = json.loads(raw)
    prior = json.loads(args.prior.read_text())
    assert prior['pin'] == PIN and prior['corpus_sha256'] == digest(raw)
    wanted = {i['sha256'] for r in corpus['rows'] for i in r['images']}
    seen = {r['git_blob']:r['sha256'] for r in prior['scanned']}
    scanned = prior['scanned'][:]
    matches = {}
    for row in prior['rows']:
        for img in row['images']:
            if img['match']:
                assert digest(Path(img['match']['local_path']).read_bytes()) == img['sha256']
                matches[img['sha256']] = img['match']
    paths = defaultdict(list)
    for line in subprocess.check_output(['git','ls-tree','-r',PIN,'runs/920'],text=True).splitlines():
        meta,path = line.split('\t',1)
        if Path(path).suffix.lower() in ('.png','.jpg','.jpeg','.webp'):
            paths[meta.split()[2]].append(path)
    local = {}
    proc = subprocess.Popen(['git','cat-file','--batch-all-objects',
        '--batch-check=%(objectname) %(objecttype) %(objectsize)'],stdout=subprocess.PIPE,text=True)
    for line in proc.stdout:
        oid,typ,size = line.split()
        if oid in paths and typ == 'blob': local[oid] = int(size)
    assert proc.wait() == 0
    def priority(oid):
        selected = [p for p in paths[oid] if any(p.split('/')[2].startswith(x) for x in RUN_PREFIXES)]
        rank = min((i for p in selected for i,x in enumerate(RUN_PREFIXES) if p.split('/')[2].startswith(x)),default=100)
        return (oid not in local, rank, sorted(paths[oid])[0])
    pending = sorted((o for o in paths if o not in seen),key=priority)
    plan = [{'git_blob':o,'paths':paths[o],'local_bytes':local.get(o),
             'network_eligible':priority(o)[1]<100} for o in pending]
    write(args.output/'plan.json',{'pin':PIN,'prior_sha256':digest(args.prior.read_bytes()),
        'network_byte_limit':args.network_bytes,'per_image_limit':8*1024**2,
        'reserve_bytes':5*1024**3,'eligible_run_prefixes':RUN_PREFIXES,'images':plan,
        'selection_basis':'Published run and unit metadata; no outcome or preference scores'})
    attempts=[];network_bytes=0;local_bytes=0;retained_bytes=0;stop=None
    for item in plan:
        oid=item['git_blob'];path=sorted(item['paths'])[0]
        if oid not in local and not item['network_eligible']:continue
        if oid not in local and network_bytes>=args.network_bytes:
            stop='bounded_network_read_limit';break
        row={'git_blob':oid,'path':path,'transport':'local_git' if oid in local else 'public_pinned_http'}
        try:
            if oid in local:
                if local[oid]>8*1024**2:raise ValueError('Local image exceeds per-file bound')
                data=subprocess.check_output(['git','cat-file','blob',oid])
                local_bytes+=len(data)
            else:
                url='https://raw.githubusercontent.com/sugusu/scenefunc/'+PIN+'/'+quote(path,safe='/')
                bound=min(8*1024**2,args.network_bytes-network_bytes)
                with urlopen(url,timeout=25) as response:data=response.read(bound+1)
                network_bytes+=len(data)
                if len(data)>bound:raise ValueError('Image exceeds remaining/per-file read bound')
            git_hash=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if git_hash!=oid:raise ValueError('Pinned Git blob mismatch')
            sha=digest(data);row.update(status='hashed',sha256=sha,bytes=len(data),wanted=sha in wanted)
            scanned.append({'git_blob':oid,'path':path,'bytes':len(data),'sha256':sha})
            seen[oid]=sha
            if sha in wanted and sha not in matches:
                if shutil.disk_usage(args.output).free-len(data)<5*1024**3:
                    raise ValueError('Disk reserve prevents writing this exact match')
                target=cache/(sha+Path(path).suffix.lower())
                with target.open('xb') as stream:stream.write(data)
                retained_bytes+=len(data)
                matches[sha]={'sha256':sha,'git_blob':oid,'git_paths':item['paths'],
                    'local_path':str(target.resolve()),'bytes':len(data)}
                print('Recovered exact view '+sha[:12]+' from '+path,flush=True)
        except Exception as exc:
            row.update(status='failed',error=str(exc))
        attempts.append(row)
        write(args.output/'attempts.json',attempts)
        if len(attempts)%10==0:print(f'Checked {len(attempts)} new blobs; matches {len(matches)}/{len(wanted)}; HTTP {network_bytes} bytes',flush=True)
        if len(matches)==len(wanted):break
    rows=[]
    for unit in corpus['rows']:
        images=[{**i,'match':matches.get(i['sha256'])} for i in unit['images']]
        rows.append({'unit_id':unit['id'],'family_id':unit['family_id'],'images':images,
            'complete_five_views':len(images)==5 and all(i['match'] for i in images)})
    result={'pin':PIN,'corpus_path':CORPUS,'corpus_sha256':digest(raw),
        'source_units':len(rows),'image_references':sum(len(r['images']) for r in rows),
        'unique_required_images':len(wanted),'unique_exact_matches':len(matches),
        'complete_units':sum(r['complete_five_views'] for r in rows),
        'complete_unit_ids':[r['unit_id'] for r in rows if r['complete_five_views']],
        'prior_lookup':str(args.prior),'prior_sha256':digest(args.prior.read_bytes()),
        'new_blobs_attempted':len(attempts),'unique_blobs_scanned':len(scanned),
        'network_bytes_read':network_bytes,'local_bytes_read':local_bytes,'retained_bytes':retained_bytes,
        'stop_reason':stop,'scope':'Bounded search of cached blobs and selected other published runs; not source-host filesystem recovery',
        'published_image_search_complete':len(seen)==len(paths),'rows':rows,'scanned':scanned,
        'wall_seconds':time.perf_counter()-tick,'model_calls':0,'training_updates':0,'rendered_replacements':0}
    write(args.output/'image_availability.json',result)
    missing=defaultdict(list);source_paths={}
    for row in rows:
        for i in row['images']:
            if not i['match']:missing[i['sha256']].append(row['unit_id']);source_paths[i['sha256']]=i['path']
    write(args.output/'REQUIRED_REMOTE_EXPORT.json',{'pin':PIN,'corpus_sha256':digest(raw),
        'images':[{'sha256':s,'source_path':source_paths[s],'required_by':ids} for s,ids in sorted(missing.items())],
        'scope':'Still required exact original images; no claim that source-host files are absent'})
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','scanned')},ensure_ascii=False))


if __name__=='__main__':main()
