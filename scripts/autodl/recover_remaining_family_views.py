"""Resume unscanned published family images; never rerender or alter a corpus.

Only selected project-family runs not covered by the previous bounded search.
All downloaded bytes are held transiently; only exact required images persist.
An unmatched scene image is never a training substitute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote
from urllib.request import urlopen


FAMILIES=('P920-MILAN-BASE-', 'P920-MILAN-TASK-', 'P920-STORAGEGATE-',
          'P920-STORAGE-v1_', 'P920-MODEBASE-', 'P920-FPOLICY-', 'P920-FCHOICE-')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--pixel',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cache=a.out/'cache';cache.mkdir()
    old=json.loads(a.prior.read_text());plan=json.loads(a.plan.read_text())
    pixel=json.loads(a.pixel.read_text());assert old['pin']==plan['pin']==pixel['pin']
    seen={x['git_blob'] for x in old['scanned']}
    candidates=[x for x in plan['images'] if x['git_blob'] not in seen and
                any(any(p.split('/')[2].startswith(prefix) for prefix in FAMILIES) for p in x['paths'])]
    targets={im['sha256']:im for u in pixel['material'] for im in u['images']
             if not im['available_with_explicit_equivalence']}
    write(a.out/'plan.json',{'pin':old['pin'],'candidate_runs':FAMILIES,
        'selection':'Same source project families; no quality scores or success filtering',
        'candidate_blobs':candidates,'unique_missing_targets':len(targets),
        'prior_sha256':digest(a.prior.read_bytes()),'pixel_sha256':digest(a.pixel.read_bytes()),
        'max_total_bytes':128*1024**2,'max_image_bytes':8*1024**2,'reserve_bytes':5*1024**3})
    rows=[];found={};used=0;start=time.monotonic()
    for candidate in candidates:
        path=sorted(candidate['paths'])[0]
        rec={'git_blob':candidate['git_blob'],'path':path}
        try:
            cap=min(8*1024**2,128*1024**2-used)
            if cap<=0:raise RuntimeError('Fixed total read cap reached')
            url='https://raw.githubusercontent.com/sugusu/scenefunc/'+old['pin']+'/'+quote(path,safe='/')
            with urlopen(url,timeout=30) as response:raw=response.read(cap+1)
            used+=len(raw)
            if len(raw)>cap:raise RuntimeError('Per-file or remaining read cap exceeded')
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==candidate['git_blob']
            sha=digest(raw);rec.update(status='hashed',bytes=len(raw),sha256=sha,required_exact_match=sha in targets)
            if sha in targets and sha not in found:
                if shutil.disk_usage(cache).free-len(raw)<5*1024**3:raise RuntimeError('Disk reserve prevents retaining match')
                dest=cache/(sha+Path(path).suffix.lower());dest.write_bytes(raw)
                found[sha]={'local_path':str(dest.resolve()),'sha256':sha,'git_blob':candidate['git_blob'],
                           'git_paths':candidate['paths'],'bytes':len(raw)}
                print('Recovered',sha[:12],path,flush=True)
        except Exception as e:rec.update(status='failed',error=type(e).__name__+': '+str(e))
        rows.append(rec);write(a.out/'attempts.json',rows)
        if len(rows)%8==0:print('checked',len(rows),'of',len(candidates),'matches',len(found),'bytes',used,flush=True)
        if used>=128*1024**2:break
    units=[]
    for unit in pixel['material']:
        images=[]
        for im in unit['images']:
            item=dict(im)
            if item['sha256'] in found:
                item.update(match=found[item['sha256']],available_with_explicit_equivalence=True)
            images.append(item)
        units.append({**unit,'images':images,'complete_five_views':all(im['available_with_explicit_equivalence'] for im in images)})
    result={'pin':old['pin'],'new_exact_matches':len(found),'new_targets':found,
        'complete_units_before':sum(u['complete_five_views'] for u in pixel['material']),
        'complete_units_after':sum(u['complete_five_views'] for u in units),'material':units,
        'candidate_count':len(candidates),'attempted':len(rows),'read_bytes':used,
        'seconds':time.monotonic()-start,'new_model_calls':0,'scene_edits':0,'training_updates':0,
        'all_remaining_published_images_searched':False,'scope':'Seven metadata-selected unscanned same-family run prefixes; no substitutes or altered targets.'}
    write(a.out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['material','new_targets']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
