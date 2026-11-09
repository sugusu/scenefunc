"""Decode archived views and distinguish file, RGB, and model-input identity.

No rendering or model calls. Original RGB hashes come from a pinned source
audit and are tied to the original corpus file hash; they are not inferred
from a visually similar image. Any pixel-equivalent recovery is explicit.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote
from urllib.request import urlopen

from PIL import Image
from audit_material_routing import PIN, CORPUS, digest, write

AUDIT = 'runs/920/P920-RESTOREBRIDGE-v1_20261108T062410Z/NATIVE_AUDIT.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--availability', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output/'cache'; cache.mkdir()
    (args.output/'.gitignore').write_text('cache/\n')
    tick = time.perf_counter()
    sources = {}
    def read(path):
        data = subprocess.check_output(['git','show',PIN+':'+path])
        sources[path] = digest(data)
        return json.loads(data)
    audit = read(AUDIT)
    corpus = read(CORPUS)
    original = {}
    for r in audit['rows']:
        for im in r['images']:
            h = im['original_hashes']; sha = h['file_sha256']
            val = {'rgb_sha256':h['rgb_sha256'], 'size':h['size']}
            if sha in original: assert original[sha] == val
            original[sha] = val
    inputs = json.loads(args.inputs.read_text())
    available = json.loads(args.availability.read_text())
    wanted = {i['sha256'] for r in corpus['rows'] for i in r['images']}
    assert set(original) <= wanted
    candidates = {}
    for r in inputs['rows']:
        for v in r['images']:
            for s in v['independently_hashed_matching_sources'][:1]:
                candidates[v['rollout_sha256']] = s
    local = set()
    p = subprocess.Popen(['git','cat-file','--batch-all-objects',
        '--batch-check=%(objectname) %(objecttype) %(objectsize)'],stdout=subprocess.PIPE,text=True)
    for line in p.stdout:
        oid,typ,size = line.split()
        if typ=='blob': local.add(oid)
    assert p.wait()==0
    decoded = {}; recovered = {}; attempts = []; network = 0
    for sha,s in candidates.items():
        item = dict(s, file_sha256=sha)
        try:
            if s['git_blob'] in local:
                data = subprocess.check_output(['git','cat-file','blob',s['git_blob']])
                item['transport'] = 'local_git'
            else:
                remaining = 96*1024**2-network
                if remaining<=0: raise ValueError('Bounded public image read budget exhausted')
                url='https://raw.githubusercontent.com/sugusu/scenefunc/'+PIN+'/'+quote(s['path'],safe='/')
                bound=min(8*1024**2,remaining)
                with urlopen(url,timeout=25) as response: data=response.read(bound+1)
                network+=len(data); item['transport']='public_pinned_http'
                if len(data)>bound: raise ValueError('Image exceeds remaining read bound')
            assert digest(data)==sha
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==s['git_blob']
            with Image.open(io.BytesIO(data)) as im:
                rgb=im.convert('RGB')
                proof={'size':list(rgb.size),'rgb_sha256':digest(rgb.tobytes())}
                item.update(proof, status='decoded',bytes=len(data),mode=im.mode,
                            png_metadata_keys=sorted(im.info))
            decoded[sha]=proof
            matches=[h for h,ref in original.items() if ref==proof]
            if any(h not in recovered for h in matches):
                if shutil.disk_usage(cache).free-len(data)<5*1024**3:
                    raise ValueError('5 GiB reserve prevents caching verified equivalent')
                path=cache/(sha+'.png'); path.write_bytes(data)
                for h in matches:
                    if h not in recovered:
                        recovered[h]=dict(original_file_sha256=h,replacement_file_sha256=sha,
                            local_path=str(path.resolve()),git_path=s['path'],git_blob=s['git_blob'],
                            **proof,proof_source=AUDIT,
                            identity='exact_decoded_RGB; original_PNG_bytes_not_recovered')
        except Exception as exc: item.update(status='unverified',error=str(exc))
        attempts.append(item)
        write(args.output/'image_attempts.json',attempts)
        if len(attempts)%10==0: print('Decoded',len(decoded),'/',len(candidates),flush=True)
    rows=[]; groups=defaultdict(list)
    for r in inputs['rows']:
        request=read(r['request_path'])
        assert sources[r['request_path']]==r['request_sha256']
        norm=copy.deepcopy(request); index=0; imgs=[]; complete=True
        for message in norm:
            if not isinstance(message['content'],list):continue
            for part in message['content']:
                if part.get('type')!='image':continue
                view=r['images'][index]; index+=1
                proof=decoded.get(view['rollout_sha256'])
                ref=original.get(view['original_training_sha256'])
                part['image']=proof
                complete=complete and proof is not None
                imgs.append(dict(index=view['index'],decoded=proof,
                    original_rgb_reference=ref,
                    same_rgb_as_training=(proof==ref) if proof and ref else None))
        assert index==5
        signature=digest(json.dumps(norm,sort_keys=True).encode()) if complete else None
        row=dict(unit_id=r['unit_id'],arm=r['arm'],images=imgs,
                 complete_first_views=complete,nonimage_and_rgb_signature=signature,
                 original_user_text_equal=r['user_text_equal'])
        rows.append(row); groups[r['unit_id']].append(row)
    comparisons=[]
    for unit,rs in groups.items():
        known=[r for r in rs if r['complete_first_views']]
        comparisons.append(dict(unit_id=unit,independently_decoded_arms=len(known),
            available_arm_inputs_equal=len({r['nonimage_and_rgb_signature'] for r in known})<=1,
            all_three_arms_independently_verified=(len(known)==3 and len({r['nonimage_and_rgb_signature'] for r in known})==1)))
    material=[]
    for r in available['rows']:
        images=[]
        for im in r['images']:
            item=copy.deepcopy(im)
            item['pixel_equivalent_recovery']=recovered.get(im['sha256'])
            item['available_with_explicit_equivalence']=bool(im['match'] or item['pixel_equivalent_recovery'])
            images.append(item)
        material.append(dict(unit_id=r['unit_id'],family_id=r['family_id'],images=images,
            complete_five_views=all(i['available_with_explicit_equivalence'] for i in images)))
    result=dict(schema='explicit-rgb-equivalence-v1',pin=PIN,source_sha256=sources,
        local_source_sha256={str(p):digest(p.read_bytes()) for p in (args.inputs,args.availability)},
        rows=rows,conditions=comparisons,recovered_original_image_signatures=recovered,
        material=material,complete_units_before=available['complete_units'],
        complete_units_after_explicit_rgb_recovery=sum(r['complete_five_views'] for r in material),
        original_rgb_signature_count=len(original),decoded_unique_files=len(decoded),
        candidates=len(candidates),network_bytes_read=network,
        png_byte_identity_not_required_for_rgb_identity=True,
        model_tensor_identity='not_tested_in_this_script',
        limitation='Original RGB signatures are archived source-host evidence; replacement RGB bytes independently decoded here. Missing first views remain unverified. No original dataset files modified.',
        wall_seconds=time.perf_counter()-tick,model_calls=0,training_updates=0,native_scene_calls=0)
    write(args.output/'pixel_equivalence.json',result)
    print(json.dumps({k:result[k] for k in ('conditions','complete_units_before','complete_units_after_explicit_rgb_recovery','original_rgb_signature_count','decoded_unique_files','candidates','network_bytes_read','wall_seconds')}))


if __name__=='__main__':main()
