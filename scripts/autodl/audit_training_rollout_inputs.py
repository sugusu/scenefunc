"""Distinguish same task/state from identical training and evaluation inputs.

Reads original matched material and all 12 saved first requests. Evaluation
images are checked against independently hashed published blobs, never used as
silent replacements for unavailable training views. No model or scene calls.
"""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import subprocess

from audit_material_routing import CORPUS, PIN, digest, write
from audit_decision_phase_support import raw_messages

BASE = 'runs/920/P920-MATCHEDROLL-v1_20261108T062940Z'


def text_content(messages):
    return '\n'.join(c['text'] for m in messages if isinstance(m['content'],list)
                     for c in m['content'] if 'text' in c)


def extract_state(text):
    return json.JSONDecoder().raw_decode(text.split('CURRENT NATIVE STATE:\n',1)[1])[0]


def differences(a,b,path=()):
    if type(a)!=type(b): return [{'path':list(path),'before':a,'after':b}]
    if isinstance(a,dict):
        out=[]
        for k in sorted(a.keys()|b.keys()):
            out.extend(differences(a.get(k),b.get(k),path+(k,)))
        return out
    if isinstance(a,list):
        if len(a)!=len(b):return [{'path':list(path),'before':a,'after':b}]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+(i,))]
    return [] if a==b else [{'path':list(path),'before':a,'after':b}]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--availability',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    hashes={}
    def source(path):
        data=subprocess.check_output(['git','show',PIN+':'+path]);hashes[path]=digest(data)
        return json.loads(data)
    corpus={u['id']:u for u in source(CORPUS)['rows']}
    material_path=Path('runs/autodl/20261110_material_routing/actual/prospective_components.jsonl')
    material={r['unit_id']:r for r in map(json.loads,material_path.read_text().splitlines())
              if r['arm']=='separate_tagged' and r['component']=='action'}
    availability=json.loads(args.availability.read_text())
    blobs={r['git_blob']:r['sha256'] for r in availability['scanned']}
    known_sha=defaultdict(list)
    for item in availability['scanned']:
        known_sha[item['sha256']].append({'path':item['path'],'git_blob':item['git_blob']})
    tree={}
    for line in subprocess.check_output(['git','ls-tree','-r',PIN,BASE],text=True).splitlines():
        meta,path=line.split('\t',1);tree[path]=meta.split()[2]
    plan=source(BASE+'/BATCH_PLAN.json');rows=[];groups=defaultdict(list)
    for cell in plan['cells']:
        ident=cell['condition'];name=cell['cell_id']
        reference=raw_messages(material[ident])
        request_path=BASE+'/cells/'+name+'/recorded/round_000/request.json'
        request=source(request_path)
        receipt=source(BASE+'/cells/'+name+'/recorded/round_000/model/RESULT.json')
        assert hashes[request_path]==receipt['request_sha256']
        before=text_content(reference);after=text_content(request)
        ca,cb=extract_state(before),extract_state(after)
        delta=differences(ca,cb)
        views=[]
        for i,(old,new) in enumerate(zip(corpus[ident]['images'],receipt['encoding']['images'])):
            path=BASE+'/cells/'+name+'/recorded/views/'+str(i)+'_'+Path(new['origin']).name
            published=blobs.get(tree.get(path))
            if published is not None:assert published==new['sha256'],path
            views.append({'index':i,'original_training_sha256':old['sha256'],
                'rollout_sha256':new['sha256'],'byte_equal':old['sha256']==new['sha256'],
                'expected_literal_archive_path':path,'literal_path_present':path in tree,
                'git_blob':tree.get(path),
                'independently_hashed_matching_sources':known_sha.get(new['sha256'],[]),
                'published_bytes_independently_verified':new['sha256'] in known_sha})
        assert len(views)==len(corpus[ident]['images'])==len(receipt['encoding']['images'])==5
        row={'unit_id':ident,'arm':cell['arm'],'request_path':request_path,
             'request_sha256':hashes[request_path],
             'system_text_equal':reference[0]==request[0],
             'user_text_equal':before==after,'training_user_chars':len(before),'rollout_user_chars':len(after),
             'context_changed_keys':[k for k in ca.keys()|cb.keys() if ca.get(k)!=cb.get(k)],
             'context_differences':delta,
             'original_task_equal':ca.get('original_request')==cb.get('original_request'),
             'activity_requirement_equal':ca.get('declared_activity_requirements')==cb.get('declared_activity_requirements'),
             'measurements_equal':ca.get('available_measurement')==cb.get('available_measurement'),
             'images':views,'rollout_input_tokens':receipt['encoding']['input_tokens'],
             'note':'Image hash difference does not establish perceptual difference, nor explain policy failure.'}
        rows.append(row)
        normalized=copy.deepcopy(request)
        for message in normalized:
            if isinstance(message['content'],list):
                image_i=0
                for part in message['content']:
                    if part.get('type')=='image':
                        part['image']=receipt['encoding']['images'][image_i]['sha256'];image_i+=1
        groups[ident].append((cell['arm'],digest(json.dumps(normalized,sort_keys=True).encode())))
    result={'pin':PIN,'rows':rows,'source_sha256':hashes,
        'local_source_sha256':{str(p):digest(p.read_bytes()) for p in (material_path,args.availability)},
        'conditions':len(groups),'first_requests':len(rows),
        'same_first_request_with_image_file_hashes_across_arms':all(len({h for a,h in g})==1 for g in groups.values()),
        'normalized_request_hashes':dict(groups),
        'identical_training_user_text_requests':sum(r['user_text_equal'] for r in rows),
        'identical_training_image_references':sum(v['byte_equal'] for r in rows for v in r['images']),
        'total_image_references':sum(len(r['images']) for r in rows),
        'published_image_references_verified':sum(v['published_bytes_independently_verified'] for r in rows for v in r['images']),
        'new_model_calls':0,'new_training_updates':0,'new_native_scene_calls':0,
        'conclusion':'This check compares text and image FILE bytes only. Different PNG hashes do not establish different decoded pixels or model tensors. Across-arm pixel equivalence requires the separate decode audit. Training text and original image identity are reported separately; no frozen dataset replacement is performed.'}
    write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256','local_source_sha256','normalized_request_hashes')}))


if __name__=='__main__':main()
