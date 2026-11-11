"""Acquire two prospectively selected official rows and build native prompts.

Only public metadata and the six specified images, not a model or full dataset.
Reference rankings never enter the observation or request.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request
from cpu_continuation import read, dump, sha

PIN='e50788bdbe74644d98e62929cd24218c9110e812'
BASE=f'https://huggingface.co/datasets/XuanyuLiu/Scene-OTC-Bench/resolve/{PIN}/'
W=Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
SOURCE=W.parent/'systematic_scene_literature_20261029/scene_designer_context_review_20261102/code/src_critic/grpo_advantage.py'


def fetch(url,path,expected,limit):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert sha(path)==expected
        return dict(path=str(path),url=url,sha256=expected,already_present=True)
    # Honor the user's existing SSH-forwarded proxy. Direct HF DNS is not a
    # usable route on this host; no token is required for these public files.
    op=urllib.request.build_opener(urllib.request.ProxyHandler())
    with op.open(url,timeout=30) as f:data=f.read(limit+1)
    assert len(data)<=limit,'Bounded download exceeded'
    assert hashlib.sha256(data).hexdigest()==expected,(url,'hash mismatch')
    path.write_bytes(data)
    return dict(path=str(path),url=url,sha256=expected,bytes=len(data))


def prompt_function():
    nodes=[]
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['_LABELS','DEFAULT_USER_TEXT'] for t in node.targets):nodes.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=='build_default_user_text':nodes.append(node)
    scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),scope)
    return scope['build_default_user_text']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--images',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    assert not list(root.glob('B*/turn_*/model_process_receipt.json')),'Inputs must be frozen before model calls'
    refs=[('scene_otc_bench.json','8afbe7f0213849466724f5d7398222c1f9b68ebe48ceb2c66e9201fe6a15e0c1'),
          ('SHA256SUMS','fe5f2e1431c7334ce2f8b5106f74ed94653060e15c49165db98759b5e612b37a')]
    receipts=[]
    for name,digest in refs:receipts.append(fetch(BASE+name,root/'inputs'/name,digest,512000))
    sums={line.split(maxsplit=1)[1].lstrip('*./'):line.split()[0] for line in (root/'inputs/SHA256SUMS').read_text().splitlines() if line.strip()}
    rows=read(root/'inputs/scene_otc_bench.json');selected=[next(r for r in rows if r['numimages']==n) for n in [2,4]]
    dump(root/'selection_before_inference.json',dict(rows=selected,source_order_preserved=True,do_not_feed_reference_ranks_to_model=True))
    get_prompt=prompt_function();cases=[]
    for row in selected:
        content=[]
        for image in row['images']:
            path=a.images.resolve()/image
            receipts.append(fetch(BASE+image,path,sums[image],8*1024**2))
            content.append(dict(type='image',image=str(path)))
        prompt=get_prompt(row['numimages']);content.append(dict(type='text',text=prompt))
        folder=root/f'B_native_n{row["numimages"]}'/'turn_00';folder.mkdir(parents=True,exist_ok=True)
        request=[dict(role='user',content=content)]
        dump(folder/'request.json',request)
        cases.append(dict(arm=folder.parent.name,id=row['id'],numimages=row['numimages'],
                          request_sha256=sha(folder/'request.json'),reference=row['ranking_best_to_worst'],
                          source_builder_sha256=sha(SOURCE),no_system=True))
    dump(root/'labels_quarantine.json',cases)
    dump(root/'acquisition.json',dict(dataset_pin=PIN,files=receipts,policy='No full dataset/model download or API call'))
    print(json.dumps(cases),flush=True)


if __name__=='__main__':main()
