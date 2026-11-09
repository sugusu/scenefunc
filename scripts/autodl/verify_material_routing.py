"""Integrity and objective-accounting check, not an efficacy experiment."""
import argparse
import ast
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import struct


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.extend(v for k,v in attrs if k=='href')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path('.'))
    ap.add_argument('--seal',action='store_true');a=ap.parse_args();repo=a.repo.resolve()
    root=repo/'runs/autodl/20261110_material_routing'
    paths=[repo/'scripts/autodl'/p for p in ['audit_material_routing.py','report_material_routing.py','verify_material_routing.py']]
    for p in paths:ast.parse(p.read_text())
    rows=[json.loads(s) for s in (root/'actual/prospective_components.jsonl').read_text().splitlines()]
    index={(r['arm'],r['unit_id'],r['component']):r for r in rows}
    if len(index)!=len(rows) or len(rows)!=64:raise ValueError('Unexpected component count')
    for r in rows:
        arm='paired_tagged' if r['arm']=='separate_tagged' else 'separate_tagged'
        paired=index[arm,r['unit_id'],r['component']]
        for key in ['messages','target_token_ids','target_component_weight','common_prompt_sha256']:
            if r[key]!=paired[key]:raise ValueError('Unintended treatment difference: '+key)
        ids=r['target_token_ids']
        if hashlib.sha256(struct.pack('<'+'q'*len(ids),*ids)).hexdigest()!=r['target_sha256']:
            raise ValueError('Target token content changed')
        if (r['component']=='decision' or r['arm']=='separate_tagged') and r['assistant_prefix']:
            raise ValueError('Unexpected reason prefix')
    for arm in ['separate_tagged','paired_tagged']:
        for component,expected in [('action',1.),('decision',5/6)]:
            total=sum(len(r['target_token_ids'])*r['target_component_weight'] for r in rows
                      if r['arm']==arm and r['component']==component)
            if not math.isclose(total,expected,rel_tol=1e-12):raise ValueError('Changed objective coefficient')
    old=json.loads((repo/'runs/autodl/20261110_objective_source_review/material_manifest.json').read_text())
    oldindex={r['id']:r for r in old['records'] if r['arm']=='separate'}
    for r in oldindex.values():
        if r['component']!='decision':continue
        other=oldindex[r['unit_id']+':action']
        if r['prompt_sha256']!=other['prompt_sha256']:raise ValueError('Prefix differs')
        for key in ['pixel_values','image_grid_thw']:
            if r['transformed_sha256'][key]!=other['transformed_sha256'][key]:
                raise ValueError('The two tasks have different visual input')
    source=json.loads((root/'actual/source_manifest.json').read_text())
    for p,digest in source['local_tokenizer_hashes'].items():
        if sha(Path(p))!=digest:raise ValueError('Historical tokenizer changed')
    report=root/'stage_report.html';links=Links();links.feed(report.read_text())
    local=[x for x in links.links if not x.startswith(('https://','http://','#'))]
    for link in local:
        if link=='verification_receipt.json':continue
        if not (root/link).exists():raise ValueError('Missing report link: '+link)
    manifest=root/'manifest.json'
    if a.seal:
        if manifest.exists():raise ValueError('Refuse to replace the manifest')
        files=[p for p in root.rglob('*') if p.is_file() and p.name not in ('manifest.json','verification_receipt.json')]+paths
        manifest.write_text(json.dumps({'files':{str(p.relative_to(repo)):sha(p) for p in sorted(files)}},indent=2)+'\n')
    files=json.loads(manifest.read_text())['files']
    for relative,digest in files.items():
        if sha(repo/relative)!=digest:raise ValueError('Delivery hash differs: '+relative)
    result={'delivery_files_verified':len(files),'prospective_components':64,
            'between_arm_material_and_observation_messages_equal':True,
            'original_same_prefix_visual_input_pairs_verified':8,
            'component_coefficient_sums':{'action':1.,'decision':5/6},
            'local_tokenizer_hashes_unchanged':True,'local_report_links':len(local),
            'new_model_calls':0,'training_updates':0,'training_ready':False,
            'scope':'Delivery and material accounting; no multimodal pixel processing or performance test.'}
    (root/'verification_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
