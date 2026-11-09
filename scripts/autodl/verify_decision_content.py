"""Original-view encoding and source checks for oracle field ablations only."""
import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import time

from audit_material_routing import digest,write
from audit_decision_content import rank_knowledge,read_explicit_plan
from routed_multimodal import resolve_messages,encode_component,tensor_digest


def main():
    ap=argparse.ArgumentParser()
    for name in ('audit','variants','availability','processor','scene-observation','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    tick=time.perf_counter();audit=json.loads(args.audit.read_text())
    for path,sha in audit['source_sha256'].items():
        b=subprocess.check_output(['git','show',path]) if ':' in path and path.split(':',1)[0]==audit['source_pin'] else Path(path).read_bytes()
        assert digest(b)==sha,path
    coverage=json.loads(Path('runs/autodl/20261110_conditional_evidence/actual/condition_coverage.json').read_text())
    source_rows={r['unit_id']:r for r in coverage['rows']}
    leaks=[]
    for row in audit['rows']:
        c=json.loads(subprocess.check_output(['git','show',audit['source_pin']+':'+source_rows[row['unit_id']]['input_path']]))['context']
        q,ranks=rank_knowledge(c)
        assert q==row['query'] and ranks==row['bm25']
        modified=copy.deepcopy(c)
        modified.update(gold_decision='WRONG',target_action={'name':'WRONG'},evaluation_label='WRONG')
        assert rank_knowledge(modified)==(q,ranks)
        leaks.append(row['unit_id'])
    # Cover both object-table schemas using a real saved object, without changing
    # any scientific sample or counting this as a new generation task.
    c={'objects':{'item_1':{'object_id':'item_1'}}}
    plan={'description':'移动现有对象','object_or_asset_references':'item_1'}
    expected=read_explicit_plan(plan,c)
    assert read_explicit_plan(plan,{'objects':list(c['objects'].values())})==expected
    assert read_explicit_plan(dict(plan,object_or_asset_references='absent'),c) is None
    assert digest(args.scene_observation.read_bytes())=='1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3'
    spec=importlib.util.spec_from_file_location('fixed_visual_contract',args.scene_observation)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(args.processor,local_files_only=True,trust_remote_code=False)
    official.configure_image_budget(processor,65536)
    views={r['unit_id']:r for r in json.loads(args.availability.read_text())['rows']}
    references=json.loads(Path('runs/autodl/20261110_objective_source_review/material_manifest.json').read_text())['records']
    old={r['unit_id']:r for r in references if r['arm']=='separate' and r['component']=='action'}
    variants=[json.loads(l) for l in args.variants.read_text().splitlines()]
    rows=[]
    for v in variants:
        row={'unit_id':v['unit_id'],'variant':v['variant']}
        assert v['evaluation_only_oracle_prefix'] and not v['training_admission'] and v['not_policy_output']
        p=Path(v['request_path']);assert digest(p.read_bytes())==v['request_sha256']
        view=views[v['unit_id']]
        if not view['complete_five_views']:
            row['status']='missing_original_views';rows.append(row);continue
        messages,paths=resolve_messages(json.loads(p.read_text())['messages'],view['images'])
        images=[]
        for path in paths:
            with Image.open(path) as image:images.append(image.convert('RGB'))
        batch,boundary=encode_component(processor,v,messages,images)
        for k in ('pixel_values','image_grid_thw'):
            assert tensor_digest(batch[k])==old[v['unit_id']]['transformed_sha256'][k]
        row.update(status='encoded',sequence_tokens=batch['input_ids'].shape[1],boundary=boundary,
                   target_tokens=int((batch['labels']!=-100).sum()),prefix_tokens=v['prefix_tokens'])
        rows.append(row)
    counts=dict(Counter(r['status'] for r in rows))
    assert counts=={'encoded':15,'missing_original_views':25}
    result={'status':'partial_oracle_diagnostic_ready','counts':counts,'rows':rows,
            'unread_label_injection_invariance':leaks,'source_hashes_rechecked':len(audit['source_sha256']),
            'original_targets_and_images_unchanged':True,'full_material_available':False,
            'input_costs_equal':False,'model_outputs_generated':False,
            'scope':'Teacher-field ablations; not a learning result or deployment policy. Do not count missing images as failures/successes or silently drop them.',
            'model_calls':0,'training_updates':0,'scene_edits':0,
            'wall_seconds':time.perf_counter()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cuda_initialized':torch.cuda.is_initialized(),
            'source_sha256':{str(p):digest(p.read_bytes()) for p in (Path(__file__),args.audit,args.variants,args.availability,args.scene_observation)}}
    write(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','counts','wall_seconds','peak_rss_kib')}))


if __name__=='__main__':main()
