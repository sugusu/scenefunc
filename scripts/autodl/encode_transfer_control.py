"""Encode the optional conditioning diagnostic on actual available five views.

Missing units stay in the denominator. Explicit RGB recovery is accepted only
with file/RGB proof, followed by exact original vision-tensor verification.
No model weights, optimizer, GPU execution or generated actions.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import resource
import time

from audit_material_routing import digest, token_hash, write
from routed_multimodal import encode_component, tensor_digest
from phase_objective import attach_objective_weights


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('material','equivalence','processor','scene-observation','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    tick=time.perf_counter()
    material=[json.loads(s) for s in args.material.read_text().splitlines()]
    evidence=json.loads(args.equivalence.read_text());views={u['unit_id']:u for u in evidence['material']}
    sourcepath=Path('runs/autodl/20261110_objective_source_review/material_manifest.json')
    old=json.loads(sourcepath.read_text())
    old={r['unit_id']:r['transformed_sha256'] for r in old['records']}
    assert digest(args.scene_observation.read_bytes())=='1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3'
    spec=importlib.util.spec_from_file_location('official_image_contract',args.scene_observation)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(args.processor,local_files_only=True,trust_remote_code=False)
    official.configure_image_budget(processor,65536)
    rows=[]
    for c in material:
        row={k:c[k] for k in ('unit_id','arm','stage','variant','target_sha256','original_weight_fraction')}
        unit=views[c['unit_id']]
        if not unit['complete_five_views']:
            row.update(status='missing_original_or_proven_equivalent_views');rows.append(row);continue
        paths=[];images=[];kinds=[]
        for view in unit['images']:
            if view['match']:
                p=Path(view['match']['local_path']);assert digest(p.read_bytes())==view['sha256']
                kind='original_bytes'
            else:
                eq=view['pixel_equivalent_recovery'];p=Path(eq['local_path'])
                assert digest(p.read_bytes())==eq['replacement_file_sha256'];kind='explicit_rgb_equivalence'
            with Image.open(p) as im:
                rgb=im.convert('RGB')
                if kind=='explicit_rgb_equivalence':
                    assert list(rgb.size)==eq['size'] and digest(rgb.tobytes())==eq['rgb_sha256']
                images.append(rgb)
            paths.append(str(p));kinds.append(kind)
        messages=copy.deepcopy(c['messages']);slot=0
        for m in messages:
            if not isinstance(m.get('content'),list):continue
            for part in m['content']:
                if part.get('type')!='image':continue
                assert part['image']==unit['images'][slot]['path']
                part['image']=paths[slot];slot+=1
        assert slot==5
        batch,boundary=encode_component(processor,c,messages,images)
        batch=attach_objective_weights(batch,c)
        hashes={k:tensor_digest(v) for k,v in batch.items()}
        assert all(hashes[k]==old[c['unit_id']][k] for k in ('pixel_values','image_grid_thw'))
        assert token_hash(batch['labels'][batch['labels']!=-100].tolist())==c['target_sha256']
        assert bool((batch['labels'][:,:boundary]==-100).all())
        row.update(status='encoded',sequence_tokens=batch['input_ids'].shape[1],
            boundary=boundary,target_tokens=int((batch['labels']!=-100).sum()),
            objective_weight_mass=float(batch['token_objective_weights'].sum()),
            original_vision_tensors_exact=True,all_context_labels_masked=True,
            view_provenance=kinds,tensor_sha256=hashes)
        rows.append(row)
        print(c['unit_id'],c['stage'],c['variant'],'encoded',flush=True)
    out={'rows':rows,'components_total':len(rows),'components_encoded':sum(r['status']=='encoded' for r in rows),
        'units_total':len(views),'units_encoded':len({r['unit_id'] for r in rows if r['status']=='encoded'}),
        'source_sha256':{str(p):digest(p.read_bytes()) for p in (args.material,args.equivalence,args.scene_observation,sourcepath)},
        'wall_seconds':time.perf_counter()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'model_calls':0,'training_updates':0,'cuda_initialized':torch.cuda.is_initialized(),
        'training_ready':False,'scope':'Actual processor, target and mask checks only. No trained diagnostic or scene improvement. Partial set is not substituted for the original corpus.'}
    write(args.output,out)
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows','source_sha256')}))


if __name__=='__main__':main()
