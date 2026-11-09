"""Validate explicit pixel-equivalent recovery through the actual CPU processor.

Does not load model weights, change training data, or create optimizer updates.
Only units newly completed by an RGB-equivalence proof are processed.
"""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import time

from audit_material_routing import digest, write
from routed_multimodal import tensor_digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('equivalence','processor','scene-observation','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    tick=time.perf_counter()
    proof=json.loads(args.equivalence.read_text())
    oldpath=Path('runs/autodl/20261110_objective_source_review/material_manifest.json')
    old=json.loads(oldpath.read_text())
    expected={r['unit_id']:r['transformed_sha256'] for r in old['records']}
    selected=[u for u in proof['material'] if u['complete_five_views']
              and any(not im['match'] for im in u['images'])]
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
    for unit in selected:
        images=[];refs=[]
        for item in unit['images']:
            if item['match']:
                p=Path(item['match']['local_path']); assert digest(p.read_bytes())==item['sha256']
                provenance='original_file_bytes'
            else:
                eq=item['pixel_equivalent_recovery'];p=Path(eq['local_path'])
                assert digest(p.read_bytes())==eq['replacement_file_sha256']
                provenance='explicit_decoded_RGB_equivalence'
            with Image.open(p) as im:
                rgb=im.convert('RGB')
                if not item['match']:
                    assert list(rgb.size)==eq['size'] and digest(rgb.tobytes())==eq['rgb_sha256']
                images.append(rgb)
            refs.append(dict(original_file_sha256=item['sha256'],used_file_sha256=digest(p.read_bytes()),
                             provenance=provenance,path=str(p)))
        batch=processor.image_processor(images=images,return_tensors='pt')
        hashes={k:tensor_digest(batch[k]) for k in ('pixel_values','image_grid_thw')}
        row=dict(unit_id=unit['unit_id'],views=refs,actual=hashes,
                 original={k:expected[unit['unit_id']][k] for k in hashes},
                 exact_original_vision_tensors=all(h==expected[unit['unit_id']][k] for k,h in hashes.items()),
                 shapes={k:list(batch[k].shape) for k in hashes})
        rows.append(row)
        print(unit['unit_id'],'original vision tensors exact:',row['exact_original_vision_tensors'],flush=True)
    result=dict(rows=rows,units_processed=len(rows),
        units_original_vision_tensors_exact=sum(r['exact_original_vision_tensors'] for r in rows),
        source_sha256={str(p):digest(p.read_bytes()) for p in (args.equivalence,oldpath,args.scene_observation)},
        processor_files={str(p):digest(p.read_bytes()) for p in args.processor.iterdir() if p.suffix=='.json'},
        versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','transformers','Pillow')},
        wall_seconds=time.perf_counter()-tick,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_calls=0,training_updates=0,cuda_initialized=torch.cuda.is_initialized(),
        full_training_corpus_ready=False,
        limitation='Exact vision tensors only. Original PNG bytes were not recovered, textual training protocol and full data readiness remain separate checks.')
    write(args.output,result)


if __name__=='__main__':main()
