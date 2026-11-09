"""CPU encoding of prospective shared-host material, retaining all missing rows."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

from phase_deployment import deployment_input
from phase_objective import PHASE_MODE, attach_objective_weights
from routed_multimodal import encode_component, tensor_digest


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    for arg in ['material','equivalence','processor','scene-observation','out']:
        ap.add_argument('--'+arg,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(a.processor,local_files_only=True,trust_remote_code=False)
    assert sha(a.scene_observation)=='1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3'
    spec=importlib.util.spec_from_file_location('official_observation',a.scene_observation)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    native.configure_image_budget(processor,65536)
    components=[json.loads(x) for x in a.material.read_text().splitlines()]
    views={u['unit_id']:u for u in json.loads(a.equivalence.read_text())['material']}
    old={r['unit_id']:r['transformed_sha256'] for r in json.loads(Path('runs/autodl/20261110_objective_source_review/material_manifest.json').read_text())['records']}
    records=[]
    for c in components:
        row={k:c[k] for k in ['unit_id','arm','stage','target_sha256']}
        unit=views[c['unit_id']]
        if not unit['complete_five_views']:
            records.append(dict(row,status='missing_original_or_proven_equivalent_views'));continue
        images=[];paths=[]
        for view in unit['images']:
            if view['match']:
                p=Path(view['match']['local_path']);assert sha(p)==view['sha256']
            else:
                eq=view['pixel_equivalent_recovery'];p=Path(eq['local_path'])
                assert sha(p)==eq['replacement_file_sha256']
            with Image.open(p) as im:rgb=im.convert('RGB')
            if not view['match']:
                assert list(rgb.size)==eq['size'] and hashlib.sha256(rgb.tobytes()).hexdigest()==eq['rgb_sha256']
            images.append(rgb);paths.append(str(p))
        messages=copy.deepcopy(c['messages']);slot=0
        for m in messages:
            if not isinstance(m.get('content'),list):continue
            for part in m['content']:
                if part.get('type')=='image':
                    assert part['image']==unit['images'][slot]['path']
                    part['image']=paths[slot];slot+=1
        assert slot==5
        cc=dict(c,component=c['stage'])
        try:
            batch,boundary=encode_component(processor,cc,messages,images)
            batch=attach_objective_weights(batch,cc)
        except ValueError as exc:
            records.append(dict(row,status='encoding_rejected',reason=str(exc)));continue
        hashes={k:tensor_digest(v) for k,v in batch.items()}
        assert all(hashes[k]==old[c['unit_id']][k] for k in ['pixel_values','image_grid_thw'])
        raw=copy.deepcopy(messages);suffix=PHASE_MODE[c['stage']]
        assert raw[0]['content'].endswith(suffix)
        raw[0]['content']=raw[0]['content'][:-len(suffix)]
        if c['arm']=='paired_phase_v3' and c['stage']=='action' and not c['assistant_prefix']:
            prefix_status='not_asserted: training has no D, paired live deployment always generates D'
        else:
            prefix=deployment_input(processor,raw,images,arm=c['arm'],stage=c['stage'],
                generated_decision=c['assistant_prefix'] or None)
            assert torch.equal(prefix['input_ids'],batch['input_ids'][:,:boundary])
            assert all(torch.equal(prefix[k],batch[k]) for k in ['pixel_values','image_grid_thw'])
            prefix_status='exact_with_recorded_D_only_as_encoding_fixture'
        row.update(status='encoded',input_tokens=boundary,sequence_tokens=batch['input_ids'].shape[1],
            tensor_sha256=hashes,all_original_vision_tensors_equal=True,
            target_token_count=int((batch['labels']!=-100).sum()),
            inherited_token_mass=float(batch['token_objective_weights'].sum()),
            deployment_prefix_check=prefix_status)
        records.append(row)
        print(c['unit_id'],c['arm'],c['stage'],row['sequence_tokens'],flush=True)
    summary=dict(components_total=len(records),components_encoded=sum(r['status']=='encoded' for r in records),
        missing_views=sum(r['status']=='missing_original_or_proven_equivalent_views' for r in records),
        encoding_rejections=sum(r['status']=='encoding_rejected' for r in records),
        complete_units=len({r['unit_id'] for r in records if r['status']=='encoded'}),
        max_sequence_tokens=max((r.get('sequence_tokens',0) for r in records),default=0),
        rows=records,wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(p):sha(p) for p in [a.material,a.equivalence,a.scene_observation]},
        training_ready=False,all_missing_units_retained=True,model_calls=0,training_updates=0,
        cuda_initialized=torch.cuda.is_initialized(),
        scope='Real multimodal input and label encoding only. Teacher D is a boundary fixture, never a model output. Shared envelope does not establish matching tasks/base prompts.')
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','source_sha256']}))


if __name__=='__main__':main()
