"""CPU regression against all 44 real cached episodes; no model weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoProcessor
    from edit_supervision import encode_single_episode
    torch.set_num_threads(2)
    plan=read(a.study/'ENCODING_PLAN.json');source=Path(plan['official_source']);sys.path.insert(0,str(source/'src'))
    spec=importlib.util.spec_from_file_location('prefix_check_official',source/'src/sft_dataset.py');up=importlib.util.module_from_spec(spec);sys.modules[spec.name]=up;spec.loader.exec_module(up)
    from scene_observation import configure_image_budget
    processor=AutoProcessor.from_pretrained(plan['model'],local_files_only=True,trust_remote_code=False);configure_image_budget(processor,65536)
    rows=[];data=read(a.study/'DATASET.json');cache=read(a.study/'ENCODING.json')
    for row in data['rows']:
        for arm,doc in row['documents'].items():
            assert sha(doc['path'])==doc['sha256']
            old=next(x for x in cache['records'] if x['id']==row['id'] and x['arm']==arm)
            assert sha(old['tensor_path'])==old['tensor_sha256']
            old_batch=torch.load(old['tensor_path'],map_location='cpu',weights_only=True)
            ex,am,dm,sample=encode_single_episode(doc['path'],a.study.resolve().parents[3],processor,up)
            _,prompt=up._build_messages(sample,no_thinking=not bool(read(doc['path'])['turns'][0].get('content','').strip()))
            text=processor.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
            # Render prompt with the same pixels using the official image path order.
            from PIL import Image
            images=[]
            for path in sample.scene_images:
                with Image.open(path) as im:images.append(im.convert('RGB'))
            enc=processor(text=[text],images=images,return_tensors='pt',padding=False)
            boundary=enc['input_ids'].shape[1]
            assert torch.equal(enc['input_ids'][0],ex['input_ids'][:boundary])
            assert torch.equal(old_batch['input_ids'][0,:boundary],ex['input_ids'][:boundary])
            assert int((ex['labels']!=-100).nonzero().flatten()[0])==boundary
            assert bool((ex['labels'][boundary:]!=-100).all())
            assert torch.equal(ex['labels'][am],old_batch['labels'][0][old_batch['action_mask'][0]])
            assert torch.equal(ex['labels'][dm],old_batch['labels'][0][old_batch['decision_mask'][0]])
            assert torch.equal(ex['pixel_values'],old_batch['pixel_values'])
            assert torch.equal(ex['image_grid_thw'],old_batch['image_grid_thw'])
            rows.append(dict(id=row['id'],arm=arm,removed_unsupervised_prefix_tokens=old_batch['input_ids'].shape[1]-len(ex['input_ids']),
                input_prompt_ids_exact=True,action_targets_exact=True,decision_targets_exact=True,five_view_pixels_exact=True,
                first_assistant_token_supervised=True,action_tokens=int(am.sum()),decision_tokens=int(dm.sum())))
    assert len(rows)==44 and not torch.cuda.is_initialized()
    result=dict(status='candidate_prefix_fix_verified_on_CPU',recorded_at=datetime.now(timezone.utc).isoformat(),
        examples=len(rows),changed_examples=sum(x['removed_unsupervised_prefix_tokens']>0 for x in rows),rows=rows,
        model_calls=0,training_updates=0,cuda_initialized=False,historical_tensors_modified=False,
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['edit_supervision.py','check_edit_supervision.py']},
        limitations=['Only assistant-prefix serialization repaired. Tool/state representations remain heterogeneous.',
            'No retraining or inference using this candidate. No design or fitting benefit established.'])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','examples','changed_examples','model_calls','training_updates']}),flush=True)


if __name__=='__main__':main()
