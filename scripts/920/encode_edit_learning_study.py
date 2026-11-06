"""Encode real official OTA examples and freeze separate action/decision masks on CPU."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();plan=read(a.plan)
    assert not a.out.exists();a.out.mkdir(parents=True)
    r=Path(__file__).resolve().parents[2];source=Path(plan['official_source']);dataset=Path(plan['dataset']['path'])
    assert sha(dataset)==plan['dataset']['sha256']
    for n,h in plan['code_sha256'].items():assert sha(r/'scripts/920'/n)==h
    for n,h in plan['official_source_hashes'].items():assert sha(source/n)==h
    for n,h in plan['processor_hashes'].items():assert sha(Path(plan['model'])/n)==h
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    import torch
    from transformers import AutoProcessor
    from decision_pairing import mask_role
    torch.set_num_threads(4);sys.path.insert(0,str(source/'src'))
    spec=importlib.util.spec_from_file_location('edit_study_official_dataset',source/'src/sft_dataset.py');up=importlib.util.module_from_spec(spec);sys.modules[spec.name]=up;spec.loader.exec_module(up)
    from scene_observation import configure_image_budget
    processor=AutoProcessor.from_pretrained(plan['model'],local_files_only=True,trust_remote_code=False);configure_image_budget(processor,plan['budget']['max_pixels'])
    data=read(dataset);records=[];padding_checks=0;dig=lambda t:hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
    print(datetime.now(timezone.utc).isoformat(),'begin44officialCPUexamples; no model weights',flush=True)
    for row in data['rows']:
        pair={};prompts={};pixels={};actions={}
        for im in row['images']:assert sha(im['path'])==im['sha256']
        for arm,ref in row['documents'].items():
            f=Path(ref['path']);assert sha(f)==ref['sha256'];doc=read(f)
            ds=up.StepwiseSFTDataset(f.parent,r.parent,processor,max_length=plan['budget']['max_sequence_tokens']);assert len(ds)==1;raw=ds[0]
            action=mask_role(raw,processor.tokenizer,'action');is_action=action['labels']!=-100;is_decision=(raw['labels']!=-100)&~is_action
            if arm=='action_only' or row['decision_target'] is None:is_decision[:]=False
            ex={k:v.clone() for k,v in raw.items()};ex['labels'][~(is_action|is_decision)]=-100
            assert len(ex['image_grid_thw'])==5
            positions=(raw['labels']!=-100).nonzero().flatten();prompt_end=int(positions[0]);prompts[arm]=raw['input_ids'][:prompt_end]
            assert not bool(is_action[:prompt_end].any()) and not bool(is_decision[:prompt_end].any())
            actions[arm]=ex['labels'][is_action];pixels[arm]=dig(ex['pixel_values']);pair[arm]=ex
            text=processor.tokenizer.decode(ex['input_ids'],skip_special_tokens=False)
            assert doc['task_context']['scene_description'] in text
            sentinel=doc.get('evaluation_only',{}).get('sentinel');assert not sentinel or sentinel not in text
            assert 'EVAL_ONLY_EDIT_STUDY' not in text
            if arm=='action_only' and row['decision_target'] is not None:
                assert json.dumps(row['decision_target'],ensure_ascii=False) not in text
            batch=up.StepwiseCollator(processor.tokenizer.pad_token_id)([ex]);batch['action_mask']=is_action.unsqueeze(0);batch['decision_mask']=is_decision.unsqueeze(0)
            target=a.out/'tensors'/arm/(row['id']+'.pt');target.parent.mkdir(parents=True,exist_ok=True);torch.save(batch,target)
            records.append(dict(id=row['id'],arm=arm,family_id=row['family_id'],tensor_path=str(target),tensor_sha256=sha(target),input_tokens=len(ex['input_ids']),action_tokens=int(is_action.sum()),decision_tokens=int(is_decision.sum()),input_sha256=dig(ex['input_ids']),action_target_sha256=dig(actions[arm]),pixel_sha256=pixels[arm],decision_text=processor.tokenizer.decode(ex['labels'][is_decision],skip_special_tokens=False),action_text=processor.tokenizer.decode(actions[arm],skip_special_tokens=False)))
        assert torch.equal(prompts['action_only'],prompts['decision_action'])
        assert torch.equal(actions['action_only'],actions['decision_action']) and pixels['action_only']==pixels['decision_action']
        batch=up.StepwiseCollator(processor.tokenizer.pad_token_id)(list(pair.values()));assert bool((batch['labels'][batch['attention_mask']==0]==-100).all());padding_checks+=1
        print(row['id'],'encoded',flush=True)
    assert len(records)==44 and not torch.cuda.is_initialized()
    totals={}
    for arm in ['action_only','decision_action']:
        rows=[x for x in records if x['arm']==arm];families=defaultdict(lambda:dict(action_tokens=0,decision_tokens=0,examples=0))
        for x in rows:
            for k in ['action_tokens','decision_tokens']:families[x['family_id']][k]+=x[k]
            families[x['family_id']]['examples']+=1
        totals[arm]=dict(examples=len(rows),images=len(rows)*5,input_tokens=sum(x['input_tokens'] for x in rows),action_tokens=sum(x['action_tokens'] for x in rows),decision_tokens=sum(x['decision_tokens'] for x in rows),maximum_sequence=max(x['input_tokens'] for x in rows),families=dict(families))
    result=dict(status='official_edit_corpus_encoded_on_CPU',recorded_at=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(a.plan),dataset_sha256=sha(dataset),records=records,totals=totals,checks=dict(identical_preassistant_prompts=True,identical_action_targets=True,identical_five_view_pixels=True,prompt_tool_receipts_padding_masked=True,no_gold_decision_in_action_only=True,official_serialization_preserved=True,padding_pairs=padding_checks),loss_contract='For each update average action-token CE within each source family and then across6families. decision_action adds lambda=1 times separately token-normalized decision CE per family. Same action exposure and action coefficients; added decision targets/compute explicitly differ.',model_loaded=False,cuda_initialized=False,optimizer_updates=0,training_authorized=False)
    write(a.out/'ENCODING.json',result);write(dataset.parent/'ENCODING.json',result)
    s=read(dataset.parent/'STATUS.json');s.update(status=result['status'],updated_at=result['recorded_at'],latest_verifiable_artifact=str(dataset.parent/'ENCODING.json'));s['completed_planned']['official_encoding']=[1,1];write(dataset.parent/'STATUS.json',s)
    write(r/'runs/920/latest.json',dict(run_id=dataset.parent.name,stage=s['status'],updated_at=s['updated_at'],run_directory=str(a.out.parent)))
    print(json.dumps(totals),flush=True)

if __name__=='__main__':main()
