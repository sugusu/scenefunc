"""Read-only audit of real native prepare/restore and condition-pair inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat
from shared_action_contract_v2 import context_from_request
from native_checkpoint import canonical_hash, rotation_roundtrip_differences
from native_project import sha256
from native_completion_supervision import build_native_request
from native_completion_supervision import native_request
from restored_edit_request import build_executed_request as build_restored_request
from run_knowledge_policy import write


def pixels(path):
    with Image.open(path) as image:
        rgb=image.convert('RGB')
        return dict(size=list(rgb.size),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),file_sha256=sha256(Path(path)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True)
    a=ap.parse_args();checks=[];rows=[];pairs=[];inputs={};requests={}
    def read(path):
        path=Path(path);inputs[str(path)]=sha256(path)
        return json.loads(path.read_text())
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    batch=read(a.run/'NATIVE_RESULT.json')
    check('repaired_native_stages_completed',batch['status']=='completed' and len(batch['stages'])==4)
    previous=read(a.previous/'NATIVE_RESULT.json')
    check('previous_failure_preserved',previous['status']=='failed' and len(previous['stages'])==5)
    check('four_reused_stages_passed',all(s['receipt']['exit_code']==0 for s in previous['stages'][:4]))
    plans=[a.previous/'native'/name/'PLAN.json' for name in ('waterloo_edit_01','waterloo_initial_work_stop')]
    plans+=sorted((a.run/'native').glob('*/PLAN.json'))
    for pp in plans:
        p=read(pp);name=p['cell_id'];root=Path(p['runtime_directory'])
        source=read(p['matched_input']['path']);expected=build_native_request(**source)[0]
        prepared=read(root/'prepare/RESULT.json');restored=read(root/'contract_check/RESULT.json')
        check(name+'_no_generation_or_edit',all(r['model_calls']==r['native_calls']==r['training_updates']==0 for r in [prepared,restored]))
        check(name+'_accepted',prepared['status']=='prepared_without_model' and restored['status']=='contract_checked_without_generation')
        for phase,filename,auditname in [('prepare','request.json','initial_audit.json'),('contract_check','restored_request.json','restored_audit.json')]:
            req=read(root/phase/filename);audit=read(root/phase/auditname)
            context=context_from_request(req)
            differences=rotation_roundtrip_differences(context_from_request(expected),context)
            check(name+'_'+phase+'_context_only_native_roundoff',all(d['path'][0]=='objects' for d in differences))
            check(name+'_'+phase+'_current_measurements',context['available_measurement']==audit['measurements'])
            check(name+'_'+phase+'_history_state_bound',(prepared if phase=='prepare' else restored)['latest_input_history']['current_native_state_sha256']==audit['state_sha256'])
            actual_images=[c['image'] for c in req[1]['content'] if c['type']=='image']
            matched=copy.deepcopy(source);matched['context']=context;matched['images']=actual_images
            metadata=(prepared if phase=='prepare' else restored)['latest_input_history']
            rebuilt,_=build_restored_request(**matched,restoration_transitions=metadata.get('restoration_transitions',[]),current_state_sha256=audit['state_sha256'])
            check(name+'_'+phase+'_input_exact_after_explicit_restoration',req==native_request(rebuilt))
            image_rows=[]
            for original,current in zip(source['images'],actual_images):
                before,after=pixels(original),pixels(current)
                with Image.open(original) as im1,Image.open(current) as im2:
                    delta=ImageStat.Stat(ImageChops.difference(im1.convert('RGB'),im2.convert('RGB')))
                    mean_delta=sum(delta.mean)/3
                image_rows.append(dict(original=original,current=current,original_hashes=before,current_hashes=after,
                    exact_pixels=before['rgb_sha256']==after['rgb_sha256'],mean_absolute_channel_difference_0_255=mean_delta))
            requests[(name,phase)]=req
            rows.append(dict(cell=name,phase=phase,state_sha256=audit['state_sha256'],requested_mode=audit['review']['requested_mode'],
                measured_status=audit['review']['measured_requirements_status'],images=image_rows,
                input_tokens=prepared['encoding']['input_tokens'] if phase=='prepare' else None,
                input_context_roundoff_differences=differences,
                restoration_differences=restored.get('restoration_differences') if phase=='contract_check' else None))
    for left,right in [('waterloo_edit_01','waterloo_initial_work_stop'),('waterloo_edit_03','waterloo_joined_host_stop')]:
        for phase in ('prepare','contract_check'):
            aa,bb=requests[(left,phase)],requests[(right,phase)]
            ca,cb=context_from_request(aa),context_from_request(bb)
            for field in ('current_activity_request','current_requested_mode'):ca.pop(field);cb.pop(field)
            check(left+'_'+phase+'_condition_only_context',ca==cb)
            ia=[c['image'] for c in aa[1]['content'] if c['type']=='image'];ib=[c['image'] for c in bb[1]['content'] if c['type']=='image']
            same=[pixels(x)['rgb_sha256']==pixels(y)['rgb_sha256'] for x,y in zip(ia,ib)]
            pairs.append(dict(left=left,right=right,phase=phase,same_five_view_pixels=same,
                condition_only_context=True,fully_matched_observation=all(same)))
    write(a.run/'NATIVE_AUDIT.json',dict(status='completed',checks=checks,rows=rows,pairs=pairs,
        input_hashes=inputs,model_calls=0,training_updates=0,scene_edits=0,
        exact_condition_pair_pixels=all(p['fully_matched_observation'] for p in pairs),
        limits='Real generation-free native restoration only. New render pixels may differ from archived training images; differences are reported, not silently treated as exact. No new model or professional result.'))
    print('Audited',len(rows),'real stages and',len(pairs),'condition/phase pairs.',flush=True)


if __name__=='__main__':main()
