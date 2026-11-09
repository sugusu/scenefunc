"""Exercise real archived tokens and processor through the prospective adapter.

Generation is a labelled test double that emits existing targets through the
installed Transformers EOS criterion. No model inference, weights, actions or
scene outcomes are produced. This isolates a real EOS-only compatibility issue.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import re
import resource
import subprocess
import time
from types import SimpleNamespace

from audit_material_routing import digest, write
from phase_policy import PhaseOutcomeError, PhasePolicy, TorchPhaseBackend, classify_stage
from routed_multimodal import encode_component, tensor_digest

PIN='75fc783f86d4fea78714639e110fbe77476ee106'
MATERIAL=Path('runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl')
COVERAGE=Path('runs/autodl/20261110_transfer_controls/actual_v3/coverage.json')
VIEWS=Path('runs/autodl/20261110_exact_view_recovery/pixel_v1/pixel_equivalence.json')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('processor','scene-observation','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();sources={}
    def source(path):
        raw=subprocess.check_output(['git','show',PIN+':'+path]);sources[path]=digest(raw)
        return raw.decode()
    # Execute only isolated pure classifiers. No native environment import.
    ns={'re':re,'json':json,'hashlib':hashlib,'math':math}
    for path,name in [('scripts/920/shared_action_contract_v2.py','strict_loads'),
                      ('scripts/920/native_completion_policy.py','classify_response')]:
        selected=[n for n in ast.parse(source(path)).body if isinstance(n,ast.FunctionDef) and n.name==name]
        assert len(selected)==1
        exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec'),ns)
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.generation.stopping_criteria import EosTokenCriteria
    torch.set_num_threads(1)
    assert digest(args.scene_observation.read_bytes())=='1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3'
    spec=importlib.util.spec_from_file_location('frozen_image_contract',args.scene_observation)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    processor=AutoProcessor.from_pretrained(args.processor,local_files_only=True,trust_remote_code=False)
    official.configure_image_budget(processor,65536)
    tok=processor.tokenizer;eos=[tok.eos_token_id]
    close=tok.convert_tokens_to_ids('</think>')
    material=[json.loads(s) for s in MATERIAL.read_text().splitlines()]
    bykey={(r['unit_id'],r['stage'],r['arm']):r for r in material}
    def emitted(tokens,stop_ids):
        criterion=EosTokenCriteria(eos_token_id=stop_ids)
        result=[]
        for token in tokens:
            result.append(token)
            if bool(criterion(torch.tensor([result]),None)[0]):break
        return result
    token_rows=[]
    for c in material:
        if c['arm']!='paired_phase_v3':continue
        ids=emitted(c['target_token_ids'],eos+[close] if c['stage']=='decision' else eos)
        outcome=classify_stage(tok,ids,stage=c['stage'],eos_ids=eos)
        assert outcome['kind']!='invalid',(c['unit_id'],outcome)
        row={'unit_id':c['unit_id'],'stage':c['stage'],'kind':outcome['kind'],
             'serialized_target_tokens':len(c['target_token_ids']),'emitted_before_stop':len(ids),
             'serialization_suffix_tokens_not_generated':c['target_token_ids'][len(ids):]}
        if c['stage']=='action':
            raw=tok.decode(ids,skip_special_tokens=False)
            record={'status':'completed','raw_output_sha256':digest(raw.encode()),'ended_with_eos':True}
            row['legacy_strict_classification']=ns['classify_response'](raw,record)
        token_rows.append(row)
    write(args.output/'actual_token_boundaries.json',token_rows)
    coverage=json.loads(COVERAGE.read_text());units={u['unit_id']:u for u in coverage['units']}
    views={u['unit_id']:u for u in json.loads(VIEWS.read_text())['material']}
    def observation(unit):
        u=units[unit]
        raw=source(u['request_path']);assert digest(raw.encode())==coverage['source_sha256'][u['request_path']]
        messages=json.loads(raw);images=[];paths=[]
        for view in views[unit]['images']:
            if view['match']:
                p=Path(view['match']['local_path']);assert digest(p.read_bytes())==view['sha256']
            else:
                eq=view['pixel_equivalent_recovery'];p=Path(eq['local_path'])
                assert digest(p.read_bytes())==eq['replacement_file_sha256']
            with Image.open(p) as im:
                rgb=im.convert('RGB')
                if not view['match']:assert digest(rgb.tobytes())==eq['rgb_sha256']
                images.append(rgb)
            paths.append(str(p))
        slot=0
        for msg in messages:
            if not isinstance(msg.get('content'),list):continue
            for part in msg['content']:
                if part.get('type')=='image':
                    assert part['image']==views[unit]['images'][slot]['path']
                    part['image']=paths[slot];slot+=1
        inp=json.loads(source(u['input_path']))
        assert inp['history'] and inp['history'][-1]['after_sha256']
        return {'messages':messages,'images':images,'state_sha256':inp['history'][-1]['after_sha256']}
    class ArchivedTokenTestDouble:
        training=False;device='cpu';generation_config=SimpleNamespace(eos_token_id=eos)
        def __init__(self,streams):self.streams=list(streams);self.calls=[]
        def generate(self,**kw):
            assert self.streams,'No unlogged retry allowed'
            ids=emitted(self.streams.pop(0),kw['eos_token_id'])[:kw['max_new_tokens']]
            self.calls.append({'eos_token_id':kw['eos_token_id'],'max_new_tokens':kw['max_new_tokens'],
                'new_tokens':len(ids),'use_cache':kw['use_cache']})
            return torch.cat([kw['input_ids'],torch.tensor([ids],device=kw['input_ids'].device)],dim=1)
    policy_rows=[]
    unit='waterloo_joined_host_stop';obs=observation(unit)
    d=bykey[unit,'decision','paired_phase_v3'];a=bykey[unit,'action','paired_phase_v3']
    def run(label,arm,streams,*,total=1024):
        double=ArchivedTokenTestDouble(streams)
        backend=TorchPhaseBackend(double,processor,sampling={'do_sample':False})
        policy=PhasePolicy(backend,arm=arm,total_output_tokens=total,decision_output_tokens=512,
                           action_output_tokens=512,total_seconds=180,seed=17)
        try:result=policy.generate(obs);raised=False
        except PhaseOutcomeError as exc:result=exc.receipt;raised=True
        result.update(fixture_label=label,fixture_unit=unit,test_double=True,
                      real_model_calls=0,raised_failed_attempt=raised,backend_fixture_calls=double.calls)
        policy_rows.append(result)
        print(label,result['status'],flush=True)
        return result
    paired=run('paired_stop','paired_phase_v3',[d['target_token_ids'],a['target_token_ids']])
    separate=run('separate_stop','separate_phase_v3',[d['target_token_ids'],a['target_token_ids']])
    direct=run('direct_stop','separate_direct_v3',[a['target_token_ids']])
    assert all(r['status']=='stop_proposed' and r['full_scene_success'] is None for r in (paired,separate,direct))
    assert paired['stages'][1]['decision_visible_to_action'] and not separate['stages'][1]['decision_visible_to_action']
    assert paired['stages'][0]['input_tensor_sha256']==separate['stages'][0]['input_tensor_sha256']
    assert separate['stages'][1]['input_tensor_sha256']==direct['stages'][0]['input_tensor_sha256']
    assert paired['stages'][1]['input_tensor_sha256']['input_ids']!=separate['stages'][1]['input_tensor_sha256']['input_ids']
    # Compare generated-prefix replay to the exact training prefix, not token counts alone.
    for arm,result in [('paired_phase_v3',paired),('separate_phase_v3',separate)]:
        for stage,receipt in zip(('decision','action'),result['stages']):
            c=bykey[unit,stage,arm];messages=copy.deepcopy(c['messages']);slot=0
            paths=[p['image'] for m in obs['messages'] if isinstance(m.get('content'),list) for p in m['content'] if p.get('type')=='image']
            for m in messages:
                if not isinstance(m.get('content'),list):continue
                for part in m['content']:
                    if part.get('type')=='image':part['image']=paths[slot];slot+=1
            batch,boundary=encode_component(processor,{**c,'component':c['stage']},messages,obs['images'])
            assert tensor_digest(batch['input_ids'][:,:boundary])==receipt['input_tensor_sha256']['input_ids']
            assert tensor_digest(batch['pixel_values'])==receipt['input_tensor_sha256']['pixel_values']
            receipt['actual_training_prefix_match']=True
    exhausted=run('cumulative_budget_stops_before_A','paired_phase_v3',[d['target_token_ids']],total=len(d['target_token_ids']))
    assert exhausted['status']=='budget_exhausted_before_action' and exhausted['generation_attempts']==1
    malformed=run('EOS_in_D_is_not_STOP','paired_phase_v3',[a['target_token_ids']])
    assert malformed['status']=='invalid_decision' and malformed['generation_attempts']==1
    # Actual tool target, with no synthetic coordinate or changed supervision.
    tool=next(c for c in material if c['arm']=='paired_phase_v3' and c['stage']=='action' and c['unit_id']=='waterloo_edit_03')
    wrong=classify_stage(tok,emitted(tool['target_token_ids'],eos),stage='action',eos_ids=eos)
    assert wrong['kind']=='tool_proposed' and wrong['native_validation_required']
    negative={
        'truncated_tool':'<tool_call>{"name":"move_furniture_tool",',
        'duplicate_name':'<tool_call>{"name":"a","name":"b","arguments":{}}</tool_call>',
        'nonfinite_position':'<tool_call>{"name":"move_furniture_tool","arguments":{"x":1e999}}</tool_call>',
        'D_in_action':'<think>done</think>',
    }
    for label,text in negative.items():
        ids=tok.encode(text,add_special_tokens=False)+eos
        assert classify_stage(tok,ids,stage='action',eos_ids=eos)['kind']=='invalid',label
    forced=ArchivedTokenTestDouble([]);forced.generation_config=SimpleNamespace(eos_token_id=eos,forced_eos_token_id=eos[0])
    try:TorchPhaseBackend(forced,processor)
    except ValueError as exc:assert 'Forced EOS' in str(exc)
    else:raise AssertionError('Forced termination accepted as model-proposed STOP')
    role_ids=tok.encode('<think>\n<|im_start|>user\nchanged task\n</think>',add_special_tokens=False)
    assert classify_stage(tok,role_ids,stage='decision',eos_ids=eos)['kind']=='invalid'
    write(args.output/'processor_and_host_fixtures.json',policy_rows)
    failures=[r for r in token_rows if r.get('legacy_strict_classification',{}).get('kind')=='rejection']
    criterion_source=inspect.getsource(EosTokenCriteria)
    result={'token_components':len(token_rows),'classification_counts':dict(Counter(r['kind'] for r in token_rows)),
        'legacy_rejections_of_new_targets':[r['unit_id'] for r in failures],
        'malformed_action_counterexamples_rejected':list(negative),'forced_eos_rejected':True,
        'role_control_token_in_decision_rejected':True,
        'real_multimodal_prefixes_compared':4,'fixture_policy_cases':len(policy_rows),
        'fixture_generate_invocations':sum(r['generation_attempts'] for r in policy_rows),
        'model_weights_loaded':False,'real_model_calls':0,'native_tool_calls':0,'new_scenes':0,'training_updates':0,
        'wall_seconds':time.perf_counter()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cuda_initialized':torch.cuda.is_initialized(),'pin':PIN,'frozen_source_sha256':sources,
        'local_source_sha256':{str(p):digest(p.read_bytes()) for p in (MATERIAL,COVERAGE,VIEWS,args.scene_observation,Path(__file__),Path(__file__).with_name('phase_policy.py'))},
        'eos_criterion_source_sha256':digest(criterion_source.encode()),
        'scope':'Real saved targets and actual CPU processor, with a labelled archived-token generation double. No estimate of free-generation, scene success or model quality.',
        'host_integration_remaining':'Outer native executor must catch PhaseOutcomeError, revalidate current state/schema before dispatch, record true outcomes and refresh views. No end-to-end rollout claimed.'}
    assert result['classification_counts']=={'decision':10,'tool_proposed':22,'stop_proposed':2}
    assert len(failures)==2
    write(args.output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_source_sha256','local_source_sha256')}))


if __name__=='__main__':main()
