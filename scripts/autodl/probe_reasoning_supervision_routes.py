"""Execute pinned author preprocessing and audit existing scene-control contexts.

No policy weights, inference, optimization, new scene labels or native edits.
Author functions are AST-extracted with unchanged method bodies; imports and
registry decorators are excluded to avoid booting robot/model dependencies.
Synthetic fixtures test serialization only. They are not robot/scene results.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Tuple

import numpy as np
import torch

from audit_conditioning_attribution import optional_shared_support, mass_by_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(path, name, namespace, methods=None):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, 'name', None) == name)
    original = ast.dump(node, include_attributes=False)
    node = copy.deepcopy(node)
    node.decorator_list = []
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        if {n.name for n in node.body} != set(methods):
            raise ValueError('Requested author methods absent')
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name], dict(path=str(path), source_sha256=sha(path),
        symbol=name, methods=sorted(methods) if methods else 'all class methods',
        original_ast_sha256=hashlib.sha256(original.encode()).hexdigest(),
        executed_ast_sha256=hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
        changes='Removed top-level imports and class registry decorators only; selected methods unchanged')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'material', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    before = {str(a.material): sha(a.material)}
    for row in json.loads((a.sources/'source_receipts.json').read_text()):
        if sha(row['local_path']) != row['sha256']:
            raise ValueError('Pinned author source changed')
        before[row['local_path']] = row['sha256']
    extraction = []
    env = dict(np=np, os=os, json=json, random=random, re=re, tempfile=tempfile,
               deepcopy=copy.deepcopy, Tuple=Tuple)
    load_path = a.sources/'source/ERVLA/src/data/transforms/load.py'
    LoadData, rec = extract(load_path, 'LoadData', env)
    extraction.append(rec)
    # Pin debug off only in this subprocess: do not create author debug flags.
    os.environ['DEBUG_COT_SAMPLE_ONCE'] = '0'
    os.environ['_add_meta_to_prompt'] = 'False'
    fields = {'Understanding':['goal'], 'Planning':['plan','subtask_name','subtask_reasoning'],
              'Grounding':['visible_objects'], 'Acting':['movement_description','gripper_pose','waypoint_px']}
    fixture = {'frame_index':0, 'instruction':{'subtask':[{'conversations':[
        {'from':'human','value':'Fixture: move the block.'}, {'from':'gpt','value':''}]}]},
        'cot':{'Understanding':{'goal':'Fixture goal'},
               'Planning':{'plan':['reach','place'],'subtask_name':'reach','subtask_reasoning':'Fixture rationale'},
               'Grounding':{'visible_objects':{'block':[1,2,3,4]}},
               'Acting':{'movement_description':'move left','gripper_pose':[1,2],'waypoint_px':[[1,2],[2,2]]}}}
    raw_fixture = copy.deepcopy(fixture)
    outputs = []
    for enabled in (True, False):
        sample = copy.deepcopy(fixture)
        if not enabled:
            sample.pop('cot')
        for probability in (0., .5, 1.):
            loader = LoadData(add_cot_prob=probability, cot_fields=fields)
            for seed in range(4):
                np.random.seed(seed); random.seed(seed)
                prompt = loader.get_prompt(sample)
                answer = prompt['conversations'][1]['value']
                mode = 'no_cot' if answer == '<cot></cot>' else 'cot'
                if not enabled or probability == 0:
                    assert mode == 'no_cot'
                if enabled and probability == 1:
                    assert mode == 'cot'
                outputs.append(dict(has_annotation=enabled, probability=probability, seed=seed,
                                    mode=mode, prompt=prompt))
        assert sample == (raw_fixture if enabled else {k:v for k,v in raw_fixture.items() if k!='cot'})
    assert fixture == raw_fixture
    ecot_path = a.sources/'source/embodied-CoT/prismatic/vla/datasets/datasets.py'
    env['abbreviate_tag'] = lambda tag: tag[0]+tag[-2]
    fn, rec = extract(ecot_path, 'reasoning_dropout', env)
    extraction.append(rec)
    os.environ.pop('EXCLUDE_TAGS', None)
    ecot_rows = []
    text = 'TASK:@fixture task@PLAN:@fixture plan@MOVE:@move left'
    for probability in (0., .5, 1.):
        for seed in range(4):
            np.random.seed(seed)
            reason, subset = fn(text, probability)
            if probability == 1:
                assert not reason
            if probability == 0:
                assert all(tag in reason for tag in ('TASK:', 'PLAN:', 'MOVE:'))
            ecot_rows.append(dict(probability=probability, seed=seed, reason=reason, subset=subset))
    # Execute the published prefix/full-cache selector, not an attention model.
    model_path = a.sources/'source/ERVLA/src/models/VLA/P2_Choice_KI.py'
    tree = ast.parse(model_path.read_text())
    cls_name = next(n.name for n in tree.body if isinstance(n, ast.ClassDef)
                    and any(isinstance(m, ast.FunctionDef) and m.name=='get_action_vlm_condition_segments' for m in n.body))
    # Its class derives from the author's architecture: supply only a harmless
    # base class because we execute this single dictionary-selection method.
    cls_node = next(n for n in tree.body if getattr(n,'name',None)==cls_name)
    for base in cls_node.bases:
        if not isinstance(base,ast.Name):
            raise ValueError('Unexpected model superclass')
        env[base.id] = object
    cls, rec = extract(model_path, cls_name, env, {'get_action_vlm_condition_segments'})
    extraction.append(rec)
    cache_routes = []
    for flag in (False, True):
        obj = object.__new__(cls); obj.knowledge_insulation = flag
        batch = {'action_segments':[[0,120]], 'action_vlm_condition_segments':[[0,80]]}
        result = obj.get_action_vlm_condition_segments(batch)
        assert result == ([[0,80]] if flag else [[0,120]])
        cache_routes.append(dict(flag=flag, selected=result, remaining_keys=sorted(batch)))
    rows = [json.loads(line) for line in a.material.read_text().splitlines()]
    idx = {(r['unit_id'],r['arm'],r['stage']):r for r in rows}
    paired = [r for r in rows if r['arm']=='paired_phase_v3']
    separate = [r for r in rows if r['arm']=='separate_phase_v3']
    for p in paired:
        s = idx[p['unit_id'],'separate_phase_v3',p['stage']]
        for field in ('messages','target_token_ids','token_objective_weights','source_token_origins'):
            assert p[field] == s[field]
        if p['stage']=='decision':
            assert p['assistant_prefix']==s['assistant_prefix']==''
        if p['stage']=='action':
            assert s['assistant_prefix']==''
    # Exact expectation/gradient identity for this repository's existing
    # ordinary mixture. Synthetic scalar logits use actual source coefficients;
    # this is NOT a measured neural loss or a scene result.
    def artificial_loss(components, parameter):
        total = parameter*0
        for c in components:
            signature=json.dumps([c['messages'],c['assistant_prefix'],c['target_token_ids']],sort_keys=True)
            digest=hashlib.sha256(signature.encode()).digest()
            feature=(int.from_bytes(digest[:2],'big')/65535)*2-1
            offset=int.from_bytes(digest[2:4],'big')/65535
            total=total+sum(c['token_objective_weights'])*torch.nn.functional.softplus(parameter*feature+offset)
        return total
    equations=[]
    for q in (.25,.5,.75):
        mixture=optional_shared_support(rows,q)
        theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        lhs=artificial_loss(mixture,theta)
        rhs=q*artificial_loss(paired,theta)+(1-q)*artificial_loss(separate,theta)
        g1=torch.autograd.grad(lhs,theta,retain_graph=True)[0]
        g2=torch.autograd.grad(rhs,theta)[0]
        old_mass=mass_by_source(paired);new_mass=mass_by_source(mixture)
        mass_error=max(abs(old_mass[k]-new_mass[k]) for k in old_mass)
        assert old_mass.keys()==new_mass.keys() and mass_error<1e-14
        assert abs(float((lhs-rhs).detach()))<1e-12 and abs(float(g1-g2))<1e-12
        equations.append(dict(conditioned_fraction=q, mixed_components=len(mixture),
            value_error=float((lhs-rhs).detach()), gradient_error=float(g1-g2), source_mass_error=mass_error))
    result=dict(status='completed',author_preprocessing_cases=len(outputs)+len(ecot_rows),
        author_cache_selector_cases=len(cache_routes),
        ervla=outputs,ecot=ecot_rows,cache_selector=cache_routes,extracted_symbols=extraction,
        current_material=dict(units=len({r['unit_id'] for r in paired}),
            decisions=sum(r['stage']=='decision' for r in paired),
            no_decision_actions=sum(r['stage']=='action' and not r['assistant_prefix'] for r in paired),
            original_target_tokens=sum(len(r['target_token_ids']) for r in paired)),
        mixture_identity=equations,
        distinction='Existing mixture retains D supervision on every labelled unit and changes A conditioning only. Author routes can remove reasoning targets and alter prompts; ERVLA also uses continuous action heads. Not an ERVLA reproduction.',
        source_sha256=before,sources_unchanged=all(sha(p)==h for p,h in before.items()),
        seconds=time.monotonic()-tick,model_calls=0,training_updates=0,scene_edits=0,
        scope='Actual author preprocessing on synthetic fixtures; actual source-component algebra. No policy, task success, causal effect or whole-room evaluation.')
    assert result['sources_unchanged']
    (a.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','author_preprocessing_cases','author_cache_selector_cases','current_material','mixture_identity','seconds')}))


if __name__=='__main__':main()
