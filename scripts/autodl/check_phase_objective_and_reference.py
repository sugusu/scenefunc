"""Verify inherited target mass and probe pinned FRODO loss methods on CPU.

The reference probe supplies scalar NLLs, not a learned language model. It is a
numerical implementation check, not reproduction of any paper's experiments.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

from audit_material_routing import digest, token_hash, write


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference-source', type=Path, required=True)
    ap.add_argument('--reference-manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--material', type=Path, default=Path('runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl'))
    args=ap.parse_args()
    old=[json.loads(x) for x in Path('runs/autodl/20261110_material_routing/actual/prospective_components.jsonl').read_text().splitlines()]
    new=[json.loads(x) for x in args.material.read_text().splitlines()]
    old_lookup={(r['unit_id'],r['component']):r for r in old if r['arm']=='paired_tagged'}
    coverage=defaultdict(list)
    weights=defaultdict(list)
    import torch
    from torch import nn
    for row in new:
        assert len(row['target_token_ids'])==len(row['source_token_origins'])==len(row['token_objective_weights'])
        for tok, origin, weight in zip(row['target_token_ids'],row['source_token_origins'],row['token_objective_weights']):
            src=old_lookup[row['unit_id'],origin['component']]
            assert tok==src['target_token_ids'][origin['index']]
            assert weight==src['target_component_weight']
            coverage[row['arm']].append((row['unit_id'],origin['component'],origin['index']))
            weights[row['arm']].append(weight)
    original_positions={(unit,part,i) for (unit,part),src in old_lookup.items() for i in range(len(src['target_token_ids']))}
    assert len(original_positions)==2975
    for arm,positions in coverage.items():
        assert len(positions)==len(set(positions)) and set(positions)==original_positions
        assert abs(sum(weights[arm])-11/6)<1e-10
    # Same NLL assigned by source-token identity, evaluated with both partitions.
    nll={k:0.2+(i%101)/100 for i,k in enumerate(sorted(original_positions))}
    old_loss=sum(nll[key]*old_lookup[key[:2]]['target_component_weight'] for key in original_positions)
    revised_loss={arm:sum(nll[k]*w for k,w in zip(coverage[arm],weights[arm])) for arm in coverage}
    assert all(abs(v-old_loss)<1e-12 for v in revised_loss.values())

    source=args.reference_source.read_text()
    manifest=json.loads(args.reference_manifest.read_text())
    pinned=next(r for r in manifest if r['local_path']==str(args.reference_source))
    assert digest(source.encode())==pinned['sha256']
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReasoningModule')
    namespace={'torch':torch,'nn':nn,'Dict':Dict,'Optional':Optional,'FRODOConfig':SimpleNamespace}
    # Execute the exact class AST, omitting file-level GPU discovery/main code.
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(args.reference_source),'exec'),namespace)
    class NLLProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.positive=nn.Parameter(torch.tensor(.2,dtype=torch.float64))
            self.negative=nn.Parameter(torch.tensor(.8,dtype=torch.float64))
            self.calls=[]
        def forward(self,input_ids,attention_mask,labels,return_dict):
            self.calls.append({'input_ids':input_ids.tolist(),'labels':labels.tolist()})
            return SimpleNamespace(loss=self.positive if input_ids[0,-1].item()==2 else self.negative)
    probe=NLLProbe()
    config=SimpleNamespace(margin=1.,lambda_lm=1.,lambda_ie=1.,lambda_margin=1.)
    module=namespace['ReasoningModule'](probe,config)
    q=torch.tensor([[7]]);mask=torch.ones_like(q);correct=torch.tensor([[2]]);wrong=torch.tensor([[3]]);answer=torch.tensor([[11]])
    actual=module.compute_margin_ranking_loss(q,mask,correct,mask,wrong,mask,answer)
    grad=torch.autograd.grad(actual,(probe.positive,probe.negative))
    intended=torch.relu(1.+probe.positive-probe.negative)
    intended_grad=torch.autograd.grad(intended,(probe.positive,probe.negative))
    assert [float(g) for g in grad]==[-1.,1.]
    assert [float(g) for g in intended_grad]==[1.,-1.]
    probe.calls=[]
    values=module(q,mask,correct,mask,answer,wrong,mask)
    assert probe.calls[0]==probe.calls[1]
    # The first two branches both consume correct R and correct Y. They do not
    # implement the paper's counterfactual likelihood with (wrong R, wrong Y).
    result={'status':'completed','fixed_source_positions':len(original_positions),
            'material_path':str(args.material),'material_sha256':digest(args.material.read_bytes()),
            'tokens_preserved_once_per_arm':True,'coefficients_preserved_per_token':True,
            'inherited_full_pass_weight_mass':11/6,'old_scalar_probe_loss':old_loss,
            'revised_scalar_probe_losses':revised_loss,
            'reference':pinned,'reference_class_lines':[node.lineno,node.end_lineno],
            'reference_margin_probe':{'positive_nll':.2,'negative_nll':.8,
               'actual_loss':float(actual.detach()),'actual_gradients':[float(g) for g in grad],
               'expected_nll_margin_loss':float(intended.detach()),'expected_gradients':[float(g) for g in intended_grad]},
            'reference_forward_probe':{'calls':probe.calls,'language_and_indirect_inputs_identical':True,
                'loss_components':{k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in values.items()}},
            'scope':'Exact pinned current-source methods with controlled scalar losses. No claim that these files produced the published paper results.',
            'model_weights_loaded':False,'trained_parameter_updates':0,'cuda_initialized':torch.cuda.is_initialized()}
    write(args.output,result)
    print(json.dumps({k:result[k] for k in ('fixed_source_positions','coefficients_preserved_per_token','old_scalar_probe_loss','revised_scalar_probe_losses','reference_margin_probe')}))


if __name__=='__main__':main()
