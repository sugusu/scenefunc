"""Check an objective implication, not a model or scene experiment.

The PINTO source's label-smoothed CE has a different confidence target from
ordinary correct-action CE. Both retain the correct class as argmax. The toy
calculation therefore does NOT establish worse task accuracy or reject PINTO.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-directory',type=Path,required=True)
    ap.add_argument('--source-manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads(args.source_manifest.read_text())
    for f in manifest['files']:
        assert hashlib.sha256((args.source_directory/f['path']).read_bytes()).hexdigest()==f['sha256']
    main_source=(args.source_directory/'main.py').read_text()
    tree=ast.parse(main_source)
    matched=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):continue
        if node.func.attr!='cross_entropy':continue
        if any(k.arg=='label_smoothing' and isinstance(k.value,ast.Attribute) and
               k.value.attr=='label_smoothing_no_inference' for k in node.keywords):
            matched.append({'line':node.lineno,'expression':ast.get_source_segment(main_source,node)})
    assert len(matched)==1
    shell=(args.source_directory/'run.sh').read_text()
    epsilon=float(re.search(r'^label_smoothing_no_inference=([0-9.]+)$',shell,re.M).group(1))
    import torch
    import torch.nn.functional as F
    rows=[]
    for choices in (2,4,5,8):
        p=torch.tensor([.99]+[.01/(choices-1)]*(choices-1),dtype=torch.float64)
        observations=[]
        for smooth in (0.,epsilon):
            z=p.log().requires_grad_()
            loss=F.cross_entropy(z[None,:],torch.tensor([0]),label_smoothing=smooth)
            loss.backward()
            q=torch.full_like(p,smooth/choices);q[0]+=1-smooth
            err=float((z.grad-(p-q)).abs().max());assert err<1e-14
            assert int(q.argmax())==0
            observations.append({'epsilon':smooth,'loss':float(loss.detach()),
                'correct_class_target':float(q[0]),'correct_logit_gradient':float(z.grad[0]),
                'analytic_gradient_max_error':err,'correct_class_still_argmax':True})
        rows.append({'choices':choices,'synthetic_predicted_correct_probability':.99,'objectives':observations})
    result={'source_commit':manifest['commit'],'matched_source_expression':matched,
        'epsilon_from_official_run_script':epsilon,'synthetic_loss_calculations':rows,
        'interpretation':'Label smoothing reduces overconfident predictions toward its softened target; it does not force the wrong class or establish worse accuracy. Draft dependence is not identical to correct environment action.',
        'three_dimensional_inference':'If only an untrusted draft changes, physical task truth does not. Evaluate actual requirement violations, not a reward for action flipping. New verified measurements must enter the observation explicitly.',
        'real_data_used_for_loss_example':False,'model_calls':0,'optimizer_updates':0,
        'new_method':False,'new_scene_results':0,'torch_version':torch.__version__}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'epsilon':epsilon,'classes':[r['choices'] for r in rows],
        'max_gradient_error':max(o['analytic_gradient_max_error'] for r in rows for o in r['objectives']),
        'model_calls':0,'optimizer_updates':0}))


if __name__=='__main__':main()
