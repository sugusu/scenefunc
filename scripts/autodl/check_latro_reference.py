"""Inspect pinned public LaTRO settings and execute only its scalar loss AST.

No upstream imports, LLM initialization, sampling, optimizer or training call.
This qualifies the published example command, not the authors' experiment runs.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    manifest=json.loads((a.source/'manifest.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((a.source/row['path']).read_bytes()).hexdigest()==row['sha256']
    config=ast.parse((a.source/'LaTRO/trainer/trainer_config.py').read_text())
    nodes=[n for n in ast.walk(config) if isinstance(n,ast.AnnAssign)
           and isinstance(n.target,ast.Name) and n.target.id=='sft_penalty']
    assert len(nodes)==1
    default=ast.literal_eval(next(k.value for k in nodes[0].value.keywords if k.arg=='default'))
    active='\n'.join(line for line in (a.source/'scripts/training.sh').read_text().splitlines()
                     if not line.lstrip().startswith('#'))
    assert '--sft_penalty' not in active and default==0
    trainer=ast.parse((a.source/'LaTRO/trainer/latro_trainer.py').read_text())
    assignments=[n for n in ast.walk(trainer) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets)]
    assert len(assignments)==1
    expression=ast.unparse(assignments[0].value)
    assert expression=='rloo_loss + args.sft_penalty * sft_loss'
    expression_code=compile(ast.Expression(assignments[0].value),'<pinned_scalar_loss>','eval')
    # Distinct reason/action parameters make the missing derivative observable;
    # shared real model parameters can still change the action distribution.
    results=[]
    for coefficient in [default,1.]:
        z=torch.tensor([.3,-.4],dtype=torch.float64,requires_grad=True)
        rp=-2.*torch.nn.functional.logsigmoid(z[0])
        ra=-torch.nn.functional.logsigmoid(z[1])
        value=eval(expression_code,{'__builtins__':{}},
                   {'rloo_loss':rp,'sft_loss':ra,'args':SimpleNamespace(sft_penalty=coefficient)})
        g=torch.autograd.grad(value,z)[0]
        results.append({'sft_penalty':coefficient,'gradient':g.tolist()})
    assert results[0]['gradient'][1]==0 and results[1]['gradient'][1]!=0
    result={'commit':manifest['commit'],'source_files_verified':len(manifest['files']),
      'sft_penalty_default':default,'active_example_overrides_sft_penalty':False,
      'loss_expression':expression,'loss_expression_line':assignments[0].lineno,
      'scalar_loss_gradient':results,
      'qualified_conclusion':'The published default/example weights the explicit answer-likelihood gradient by zero. Set the coefficient explicitly to reproduce the two-term paper objective. Actual published experiment commands and checkpoints were not verified.',
      'shared_parameter_limit':'Zero explicit action term does not mean action probabilities cannot change via shared weights.',
      'paper_replication':False,'upstream_trainer_executed':False,'model_calls':0,'optimizer_updates':0}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
