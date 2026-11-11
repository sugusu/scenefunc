"""Qualify distinct distillation objectives against upstream code and exact math.

The categorical inputs are authored mathematical witnesses, never 3D examples,
human preferences, model logits, rollouts, or measured policy improvements.
No model object is instantiated and no optimizer update occurs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import time


PIN = "6f53ef41c951de0962bbf27fe3f5819eabc8ce0e"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def divergence(p, q):
    return sum(x * math.log(x / y) for x, y in zip(p, q) if x)


def coarse_divergence(p, q, selected):
    value = sum(p[i] * math.log(p[i] / q[i]) for i in selected)
    pt = sum(v for i, v in enumerate(p) if i not in selected)
    qt = sum(v for i, v in enumerate(q) if i not in selected)
    return value + (pt * math.log(pt / qt) if pt else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    root, repo, out = args.source_root.resolve(), args.repo.resolve(), args.output.resolve()
    stage = repo / "runs/autodl/20261111_privileged_training_objectives"
    if not (out.is_relative_to(stage) or out.is_relative_to(Path("/dev/shm"))):
        raise ValueError("Write this study or RAM only")
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((root / "download_manifest.json").read_text())
    if manifest["pin"] != PIN:
        raise ValueError("Upstream revision changed")
    for row in manifest["files"]:
        if "sha256" not in row or sha(root / row["path"]) != row["sha256"]:
            raise ValueError("Missing/changed upstream file: " + row["path"])
    path = root / "opcd/verl/verl/trainer/ppo/core_algos.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kl_penalty")
    # Only the inspected original scalar/tensor kernel is loaded, not VeRL/Ray.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    ast.fix_missing_locations(module)
    import torch
    ns = dict(torch=torch)
    exec(compile(module, str(path), "exec"), ns)
    kernel = ns["kl_penalty"]
    inputs = [([0.6, 0.3, 0.1], [0.7, 0.2, 0.1]),
              ([0.90, 0.09, 0.01], [0.95, 0.04, 0.01])]
    cases = []
    for p, q in inputs:
        lp, lq = torch.tensor(p, dtype=torch.float64).log(), torch.tensor(q, dtype=torch.float64).log()
        selected = [max(range(len(p)), key=p.__getitem__)]
        full = float(kernel(lp, lq, "full"))
        partial = float(kernel(lp[selected], lq[selected], "full", False))
        conditional = float(kernel(lp[selected], lq[selected], "full", True))
        reference = divergence(p, q)
        coarse = coarse_divergence(p, q, selected)
        assert math.isclose(full, reference, rel_tol=1e-12, abs_tol=1e-12)
        assert partial < 0.0 and abs(conditional) < 1e-12 and 0 <= coarse <= full + 1e-12
        tail = [i for i in range(len(p)) if i not in selected]
        pt = sum(p[i] for i in tail)
        qt = sum(q[i] for i in tail)
        tail_kl = divergence([p[i] / pt for i in tail], [q[i] / qt for i in tail])
        assert math.isclose(full, coarse + pt * tail_kl, rel_tol=1e-12, abs_tol=1e-12)
        cases.append(dict(student=p, teacher=q, selected=selected,
                          full_original_kernel=full, independent_full_KL=reference,
                          selected_unnormalized_original_kernel=partial,
                          selected_renormalized_original_kernel=conditional,
                          aggregated_tail_reference_KL=coarse,
                          tail_chain_rule_residual=full-coarse-pt*tail_kl,
                          evidence_type="Authored finite-distribution mathematical witness, not scene data"))

    contexts = [[0.8, 0.18, 0.02], [0.02, 0.18, 0.8]]
    arithmetic = [sum(v[i] for v in contexts) / len(contexts) for i in range(3)]
    geometric = [math.exp(sum(math.log(v[i]) for v in contexts) / len(contexts)) for i in range(3)]
    total = sum(geometric)
    geometric = [v / total for v in geometric]
    gradients = {}
    # One fixed observation with hidden teacher-context uncertainty; not the
    # global optimum of a sequence or interactive on-policy objective.
    teacher = torch.tensor(contexts, dtype=torch.float64)
    for direction, optimum in [("teacher_to_student", arithmetic), ("student_to_teacher", geometric)]:
        z = torch.tensor(optimum, dtype=torch.float64).log().requires_grad_()
        logq = z.log_softmax(-1)
        if direction == "teacher_to_student":
            objective = (teacher * (teacher.log() - logq)).sum(-1).mean()
        else:
            objective = (logq.exp() * (logq - teacher.log())).sum(-1).mean()
        grad, = torch.autograd.grad(objective, z)
        assert float(grad.abs().max()) < 1e-12
        gradients[direction] = dict(optimum=optimum, objective=float(objective.detach()),
                                    max_abs_logit_gradient=float(grad.abs().max()))

    result = dict(kind="Original OPCD mathematical-kernel qualification and conditional-objective theory checks",
                  pin=PIN, upstream_definition=dict(path=str(path.relative_to(root)), start=node.lineno, end=node.end_lineno,
                                                    file_sha256=sha(path)),
                  authored_witnesses=cases, hidden_context_witness=dict(teacher_contexts=contexts, results=gradients),
                  interpretation="Forward/reverse KL and truncated/renormalized/full-vocabulary objectives are distinct; none of these witnesses is a 3D policy result",
                  standard_tail_coarsening_not_claimed_novel=True, source_kernel_calls=6, autograd_calls=2,
                  model_objects_instantiated=0, pretrained_model_forwards=0, optimizer_updates=0,
                  scene_executions=0, paid_API_calls=0, torch_version=torch.__version__,
                  seconds=time.monotonic()-start)
    # Actual prior records remain unknown as environment-correctness labels;
    # do not turn tool success flags into calibrated task rewards.
    old = repo / "runs/autodl/20261111_tool_decision_ownership/actual_audit_v1"
    calls_path, decisions_path = old / "tool_calls.json", old / "policy_decisions.json"
    calls = json.loads(calls_path.read_text())
    decisions = json.loads(decisions_path.read_text())
    result["existing_native_record_scope"] = dict(
        tool_calls=len(calls), policy_responses=len(decisions), independent_tasks=1,
        expert_correctness_known=sum(v.get("expert_correctness") is not None for v in calls),
        complete_scene_success_known=sum(v.get("complete_scene_success") is not None for v in calls),
        environment_command_advantages_not_estimated=True,
        reason="One partial historical run is not eight sampled complete trajectories with task rewards; tool success is not task correctness",
        calls_sha256=sha(calls_path), decisions_sha256=sha(decisions_path))
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("authored_witnesses", "hidden_context_witness")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
