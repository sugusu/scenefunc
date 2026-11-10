#!/usr/bin/env python3
"""Inspect a pinned SceneReVis source contract without importing its trainer.

This executes only original scalar reward and count functions, and an isolated
literal weight-selection block. Inputs are algebra probes, never scene scores.
No model, renderer, simulator, network, training, or optimizer is constructed.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_function(path, name, class_name=None):
    tree = ast.parse(Path(path).read_text())
    body = tree.body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    if isinstance(node, ast.AsyncFunctionDef):
        raise ValueError("Full async interaction execution is outside this audit")
    node = copy.deepcopy(node)
    node.decorator_list = []
    env = dict(Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
               sys=sys)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), "exec"), env)
    return env[name]


def original_weight_selector(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SceneEditingInteraction")
    fn = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "calculate_score")
    blocks = [n for n in fn.body if isinstance(n, ast.If)
              and any(isinstance(a, ast.Assign) and isinstance(a.value, ast.Dict)
                      and any(isinstance(t, ast.Name) and t.id == "weights" for t in a.targets)
                      for a in ast.walk(n))]
    if len(blocks) != 1:
        raise ValueError("Source changed: weight-selection block is not unique")
    wrapper = ast.parse("def select(turn, is_final_turn, self):\n    pass\n").body[0]
    wrapper.body = [copy.deepcopy(blocks[0]), ast.Return(value=ast.Name(id="weights", ctx=ast.Load()))]
    env = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 str(path), "exec"), env)
    return env["select"], [blocks[0].lineno, blocks[0].end_lineno]


def scalar_probe(fn, n, branch):
    def evaluate(values):
        if branch == "rollout_user_turn_rewards":
            info = {"rollout_reward_scores": {"user_turn_rewards": values}}
        elif branch == "fallback_turn_scores":
            info = {"rollout_reward_scores": {}, "turn_scores": values}
        else:
            raise ValueError(branch)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return fn("scene_editing", "unused-response", None, info)
    baseline = evaluate([0.0] * n)
    coefficients = []
    for k in range(n):
        values = [0.0] * n
        values[k] = 1.0
        coefficients.append(evaluate(values) - baseline)
    expected = ([0.0] * n if n <= 2 else
                ([0.4 / (n - 1)] * (n - 1) + [0.6]
                 if branch == "rollout_user_turn_rewards" else [0.0] * (n - 1) + [1.0]))
    expected_baseline = -1.0 if n <= 2 else 0.0
    if not math.isclose(baseline, expected_baseline, abs_tol=1e-12) or any(
            not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(coefficients, expected)):
        raise AssertionError((n, branch, baseline, coefficients))
    # A non-basis vector checks the affine reduction; values are not model rewards.
    values = [(-1) ** k * (k + 1) / (n + 1) for k in range(n)]
    predicted = baseline + sum(c * v for c, v in zip(coefficients, values))
    actual = evaluate(values)
    if not math.isclose(predicted, actual, abs_tol=1e-12):
        raise AssertionError((predicted, actual))
    return dict(branch=branch, turns=n, baseline=baseline,
                effective_coefficients=coefficients, affine_vector_check=True)


def run(source):
    reward = source / "script/RL/scene_reward.py"
    interaction = source / "verl/verl/interactions/scene_editing_interaction.py"
    score = original_function(reward, "compute_score")
    scalar = [scalar_probe(score, n, branch) for branch in
              ("rollout_user_turn_rewards", "fallback_turn_scores") for n in range(1, 8)]
    selector, lines = original_weight_selector(interaction)
    weights = {}
    for mode in ("trimesh", "voxel"):
        for stage, turn, final in (("initial", 1, False), ("intermediate", 2, False), ("final", 7, True)):
            w = selector(turn, final, SimpleNamespace(physics_mode=mode))
            if not math.isclose(sum(w.values()), 1.0, abs_tol=1e-12):
                raise AssertionError(w)
            weights[f"{mode}/{stage}"] = {k: v for k, v in w.items() if v != 0}
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    count_fn = original_function(interaction, "_calculate_object_count_reward", "SceneEditingInteraction")
    counts = []
    for n in (0, 1, 2, 3, 4, 5, 15):
        value, override = count_fn(SimpleNamespace(logger=logger), {"objects": [{} for _ in range(n)]}, True)
        if override != (n < 4):
            raise AssertionError((n, override))
        counts.append(dict(object_count=n, count_reward=value, override_total=override))
    config = (source / "script/RL/config/scene_editing_interaction_B200.yaml").read_text()
    if "enabled: false" not in config or "vlm_judge_enabled: true" not in config:
        raise AssertionError("Published config switches changed; review required")
    return {
        "scope": "Original scalar functions and literal weight-selection block only; no whole trainer or scene execution",
        "source_hashes": {str(p.relative_to(source)): sha(p) for p in (reward, interaction)},
        "reward_basis_probes": scalar,
        "weight_selector_original_lines": lines,
        "nonzero_component_weights": weights,
        "count_gate_probes": counts,
        "count_probe_boundary": "Synthetic list lengths; not rendered scenes, completeness labels, or new failures",
        "input_route_boundary": "Both branches verified. Dominant published training route unresolved without actual RL parquet/config/run logs",
        "text_feedback_master_in_published_B200_config": False,
        "vlm_judge_in_published_B200_config": True,
        "missing_feedback_boundary": "Disables injected textual feedback only; does not remove rendered observations or reward judge",
        "model_forwards": 0, "optimizer_updates": 0, "API_calls": 0,
        "new_generated_scenes": 0, "paper_result_reproduced": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "scope": result["scope"],
                      "reward_branches_verified": 2, "model_forwards": 0}))


if __name__ == "__main__":
    main()
