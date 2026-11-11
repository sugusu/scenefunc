"""Audit real public PSD repairs and run its inspected, unchanged pure kernels.

This never imports a model backend, reconstructs a benchmark environment, calls
an API, or labels any 3D scene successful. Public rows remain in RAM; persisted
tables contain only identifiers, counts, weights and evidence classifications.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time


PIN = "778be78bdac582b51a975ff819046583aad383e0"
DATA_PIN = "257d3d5a9a056fd357ac4a92c6edbc20dc0cb3b1"
STAGE = "20261111_verified_repair_supervision"
EXPECTED = {
    "repairs/river_native/repair_rows_clean.jsonl": "65a41917dcc53f93f12df6c9753e5844c6d3179ebfb272da7bede8a90737d039",
    "traces/river_native/repair_search_traces.jsonl": "3aac1030255c9a154cdeea9e126230311872770f13fe192e4c02e336c6e11584",
    "targets/river_native/top20_targets.jsonl": "5efa1ba52bfef69b3a4cab102e6058fed4beb876c4e4fd6d55f8ee738c21768a",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def records(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def original_functions(path, names):
    # AST isolation is intentional: no river_client, tokenizers or package-level
    # environment setup executes. Function bodies are not edited or retyped.
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"math": math, "CONTEXT_WINDOW": 262144}
    exec(compile(module, str(path), "exec"), ns)
    return ns, [dict(name=n.name, first_line=n.lineno, last_line=n.end_lineno) for n in nodes]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    repo, root, out = a.repo.resolve(), a.source_root.resolve(), a.output.resolve()
    assert out.is_relative_to(repo / "runs/autodl" / STAGE) or out.is_relative_to(Path("/dev/shm"))
    if out.exists():
        raise FileExistsError(out)
    assert json.loads((root / "psd_tree.json").read_text())["sha"] == PIN
    assert json.loads((root / "data_repo.json").read_text())["sha"] == DATA_PIN
    manifest = {}
    for line in (root / "data/MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        manifest[name.lstrip("*./")] = digest
    for name, digest in EXPECTED.items():
        assert sha(root / "data" / name) == digest == manifest[name], name

    repairs = list(records(root / "data/repairs/river_native/repair_rows_clean.jsonl"))
    traces = list(records(root / "data/traces/river_native/repair_search_traces.jsonl"))
    by_task = {t["task_id"]: t for t in traces}
    assert len(by_task) == len(traces) == 264
    assert len(repairs) == len({(r["task_id"], r["turn_idx"]) for r in repairs}) == 375
    trace_rows = []
    attempt_rows = []
    for t in traces:
        assert t["failed"]["checker"]["valid"] is False
        assert t["final"]["task_valid"] is True and t["final"]["checker"]["valid"] is True
        final_slate = {str(r["turn_idx"]): r["hint"] for r in t["repair_turns"]}
        executed_slate = t["final"]["slate"]
        # The released training rows were cleaned after execution. Two traces
        # retain an executed hint absent from repair_turns. Preserve both
        # populations; removing a row does not rerun the trajectory without it.
        assert all(executed_slate.get(key) == hint for key, hint in final_slate.items())
        trace_rows.append(dict(task_id=t["task_id"], category=t["category"],
            retained_repair_hint_count=len(t["repair_turns"]),
            executed_final_hint_count=len(executed_slate),
            additional_executed_hint_turns=";".join(sorted(set(executed_slate)-set(final_slate))),
            recorded_attempts=len(t["attempts"]),
            failed_original=True, author_reported_final_pass=True,
            hint_free_attempts=sum(not x["slate"] for x in t["attempts"]),
            evidence_scope="joint_slate_full_task" if len(executed_slate) > 1 else "single_hint_full_task",
            isolated_hint_causal_effect="not_estimated", locally_reexecuted=False))
        for i, attempt in enumerate(t["attempts"]):
            attempt_rows.append(dict(task_id=t["task_id"], attempt_index=i,
                slate_size=len(attempt["slate"]), author_reported_task_valid=attempt["task_valid"],
                checker_valid=(attempt.get("checker") or {}).get("valid"),
                assistant_steps=sum(len(v.get("steps", [])) for v in attempt.get("turns", [])),
                baseline_pair_id="not_provided"))

    repair_rows = []
    for r in repairs:
        t = by_task[r["task_id"]]
        assert r["accepted"] is True and r["verified_full_task"] is True
        assert r["teacher_messages"] == [*r["student_messages"], {"role": "user", "content": r["hint"]}]
        assert str(r["turn_idx"]) in t["final"]["slate"]
        assert t["final"]["slate"][str(r["turn_idx"]) ] == r["hint"]
        assert r["target_message"]["role"] == "assistant"
        assert len(r["completion_ids"]) == len(r["sample_logprobs"]) > 0
        repair_rows.append(dict(task_id=r["task_id"], turn_index=r["turn_idx"],
            retained_hint_count=len(t["repair_turns"]),
            hint_count_in_verified_slate=len(t["final"]["slate"]), completion_tokens=len(r["completion_ids"]),
            native_tools_in_target=len(r["target_message"].get("tool_calls") or []),
            teacher_exactly_student_plus_hint=True, author_tier=r.get("tier"),
            evidence_scope="joint_slate" if len(t["final"]["slate"]) > 1 else "single_hint",
            unhinted_matched_prefix_control="not_provided", full_3D_scene_result=False))

    source = root / "repo/psd/training/river_psd.py"
    assert sha(source) == "ae60d24a989f4cfdaa5730c87d143b4aff66698286053764eea61362954f14ea"
    kernel, source_ranges = original_functions(source, ["validate_target", "expanded_datums", "native_datum"])
    target_rows = []
    mass, token_mass, counts = defaultdict(float), defaultdict(float), Counter()
    total_presentation = 0
    max_error = 0.0
    example = None
    for row in records(root / "data/targets/river_native/top20_targets.jsonl"):
        kernel["validate_target"](row, 20)
        native = kernel["native_datum"](row, 20)
        expanded = kernel["expanded_datums"](row, 20)
        # Compare exact coefficients at every original prediction position,
        # including all zero-masked prompt/final positions. No proxy model used.
        assert len(expanded) == 20
        for rank, datum in enumerate(expanded):
            assert datum["input_ids"] == native["input_ids"]
            assert datum["target_tokens"] == [v[rank] for v in native["target_tokens"]]
            assert datum["weights"] == [v[rank] for v in native["weights"]]
        m = sum(sum(w) for w in native["weights"])
        expected = float(row["row_weight"]) * len(row["completion_ids"])
        max_error = max(max_error, abs(m - expected))
        assert math.isclose(m, expected, rel_tol=1e-12, abs_tol=1e-10)
        active = [i for i, w in enumerate(native["weights"]) if any(w)]
        n = len(row["student_prompt_ids"])
        assert active == list(range(n-1, n-1+len(row["completion_ids"])))
        kind = row["kind"]
        mass[kind] += float(row["row_weight"])
        token_mass[kind] += m
        counts[kind] += 1
        total_presentation += len(native["input_ids"])
        target_rows.append(dict(key=row["key"], kind=kind, task_id=row["task_id"],
            turn_index=row["turn_idx"], step_index=row["step_idx"],
            prompt_tokens=n, completion_tokens=len(row["completion_ids"]),
            row_weight=row["row_weight"], weighted_target_token_mass=m,
            packing_coefficients_identical=True))
        if example is None:
            example = {k: row[k] for k in ["key", "kind", "row_weight"]}
            example["first_cached_top20"] = row["topk_by_position"][0]
        del native, expanded
    assert counts == {"repair": 375, "preserve": 1175}
    assert all(math.isclose(mass[k], 375.0, rel_tol=1e-12) for k in counts)
    joint_rows = sum(len(t["repair_turns"]) for t in traces if len(t["final"]["slate"]) > 1)
    result = dict(kind="Public PSD evidence and unchanged target-packing qualification, not model or scene execution",
        code_pin=PIN, data_pin=DATA_PIN, checked_data_files=EXPECTED,
        public_repairs=dict(rows=len(repairs), unique_tasks=len(traces),
            retained_hint_count_distribution=dict(sorted(Counter(len(t["repair_turns"]) for t in traces).items())),
            executed_hint_count_distribution=dict(sorted(Counter(len(t["final"]["slate"]) for t in traces).items())),
            single_hint_tasks=sum(len(t["final"]["slate"]) == 1 for t in traces),
            multi_hint_tasks=sum(len(t["final"]["slate"]) > 1 for t in traces),
            traces_with_unretained_executed_hints=[v for v in trace_rows if v["additional_executed_hint_turns"]],
            rows_supported_by_joint_slate=joint_rows,
            recorded_attempts=len(attempt_rows),
            recorded_passing_attempts=sum(x["author_reported_task_valid"] for x in attempt_rows),
            hint_free_attempts=sum(x["slate_size"] == 0 for x in attempt_rows),
            warning="Released success-selected search traces do not estimate population repair yield or isolated hint treatment effects; author verifier outcomes were read, not reexecuted"),
        original_kernel=dict(path="psd/training/river_psd.py", sha256=sha(source), ranges=source_ranges,
            calls=3*len(target_rows), targets=len(target_rows), expanded_datums=20*len(target_rows),
            all_coefficients_identical=True, source_target_counts=dict(counts), source_row_mass=dict(mass),
            source_weighted_token_mass=dict(token_mass), max_weight_mass_residual=max_error,
            native_input_token_presentations=total_presentation,
            expanded_input_token_presentations=20*total_presentation,
            server_reduction_dropout_FP8_and_optimizer_equivalence_not_tested=True,
            cached_top20_is_conditional_teacher_distribution=True, original_tail_probability_not_in_cache=True),
        actual_model_objects=0, model_forwards=0, optimizer_updates=0, environment_replays=0,
        paid_API_calls=0, full_3D_generations=0, seconds=time.monotonic()-start)
    out.mkdir(parents=True)
    for name, rows in [("public_trace_census.csv", trace_rows), ("public_attempt_census.csv", attempt_rows),
                       ("public_repair_census.csv", repair_rows), ("public_target_census.csv", target_rows)]:
        csv_write(out / name, rows)
    (out / "first_cached_distribution.json").write_text(json.dumps(example, indent=2)+"\n")
    (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
