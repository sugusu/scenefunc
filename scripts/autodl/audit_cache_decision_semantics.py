"""Inspect the released image-cache metrics without loading its neural modules.

This is a source audit, not a reproduction of the paper's measurements. In
particular, correctness-indicator agreement is not exact action agreement.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time


PIN = "e9085eafcc6c83e60c548de69060c4bd5b210c96"
VISION_HASHES = {
    "esys/composable_vision.py": "2ce0c83cea26578fc1ace1fef756e40bdde15a4d39f5cc8afe60c9371d309b94",
    "esys/composable_vision_shift.py": "9fb79e3b41760d7e8afd8d2c3342cff70f7dc5f2f06064d454b41782c80e0016",
}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    out = a.output.resolve()
    study = a.repo.resolve() / "runs/autodl/20261111_sceneexplorer_objectives"
    if not (out.is_relative_to(study) or out.is_relative_to(Path("/dev/shm"))):
        raise ValueError("Write this study or RAM only")
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((a.source_root / "vision_download_manifest.json").read_text())
    if manifest["pin"] != PIN:
        raise ValueError("Source revision changed")
    rows = []
    for relative, expected in VISION_HASHES.items():
        path = a.source_root / relative
        entry = next(v for v in manifest["files"] if v["path"] == relative)
        if expected != entry["sha256"] or digest(path) != expected:
            raise ValueError("Frozen source changed: " + relative)
        tree = ast.parse(path.read_text())
        assignments = []
        agreements = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("fc", "pc", "sc") for t in node.targets):
                assignments.append(dict(line=node.lineno, expression=ast.unparse(node)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append":
                text = ast.unparse(node)
                if "['agree']" in text:
                    agreements.append(dict(line=node.lineno, expression=text))
        if not assignments or len(agreements) != 1 or "fc == " not in agreements[0]["expression"]:
            raise ValueError("Metric expression changed; re-read the new code rather than reusing this interpretation")
        rows.append(dict(path=relative, sha256=expected, correctness_assignments=assignments,
                         reported_agreement=agreements,
                         metric_interpretation="Equality of answer-substring correctness indicators, including both wrong",
                         exact_decoded_string_agreement_measured_here=False,
                         exact_executed_tool_arguments_agreement_measured_here=False))
    result = dict(kind="Pinned upstream AST metric audit; no model execution", pin=PIN, files=rows,
                  vision_task_source="gen_tasks uses PIL drawings and ten question/policy templates in the inspected branch",
                  scope="These two scripts only. Other experiments and the complete paper may use different metrics/data.",
                  inference="Reported agreement in these scripts alone cannot certify identical action JSON or complete scene traces",
                  unchanged=all(digest(a.source_root / r) == h for r, h in VISION_HASHES.items()),
                  model_forwards=0, training_updates=0, new_scene_trials=0, paid_API_calls=0,
                  seconds=time.monotonic() - start)
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
