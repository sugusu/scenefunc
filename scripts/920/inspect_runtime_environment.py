"""Inspect installed metadata without importing model, geometry or server code."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    data = args.requirements.read_bytes()
    rows = []
    for line in data.decode().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        applies = requirement.marker is None or requirement.marker.evaluate()
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            version = None
        rows.append({"package": requirement.name, "required": str(requirement.specifier),
                     "installed": version, "applies": applies,
                     "satisfies": not applies or
                     (version is not None and requirement.specifier.contains(version))})
    result = {"started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Distribution metadata and top-level specs only; no module imports for inspected dependencies.",
              "python": sys.version.split()[0], "environment": sys.executable,
              "requirements_path": str(args.requirements.resolve()),
              "requirements_sha256": hashlib.sha256(data).hexdigest(),
              "requirements": rows,
              "additional_modules": {name: util.find_spec(name) is not None
                                     for name in ("bpy", "pydrake", "scenesmith")},
              "official_preflight_executed": False, "full_runtime_ready": None,
              "model_calls": 0, "training": False, "services_started": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "python": result["python"],
                      "requirements_satisfied": sum(x["satisfies"] for x in rows),
                      "requirements_total": len(rows),
                      "additional_modules": result["additional_modules"]}))


if __name__ == "__main__":
    main()
