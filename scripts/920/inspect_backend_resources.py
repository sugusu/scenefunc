"""Read fixed Hydra defaults and local resources without constructing agents."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenesmith", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(datetime.now(timezone.utc).isoformat(), "inspect declared backend defaults and local files", flush=True)
    root = args.scenesmith.resolve()
    # Deliberately inspect declared defaults, without reading credentials or
    # exposing endpoint overrides from the user's environment.
    for key in ("SCENESMITH_MODEL", "SS_GEOMETRY_PORT", "SS_HSSD_PORT",
                "SS_ARTICULATED_PORT", "SS_OBJAVERSE_PORT", "SS_MATERIALS_PORT"):
        os.environ.pop(key, None)
    with initialize_config_dir(config_dir=str(root / "configurations"), version_base=None):
        cfg = compose(config_name="scene_designer")
    rows, agents = [], {}
    def select(key):
        return OmegaConf.select(cfg, key)
    def resource(key, required_by):
        value = select(key)
        if value is None:
            return
        path = root / str(value)
        rows.append({"configuration_key": key, "configured_path": str(value),
                     "resolved_path": str(path), "exists": path.exists(),
                     "required_by": required_by, "content_validated": False})
    for stage in ("furniture_agent", "wall_agent", "ceiling_agent", "manipuland_agent"):
        prefix = stage + ".asset_manager"
        general = str(select(prefix + ".general_asset_source"))
        agents[stage] = {"general_asset_source": general,
                         "backend": select(prefix + ".backend"),
                         "router_enabled": bool(select(prefix + ".router.enabled")),
                         "articulated_enabled": bool(select(prefix + ".router.strategies.articulated.enabled")),
                         "image_generation_backend": select(prefix + ".image_generation.backend"),
                         "auxiliary_model": select(stage + ".openai.model")}
        for name in ("sam3_checkpoint", "sam3d_checkpoint"):
            resource(prefix + ".sam3d." + name, "default_geometry_generation" if general == "generated" else "inactive_geometry_option")
        for name in ("data_path", "preprocessed_path"):
            resource(prefix + ".hssd." + name, "default_hssd_retrieval" if general == "hssd" else "alternative_hssd_option")
        for source in ("partnet_mobility", "artvip"):
            active = agents[stage]["articulated_enabled"] and bool(select(prefix + ".articulated.sources." + source + ".enabled"))
            for name in ("data_path", "embeddings_path"):
                resource(prefix + ".articulated.sources." + source + "." + name,
                         "default_articulated_retrieval" if active else "inactive_articulated_option")
    for name in ("data_path", "embeddings_path"):
        resource("experiment.materials_retrieval_server." + name, "default_materials_retrieval")
    asset_lock = args.experiment_root / "scenefunc/paris/config/assets.lock.json"
    lock = json.loads(asset_lock.read_text())
    cache = (args.experiment_root / "resources/scenefunc/hssd-models").resolve()
    present, matching = 0, 0
    for item in lock["rows"]:
        path = (cache / item["path"]).resolve()
        if not path.is_relative_to(cache):
            raise ValueError("Asset lock path escaped the configured cache")
        if path.is_file():
            present += 1
            if path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]:
                matching += 1
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "configuration": "scene_designer declared defaults",
        "source_config_hashes": {p.relative_to(root).as_posix(): digest(p) for p in sorted((root / "configurations").rglob("*.yaml"))},
        "agents": agents, "resources": rows,
        "local_asset_cache": {"path": str(cache), "lock_sha256": digest(asset_lock),
                              "declared_original_candidates": lock["candidate_pool_count"],
                              "declared_selected_assets": lock["selected_asset_count"],
                              "files_present": present, "files_matching_locked_hashes": matching,
                              "full_hssd_dataset_or_retrieval_index": False},
        "default_pipeline_resource_ready": not any(not r["exists"] for r in rows if r["required_by"].startswith("default_")),
        "services_started": 0, "model_calls": 0, "training_updates": 0,
        "limitations": ["File existence does not validate resource content or a running service.",
                        "Existing selected assets do not reproduce the original candidate pool or full HSSD retrieval.",
                        "Auxiliary image generation and VLM requests are separate from the local Scene-Builder policy.",
                        "Agent construction can start Blender/collision services; none was constructed here."],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"resource_rows": len(rows), "matching_local_assets": matching,
                      "default_pipeline_resource_ready": result["default_pipeline_resource_ready"]}), flush=True)


if __name__ == "__main__":
    main()
