"""Native construction from a fixed portable asset catalog.

Dedicated scope keeps existing revision checkpoints and adapter unchanged.
Native placement/memory/rendering are reused; catalog geometry is pre-acquired,
not generated, articulated, or validated for whole-room usability.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import types

from native_project import import_project, read_json, sha256, contained_path, native_paths
from native_checkpoint import (SESSION_SCHEMA, canonical_hash, config_hash,
                               load_session_record, runtime_fingerprint, save_checkpoint)


TOOLS = ("get_current_scene_state", "add_furniture_to_scene_tool", "move_furniture_tool", "remove_furniture_tool", "check_facing_tool")
SCOPE = "catalog_furniture_construction_v1"


def read_catalog(project):
    """Validate portable templates without exposing local paths in model context."""
    relative = project.manifest['inputs'].get('asset_catalog')
    if not relative or relative not in project.manifest['files']:
        raise ValueError('A hashed asset catalog is required')
    catalog = read_json(contained_path(project.directory, relative))
    if catalog.get('schema') != 'scenefunc-native-catalog-v1' or not catalog.get('entries'):
        raise ValueError('Unsupported or empty catalog')
    ids = set()
    for entry in catalog['entries']:
        template = entry['template']; rid = template['object_id']
        if rid in ids or template['object_type'] != 'furniture':
            raise ValueError('Catalog IDs must be unique furniture templates')
        ids.add(rid)
        if template['support_surfaces'] or template['immutable'] or template['placement_info'] is not None:
            raise ValueError('Current catalog scope supports floor-based unarticulated templates only')
        if template['transform'] != {'translation': [0.0, 0.0, 0.0], 'rotation_wxyz': [1.0, 0.0, 0.0, 0.0]}:
            raise ValueError('Catalog templates must have canonical zero poses')
        if template['metadata'].get('catalog_registry_id') != rid or not entry['binding'].get('asset_id'):
            raise ValueError('Catalog identity missing')
        if not template.get('geometry_path') or not template.get('sdf_path'):
            raise ValueError('Real geometry and SDF required')
        native_paths(template, project.directory, project.room.scene_dir, project.manifest['files'])
    return catalog


def native_room_summary(scene):
    """Describe supplied geometry without inventing four enclosing walls."""
    geometry = scene.room_geometry
    walls = []
    for wall in geometry.walls:
        bounds = wall.compute_world_bounds()
        walls.append({"object_id": str(wall.object_id), "immutable": wall.immutable,
                      "bounds_min": bounds[0].tolist() if bounds is not None else None,
                      "bounds_max": bounds[1].tolist() if bounds is not None else None})
    return "## SUPPLIED ROOM GEOMETRY (native room coordinates, metres):\n" + json.dumps({
        "room_id": scene.room_id, "room_type": scene.room_type,
        "recorded_length": geometry.length, "recorded_width": geometry.width,
        "wall_count": len(walls), "wall_objects": walls,
        "recorded_openings": scene.to_state_dict()["room_geometry"]["openings"],
        "completeness": "Only the supplied structure is described; enclosure is not inferred.",
    }, ensure_ascii=False)


def create_app(designer: Path, scenesmith: Path):
    os.environ["SCENESMITH_ROOT"] = str(scenesmith.resolve())
    sys.path.insert(0, str(designer.resolve()))
    from src import runtime  # Adds the fixed runtime's source directories.
    import tool_server as server
    from agent_rollout_decoupled import RolloutSession
    from Designer.replay_chained import compose_full_cfg
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    if getattr(server.app.state, "scenefunc_catalog_installed", False):
        raise RuntimeError("Catalog-project routes already installed in this process")

    class CatalogSession(RolloutSession):
        def __init__(self, project, config):
            super().__init__(project.directory, config, project.task["design_request"], stub_assets=False)
            self.project = project
            self.resume_record = load_session_record(project)
            self.catalog = read_catalog(project)
            self.catalog_sha256 = canonical_hash(self.catalog)
            self.fingerprint = runtime_fingerprint(designer, scenesmith)
            self.fingerprint['adapter/catalog_project_session.py'] = sha256(Path(__file__))
            self.fingerprint['scenesmith/agent_utils/asset_registry.py'] = sha256(scenesmith/'scenesmith/agent_utils/asset_registry.py')
            if self.resume_record and (self.resume_record['scope'] != SCOPE or
                    self.resume_record.get('catalog_sha256') != self.catalog_sha256):
                raise ValueError('Checkpoint catalog/scope differs')
            if self.resume_record and self.resume_record["runtime_fingerprint"] != self.fingerprint:
                raise ValueError("Checkpoint runtime code differs; explicit migration is required")
            self.scene = project.room
            self.scene._scenefunc_structure_labels = True
            self.house_layout = project.house.layout
            self.room_geometry = self.scene.room_geometry
            self.scene.action_log_path = self.scene.scene_dir / "native_actions.json"
            self.room_summary = native_room_summary(self.scene)
            self.cfg, self.cfg_dict = compose_full_cfg(config, self.scene_dir)
            # This phase places catalog assets and edits native objects. No router or generation
            # backends are started; unavailable tools are removed from both the
            # dispatch map and advertised schemas.
            cfg = self.cfg_dict["furniture_agent"]
            cfg["asset_manager"]["router"]["enabled"] = False
            for strategy in cfg["asset_manager"]["router"]["strategies"].values():
                strategy["enabled"] = False
            cfg["rendering"]["blender_server_port_range"] = [18100, 18120]
            cfg["collision_geometry"]["server_port_range"] = [18130, 18150]
            self.cfg_dict["experiment"]["num_workers"] = max(1, ((os.cpu_count() or 1) + 3) // 4)
            self.effective_config_hash = config_hash(self.cfg_dict, self.scene_dir)
            if self.resume_record and (self.resume_record["config_name"] != config
                    or self.resume_record["effective_config_sha256"] != self.effective_config_hash
                    or self.resume_record["tool_names"] != list(TOOLS)):
                raise ValueError("Checkpoint configuration/tool scope differs")
            self.receipts = copy.deepcopy(self.resume_record["receipts"]) if self.resume_record else []
            self.restoration_transitions = copy.deepcopy(self.resume_record["restoration_transitions"]) if self.resume_record else []
            self.initial_state_sha256 = (self.resume_record["initial_state_sha256"] if self.resume_record
                                         else canonical_hash(self.scene.to_state_dict()))
            if self.receipts:
                (self.scene_dir / "session_receipts.jsonl").write_text("".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in self.receipts))
            self.active_environment = None
            self.fp_mat_server = None

        def capture_runtime(self, env):
            from omegaconf import OmegaConf
            return {"schema": SESSION_SCHEMA, "stage": env.stage, "room_id": self.scene.room_id,
                    "step_idx": env.step_idx, "memory": copy.deepcopy(env.mem),
                    "receipts": copy.deepcopy(self.receipts),
                    "restoration_transitions": copy.deepcopy(self.restoration_transitions),
                    "initial_state_sha256": self.initial_state_sha256,
                    "config_name": self.config, "effective_config_sha256": self.effective_config_hash,
                    "tool_names": list(TOOLS), "runtime_fingerprint": self.fingerprint,
                    "catalog_sha256": self.catalog_sha256,
                    "active_noise_profile": OmegaConf.to_container(env.agent.furniture_tools.active_noise_profile),
                    "scope": "catalog_furniture_construction_v1",
                    "recovery_boundary": "between completed tool batches; no in-flight operation recovery"}

        def checkpoint(self, destination):
            if self.active_environment is None:
                if self.resume_record is None:
                    raise ValueError("Start the furniture stage before checkpointing")
                record = self.resume_record
            else:
                record = self.capture_runtime(self.active_environment)
            return save_checkpoint(self.project, destination, record)

        def current_context(self):
            context = self.project.visible_initial_context()
            context["asset_bindings"] = {key: value for key, value in context["asset_bindings"].items()
                                         if key in context["objects"]}
            context["prior_tool_receipts"] = copy.deepcopy(self.receipts[-4:])
            context["history_status"] = "recorded_native_session" if self.receipts else "no_prior_tool_batches"
            context["total_tool_batches"] = len(self.receipts)
            context['asset_catalog'] = [{
                'asset_id': entry['template']['object_id'], 'source_asset_id': entry['binding']['asset_id'],
                'name': entry['template']['name'], 'description': entry['template']['description'],
                'bbox_min': entry['template']['bbox_min'], 'bbox_max': entry['template']['bbox_max'],
                'limitations': entry['limitations'],
            } for entry in self.catalog['entries']]
            return context

        def start(self):
            raise RuntimeError("An imported session must not regenerate its floor plan")

        def make_floor_plan_env(self):
            raise ValueError("Catalog construction fixes the supplied floor plan")

        def make_roomscene_env(self, stage, render_gpu_id, logger, skip_render=False):
            if stage != "furniture" or skip_render or render_gpu_id is not None:
                raise ValueError("Current catalog contract: furniture, real five views, CPU rendering")
            if os.environ.get("OPENSCENEAGENT_STUB_ASSETS") == "1":
                raise ValueError("Stub execution is outside the existing-room contract")
            env = super().make_roomscene_env(stage, render_gpu_id, logger, skip_render=False)
            self.active_environment = env
            from scenesmith.agent_utils.room import SceneObject
            registry = env.agent.furniture_tools.asset_manager.registry
            if registry.size():
                raise ValueError('Catalog stage requires a fresh native registry')
            for entry in self.catalog['entries']:
                template = SceneObject.from_dict(entry['template'], scene_dir=self.scene.scene_dir)
                registry.register(template)
            if self.resume_record is not None:
                from omegaconf import OmegaConf
                env.step_idx = self.resume_record["step_idx"]
                env.mem = copy.deepcopy(self.resume_record["memory"])
                env.agent.furniture_tools.active_noise_profile = OmegaConf.create(self.resume_record["active_noise_profile"])
            missing = set(TOOLS) - set(env.tool_map)
            if missing:
                env.cleanup()
                raise ValueError(f"Required real native tools absent: {sorted(missing)}")
            env.tool_map = {name: env.tool_map[name] for name in TOOLS}
            schemas = [{"name": name, "description": tool.description,
                        "parameters": tool.params_json_schema} for name, tool in env.tool_map.items()]
            schema_text = json.dumps(schemas, ensure_ascii=False, sort_keys=True)
            system = ("Construct and revise the supplied room for its original design request using the fixed asset catalog. "
                      "Preserve existing object identities and immutability constraints; use catalog asset_id for native additions. "
                      "Inspect the current state, latest real receipts and five views. "
                      "A successful edit does not establish collision freedom or activity suitability. "
                      "Report unverified requirements as unknown.\nAvailable executable tools:\n" + schema_text)
            initial_context = self.current_context()
            (self.scene_dir / "session_initial_context.json").write_text(json.dumps(initial_context, indent=2) + "\n")
            (self.scene_dir / "session_configuration.json").write_text(json.dumps({
                "scope": "catalog_furniture_construction_v1", "config": self.config,
                "effective_config": self.cfg_dict, "tool_names": list(TOOLS),
                "source_manifest_sha256": sha256(self.project.directory / "manifest.json"),
                "backend_generation_started": False, "original_request": self.project.task["original_request"],
            }, indent=2) + "\n")
            original_observation = env.observation
            original_step = env.step
            original_cleanup = env.cleanup

            def observation(bound_env):
                before = self.scene.to_state_dict()
                context = self.current_context()
                bound_env.room_prompt = self.project.task["design_request"] + "\nCURRENT NATIVE STATE:\n" + json.dumps(context, ensure_ascii=False)
                bound_env.system = system + ("\n" + self.room_summary if self.room_summary else "")
                obs = original_observation()
                after = self.scene.to_state_dict()
                if canonical_hash(before) != canonical_hash(after):
                    raise RuntimeError("Scene changed while producing an observation")
                evidence = {"step_idx": bound_env.step_idx, "state_sha256": canonical_hash(before),
                            "images": [{"path": p, "sha256": sha256(Path(p))} for p in obs["image_paths"]],
                            "current_context_sha256": canonical_hash(context), "native_state": before}
                target = bound_env.steps_dir / f"step_{bound_env.step_idx:03d}.state_binding.json"
                target.write_text(json.dumps(evidence, indent=2) + "\n")
                obs["state_binding"] = str(target)
                return obs

            async def step(bound_env, tool_calls):
                before = self.scene.to_state_dict()
                result = await original_step(tool_calls)
                # Bind actual newly created IDs after native placement, never guessed IDs.
                bindings = self.project.runtime['rooms'][self.scene.room_id]['asset_bindings']
                catalog_by_id = {e['template']['object_id']: e for e in self.catalog['entries']}
                for oid, obj in self.scene.objects.items():
                    if str(oid) in bindings:
                        continue
                    registry_id = obj.metadata.get('catalog_registry_id')
                    if registry_id not in catalog_by_id:
                        raise RuntimeError('Native addition lacks catalog provenance')
                    entry = catalog_by_id[registry_id]
                    bindings[str(oid)] = {**copy.deepcopy(entry['binding']),
                        'registry_id': registry_id, 'created_by': 'native_add_furniture_to_scene_tool'}
                record = {"step_idx": bound_env.step_idx, "tool_calls": tool_calls,
                          "results": result, "before_sha256": canonical_hash(before),
                          "after_sha256": canonical_hash(self.scene.to_state_dict())}
                self.receipts.append(copy.deepcopy(record))
                with (self.scene_dir / "session_receipts.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                return result

            def cleanup(bound_env):
                if self.active_environment is bound_env:
                    self.resume_record = self.capture_runtime(bound_env)
                    self.active_environment = None
                original_cleanup()

            env.observation = types.MethodType(observation, env)
            env.step = types.MethodType(step, env)
            env.cleanup = types.MethodType(cleanup, env)
            return env

        def close(self):
            if self.active_environment is not None:
                self.active_environment.cleanup()
                self.active_environment = None
            super().close()

    class CatalogRequest(BaseModel):
        bundle_directory: str
        output_directory: str
        config: str = "scene_designer"

    class CheckpointRequest(BaseModel):
        output_directory: str

    # Serialize the upstream singleton application's requests. Concurrent
    # observation and edit would invalidate image/state provenance.
    serial = asyncio.Lock()
    @server.app.middleware("http")
    async def serialize_existing_requests(request, call_next):
        async with serial:
            if isinstance(server._state.session, CatalogSession):
                allowed = {"/healthz", "/session/catalog", "/session/stage", "/session/observation",
                           "/session/step", "/session/cleanup", "/session/close", "/session/checkpoint"}
                if request.url.path not in allowed:
                    return JSONResponse(status_code=409, content={"detail": "Endpoint not validated for imported-state metadata"})
            return await call_next(request)

    def new_existing_session(req):
        if server._state.session is not None:
            raise HTTPException(409, "Close the current session explicitly before importing another project")
        try:
            project = import_project(Path(req.bundle_directory), Path(req.output_directory))
            session = CatalogSession(project, req.config)
        except (ValueError, KeyError, OSError) as error:
            raise HTTPException(422, str(error)) from error
        server._state.session = session
        server._state.scene_dir = project.directory
        return {"status": "state_loaded", "room_id": session.scene.room_id,
                "project_id": project.manifest["project"]["project_id"],
                "scene_directory": str(project.directory), "backend_generation_started": False,
                "initial_object_ids": sorted(map(str, session.scene.objects)),
                "resumed_step_idx": session.resume_record["step_idx"] if session.resume_record else None,
                "scope": "catalog_furniture_construction_v1"}

    def checkpoint_existing_session(req):
        session = server._state.session
        if not isinstance(session, CatalogSession):
            raise HTTPException(409, "No catalog-project session to checkpoint")
        try:
            return session.checkpoint(Path(req.output_directory))
        except (ValueError, KeyError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    # The class is local; give FastAPI the evaluated annotation explicitly.
    new_existing_session.__annotations__["req"] = CatalogRequest
    server.app.post("/session/catalog")(new_existing_session)
    checkpoint_existing_session.__annotations__["req"] = CheckpointRequest
    server.app.post("/session/checkpoint")(checkpoint_existing_session)
    server.app.state.scenefunc_catalog_installed = True
    return server.app, server
