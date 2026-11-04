"""Add native existing-room input to the fixed Scene-Designer HTTP application.

Current executable scope is an explicit existing-object furniture revision.
The actual upstream session, agents, tools, memory and five-view renderer are
used. Acquisition and other stages need their resource contracts separately.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import types

from native_project import import_project, read_json, sha256


TOOLS = ("get_current_scene_state", "move_furniture_tool", "remove_furniture_tool", "check_facing_tool")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def create_app(designer: Path, scenesmith: Path):
    os.environ["SCENESMITH_ROOT"] = str(scenesmith.resolve())
    sys.path.insert(0, str(designer.resolve()))
    from src import runtime  # Adds the fixed runtime's source directories.
    import tool_server as server
    from agent_rollout_decoupled import RolloutSession
    from Designer.replay_chained import compose_full_cfg
    from room_summary import from_house_layout_dict
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    if getattr(server.app.state, "scenefunc_existing_installed", False):
        raise RuntimeError("Existing-project routes already installed in this process")

    class ExistingSession(RolloutSession):
        def __init__(self, project, config):
            super().__init__(project.directory, config, project.task["design_request"], stub_assets=False)
            self.project = project
            self.scene = project.room
            self.house_layout = project.house.layout
            self.room_geometry = self.scene.room_geometry
            self.scene.action_log_path = self.scene.scene_dir / "native_actions.json"
            self.room_summary = from_house_layout_dict(self.house_layout.to_dict(), self.scene.room_id)
            self.cfg, self.cfg_dict = compose_full_cfg(config, self.scene_dir)
            # This phase edits existing native objects. No router or generation
            # backends are started; unavailable tools are removed from both the
            # dispatch map and advertised schemas.
            cfg = self.cfg_dict["furniture_agent"]
            cfg["asset_manager"]["router"]["enabled"] = False
            for strategy in cfg["asset_manager"]["router"]["strategies"].values():
                strategy["enabled"] = False
            cfg["rendering"]["blender_server_port_range"] = [18100, 18120]
            cfg["collision_geometry"]["server_port_range"] = [18130, 18150]
            self.cfg_dict["experiment"]["num_workers"] = max(1, ((os.cpu_count() or 1) + 3) // 4)
            self.active_environment = None

        def start(self):
            raise RuntimeError("An imported session must not regenerate its floor plan")

        def make_floor_plan_env(self):
            raise ValueError("Existing-room scope fixes the supplied floor plan")

        def make_roomscene_env(self, stage, render_gpu_id, logger, skip_render=False):
            if stage != "furniture" or skip_render or render_gpu_id is not None:
                raise ValueError("Current existing-room contract: furniture, real five views, CPU rendering")
            if os.environ.get("OPENSCENEAGENT_STUB_ASSETS") == "1":
                raise ValueError("Stub execution is outside the existing-room contract")
            env = super().make_roomscene_env(stage, render_gpu_id, logger, skip_render=False)
            self.active_environment = env
            missing = set(TOOLS) - set(env.tool_map)
            if missing:
                env.cleanup()
                raise ValueError(f"Required real native tools absent: {sorted(missing)}")
            env.tool_map = {name: env.tool_map[name] for name in TOOLS}
            schemas = [{"name": name, "description": tool.description,
                        "parameters": tool.params_json_schema} for name, tool in env.tool_map.items()]
            schema_text = json.dumps(schemas, ensure_ascii=False, sort_keys=True)
            system = ("Revise the supplied room for its original design request using the existing objects. "
                      "Keep the recorded object identities and immutability constraints. "
                      "Inspect the current state, latest real receipts and five views. "
                      "A successful edit does not establish collision freedom or activity suitability. "
                      "Report unverified requirements as unknown.\nAvailable executable tools:\n" + schema_text)
            initial_context = self.project.visible_initial_context()
            (self.scene_dir / "session_initial_context.json").write_text(json.dumps(initial_context, indent=2) + "\n")
            (self.scene_dir / "session_configuration.json").write_text(json.dumps({
                "scope": "existing_objects_furniture_revision", "config": self.config,
                "effective_config": self.cfg_dict, "tool_names": list(TOOLS),
                "source_manifest_sha256": sha256(self.project.directory / "manifest.json"),
                "backend_generation_started": False, "original_request": self.project.task["original_request"],
            }, indent=2) + "\n")
            original_observation = env.observation
            original_step = env.step

            def observation(bound_env):
                before = self.scene.to_state_dict()
                context = self.project.visible_initial_context()
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
                record = {"step_idx": bound_env.step_idx, "tool_calls": tool_calls,
                          "results": result, "before_sha256": canonical_hash(before),
                          "after_sha256": canonical_hash(self.scene.to_state_dict())}
                with (self.scene_dir / "session_receipts.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                return result

            env.observation = types.MethodType(observation, env)
            env.step = types.MethodType(step, env)
            return env

        def close(self):
            if self.active_environment is not None:
                self.active_environment.cleanup()
                self.active_environment = None
            super().close()

    class ExistingRequest(BaseModel):
        bundle_directory: str
        output_directory: str
        config: str = "scene_designer"

    # Serialize the upstream singleton application's requests. Concurrent
    # observation and edit would invalidate image/state provenance.
    serial = asyncio.Lock()
    @server.app.middleware("http")
    async def serialize_existing_requests(request, call_next):
        async with serial:
            if isinstance(server._state.session, ExistingSession):
                allowed = {"/healthz", "/session/existing", "/session/stage", "/session/observation",
                           "/session/step", "/session/cleanup", "/session/close"}
                if request.url.path not in allowed:
                    return JSONResponse(status_code=409, content={"detail": "Endpoint not validated for imported-state metadata"})
            return await call_next(request)

    def new_existing_session(req):
        if server._state.session is not None:
            raise HTTPException(409, "Close the current session explicitly before importing another project")
        try:
            project = import_project(Path(req.bundle_directory), Path(req.output_directory))
            session = ExistingSession(project, req.config)
        except (ValueError, KeyError, OSError) as error:
            raise HTTPException(422, str(error)) from error
        server._state.session = session
        server._state.scene_dir = project.directory
        return {"status": "state_loaded", "room_id": session.scene.room_id,
                "project_id": project.manifest["project"]["project_id"],
                "scene_directory": str(project.directory), "backend_generation_started": False,
                "initial_object_ids": sorted(map(str, session.scene.objects)),
                "scope": "existing_objects_furniture_revision"}

    # The class is local; give FastAPI the evaluated annotation explicitly.
    new_existing_session.__annotations__["req"] = ExistingRequest
    server.app.post("/session/existing")(new_existing_session)
    server.app.state.scenefunc_existing_installed = True
    return server.app, server
