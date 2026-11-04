"""Exercise real existing-room HTTP endpoints, native tools and five views."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from existing_project_session import create_app, canonical_hash
from native_project import sha256


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designer", type=Path, required=True)
    p.add_argument("--scenesmith", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(), "begin real session/HTTP/tools/views audit", flush=True)
    # These are local client placeholders, not credentials. All model routes
    # point to an unused loopback port, and model tools are absent from scope.
    for name in ("SCENESMITH_LLM_BASE_URL", "SCENESMITH_LLM_API_KEY", "OPENAI_API_KEY",
                 "GOOGLE_API_KEY", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        os.environ.pop(name, None)
    os.environ.update(OPENAI_API_KEY="disabled-local-validation", OPENAI_BASE_URL="http://127.0.0.1:9/v1",
                      OPENAI_AGENTS_DISABLE_TRACING="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", NO_PROXY="127.0.0.1,localhost", SCENE_RENDER_THREADS="4",
                      SCENE_RENDER_SAMPLES="8", OPENSCENEAGENT_STUB_ASSETS="0")
    cpus = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cpus)
    records, checks = [], []
    result = {"status": "running", "cpu_affinity": cpus, "model_calls": 0, "training_updates": 0,
              "asset_generation_calls": 0, "scientific_quality_evaluation": False,
              "checks": checks, "requests": records}
    def check(name, condition, evidence=None):
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})
        if not condition:
            raise AssertionError(name)
    client = server = None
    try:
        app, server = create_app(args.designer, args.scenesmith)
        from fastapi.testclient import TestClient
        from agent_rollout_decoupled import RolloutSession
        from PIL import Image
        client = TestClient(app)
        def post(route, payload=None, expected=200):
            start = time.monotonic()
            response = client.post(route, json=payload or {})
            data = response.json()
            records.append({"route": route, "request": payload, "status_code": response.status_code,
                            "elapsed_seconds": time.monotonic()-start, "response": data})
            dump(args.out / "HTTP_REQUESTS.json", records)
            if response.status_code != expected:
                raise RuntimeError(f"{route}: expected {expected}, got {response.status_code}: {data}")
            return data
        loaded = post("/session/existing", {"bundle_directory": str(args.bundle.resolve()),
                                           "output_directory": str((args.out / "house").resolve())})
        session = server._state.session
        check("real_rollout_session", isinstance(session, RolloutSession))
        check("selected_native_room_attached", loaded["room_id"] == "store_b" and session.scene is session.project.room)
        check("same_layout_geometry_binding", session.house_layout.room_geometries[session.scene.room_id] is session.room_geometry)
        check("cold_load_has_no_generation_backend", session.servers is None and session.fp_agent is None)
        post("/session/existing", {"bundle_directory": str(args.bundle.resolve()),
                                    "output_directory": str((args.out / "second").resolve())}, expected=409)
        check("active_session_not_replaced", server._state.session is session and not (args.out / "second").exists())
        post("/session/branch/evaluate", {"branches": []}, expected=409)
        post("/session/stage", {"stage": "furniture", "stub_assets": False, "skip_render": False})
        env = server._state.env
        check("actual_upstream_furniture_agent", type(env.agent).__module__ == "scenesmith.furniture_agents.stateful_furniture_agent")
        services = {}
        for name in ("blender_server", "collision_server"):
            service = getattr(env.agent, name)
            process = service._server_process
            services[name] = {"pid": process.pid, "running": process.poll() is None,
                              "start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split(") ",1)[1].split()[19]}
        dump(args.out / "SERVICES_STARTED.json", services)
        check("real_local_services_started", all(v["running"] for v in services.values()))
        before = post("/session/observation")
        dump(args.out / "observation_before.json", before)
        check("five_before_views", len(before["image_paths"]) == 5)
        for path in before["image_paths"]:
            with Image.open(path) as im: im.verify()
        check("explicit_initial_object_visible", "table_0" in before["room_prompt"] and "fixture_box_1" in before["room_prompt"])
        check("scoring_metadata_not_visible", "DO_NOT_EXPOSE_TARGET" not in json.dumps(before))
        check("generation_not_advertised", "generate_assets" not in before["system"])
        initial_state = session.scene.to_state_dict()
        move = {"name": "move_furniture_tool", "arguments": {"object_id": "table_0", "x": -0.7, "y": 0.6,
                                                                  "z": 0.35, "roll": 0.0, "pitch": 0.0, "yaw": 30.0}}
        moved = post("/session/step", {"tool_calls": [move]})
        check("native_tool_move_succeeded", moved["results"][0]["output"]["success"] is True)
        moved_state = session.scene.to_state_dict()
        check("native_move_changes_state", canonical_hash(initial_state) != canonical_hash(moved_state))
        denied = dict(move, arguments=dict(move["arguments"], object_id="north_wall"))
        rejected = post("/session/step", {"tool_calls": [denied]})
        check("native_immutable_rejection", rejected["results"][0]["output"]["success"] is False)
        check("rejected_edit_preserves_state", canonical_hash(session.scene.to_state_dict()) == canonical_hash(moved_state))
        after = post("/session/observation")
        dump(args.out / "observation_after.json", after)
        check("five_after_views", len(after["image_paths"]) == 5)
        for path in after["image_paths"]:
            with Image.open(path) as im: im.verify()
        check("new_views_bound_to_changed_state", canonical_hash(json.loads(Path(after["state_binding"]).read_text())["native_state"]) == canonical_hash(moved_state))
        check("actual_images_change", any(sha256(Path(a)) != sha256(Path(b)) for a,b in zip(before["image_paths"], after["image_paths"])))
        check("live_object_position_visible", '"translation": [-0.7, 0.6, 0.35]' in after["room_prompt"])
        check("real_receipts_present", len((session.scene_dir / "session_receipts.jsonl").read_text().splitlines()) == 2)
        dump(args.out / "final_house_state.json", session.project.house.to_state_dict())
        result.update(status="passed", checks_passed=len(checks), tool_calls_attempted=2, observations=2,
                      final_state_sha256=canonical_hash(moved_state), limitations=[
                          "Synthetic two-room input; no real-project or design-quality result.",
                          "Existing-object furniture stage only; acquisition, wall/ceiling stages and branch restore not validated.",
                          "Move success is not a collision/whole-room usability judgment; no physics score reported."])
    except BaseException as error:
        result.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        if server is not None:
            server._close_all()
        if client is not None:
            client.close()
        # A native export crash may leave a child server alive. Only terminate
        # descendants in this audit's dedicated process group, never other jobs.
        import psutil
        remaining = []
        for process in psutil.process_iter(["pid"]):
            try:
                if process.pid != os.getpid() and os.getpgid(process.pid) == os.getpgrp():
                    remaining.append(process)
                    process.terminate()
            except (ProcessLookupError, PermissionError, psutil.Error):
                pass
        gone, alive = psutil.wait_procs(remaining, timeout=3)
        for process in alive:
            try: process.kill()
            except psutil.Error: pass
        result["extra_owned_children_cleaned"] = [p.pid for p in remaining]
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        dump(args.out / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": len(checks), "failure": result.get("failure")}), flush=True)


if __name__ == "__main__":
    main()
