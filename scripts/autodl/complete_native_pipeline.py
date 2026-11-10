"""Extend the inspected decoupled controller with a common manipuland stage.

Floor/furniture/wall/ceiling use the same native constructors and driver loops.
Small objects use the explicit common callback before session teardown. This
does not certify assets, providers, rendering, or final scene requirements.
"""
from datetime import datetime
from pathlib import Path
import time


async def complete_native_pipeline(native, args, *, completion_driver, ledger):
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    scene_dir=args.results_dir/('agent_rollout_'+stamp)/'scene_000'
    scene_dir.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter();human=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    session=native.RolloutSession(scene_dir,args.config,args.prompt)
    try:
        session.start()
        policy=native.ModelPolicy(args.model_dir,max_new_tokens=args.max_new_tokens)
        floor=session.make_floor_plan_env()
        await native.driver_loop(env=floor,policy=policy,max_steps=args.max_steps_per_stage)
        floor.finalize_stage()
        if not session.handover_after_floor_plan(args.room):
            raise RuntimeError('Native floor handover failed')
        logger=native.ConsoleLogger(output_dir=session.scene.scene_dir)
        for stage in ('furniture','wall','ceiling'):
            env=session.make_roomscene_env(stage,args.render_gpu_id,logger)
            try:
                await native.driver_loop(env=env,policy=policy,max_steps=args.max_steps_per_stage)
                env.finalize_stage()
            finally:
                env.cleanup()
        with ledger.stage('manipuland'):
            with ledger.operation('finalize','manipuland') as row:
                result=await completion_driver(native,session,policy,args)
                if not isinstance(result,dict):
                    raise TypeError('Common completion stage must return an execution receipt')
                row['stage_receipt']=result
    finally:
        try:
            native.write_run_metrics(scene_dir.parent,tick,human,args)
        finally:
            session.close()


def common_manipuland_driver(tool_binding, *, max_steps_per_parent=4,
                            max_tool_calls=48, max_seconds=600):
    """Build the real native agent only after outer service qualification.

    Uses the already-loaded base through the local tool binding; no API client,
    adapter update, native SDK Runner, or second policy model is introduced.
    Native constructors may require services: outer preflight remains required.
    """
    async def run(native,session,policy,args):
        if tool_binding.backend is None:
            raise RuntimeError('Shared local base model has not been bound')
        from scenesmith.manipuland_agents.stateful_manipuland_agent import StatefulManipulandAgent
        from native_manipuland_runner import NativeStageReceipt, run_native_manipulands
        from local_manipuland_workflow import LocalManipulandWorkflow
        out=session.scene.scene_dir/'rollout_manipuland'
        logger=native.ConsoleLogger(output_dir=session.scene.scene_dir)
        agent=native.BaseExperiment.build_manipuland_agent(cfg_dict=session.cfg_dict,
            compatible_agents={'stateful_manipuland_agent':StatefulManipulandAgent},
            logger=logger,render_gpu_id=args.render_gpu_id)
        try:
            receipt=NativeStageReceipt(out/'stage_receipt.json')
            workflow=LocalManipulandWorkflow(tool_binding.backend,out=out/'local_policy',
                requested_model=agent.cfg.openai.model,max_steps_per_parent=max_steps_per_parent,
                max_tool_calls=max_tool_calls,max_seconds=max_seconds)
            session.scene=await run_native_manipulands(agent,session.scene,
                workflow=workflow,receipt=receipt)
            logger.log_scene(scene=session.scene,name='scene_after_manipulands')
            export=session.scene.scene_dir/'final_manipuland'/'manipuland.blend'
            native._save_roomscene_blend(agent,export)
            return dict(status=receipt.data['status'],receipt=str(receipt.path),
                export=str(export),full_scene_success=None,
                scope='Common local-policy adaptation, not upstream multi-agent SDK reproduction')
        finally:
            agent.cleanup()
    return run
