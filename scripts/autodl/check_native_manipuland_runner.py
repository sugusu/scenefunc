"""Execute the pinned upstream stage body with explicit service/geometry doubles.

This is a lifecycle/exception test, not a generated scene or a placement score.
"""
import argparse
import ast
import asyncio
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from native_manipuland_runner import NativeStageReceipt, run_native_manipulands


class Scene:
    def __init__(self, objects, generation=0):
        self.objects=objects;self.generation=generation;self.scene_dir=Path('/not-used')
    def get_object(self, key):return self.objects.get(str(key))


async def case(source, out, *, mode):
    events=[]
    logger=logging.getLogger('native_stage_fixture_'+mode)
    logger.setLevel(logging.DEBUG); logger.propagate=False
    code=ast.parse(source.read_text())
    cls=next(n for n in code.body if isinstance(n,ast.ClassDef) and n.name=='StatefulManipulandAgent')
    fn=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='add_manipulands')
    fn.returns=None
    for arg in fn.args.args:arg.annotation=None
    mod=ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    def postprocess(full_scene, furniture_id, **kwargs):
        events.append(['postprocess',str(furniture_id)])
        return Scene(full_scene.objects,full_scene.generation+1)
    scope=dict(console_logger=logger, custom_span=lambda **_:nullcontext(),
        SupportSurfaceExtractionConfig=SimpleNamespace(from_config=lambda cfg:cfg),
        extract_and_propagate_support_surfaces=lambda **k:k['furniture_object'].support_surfaces,
        apply_per_furniture_postprocessing=postprocess)
    exec(compile(mod,str(source),'exec'),scope)
    class Agent:
        add_manipulands=scope['add_manipulands']
        def __init__(self):
            self.cfg=OmegaConf.create(dict(context_furniture=dict(enabled=False),
                context_image_generation=dict(enabled=False), support_surface_extraction={},
                per_furniture_postprocessing=dict(enabled=True,simulation=dict(save_html=False))))
            self.rendering_manager=SimpleNamespace(clear_cache=lambda:events.append(['clear_cache']))
        async def _analyze_furniture_for_placement(self,scene):
            ids=[] if mode=='empty' else ['parent_a'] if mode=='clean' else ['parent_a','parent_b','missing_parent','parent_c']
            return [SimpleNamespace(furniture_id=i,suggested_items=['fixture object'],
                prompt_constraints='fixture constraint',style_notes='fixture style',context_furniture_ids=[]) for i in ids]
        def _setup_furniture_agents(self,**kwargs):raise AssertionError('Original SDK agent setup reached')
        async def _run_furniture_workflow(self,*args):raise AssertionError('Original SDK Runner reached')
        def _setup_furniture_context(self,selection):self.current_furniture_selection=selection
        def _generate_manipuland_context_image(self):
            assert not self.cfg.context_image_generation.enabled
            return None
        def _initialize_checkpoint_state(self):events.append(['checkpoint_reset'])
        def _create_designer_tools(self,*,current_furniture_id,support_surfaces):
            assert support_surfaces
            events.append(['native_tool_builder_double',str(current_furniture_id)])
            return [SimpleNamespace(name='fixture_native_placement_tool')]
    agent=Agent()
    scene=Scene({i:SimpleNamespace(description='fixture furniture',
        support_surfaces=[] if i=='parent_b' else [SimpleNamespace(surface_id='surface_'+i)])
        for i in ('parent_a','parent_b','parent_c')})
    async def workflow(actual_agent, furniture_id, designer_tools):
        assert actual_agent is agent
        assert [t.name for t in designer_tools]==['fixture_native_placement_tool']
        events.append(['workflow_double',str(furniture_id)])
        if furniture_id=='parent_c':raise RuntimeError('Deliberate workflow failure')
        return dict(model_calls=0,tool_calls=0,scope='No inference or geometry in fixture')
    receipt=NativeStageReceipt(out/'receipt.json')
    result=await run_native_manipulands(agent,scene,workflow=workflow,receipt=receipt)
    assert all(n not in agent.__dict__ for n in ('_setup_furniture_agents',
        '_run_furniture_workflow','_analyze_furniture_for_placement'))
    assert not logger.handlers and result is agent.scene
    status=receipt.data['status'];rows=receipt.data['selection']
    if mode=='clean':
        assert status=='native_workflows_returned_quality_unassessed'
        assert result is not scene and result.generation==1
        assert rows[0]['status']=='workflow_returned'
    elif mode=='partial':
        assert status=='returned_with_unresolved_items'
        assert [r['status'] for r in rows]==['workflow_returned',
            'native_skipped_or_preparation_failed','native_skipped_or_preparation_failed','workflow_failed']
        assert receipt.data['native_errors'] and len(receipt.data['native_warnings'])==2
        assert result.generation==1
    else:
        assert status=='selection_empty_unverified' and result is scene
    assert receipt.data['full_scene_success'] is None
    (out/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    return dict(mode=mode,status=status,actual_model_calls=0,actual_geometry_calls=0,
        actual_new_scenes=0,scope='Pinned native stage body with doubles only')


async def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for mode in ('clean','partial','empty'):
        rows.append(await case(a.source,a.out/mode,mode=mode))
    summary=dict(status='passed',source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        cases=rows,scope='Native control-flow integration; no SDK/model/physical generation',
        full_runtime_ready=False)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
