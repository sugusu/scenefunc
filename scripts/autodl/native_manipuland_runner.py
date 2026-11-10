"""Reuse native selection/support/postprocessing with a common policy callback.

No constructor, assets, coordinates, model client or solver is provided here.
The caller must supply a qualified native agent and an actual async policy
workflow. This replaces the native SDK planner/designer/critic loop, so it is
an explicit common-baseline adaptation, NOT an unmodified SceneSmith run.
"""
import inspect
import json
import logging
from pathlib import Path
import time

from native_service_contract import selected


class NativeStageReceipt:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = dict(status='not_started', selection=[], native_warnings=[],
            native_errors=[], full_scene_success=None,
            scope='Native parent selection/support/postprocessing; common injected workflow. Returned workflow is not placement success.')
        self.save()

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        tmp.replace(self.path)


class _LogCapture(logging.Handler):
    def __init__(self, receipt):
        super().__init__(logging.WARNING)
        self.receipt = receipt

    def emit(self, record):
        target = 'native_errors' if record.levelno >= logging.ERROR else 'native_warnings'
        self.receipt.data[target].append(record.getMessage())
        self.receipt.save()


async def run_native_manipulands(agent, scene, *, workflow, receipt):
    """Call the original add_manipulands; replace only SDK setup/workflow hooks.

    workflow(agent, furniture_id, designer_tools) must consume actual current
    observations and execute native tools. It must return a JSON-serializable
    accounting receipt, not a success oracle. No scene-specific plan is added.
    Native swallowed exceptions/skips remain visible and never imply success.
    """
    if not (inspect.iscoroutinefunction(workflow) or
            inspect.iscoroutinefunction(getattr(workflow, '__call__', None))):
        raise TypeError('An actual async workflow implementation is required')
    if selected(agent.cfg, 'context_image_generation.enabled', False):
        raise ValueError('Image context generation is not covered by this local workflow')
    names = ('_analyze_furniture_for_placement', '_setup_furniture_agents',
             '_run_furniture_workflow')
    originals = {name: getattr(agent, name) for name in names}
    owned = {name: name in agent.__dict__ for name in names}
    owned_values = {name: agent.__dict__.get(name) for name in names}
    rows = {}; tools = {}

    async def analyze(current_scene):
        selections = await originals['_analyze_furniture_for_placement'](current_scene)
        for selection in selections:
            key = str(selection.furniture_id)
            if key in rows:
                raise ValueError('Duplicate parent selection: '+key)
            row = dict(parent_id=key, status='selected',
                suggested_items=selection.suggested_items,
                prompt_constraints=selection.prompt_constraints,
                style_notes=selection.style_notes)
            rows[key] = row; receipt.data['selection'].append(row)
        receipt.save()
        return selections

    def setup(furniture_id, furniture_description):
        key = str(furniture_id); row = rows[key]
        row['status'] = 'preparing_tools'; receipt.save()
        try:
            parent = agent.scene.get_object(furniture_id)
            if parent is None or not parent.support_surfaces:
                raise ValueError('No actual parent support surfaces at tool construction')
            surfaces = {str(s.surface_id): s for s in parent.support_surfaces}
            tools[key] = agent._create_designer_tools(
                current_furniture_id=furniture_id, support_surfaces=surfaces)
            row.update(status='tools_prepared',
                tool_names=[t.name for t in tools[key]],
                support_surface_ids=list(surfaces))
        except Exception as exc:
            row.update(status='tool_preparation_failed', error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            receipt.save()

    async def execute(furniture_id):
        key = str(furniture_id); row = rows[key]
        row['status'] = 'workflow_running'; receipt.save()
        tick = time.monotonic()
        try:
            result = await workflow(agent, furniture_id, tools[key])
            if not isinstance(result, dict):
                raise TypeError('Workflow must return its execution/accounting receipt')
            # Fail locally for nonfinite/unserializable results rather than
            # losing the journal during later error reporting.
            json.dumps(result, allow_nan=False)
            row.update(status='workflow_returned', workflow=result)
        except Exception as exc:
            row.update(status='workflow_failed', error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            row['workflow_seconds'] = time.monotonic()-tick
            receipt.save()

    # Native logger belongs to the method's actual module, not the wrapper.
    native_logger = agent.add_manipulands.__func__.__globals__.get('console_logger')
    if not isinstance(native_logger, logging.Logger):
        raise ValueError('Cannot observe native swallowed exceptions without its logger')
    capture = _LogCapture(receipt)
    native_logger.addHandler(capture)
    agent._analyze_furniture_for_placement = analyze
    agent._setup_furniture_agents = setup
    agent._run_furniture_workflow = execute
    receipt.data['status']='running'; receipt.save(); tick=time.monotonic()
    try:
        await agent.add_manipulands(scene=scene)
        for key,row in rows.items():
            if row['status'] == 'selected':
                row['status']='native_skipped_or_preparation_failed'
        incomplete = any(r['status'] != 'workflow_returned' for r in rows.values())
        receipt.data['status'] = ('selection_empty_unverified' if not rows else
            'returned_with_unresolved_items' if incomplete or receipt.data['native_errors'] else
            'native_workflows_returned_quality_unassessed')
        # Native postprocessing may assign a new RoomScene instance. Return
        # that instance so the final exporter cannot silently use the old one.
        return agent.scene
    except BaseException as exc:
        receipt.data.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        receipt.data['wall_seconds']=time.monotonic()-tick
        native_logger.removeHandler(capture)
        for name in names:
            if owned[name]:setattr(agent, name, owned_values[name])
            else:delattr(agent, name)
        receipt.save()
