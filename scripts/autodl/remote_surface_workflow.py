"""Same task-grounded native surface loop with an explicit remote backend.

Only initialization/transport changes; inherited real observation, invocation,
schema checking and source-task transform stay the existing common baseline.
No fake model/processor facade and no label/coordinate/asset answer injection.
"""
from pathlib import Path
from local_manipuland_workflow import sdk_context
from task_grounded_completion import TaskGroundedManipulandWorkflow,task_context


class RemoteSurfaceWorkflow(TaskGroundedManipulandWorkflow):
    def __init__(self,backend,*,out,requested_model,original_user_request,
                 max_steps_per_parent=2,max_tool_calls=24,max_seconds=300):
        if not callable(getattr(backend,'create_completion',None)) or not isinstance(backend.rows,list):
            raise TypeError('Explicit metered completion transport required')
        if min(max_steps_per_parent,max_tool_calls,max_seconds)<=0:raise ValueError('Positive budgets required')
        task_context(original_user_request,'')
        self.backend=backend;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.requested_model=requested_model;self.steps=max_steps_per_parent
        self.cap=max_tool_calls;self.seconds=max_seconds;self.context_factory=sdk_context
        self.rows=[];self.tool_calls=[];self.started=None
        self.original_user_request=original_user_request
