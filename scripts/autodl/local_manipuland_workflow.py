"""Common bounded local policy for native surface-object tools.

An explicit baseline adaptation: single local policy replaces the upstream
SDK planner/designer/critic, identically for all experimental arms. No scene
answers, asset IDs, coordinates or evaluator labels are supplied by this code.
"""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import time

from local_tool_vlm import LocalToolVLMBackend
from phase_policy import strict_json


def output_value(value):
    """Preserve SDK typed text/images; reject unknown objects instead of str()."""
    if is_dataclass(value) and not isinstance(value, type):
        return output_value(asdict(value))
    if callable(getattr(value, 'model_dump', None)):
        return output_value(value.model_dump(mode='json', exclude_none=True))
    if isinstance(value, dict):
        if any(not isinstance(k,str) for k in value):
            raise TypeError('Tool output dictionary keys must be strings')
        return {k:output_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [output_value(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        json.dumps(value,allow_nan=False)
        return value
    raise TypeError('Unsupported tool output type: '+type(value).__name__)


def observation_parts(value):
    values=output_value(value)
    if not isinstance(values,list):
        raise ValueError('Native observe_scene must return structured image/text list')
    parts=[]
    for item in values:
        if not isinstance(item,dict):
            raise TypeError('Observation item lost structured media')
        if item.get('type') in ('image','image_url','input_image') and isinstance(item.get('image_url'),str):
            if set(item)-{'type','image_url','detail','file_id'} or item.get('file_id') is not None:
                raise ValueError('Unsupported image fields; no silent media loss')
            if not item['image_url'].startswith(('data:image/png;base64,','data:image/jpeg;base64,','data:image/webp;base64,')):
                raise ValueError('Native embedded image required')
            image=dict(url=item['image_url'])
            if item.get('detail') is not None:
                if item['detail'] not in ('auto','low','high'):
                    raise ValueError('Unknown image detail')
                image['detail']=item['detail']
            parts.append(dict(type='image_url',image_url=image))
        elif item.get('type') in ('text','input_text') and isinstance(item.get('text'),str):
            if set(item)-{'type','text'}:raise ValueError('Unsupported text fields')
            parts.append(dict(type='text',text=item['text']))
        else:
            raise ValueError('Unsupported native observation item')
    if not any(p['type']=='image_url' for p in parts):
        raise ValueError('No actual image in native observation; do not infer from text alone')
    return parts


def sdk_context(**kwargs):
    from agents.tool_context import ToolContext
    return ToolContext(context=None, **kwargs)


class LocalManipulandWorkflow:
    def __init__(self, backend, *, out, requested_model, max_steps_per_parent=4,
                 max_tool_calls=48, max_seconds=600, context_factory=sdk_context):
        if not isinstance(backend,LocalToolVLMBackend):
            raise TypeError('Already-loaded LocalToolVLMBackend required; no remote fallback')
        if min(max_steps_per_parent,max_tool_calls,max_seconds)<=0:
            raise ValueError('Positive shared stage budget required')
        self.backend=backend;self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=False)
        self.requested_model=requested_model;self.steps=max_steps_per_parent
        self.cap=max_tool_calls;self.seconds=max_seconds;self.context_factory=context_factory
        self.rows=[];self.tool_calls=[];self.started=None

    def save(self):
        result=dict(turns=self.rows,tool_calls=self.tool_calls,
            full_scene_success=None,tool_call_limit=self.cap,
            scope='Common local surface-object policy. Backend ledger includes actual generation/token costs; observations/tool calls count against this shared stage budget.')
        tmp=self.out/'workflow.tmp'
        tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        tmp.replace(self.out/'workflow.json')

    def check_time(self):
        if self.started is None:self.started=time.monotonic()
        if time.monotonic()-self.started>=self.seconds:
            raise RuntimeError('Common manipuland stage time budget exhausted')

    async def invoke(self, tool_map, name, arguments):
        self.check_time()
        if name not in tool_map:raise ValueError('Unadvertised native tool: '+name)
        if len(self.tool_calls)>=self.cap:raise RuntimeError('Common manipuland tool budget exhausted')
        row=dict(index=len(self.tool_calls),name=name,arguments=arguments,status='started')
        self.tool_calls.append(row);self.save();tick=time.monotonic()
        try:
            args=json.dumps(arguments,ensure_ascii=False,allow_nan=False)
            ctx=self.context_factory(tool_name=name,tool_call_id='manipuland_'+str(row['index']),tool_arguments=args)
            result=await tool_map[name].on_invoke_tool(ctx,args)
            result=output_value(result)
            # Images remain in the native render directory. Store payload hashes,
            # not a second base64 copy, and preserve text/structured tool results.
            if name=='observe_scene':
                parts=observation_parts(result)
                row['output']=[dict(type=p['type'],sha256=hashlib.sha256(p['image_url']['url'].encode()).hexdigest())
                    if p['type']=='image_url' else p for p in parts]
            else:row['output']=result
            row['status']='returned'
            self.check_time()
            return result
        except BaseException as exc:
            row.update(status='raised',error=type(exc).__name__+': '+str(exc));raise
        finally:
            row['seconds']=time.monotonic()-tick;self.save()

    async def __call__(self, agent, furniture_id, designer_tools):
        self.check_time();first=len(self.rows);first_tool=len(self.tool_calls)
        first_model=len(self.backend.rows)
        tool_map={t.name:t for t in designer_tools}
        if len(tool_map)!=len(designer_tools):raise ValueError('Duplicate native tool names')
        if not {'observe_scene','get_current_scene_state'}<=set(tool_map):
            raise ValueError('Native observation and actual state tools are required')
        schemas=[dict(name=t.name,description=t.description,parameters=t.params_json_schema) for t in designer_tools]
        system=('You operate native 3D surface-object tools. Preserve the original room request and existing objects. '
            'Parent suggestions are model proposals, not additional user requirements. Use actual object/surface IDs from tool state. '
            'Return exactly one JSON object: {"stop": boolean, "reason": string, "tool": string or null, "arguments": object}. '
            'For stop=true use tool=null and arguments={}. Stop is a policy decision, not proof of success. '
            'For stop=false select one advertised tool. Do not invent results. Native tool definitions: '+json.dumps(schemas,ensure_ascii=False))
        feedback=[];stop=False
        for step in range(self.steps):
            # Re-observe after actual tool execution; no correct-state reset.
            obs=await self.invoke(tool_map,'observe_scene',{})
            state=await self.invoke(tool_map,'get_current_scene_state',{})
            selection=agent.current_furniture_selection
            context=dict(original_room_request=agent.scene.text_description,
                current_parent_id=str(furniture_id),actual_state=state,
                native_selection_proposal=dict(suggested_items=selection.suggested_items,
                    prompt_constraints=selection.prompt_constraints,style_notes=selection.style_notes),
                previous_execution_feedback=feedback)
            parts=observation_parts(obs)+[dict(type='text',text=json.dumps(context,ensure_ascii=False))]
            messages=[dict(role='system',content=system),dict(role='user',content=parts)]
            row=dict(parent_id=str(furniture_id),step=step,status='started',
                request_sha256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest(),
                public_context=context,tool_schema_sha256=hashlib.sha256(json.dumps(schemas,sort_keys=True).encode()).hexdigest())
            self.rows.append(row);self.save()
            try:
                raw=self.backend.create_completion(model=self.requested_model,messages=messages,
                    reasoning_effort='low',verbosity='low',response_format={'type':'json_object'})
                row['raw_response']=raw
                action=strict_json(raw)
                if (not isinstance(action,dict) or set(action)!={'stop','reason','tool','arguments'} or
                    type(action['stop']) is not bool or not isinstance(action['reason'],str) or
                    not isinstance(action['arguments'],dict)):
                    raise ValueError('Invalid common-policy action schema')
                if action['stop']:
                    if action['tool'] is not None or action['arguments']:
                        raise ValueError('Stop must not hide an executable action')
                    row['status']='policy_stop';stop=True;break
                if not isinstance(action['tool'],str) or action['tool'] not in tool_map:
                    raise ValueError('Unadvertised native tool')
                result=await self.invoke(tool_map,action['tool'],action['arguments'])
                # Avoid base64 media being duplicated into textual feedback.
                result=self.tool_calls[-1]['output']
                feedback.append(dict(tool=action['tool'],arguments=action['arguments'],result=result))
                row['status']='action_returned'
            except (ValueError,TypeError) as exc:
                row.update(status='rejected',error=type(exc).__name__+': '+str(exc))
                feedback.append(dict(rejection=row['error']))
            finally:self.save()
        return dict(status='policy_stop' if stop else 'parent_step_budget_exhausted',
            policy_calls=len(self.rows)-first,tool_calls=len(self.tool_calls)-first_tool,
            backend_calls_including_nested_tools=len(self.backend.rows)-first_model,
            full_scene_success=None,receipt=str(self.out/'workflow.json'))
