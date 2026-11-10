"""Validate actual image decoding and callback feedback with explicit doubles."""
import argparse
import asyncio
import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from local_manipuland_workflow import LocalManipulandWorkflow, observation_parts, output_value
from local_tool_vlm import LocalToolVLMBackend, prepare_tool_messages


@dataclass
class ImageDouble:
    image_url:str
    type:str='image'


@dataclass
class TextDouble:
    text:str
    type:str='text'


async def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    raw=a.image.read_bytes();url='data:image/png;base64,'+base64.b64encode(raw).decode()
    state={'fixture_moves':0};seen=[]
    class BackendDouble(LocalToolVLMBackend):
        def __init__(self):self.rows=[]
        def create_completion(self,**kwargs):
            converted,images,records=prepare_tool_messages(kwargs['messages'],kwargs['response_format'])
            try:
                assert len(images)==1 and records[0]['sha256']==hashlib.sha256(raw).hexdigest()
                context=json.loads(kwargs['messages'][-1]['content'][-1]['text'])
                seen.append(context['actual_state']['fixture_moves'])
                self.rows.append(dict(scope='Scripted output, no inference',images=records))
                if len(self.rows)==1:return '{"stop":"wrong-type"}'
                if context['actual_state']['fixture_moves']==0:
                    return json.dumps(dict(stop=False,reason='fixture edit',tool='fixture_move',arguments={}))
                return json.dumps(dict(stop=True,reason='fixture stop',tool=None,arguments={}))
            finally:
                for image in images:image.close()
    def tool(name,callback):
        async def invoke(ctx,args):
            assert ctx.tool_name==name
            return callback()
        return SimpleNamespace(name=name,description='Test double, no geometry',
            params_json_schema={'type':'object','properties':{},'additionalProperties':False},on_invoke_tool=invoke)
    def move():
        state['fixture_moves']+=1
        return dict(success=True,scope='Integer fixture mutation only')
    tools=[tool('observe_scene',lambda:[ImageDouble(url),TextDouble('Previously rendered image used for codec testing only')]),
           tool('get_current_scene_state',lambda:dict(state)),tool('fixture_move',move)]
    backend=BackendDouble()
    workflow=LocalManipulandWorkflow(backend,out=a.out/'workflow',requested_model='fixture',
        max_steps_per_parent=3,max_tool_calls=8,context_factory=lambda **kw:SimpleNamespace(**kw))
    agent=SimpleNamespace(scene=SimpleNamespace(text_description='Fixture workflow; not a generated scene'),
        current_furniture_selection=SimpleNamespace(suggested_items=[],prompt_constraints='fixture only',style_notes=''))
    result=await workflow(agent,'fixture_parent',tools)
    assert result['status']=='policy_stop' and result['policy_calls']==3 and result['tool_calls']==7
    assert seen==[0,0,1] and len(backend.rows)==3
    assert [r['status'] for r in workflow.rows]==['rejected','action_returned','policy_stop']
    for bad in ([str(ImageDouble(url))],[{'type':'image','image_url':'https://example.invalid/a.png'}],
                [{'type':'text','text':'No image'}]):
        try:observation_parts(bad)
        except (TypeError,ValueError):pass
        else:raise AssertionError('Unsupported media accepted')
    try:output_value(object())
    except TypeError:pass
    else:raise AssertionError('Opaque object converted to string')
    # Shared budget is cumulative across parents, not reset for each surface.
    try:await workflow(agent,'second_parent',tools)
    except RuntimeError as exc:assert 'tool budget' in str(exc)
    else:raise AssertionError('Stage budget silently reset')
    summary=dict(status='passed',actual_image_decoding=True,
        image_path=str(a.image),image_sha256=hashlib.sha256(raw).hexdigest(),
        observed_fixture_states=seen,scripted_backend_calls=len(backend.rows),
        actual_language_model_calls=0,actual_native_sdk_tools=0,actual_geometry_calls=0,
        actual_new_scenes=0,paid_api_calls=0,shared_budget_enforced=True,
        sdk_types_tested=False,
        scope='Dataclass output doubles and actual PIL image decoding; no SDK installation or inference',
        result=result)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
