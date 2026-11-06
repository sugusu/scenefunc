"""Recover actual call accounting after a probe validator failure; no model calls."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from jsonschema import Draft202012Validator
from shared_action_contract import schemas_from_observation, strict_loads


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists()
    protocol=read(a.run/'PROTOCOL.json');study=Path(protocol['source_study'])
    schema_source=a.run/'inputs/clothing_build_07/request.json'
    schemas=schemas_from_observation(dict(system=read(schema_source)[0]['content']))
    byname={s['name']:s['parameters'] for s in schemas}
    rows=[];seconds=0.
    for cell in protocol['cells']:
        directory=Path(cell['plan']).parent;result_path=directory/'model/RESULT.json'
        if not result_path.exists():
            rows.append(dict(id=cell['id'],status='not_started'));continue
        result=read(result_path);receipt=read(directory/'receipt/receipt.json');seconds+=receipt['elapsed_seconds']
        rawpath=directory/'model/raw_output.txt';raw=rawpath.read_text()
        blocks=re.findall(r'<tool_call>(.*?)</tool_call>',raw,re.S)
        row=dict(id=cell['id'],status=result['status'],model_calls=result['model_calls'],output_sha256=sha(rawpath),
            explicit_think_block=bool(re.search(r'<think>.*?</think>',raw,re.S)),validation_errors=[],schema_valid=False,
            scene_executed=False,adapter_reload_exact=result.get('adapter_reload_exact'))
        if len(blocks)==1:
            call=strict_loads(blocks[0]);row['call']=call
            if call.get('name') in byname:
                row['validation_errors']=[e.message for e in Draft202012Validator(byname[call['name']]).iter_errors(call['arguments'])]
                row['schema_valid']=not row['validation_errors']
            else:row['validation_errors'].append('Unknown tool')
            context=read(Path(cell['request']).with_name('CONTEXT.json'));objects=context['objects']
            ids=set(objects if isinstance(objects,dict) else (o['object_id'] for o in objects))
            row['known_object_references']=all(v in ids for k,v in call['arguments'].items() if k in ('object_id','object_a_id','object_b_id'))
            asset=call['arguments'].get('asset_id')
            row['known_asset_reference']=None if asset is None else asset in {x['asset_id'] for x in context.get('asset_catalog',[])}
            row['legal_candidate']=row['schema_valid'] and row['known_object_references'] and row['known_asset_reference'] is not False
        else:row['validation_errors'].append('Expected one tool block')
        rows.append(row)
    representations=[]
    for row in read(study/'DATASET.json')['rows']:
        docpath=Path(row['documents']['action_only']['path']);doc=read(docpath);context=json.loads(doc['task_context']['scene_description'])
        representations.append(dict(id=row['id'],source_sha256=sha(docpath),has_full_tool_schemas='Available executable tools:' in doc['system'],
            objects_container=type(context['objects']).__name__,asset_catalog_present='asset_catalog' in context))
    result=dict(recorded_at=datetime.now(timezone.utc).isoformat(),status='posthoc_accounting_complete_probe_stopped_after_validator_failure',
        protocol_sha256=sha(a.run/'PROTOCOL.json'),planned_calls=6,actual_model_calls=sum(x.get('model_calls',0) for x in rows),
        actual_model_subprocess_seconds=seconds,not_started=sum(x['status']=='not_started' for x in rows),
        explicit_think_blocks=sum(x.get('explicit_think_block',False) for x in rows),legal_candidates=sum(x.get('legal_candidate',False) for x in rows),
        training_updates=0,native_edits=0,rows=rows,training_representations=representations,
        representation_summary=dict(full_tool_schema=sum(x['has_full_tool_schemas'] for x in representations),
            text_only_signature=sum(not x['has_full_tool_schemas'] for x in representations),objects_container=dict(Counter(x['objects_container'] for x in representations))),
        schema_validation_provenance=dict(source=str(schema_source),sha256=sha(schema_source),posthoc_not_added_to_Waterloo_input=True,
            limit='Native parameter schema from the recorded AIR prompt used only to inspect saved output; not a replay or a new model comparison.'),
        runner_failure='The orchestrator assumed every training prompt had Available executable tools JSON and dict objects. Waterloo uses a textual move signature and list objects. It failed after the fourth model call, before validation.',
        original_summary_correction='Original SUMMARY.json counts three validated rows, not all four actual model calls. This file is authoritative for actual call/time accounting.',
        interpretation='Three-arm AIR block completed: all three generate explicit think but use an unavailable source asset ID. Fourth/base Waterloo output has rotation instead of yaw. Two adapters on Waterloo were never called. No full two-example comparison, training fit, room benefit, or causal explanation established.')
    assert result['actual_model_calls']==4 and result['not_started']==2
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['actual_model_calls','actual_model_subprocess_seconds','explicit_think_blocks','legal_candidates','representation_summary']}))


if __name__=='__main__':main()
