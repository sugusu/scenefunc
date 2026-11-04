"""Apply at most twelve model calls to an immutable draft using fixed assets."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
import traceback


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-id',default='P920-DD-v1_20261104T083214Z')
    p.add_argument('--label',choices=['A_G','A_U','B_G','B_U'],required=True)
    a=p.parse_args()
    run=a.root/'runtime'/a.run_id
    draft=run/'drafts'/a.label[0]
    model=run/'model_runs'/('formal_'+a.label)
    out=run/'revisions'/a.label
    out.mkdir(parents=True,exist_ok=False)
    result=read(model/'result.json')
    report={'label':a.label,'source_draft':str(draft/'checkpoint.json'),
            'source_checkpoint_sha256':hashlib.sha256((draft/'checkpoint.json').read_bytes()).hexdigest(),
            'model_status':result['status'],'tool_calls_attempted':0,'tool_calls_successful':0,'tool_calls_failed':0,
            'asset_scope':'Same fixed 23 selected meshes, identical G/U inventory; not 436-candidate retrieval reproduction'}
    calls=read(model/'parsed_calls.json') if (model/'parsed_calls.json').exists() else []
    error=None
    if result['status']=='completed' and len(calls)<=12:
        try:
            source=run/'work/paris_local_run_results_20261104'
            sys.path.insert(0,str(source))
            import typed_retrieval
            original_rank=typed_retrieval.rank
            lock=read(a.root/'scenefunc/paris/config/assets.lock.json')
            allowed={r['id']:r for r in lock['rows'] if r['selected_records']}
            for row in allowed.values():
                file=a.root/'resources/scenefunc/hssd-models'/row['path']
                assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
            def fixed_rank(description,name,wanted,catalog,measure):
                return original_rank(description,name,wanted,[r for r in catalog if r['id'] in allowed],measure)
            typed_retrieval.rank=fixed_rank
            sys.argv=['dispatch_corrected.py','--resume',str(draft/'checkpoint.json'),'--calls',str(model/'parsed_calls.json'),'--output',str(out/'tools')]
            runpy.run_path(str(source/'dispatch_corrected.py'),run_name='__main__')
            events=read(out/'tools/tool_responses.json')
            assert len(events)==len(calls)<=12
            report.update(tool_calls_attempted=len(events),tool_calls_successful=sum(e['response'].get('success') is True for e in events),
                          tool_calls_failed=sum(e['response'].get('success') is not True for e in events),
                          final_directory=str(out/'tools'),status='executed',new_tool_events=events)
        except Exception as exception:
            error={'type':type(exception).__name__,'message':str(exception),'traceback':traceback.format_exc()}
    else:
        error={'type':'ModelProgramRejected','message':result['status']}
    if error:
        # Failed programs do not disappear from the comparison. Retain the exact
        # starting draft as their evaluated outcome, with the failure attached.
        final=out/'retained_draft';final.mkdir()
        for name in ['checkpoint.json','asset_records.json']:
            shutil.copyfile(draft/name,final/name)
        (final/'partial_scene.glb').symlink_to(draft/'partial_scene.glb')
        report.update(status='failed_retained_draft',error=error,final_directory=str(final))
    (out/'application.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='new_tool_events'},ensure_ascii=False))
    return 0 if report['status']=='executed' else 1


if __name__=='__main__':
    raise SystemExit(main())
