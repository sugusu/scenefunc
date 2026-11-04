"""Copy only explicit small shareable artifacts; retain weights/meshes locally."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();run_id='P920-DD-v1_20261104T083214Z';run=root/'runtime'/run_id;pub=root/'scenefunc/runs/920'/run_id
    result=read(run/'RESULTS.json')
    def copy(src,dst):
        assert src.stat().st_size<3*1024**2,src
        assert src.suffix not in ['.glb','.blend','.safetensors','.pt','.key','.pem']
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    copy(run/'RESULTS.json',pub/'RESULTS.json')
    copy(run/'comparison_inputs/PROTOCOL.json',pub/'COMPARISON_PROTOCOL.json')
    for f in ['GEOMETRY_ENVIRONMENT.json','geometry311_install.log','networkx_install.log','geometry_pinned_install.log']:
        copy(root/'preflight'/f,pub/'evidence'/f)
    artifacts=[]
    for row in result['rows']:
        label=row['label'];final=Path(row['final_directory'])
        dst=pub/'drafts'/label[0] if label.endswith('draft') else pub/'revisions'/label
        for name in ['audit.json','use_measurement.json']:copy(final/name,dst/name)
        copy(final/'views/receipt.json',dst/'render_receipt.json')
        copy(final/'views/current_state.topdown.png',dst/'topdown.png')
        if not label.endswith('draft'):
            copy(run/'revisions'/label/'application.json',dst/'application.json')
            copy(final/'tool_responses.json',dst/'tool_responses.json')
        artifacts.append({'label':label,'scene_path':str(final/'partial_scene.glb'),'scene_sha256':sha(final/'partial_scene.glb'),
                          'scene_bytes':(final/'partial_scene.glb').stat().st_size,'checkpoint_path':str(final/'checkpoint.json'),
                          'checkpoint_sha256':sha(final/'checkpoint.json'),'topdown':str((dst/'topdown.png').relative_to(pub)),
                          'recovery':'Rebuild the matching historical unrevised checkpoint from the locked licensed assets, then replay this label\'s saved parsed model calls with apply_revision.py; never rerun the model to reconstruct this result.'})
    for directory in (run/'model_runs').iterdir():
        if not directory.is_dir():continue
        for name in ['result.json','launch.json','raw_output.txt','output_ids.json','parsed_calls.json','model.log']:
            if (directory/name).exists():copy(directory/name,pub/'model_runs'/directory.name/name)
    for label in ['A_G','A_U','B_G','B_U']:
        for name in ['receipt.json','feedback.json']:
            copy(run/'comparison_inputs'/label/name,pub/'comparison_inputs'/label/name)
    for directory in (run/'execution').iterdir():
        if not directory.is_dir():continue
        copy(directory/'receipt.json',pub/'execution'/directory.name/'receipt.json')
        if 'measure_' in directory.name or directory.name.startswith(('apply_','score_')):
            copy(directory/'stage.log',pub/'execution'/directory.name/'stage.log')
    write(pub/'ARTIFACTS.json',{'server_root':str(root),'files':artifacts,'large_files_in_git':False})
    status=read(pub/'STATUS.json');status.update(updated_at=datetime.now(timezone.utc).isoformat(),stage='ab_comparison_complete',status='partial',
        activity='All four formal attempts and six scene scores finished; final artifact acceptance pending',
        latest_verifiable_artifacts=['RESULTS.json','ARTIFACTS.json','COMPARISON_PROTOCOL.json'],
        next_step='Verify the committed evidence and final local artifacts; publish final acceptance.',
        failure_reasons=[{'code':'UNSUPPORTED_PHYSICS_TOOL','count':3,'effect':'No state change in A-G, A-U, B-U'},
                         {'code':'NONEXISTENT_OBJECT_ID','count':1,'effect':'B-G bookcase removal rejected; other five deletions executed'}],
        final_experiment_exit_status='All four model children exited 0; all scene postprocessing exited 0; four tool calls returned failure',
        formal_capacity_validated='Actual four inputs completed on one authorized 3090 without OOM; output cap 2048, realized outputs 249/380/291/284. Capacity beyond realized lengths is estimated, not exhaustively measured.')
    status['completed_planned']['formal_revision_rounds']={'completed':4,'planned':4}
    status['completed_planned']['furnished_scenes']={'completed':6,'planned':6}
    status['actual_tool_calls']={'attempted':9,'successful':5,'failed':4}
    status['engineering_attempts_total']={'attempted':3,'saved_success':2,'failed_serialization':1,'formal':False}
    status['scientific_conclusion']='No meaningful use repair observed. A is unchanged; B-U is unchanged; B-G deletes required dining furniture, so fewer collisions are not a quality gain. No training or method superiority established.'
    write(pub/'STATUS.json',status)
    write(pub.parent/'latest.json',{'run_id':run_id,'stage':status['stage'],'status':status['status'],'updated_at':status['updated_at'],'run_directory':str(root),'status_path':str((pub/'STATUS.json').relative_to(root/'scenefunc'))})
    records=[]
    for f in sorted(pub.rglob('*')):
        if f.is_file() and f.name not in ['STATUS.json','README.md','FINAL_ACCEPTANCE.json','MANIFEST.json']:
            records.append({'path':str(f.relative_to(pub)),'bytes':f.stat().st_size,'sha256':sha(f)})
    write(pub/'MANIFEST.json',{'files':records,'exclusions':['STATUS.json','README.md','FINAL_ACCEPTANCE.json','MANIFEST.json']})
    print(json.dumps({'evidence_files':len(records),'formal_attempts':4,'scenes':6}))


if __name__=='__main__':main()
