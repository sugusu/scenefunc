"""Execute, independently measure and render retained outcomes of formal calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--labels',nargs='+',default=['A_G','A_U','B_G','B_U'])
    a=p.parse_args();root=a.root.resolve();scripts=Path(__file__).resolve().parent
    run=root/'runtime/P920-DD-v1_20261104T083214Z';geo=root/'envs/geometry311/bin/python'
    def stage(name,command):
        directory=run/'execution'/name
        if directory.exists():
            record=json.loads((directory/'receipt.json').read_text())
            assert record['status']!='running','Inspect active/interrupted stage before resuming'
            return record['exit_code']
        return subprocess.run([sys.executable,str(scripts/'run_stage.py'),'--receipt-dir',str(directory),'--timeout','900','--',*map(str,command)],check=False).returncode
    def process(label):
        assert label in ['A_G','A_U','B_G','B_U']
        model=json.loads((run/'model_runs'/('formal_'+label)/'result.json').read_text())
        assert model['status']!='running'
        stage('apply_'+label,[geo,scripts/'apply_revision.py','--root',root,'--label',label])
        report=json.loads((run/'revisions'/label/'application.json').read_text())
        final=Path(report['final_directory'])
        audit=stage('audit_'+label,[geo,run/'work/paris_local_run_results_20261104/audit_scene.py','--checkpoint',final/'checkpoint.json','--out',final/'audit.json'])
        measure=stage('measure_'+label,[geo,scripts/'measure_scene.py','--checkpoint',final/'checkpoint.json','--work',run/'work','--out',final/'use_measurement.json'])
        render=stage('render_'+label,['/data/wyh/tools/blender-4.2-lts/blender-4.2.23-linux-x64/blender','-b','--python',scripts/'render_with_blender.py','--','--work',run/'work','--site-packages',root/'envs/geometry311/lib/python3.11/site-packages','--scene',final/'partial_scene.glb','--out',final/'views'])
        return {'label':label,'application_status':report['status'],'final_directory':str(final),'audit_exit':audit,'measure_exit':measure,'render_exit':render}
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(process,a.labels))
    summary=run/('postprocess_'+'_'.join(a.labels)+'.json')
    assert not summary.exists()
    summary.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    main()
