"""Exactly four formal attempts, sequentially guarded; never redraw failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-id',default='P920-DD-v1_20261104T083214Z');a=p.parse_args()
    base=a.root/'runtime'/a.run_id/'model_runs'
    rows=[]
    for label in ['A_G','A_U','B_G','B_U']:
        target=base/('formal_'+label)
        if target.exists():
            launch=json.loads((target/'launch.json').read_text())
            assert launch['status']!='running','Inspect active or interrupted attempt; never restart it automatically'
            code=launch['exit_code']
            print('keep existing attempt',label,code,flush=True)
        else:
            command=[sys.executable,str(Path(__file__).with_name('comparison_model.py')),'--root',str(a.root),'--run-id',a.run_id,'--kind','formal','--label',label]
            code=subprocess.run(command,check=False).returncode
        rows.append({'label':label,'guard_exit_code':code})
    (base/'formal_execution_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows),flush=True)
    return 0 if all(row['guard_exit_code']==0 for row in rows) else 1


if __name__=='__main__':
    raise SystemExit(main())
