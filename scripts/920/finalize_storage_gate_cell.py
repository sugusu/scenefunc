"""Verify and export one finished cell; never rerun inference or edits."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();R=Path(__file__).resolve().parents[2];P=a.public.resolve();T=R.parent/'runtime'/P.name;A=a.arm
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(P/A/'PLAN.json');res=read(T/A/'RESULT.json');assert res['status']=='completed_pending_review';bundle=T/A/'final_checkpoint';native=str(R.parent/'envs/scene_designer311_native/bin/python')
restore=dict(bundle=str(bundle),manifest_sha256=sha(bundle/'manifest.json'),designer=plan['designer'],scenesmith=plan['scenesmith'],expected_tool_calls=res['native_calls'],code_sha256={**plan['code_sha256'],'verify_storage_gate_terminal.py':sha(R/'scripts/920/verify_storage_gate_terminal.py')})
export=dict(bundle=str(bundle),manifest_sha256=sha(bundle/'manifest.json'),code_sha256=plan['code_sha256'],include_object_types=sorted({x['object_type'] for x in read(T/A/'final_state.json')['objects'].values()}),budget=dict(output_max_bytes=100000000))
for kind,cfg,script in [('restore',restore,'verify_storage_gate_terminal.py'),('export',export,'export_catalog_geometry.py')]:
 path=P/A/(kind.upper()+'_PLAN.json')
 with path.open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
 subprocess.run([sys.executable,str(R/'scripts/920/run_stage.py'),'--receipt-dir',str(T/(A+'_'+kind+'_receipt')),'--timeout','180','--',native,str(R/'scripts/920'/script),'--plan',str(path),'--out',str(T/(A+'_'+kind))],check=True,cwd=R)
subprocess.run([native,str(R/'scripts/920/archive_storage_gate_cell.py'),'--public',str(P),'--arm',A],check=True,cwd=R)
