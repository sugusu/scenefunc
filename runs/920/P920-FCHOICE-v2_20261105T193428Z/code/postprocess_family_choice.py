"""Run bounded endpoint audits/export/cold restore for one completed candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--candidate',choices=['perimeter','divider'],required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];p=repo/'runs/920'/a.run;t=repo.parent/'runtime'/a.run/a.candidate
    native=json.loads((t/'native_stage/receipt.json').read_text());assert native['exit_code']==0
    base=json.loads((p/(a.candidate+'_PLAN.json')).read_text());bundle=t/'native/checkpoint'
    python='/data/wyh/conda/envs/c15_genrec/bin/python'
    geo=str(repo.parent/'envs/geometry311/bin/python');np=str(repo.parent/'envs/scene_designer311_native/bin/python')
    out=p/a.candidate;out.mkdir(exist_ok=True)
    def lock(names): return {n:sha(repo/'scripts/920'/n) for n in names}
    export=dict(bundle=str(bundle),manifest_sha256=sha(bundle/'manifest.json'),
        include_object_types=['furniture'],budget=dict(output_max_bytes=50000000),
        code_sha256=lock(['export_catalog_geometry.py','native_project.py','inspect_native_mesh_contacts.py']))
    restore=dict(bundle=str(bundle),manifest_sha256=sha(bundle/'manifest.json'),designer=base['designer'],
        scenesmith=base['scenesmith'],expected_tool_calls=1,expected_catalog_entries=8,
        code_sha256=lock(['restore_terminal_catalog.py','catalog_project_session.py','native_project.py',
                         'native_checkpoint.py','existing_project_session.py']))
    for name,value in [('EXPORT_PLAN.json',export),('RESTORE_PLAN.json',restore)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    jobs=[('audit',60,[geo,'scripts/920/audit_family_choice_geometry.py','--plan',str(p/'AUDIT_PLAN.json'),
            '--bundle',str(bundle),'--out',str(t/'FINAL_AUDIT.json')]),
        ('export',90,[geo,'scripts/920/export_catalog_geometry.py','--plan',str(out/'EXPORT_PLAN.json'),'--out',str(t/'export')]),
        ('restore',90,[np,'scripts/920/restore_terminal_catalog.py','--plan',str(out/'RESTORE_PLAN.json'),'--out',str(t/'cold_restore')])]
    for label,seconds,command in jobs:
        print(datetime.now(timezone.utc).isoformat(),a.candidate,label,flush=True)
        subprocess.run([python,'scripts/920/run_stage.py','--receipt-dir',str(t/(label+'_stage')),
            '--timeout',str(seconds),'--',*command],cwd=repo,check=True)


if __name__=='__main__':main()
