"""Package review evidence without duplicating large native scene companions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


NAMES='''audit_native_collision_frames audit_redundant_old_archives certify_native_floor_contacts
check_native_grpo_flat_group clear_rebuildable_python_caches compact_verified_297_bundle complete_native_model_host
diagnose_native_sdf_namespace encode_live_furniture_observation export_complete_trial_glb native_trial_evidence
owned_native_retrieval prepare_complete_native_runtime prepare_complete_native_trial prepare_corrected_native_trial
prepare_live_furniture_observation probe_complete_native_services probe_native_floor_binding probe_native_mutable_asset_rollback
recover_native_partial_glb remote_surface_workflow repack_native_trial_evidence replay_partial_native_trial
report_complete_native_trial restore_native_floor_binding retire_unused_cuda118_compiler run_complete_native_trial
package_complete_native_delivery verify_complete_native_delivery'''.split()


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);a=p.parse_args()
    stage=a.stage.resolve();repo=Path(__file__).resolve().parents[2]
    manifest=stage/'delivery_manifest.json';archive=stage/'evidence.zip'
    if manifest.exists() or archive.exists():raise FileExistsError('No overwrite of published delivery')
    rows=[];sources=[]
    for name in NAMES:
        path=repo/'scripts/autodl'/(name+'.py');ast.parse(path.read_text())
        sources.append(dict(path=str(path.relative_to(repo)),bytes=path.stat().st_size,sha256=sha(path)))
    for path in sorted(stage.rglob('*')):
        if not path.is_file() or path.is_symlink():continue
        rel=path.relative_to(stage)
        if rel.parts[0]=='runtime' or path.name in ('delivery_manifest.json','evidence.zip','delivery_verification.json'):continue
        if '__pycache__' in rel.parts:continue
        large=path.suffix in ('.xz','.glb') or (rel.parts[0].startswith('services_actual_') and 'views' in rel.parts)
        rows.append(dict(path=str(path.relative_to(repo)),bytes=path.stat().st_size,sha256=sha(path),
                         disposition='local_companion' if large else 'review_package_and_git'))
    doc=dict(stage=str(stage),files=rows,scripts=sources,scope='Full local companions plus compact reproducible review package',
             no_large_scene_or_model_payload_in_review_zip=True,
             original_file_bytes_retained_in_companions=True,compressed_original_ZIP_byte_identity_not_promised=True,
             goal_achieved=False)
    manifest.write_text(json.dumps(doc,indent=2)+'\n')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.write(manifest,arcname=str(manifest.relative_to(repo)))
        for row in rows:
            # Images stay next to the HTML. Avoid a second PNG copy in evidence.
            if row['disposition']=='review_package_and_git' and not row['path'].endswith('.png'):
                z.write(repo/row['path'],arcname=row['path'])
        for row in sources:z.write(repo/row['path'],arcname=row['path'])
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in rows+sources:
            if row['path'] in z.namelist():assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    assert shutil.disk_usage(stage).free>=5*1024**3
    print(json.dumps(dict(files=len(rows),scripts=len(sources),zip_bytes=archive.stat().st_size,
                         archive_sha256=sha(archive),free_GiB=shutil.disk_usage(stage).free/2**30)))


if __name__=='__main__':main()
