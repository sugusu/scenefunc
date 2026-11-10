"""Audit explicitly authorized old redundant bundles, never raw result files.

Delete only an archive whose EVERY member is independently retained byte-for-
byte in its existing directory tree. Preserve old source/raw/negative evidence,
archive metadata and a reconstruction map. No hardlinks or weight deletion.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ARCHIVES=[
 '/root/autodl-tmp/scene_generation_next_20261016/work/required_instance_reexecution_400_20261019/evidence.zip',
 '/root/autodl-tmp/scene_generation_next_20261016/work/297_actual_audit/evidence.zip',
 '/root/autodl-tmp/scene_generation_next_20261016/work/coupled_completion_429_20261020/baseline_evidence.zip',
 '/root/autodl-tmp/scene_generation_next_20261016/work/coupled_completion_429_20261020/first_branch_evidence.zip']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--delete-authorized',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);cache={};rows=[]
    before=shutil.disk_usage(a.out).free
    for name in ARCHIVES:
        path=Path(name)
        if not path.exists():
            rows.append(dict(path=name,status='already_absent_no_action'));continue
        archive_sha=sha(path);stat=path.stat();index=defaultdict(list)
        for candidate in path.parent.rglob('*'):
            if candidate.is_file() and not candidate.is_symlink() and candidate!=path:
                index[(candidate.name,candidate.stat().st_size)].append(candidate)
        record=dict(path=name,bytes=stat.st_size,sha256=archive_sha,
            mode=stat.st_mode,mtime_ns=stat.st_mtime_ns,members=[])
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                info=dict(name=item.filename,bytes=item.file_size,date_time=list(item.date_time),
                    compress_type=item.compress_type,external_attr=item.external_attr,
                    extra_hex=item.extra.hex(),comment_hex=item.comment.hex())
                if item.is_dir():info.update(status='directory');record['members'].append(info);continue
                data=z.read(item);target=hashlib.sha256(data).hexdigest();matches=[]
                for candidate in index[(Path(item.filename).name,item.file_size)]:
                    key=str(candidate)
                    if key not in cache:cache[key]=sha(candidate)
                    if cache[key]==target:matches.append(key)
                info.update(sha256=target,retained_matching_files=matches,
                    status='retained_exact_copy' if matches else 'unique_or_not_located_keep_archive')
                record['members'].append(info)
        safe=bool(record['members']) and all(r['status'] in ('directory','retained_exact_copy') for r in record['members'])
        record['eligible']=safe
        record['status']='verified_redundant' if safe else 'retained_contains_unmatched_members'
        rows.append(record)
    audit=dict(scope='Only four old redundant bundle candidates; all old raw/source files remain read-only',
        authority='User explicitly authorized deleting genuinely unused old files; no deletion of unique evidence',
        archives=rows,free_before=before,deleted_archives=[])
    # Publish the complete proof before any deletion.
    receipt=a.out/'audit.json';receipt.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    if a.delete_authorized:
        for row in rows:
            if not row.get('eligible'):continue
            path=Path(row['path']);assert sha(path)==row['sha256']
            for m in row['members']:
                if m['status']=='directory':continue
                assert any(Path(p).is_file() and sha(p)==m['sha256'] for p in m['retained_matching_files'])
            # No old raw/source file is removed or rewritten.
            path.unlink();audit['deleted_archives'].append(row['path'])
            receipt.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    audit.update(free_after=shutil.disk_usage(a.out).free,
        deleted_archive_bytes=sum(r['bytes'] for r in rows if r['path'] in audit['deleted_archives']),
        deleted_raw_files=0,weights_deleted=0,filesystem_hardlinks_used=False,
        compressed_byte_identical_regeneration_claimed=False)
    receipt.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(deleted=audit['deleted_archives'],freed_bytes=audit['deleted_archive_bytes'],
        free_after_GiB=audit['free_after']/2**30,retained_unmatched=[r['path'] for r in rows if r.get('eligible') is False])))


if __name__=='__main__':main()
