"""Retire only the dormant old CUDA11.8 compiler, retaining install provenance.

The active V100 inference runtime uses its own CUDA12.8 libraries. No model,
native/eval Python, research source or scene result is removed. Re-running the
old CUDA11.8 build branch will require reinstalling these exact packages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/root/autodl-tmp/layout_experience_pilot/envs/cuda118')
PARTS = ('bin', 'include', 'lib', 'nvvm')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    active = []
    for process in Path('/proc').glob('[0-9]*'):
        for kind in ('maps', 'cmdline'):
            try:
                if str(ROOT).encode() in (process/kind).read_bytes():
                    active.append(dict(pid=int(process.name), kind=kind))
            except (OSError, PermissionError):
                pass
    if active:
        (a.out/'active_use.json').write_text(json.dumps(active))
        raise RuntimeError('Old compiler is referenced by a live process; no deletion')
    package_paths = set()
    packages = []
    for source in sorted((ROOT/'conda-meta').glob('*.json')):
        d = json.loads(source.read_text())
        assert d['name'] in ('cuda-nvcc', 'cuda-cccl', 'cuda-cudart', 'cuda-cudart-dev')
        assert d['version'] == '11.8.89'
        package_paths.update(d['files'])
        dest = a.out/'conda-meta'/source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
        packages.append(dict(name=d['name'], version=d['version'], url=d['url'],
                             package_md5=d['md5'], metadata_sha256=sha(source)))
    assert len(packages) == 4
    for source in (ROOT/'LICENSE', ROOT/'conda-meta/history'):
        shutil.copyfile(source, a.out/source.name)
    rows = []
    for part in PARTS:
        for path in sorted((ROOT/part).rglob('*')):
            if path.is_dir() and not path.is_symlink():
                continue
            rel = str(path.relative_to(ROOT))
            assert rel in package_paths, ('Unrecognized installation file', rel)
            if path.is_symlink():
                rows.append(dict(path=rel, symlink=str(path.readlink())))
            else:
                st = path.stat()
                rows.append(dict(path=rel, bytes=st.st_size, sha256=sha(path),
                                 inode=st.st_ino, mtime_ns=st.st_mtime_ns))
    before = shutil.disk_usage(a.out).free
    receipt = dict(authority='User authorized deleting genuinely unused old installations',
                   root=str(ROOT), packages=packages, files=rows,
                   active_references=[], status='verified_before_delete',
                   requires_reinstallation_for_old_cuda118_builds=True,
                   active_inference_runtime_unchanged=True,
                   original_experiment_results_deleted=0, model_weights_deleted=0)
    target = a.out/'receipt.json'
    target.write_text(json.dumps(receipt, indent=2)+'\n')
    for row in rows:
        path = ROOT/row['path']
        if 'sha256' in row:
            assert sha(path) == row['sha256']
        else:
            assert str(path.readlink()) == row['symlink']
    for part in PARTS:
        shutil.rmtree(ROOT/part)
    receipt.update(status='unused_compiler_removed_metadata_retained',
                   free_before=before, free_after=shutil.disk_usage(a.out).free,
                   installation_file_bytes=sum(r.get('bytes', 0) for r in rows))
    target.write_text(json.dumps(receipt, indent=2)+'\n')
    (a.out/'REINSTALL.txt').write_text(
        'Old CUDA11.8 compiler only; not needed by the active CUDA12.8 inference runtime.\n'
        'If revisiting that old build, create a fresh conda prefix, using these exact\n'
        'package URLs and checking each retained package MD5 before extraction.\n'
        + '\n'.join(x['url']+'  md5='+x['package_md5'] for x in packages)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','packages')}))


if __name__ == '__main__':
    main()
