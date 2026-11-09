"""Run on the source host to export only SHA-locked missing observation images.

This script never renders replacements or reads model weights/scene assets.
It does not transfer the archive or start an experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-bytes', type=int, default=256 * 1024 ** 2)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    checked = []
    total = 0
    for row in manifest['images']:
        path = Path(row['source_path'])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Original view changed: ' + str(path))
        total += len(raw)
        if total > a.max_bytes:
            raise ValueError('Archive input limit exceeded before creating output')
        checked.append((row, path, len(raw)))
    if not a.check_only:
        # x: never silently replace an earlier evidence archive.
        with tarfile.open(a.output, 'x') as archive:
            for row, path, size in checked:
                # Recheck at packaging time to catch intervening changes.
                if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError('View changed between validation and export')
                info = tarfile.TarInfo(row['sha256'] + path.suffix.lower())
                info.size = size
                info.mode = 0o644
                with path.open('rb') as handle:
                    archive.addfile(info, handle)
    print(json.dumps({'verified_original_images': len(checked), 'bytes': total,
                      'archive_created': not a.check_only, 'model_calls': 0,
                      'training_updates': 0, 'rendered_replacements': 0}))


if __name__ == '__main__':
    main()
