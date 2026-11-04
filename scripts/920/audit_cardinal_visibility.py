"""Render a saved native scene once and retain camera-side visibility evidence."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--renderer', type=Path, required=True)
    p.add_argument('--blend', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    before = digest(args.blend)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), 'begin saved native cardinal render', flush=True)
    result = {'status': 'running', 'model_calls': 0, 'training_updates': 0,
              'source_blend_sha256': before, 'renderer_sha256': digest(args.renderer),
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'images': {},
              'visual_acceptance': 'pending', 'scope': 'synthetic one-wall saved scene'}
    try:
        sys.argv = [str(args.renderer), str(args.blend), '--out', str(args.out),
                    '--views', 'east', 'north', 'west', 'south']
        runpy.run_path(str(args.renderer), run_name='__main__')
        from PIL import Image
        for view in ('east', 'north', 'west', 'south'):
            image = args.out / f'{args.blend.stem}.{view}.png'
            with Image.open(image) as im:
                size = list(im.size)
                im.verify()
            visibility = json.loads(Path(str(image) + '.visibility.json').read_text())
            # The archived blend inspection identifies mesh1 as the north wall
            # and mesh2 as the table, independently of the renderer predicate.
            expected = ['mesh1'] if view == 'north' else []
            assert visibility['hidden_panel_names'] == expected, (view, visibility)
            assert visibility['blend_sha256'] == before
            result['images'][view] = {'path': str(image), 'sha256': digest(image),
                                      'size': size, 'visibility': visibility}
        assert digest(args.blend) == before, 'source blend was modified'
        result.update(status='passed', source_blend_unchanged=True,
                      render_count=4, visible_table_in_all_views='requires_visual_review')
    except Exception as error:
        result.update(status='failed', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['ended_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (args.out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
