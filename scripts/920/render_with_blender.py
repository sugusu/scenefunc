"""Run the pinned five-view code inside the existing Blender, on CPU only.

The original render_state subprocess invocation is dispatched in-process because
bpy is embedded in this installation. The outer run_stage enforces the timeout.
No camera, geometry, image-size, sample-count or rendering algorithm is changed.
"""
import argparse
import contextlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace


def main():
    sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index('--')+1:]
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--site-packages', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.site_packages))
    renderer = a.work / 'native_observation_bridge_20261104/source/tools/render_five_views.py'
    original_run = subprocess.run
    def inline(command, **options):
        if len(command) < 2 or Path(command[1]) != renderer:
            return original_run(command, **options)
        old = sys.argv
        sys.argv = command[1:]
        try:
            with contextlib.redirect_stdout(options['stdout']), contextlib.redirect_stderr(options['stdout']):
                runpy.run_path(str(renderer), run_name='__main__')
        finally:
            sys.argv = old
        return SimpleNamespace(returncode=0)
    subprocess.run = inline
    sys.argv = ['render_state.py', '--scene', str(a.scene), '--out', str(a.out), '--transient-blend']
    runpy.run_path(str(a.work / 'paris_local_run_results_20261104/render_state.py'), run_name='__main__')
    import bpy
    record = {'blender_version': bpy.app.version_string, 'execution': 'native renderer inside existing Blender; outer stage timeout', 'CPU_only': True}
    (a.out / '920_renderer_adapter.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
