"""Check actual SDF boundaries/apertures against the declared input task.

This validates the construction input only; no usability or building-code label.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'inspect actual SDF against task assumptions', flush=True)
    task = json.loads(a.task.read_text())
    state = json.loads((a.bundle/'house_state.json').read_text())
    rid = 'whole_room_transfer_01'
    root = ET.parse(a.bundle/('room_'+rid)/'room.sdf')
    boxes = {}
    for node in root.findall('.//collision'):
        pose = np.fromstring(node.findtext('pose'), sep=' ')
        size = np.fromstring(node.findtext('geometry/box/size'), sep=' ')
        if len(pose) != 6 or np.any(pose[3:] != 0) or len(size) != 3:
            raise ValueError('Unsupported collision primitive')
        boxes[node.attrib['name']] = (pose[:3]-size/2, pose[:3]+size/2)
    def faces(prefix, axis, edge):
        return {round(bounds[edge][axis], 8) for name, bounds in boxes.items()
                if name.startswith(prefix+'_')}
    assert faces('west', 0, 1) == {-3.}
    assert faces('east', 0, 0) == {3.}
    assert faces('south', 1, 1) == {-2.5}
    assert faces('north', 1, 0) == {2.5}
    assert faces('bath_west', 0, 1) == {1.2}
    assert faces('bath_north', 1, 0) == {-.5}
    floor = boxes['floor']
    assert np.allclose(floor, [[-3., -2.5, -.05], [3., 2.5, 0.]], atol=1e-12)
    core = task['geometry_assumptions']['fixed_service_core']['rectangle']
    assert np.allclose([core[2]-core[0], core[3]-core[1]], [1.8, 2.])
    def hits(point):
        point = np.array(point)
        return [name for name, (lo, hi) in boxes.items()
                if np.all(point > lo+1e-8) and np.all(point < hi-1e-8)]
    probes = []
    for name, center, closed_z in [
        ('entry', [-2.35, -2.525, 1.], 2.2),
        ('terrace_access', [2.25, 2.525, 1.], 2.2),
        ('bath_access', [1.175, -1.55, 1.], 2.2),
        ('window', [-1.7, 2.525, 1.55], .8),
    ]:
        assert not hits(center), (name, hits(center))
        above = center[:2] + [closed_z]
        assert hits(above), name
        probes.append({'name': name, 'open_center': center, 'solid_frame_probe': above,
                       'solid_hits': hits(above)})
    geometry = state['rooms'][rid]['room_geometry']
    assert (geometry['length'], geometry['width'], geometry['wall_height']) == (6, 5, 2.7)
    catalog = json.loads((a.bundle/'catalog.json').read_text())
    assert len(catalog['entries']) == 16
    assert all(x['template']['transform']['translation'] == [0., 0., 0.] for x in catalog['entries'])
    report = {'status': 'declared_input_geometry_checks_passed',
              'outer_clear_dimensions_m': [6, 5], 'bath_clear_dimensions_m': [1.8, 2],
              'aperture_probes': probes, 'catalog_entries': 16,
              'not_verified': ['render appearance', 'door mechanics', 'whole-room feasibility',
                               'fixture use', 'structural safety', 'model generation'],
              'model_calls': 0, 'training_updates': 0}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
