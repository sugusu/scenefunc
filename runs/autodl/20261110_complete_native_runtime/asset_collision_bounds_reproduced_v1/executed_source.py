"""Reproduce collision-part bounds from the persisted unmodified scene bundle."""
import argparse
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from native_trial_evidence import restore_solid_evidence,relocate_restored_paths,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    stage=a.stage.resolve();a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='native_collision_bounds_',dir='/dev/shm') as scratch:
        root=Path(scratch)/'restored';proof=restore_solid_evidence(stage/'evidence_solid_v2/receipt.json',root)
        relocate_restored_paths(root,proof['original_root'])
        state=json.loads((root/'last_actual_room_state.json').read_text())
        for oid,ob in state['objects'].items():
            if ob['object_type']!='furniture':continue
            sdf_path=Path(ob['sdf_path']);parts=[]
            for collision in ET.parse(sdf_path).getroot().findall('.//collision'):
                mesh=collision.find('geometry/mesh')
                if mesh is None:continue
                # The actual native parts have no additional relative pose.
                assert collision.find('pose') is None
                path=(sdf_path.parent/mesh.findtext('uri')).resolve()
                scale=[float(x) for x in mesh.findtext('scale','1 1 1').split()]
                verts=[[float(v)*scale[i] for i,v in enumerate(line.split()[1:4])]
                       for line in path.read_text().splitlines() if line.startswith('v ')]
                if not verts:continue
                parts.append(dict(name=collision.attrib['name'],mesh_path=str(path.relative_to(root)),sha256=sha(path),
                                  scale=scale,min=[min(v[i] for v in verts) for i in range(3)],
                                  max=[max(v[i] for v in verts) for i in range(3)]))
            transform=json.loads(json.dumps(ob['transform']))
            rows.append(dict(object_id=oid,scene_bbox_min=ob['bbox_min'],scene_bbox_max=ob['bbox_max'],
                             collision_union_min=[min(x['min'][i] for x in parts) for i in range(3)],
                             collision_union_max=[max(x['max'][i] for x in parts) for i in range(3)],
                             scene_transform=transform,parts=parts))
        result=dict(source_zip_SHA256=proof['original_zip_sha256'],objects=rows,scene_coordinates_modified=False,model_calls=0)
        original=json.loads((stage/'asset_collision_frame_audit/geometry_bounds.json').read_text())
        assert result==original
        (a.out/'geometry_bounds.json').write_text(json.dumps(result,indent=2)+'\n')
        summary=dict(status='persisted_bundle_bounds_reproduced',objects=len(rows),
                     identical_to_first_measured_bounds=True,source_archive=proof['archive'],
                     source_archive_SHA256=proof['archive_sha256'],model_calls=0)
        (a.out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
