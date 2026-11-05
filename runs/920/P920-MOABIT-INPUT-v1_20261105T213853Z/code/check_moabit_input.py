"""Independently inspect native SDF faces and apertures against input assumptions."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    task=json.loads(a.task.read_text());g=task['geometry_assumptions'];L,W=g['interior_rectangle'][2:];H=g['height']
    rid=task['task_id'];house=json.loads((a.bundle/'house_state.json').read_text());state=house['rooms'][rid]
    shapes={}
    for node in ET.parse(a.bundle/('room_'+rid)/'room.sdf').findall('.//collision'):
        pose=np.fromstring(node.findtext('pose'),sep=' ');size=np.fromstring(node.findtext('geometry/box/size'),sep=' ')
        assert len(pose)==6 and not np.any(pose[3:]) and len(size)==3
        shapes[node.attrib['name']]=(pose[:3]-size/2,pose[:3]+size/2)
    for prefix,axis,side,value in [('west',0,1,-L/2),('east',0,0,L/2),('south',1,1,-W/2),('north',1,0,W/2),('bath_west',0,1,L/2-1.5),('bath_north',1,0,-W/2+1.8)]:
        faces=[v[side][axis] for k,v in shapes.items() if k.startswith(prefix+'_')]
        assert faces and np.allclose(faces,value,atol=1e-10,rtol=0),(prefix,faces,value)
    assert np.allclose(shapes['floor'],[[-L/2,-W/2,-.05],[L/2,W/2,0]],rtol=0,atol=1e-10)
    assert np.isclose(L*W,21) and state['room_geometry']['wall_height']==H
    def hits(x):
        x=np.array(x);return [k for k,(lo,hi) in shapes.items() if np.all(x>=lo-1e-8) and np.all(x<=hi+1e-8)]
    checks=[]
    for label,point,frame_z in [('entry',[-L/2+.65,-W/2-.025,1.],2.2),('window',[-L/2+1.3,W/2+.025,1.55],.8),('bath_access',[L/2-1.525,-W/2+.9,1.],2.2)]:
        assert not hits(point),(label,hits(point))
        frame=point[:2]+[frame_z];assert hits(frame)
        checks.append(dict(opening=label,open_point=point,solid_probe=frame,solid_hits=hits(frame)))
    furniture=[k for k,v in state['objects'].items() if v['object_type']=='furniture'];assert not furniture
    catalog=json.loads((a.bundle/'catalog.json').read_text());assert len(catalog['entries'])==16
    assert all(e['template']['transform']=={'translation':[0.,0.,0.],'rotation_wxyz':[1.,0.,0.,0.]} for e in catalog['entries'])
    result=dict(status='declared_input_sdf_verified',variant=g['sensitivity']['variant'],clear_dimensions_m=[L,W,H],area_m2=L*W,
        bath_clear_dimensions_m=[1.5,1.8],apertures=checks,catalog_entries=16,furniture_instances=0,
        model_calls=0,training_updates=0,whole_room_feasibility='unverified',
        limits=['Open apertures without moving door leaves; privacy/operation unknown.','Bounds and catalog are research inputs, not original project geometry.','No actual furnished layout or functional success.'])
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
