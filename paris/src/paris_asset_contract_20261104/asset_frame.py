"""Resolve data-coordinate frames from recorded sources, not inferred layouts.
The Habitat asset frame is not a claim about drawers, sitting direction or use.
"""
import hashlib,json,math
from pathlib import Path
P=Path(__file__).resolve().parent
def parse(v):
    if isinstance(v,str):v=[float(s) for s in v.split(',')]
    if not isinstance(v,list) or len(v)!=3 or any(not isinstance(x,(float,int)) or not math.isfinite(x) for x in v):raise ValueError('Invalid axis vector')
    return [float(x) for x in v]
def frame_for(row):
    if row.get('up') and row.get('front'):
        return {'up':parse(row['up']),'front':parse(row['front']),'source':'explicit HSM preprocessed HSSD per-object axes','semantic_front_verified':False}
    path=P/'official_configs'/(row['id']+'.json')
    if not path.is_file():return None
    raw=path.read_bytes();d=json.loads(raw)
    if Path(d['render_asset']).name!=row['id']+'.glb':raise ValueError('Official config/model ID mismatch')
    return {'up':parse(d['up']),'front':parse(d['front']),
            'source':'official hssd-hab same-ID asset coordinate frame; corresponding uncompressed HSSD model',
            'source_path':str(path),'source_sha256':hashlib.sha256(raw).hexdigest(),
            'repository_revision':'4369cb9876214c7fbebcf552eb532380e4d287e4',
            'semantic_front_verified':False,
            'scope':'Coordinate conversion only. Functional access direction and per-asset geometric equivalence are not universally validated.'}
