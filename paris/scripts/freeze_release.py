"""Maintainer-only content manifest rebuild after documented package changes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
files=[]
for p in sorted(ROOT.rglob('*')):
 if p.is_file() and p.name!='release_manifest.json' and '__pycache__' not in p.parts:
  files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'release_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'frozen_files':len(files)}))
