"""Reproduce presence-only palette audit; categories do not verify room geometry."""
import argparse,collections,hashlib,json,re
from pathlib import Path
from PIL import Image

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[2];pub=root/'runs/920'/a.run_id;record=json.loads((pub/'PALETTE_CHECK.json').read_text());protocol=json.loads((pub/'PROTOCOL.json').read_text());index=Path(protocol['source_index']['path']);assert hashlib.sha256(index.read_bytes()).hexdigest()==record['source_record_index_sha256'];rows=json.loads(index.read_text())['records'];matches={fn:[] for fn in record['mapping']}
 for row in rows:
  counts={fn:int(n) for n,fn in re.findall(r'(\d+)_(living_room|bedroom|bathroom|kitchen|balcony)',row['category'])}
  with Image.open(root.parent/row['files']['plan']['path']) as image:colors=collections.Counter(image.getdata())
  for fn,item in record['mapping'].items():
   if bool(counts.get(fn,0))!=bool(colors[tuple(item['rgba'])]):matches[fn].append(row['record_id'])
 for fn,item in record['mapping'].items():assert set(matches[fn])==set(item['mismatches']) and len(rows)-len(matches[fn])==item['agreements']
 selected={r['record_id'] for r in protocol['selected']};assert not any(selected&set(x) for x in matches.values())
 print(json.dumps({'status':'verified','records':len(rows),'selected_cases_without_presence_conflicts':len(selected),'scope':'Inferred palette/presence only, not official semantic legend, exact counts or professional labels.'}))
if __name__=='__main__':main()
