"""Read-only post hoc argument review against the actual advertised schemas."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import jsonschema
from native_project import sha256

p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=json.loads(a.plan.read_text());root=Path(plan['runtime_directory']);report_path=root/'report/REPORT.json';report=json.loads(report_path.read_text())
obs_path=Path(plan['prepared_directory'])/'observation.json';obs=json.loads(obs_path.read_text())
schemas,_=json.JSONDecoder().raw_decode(obs['system'].split('Available executable tools:\n',1)[1]);schemas={x['name']:x['parameters'] for x in schemas}
rows=[]
for action in report['actions']:
 call=action['requested'];errors=[e.message for e in jsonschema.Draft202012Validator(schemas[call['name']]).iter_errors(call['arguments'])]
 rows.append({'turn':action['turn'],'tool':call['name'],'arguments':call['arguments'],'schema_errors':errors,'native_accepted':action['native_success']})
result={'reviewed_at':datetime.now(timezone.utc).isoformat(),'scope':'Post hoc check of recorded arguments against actually advertised schemas; no replay, score or prompt change.','recorded_calls':len(rows),'schema_mismatch_calls':sum(bool(x['schema_errors']) for x in rows),'counts':dict(Counter(x['tool'] for x in rows)),'calls':rows,'source_report_sha256':sha256(report_path),'source_observation_sha256':sha256(obs_path),'script_sha256':sha256(Path(__file__))}
with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ['recorded_calls','schema_mismatch_calls','counts']},ensure_ascii=False))
