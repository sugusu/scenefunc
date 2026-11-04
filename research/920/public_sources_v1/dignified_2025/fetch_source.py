import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import requests
p=Path(__file__).resolve().parent;plan=json.loads((p/"PLAN.json").read_text());records=[]
for item in plan["requests"]:
 target=Path(item["destination"]);assert not target.exists()
 response=requests.get(item["url"],stream=True,timeout=plan["budget"]["seconds_each"],proxies={"http":"http://127.0.0.1:19998","https":"http://127.0.0.1:19998"})
 response.raise_for_status();data=bytearray()
 for block in response.iter_content(65536):
  data.extend(block)
  if len(data)>plan["budget"]["maximum_bytes_each"]:raise ValueError("Bounded source size exceeded")
 if item["kind"]=="pdf" and not data.startswith(b"%PDF-"):raise ValueError("Not a PDF")
 with target.open("xb") as f:f.write(data)
 records.append({**item,"resolved_url":response.url,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"status":response.status_code})
 result={"recorded_at":datetime.now(timezone.utc).isoformat(),"status":"completed","sources":records,"model_calls":0,"contacted_authors":False}
 (p/"ACQUISITION_RESULT.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(records[-1]),flush=True)
