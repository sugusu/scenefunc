import importlib, importlib.metadata, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "resources/full_runtime_20261104/scenesmith"))
rows=[]
for name in ["torch", "transformers", "bpy", "pydrake.all", "scenesmith.agent_utils.house"]:
    start=time.monotonic()
    print("import",name,flush=True)
    module=importlib.import_module(name)
    rows.append({"module":name,"file":module.__file__,"seconds":time.monotonic()-start})
import torch
result={"imports":rows,"cuda_initialized":torch.cuda.is_initialized(),"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}}
Path(__file__).with_name("MODULE_IMPORTS.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"imports_passed":len(rows),"cuda_initialized":torch.cuda.is_initialized()}),flush=True)
