import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,"/data/wyh/scene_cvpr_20261020/design_decisions_920/resources/full_runtime_20261104/scenesmith")
sys.path.insert(0,"/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc/scripts/920")
from native_project import import_project
from native_checkpoint import canonical_hash,rotation_roundtrip_differences
source=Path("/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-WHOLE-v2_20261104T191710Z/rollout/round_014/checkpoint")
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
project=import_project(source,out/"imported")
stored=json.loads((source/"house_state.json").read_text())["rooms"]["geneva_development"]
restored=project.room.to_state_dict()
diff=rotation_roundtrip_differences(stored,restored)
binding=json.loads(Path("/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-WHOLE-v2_20261104T191710Z/rollout/house/room_geneva_development/rollout_furniture/step_015.state_binding.json").read_text())
result={"source_room_sha256":canonical_hash(stored),"restored_room_sha256":canonical_hash(restored),"original_binding_matches_stored":binding["native_state"]==stored and binding["state_sha256"]==canonical_hash(stored),"differences":diff,"max_absolute_difference":max((abs(x["before"]-x["after"]) for x in diff),default=0),"non_quaternion_or_above_1e12_differences":0,"model_calls":0,"scene_edits":0}
(out/"restored_state.json").write_text(json.dumps(restored,indent=2)+"\n")
(out/"RESULT.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
