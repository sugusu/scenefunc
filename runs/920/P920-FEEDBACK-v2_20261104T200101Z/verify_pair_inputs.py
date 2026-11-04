import copy,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
T=Path(json.loads((P/"G/POLICY_PLAN.json").read_text())["runtime_directory"]).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plans={a:json.loads((P/a/"POLICY_PLAN.json").read_text()) for a in ["G","U"]}
prep={a:json.loads((T/a/"prepare/RESULT.json").read_text()) for a in plans}
req={a:json.loads((T/a/"prepare/request.json").read_text()) for a in plans}
protocol=json.loads((P/"RUN_PROTOCOL.json").read_text())
common=copy.deepcopy(req);feedback={}
for a in plans:
 text=common[a][1]["content"][-1]["text"]
 before,raw=text.split("\n\n## Independent current-state feedback\n",1)
 common[a][1]["content"][-1]["text"]=before;feedback[a]=json.loads(raw)
normalized=copy.deepcopy(plans)
for v in normalized.values():
 for k in ["feedback_arm","public","runtime","public_directory","runtime_directory"]:v.pop(k)
checks={
 "both_native_preparations_passed":all(x["status"]=="prepared_without_model" for x in prep.values()),
 "zero_model_and_tool_calls":all(x["model_calls"]==x["tool_calls"]==0 for x in prep.values()),
 "same_actual_restored_state":prep["G"]["state_sha256"]==prep["U"]["state_sha256"],
 "same_prepared_native_state":json.loads((T/"G/prepare/checkpoint/house_state.json").read_text())==json.loads((T/"U/prepare/checkpoint/house_state.json").read_text()),
 "same_views_and_common_messages":common["G"]==common["U"],
 "same_geometry_feedback":feedback["G"]["geometry"]==feedback["U"]["geometry"],
 "only_U_gets_use_feedback":set(feedback["G"])=={"geometry"} and set(feedback["U"])=={"geometry","use"},
 "same_plan_except_arm_and_output_paths":normalized["G"]==normalized["U"],
 "plans_match_frozen_protocol":all(sha(P/a/"POLICY_PLAN.json")==protocol["arms"][a]["policy_plan_sha256"] for a in plans),
 "same_encoded_image_evidence":prep["G"]["encoding"]["images"]==prep["U"]["encoding"]["images"],
 "both_within_original_input_ceiling":all(x["encoding"]["input_tokens"]<=12288 for x in prep.values()),
 "no_hidden_extra_budget":all(x["budget"]["tool_calls"]==x["budget"]["model_calls"]==9 and x["prior_tool_calls"]==15 for x in plans.values()),
 "same_numerical_restore_record":prep["G"]["view_reuse_equivalence"]==prep["U"]["view_reuse_equivalence"]}
r={"recorded_at":datetime.now(timezone.utc).isoformat(),"status":"passed" if all(checks.values()) else "failed","checks":checks,"input_tokens":{a:x["encoding"]["input_tokens"] for a,x in prep.items()},"model_calls":0,"tool_calls":0,"training_updates":0,"limitations":"U has additional information and input tokens by design; no FLOP equality claim. Native quaternion roundtrip recorded, not byte-identical to original serialized source state.","inputs":{str(T/a/"prepare/RESULT.json"):sha(T/a/"prepare/RESULT.json") for a in plans},"script_sha256":sha(Path(__file__))}
with (P/"PAIR_INPUT_AUDIT.json").open("x") as f:json.dump(r,f,indent=2);f.write("\n")
print(json.dumps(r),flush=True)
assert all(checks.values())
