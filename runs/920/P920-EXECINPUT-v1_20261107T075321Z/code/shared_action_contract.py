"""Opt-in shared output boundary for future catalog policies; no design oracle.

Historical rollouts do not import this module. A host must construct a fresh
contract from each actual observation, preserve raw output, count rejected calls,
and validate against its current context before dispatch. Token constraints are
an aid; strict final validation remains mandatory. No retries or repairs occur.
"""
import copy
import hashlib
import json
import math


TOOL_NAMES = frozenset(("get_current_scene_state", "add_furniture_to_scene_tool",
                        "move_furniture_tool", "remove_furniture_tool", "check_facing_tool"))
DONE = "__done__"


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_loads(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("Duplicate JSON key: " + key)
            out[key] = value
        return out

    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def schemas_from_observation(observation):
    text = observation["system"].split("Available executable tools:\n", 1)[1]
    return json.JSONDecoder().raw_decode(text.lstrip())[0]


def context_from_request(messages):
    candidates = []
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        text = content if isinstance(content, str) else "\n".join(
            part.get("text", "") for part in content)
        if "CURRENT NATIVE STATE:" in text:
            candidates.append(json.JSONDecoder().raw_decode(
                text.split("CURRENT NATIVE STATE:", 1)[1].lstrip())[0])
    if len(candidates) != 1:
        raise ValueError("Expected exactly one actual current native state")
    return candidates[0]


class ActionContract:
    def __init__(self, tools, context):
        from jsonschema import Draft202012Validator
        names = [tool["name"] for tool in tools]
        if len(names) != len(TOOL_NAMES) or set(names) != TOOL_NAMES:
            raise ValueError("Expected the five catalog tools exactly once")
        self.context_sha256 = digest(context)
        assets = sorted(item["asset_id"] for item in context["asset_catalog"])
        if len(assets) != len(set(assets)):
            raise ValueError("Duplicate catalog IDs")
        objects = sorted(oid for oid, obj in context["objects"].items()
                         if obj["object_type"] == "furniture")
        for oid in objects:
            if context["objects"][oid]["object_id"] != oid:
                raise ValueError("Object identity mismatch")
        branches = []
        for tool in tools:
            args = copy.deepcopy(tool["parameters"])
            Draft202012Validator.check_schema(args)
            if args.get("type") != "object" or args.get("additionalProperties") is not False:
                raise ValueError("Native argument object must reject additional properties")
            feasible = True
            for field, spec in args["properties"].items():
                choices = assets if field == "asset_id" else objects if field in (
                    "object_id", "object_a_id", "object_b_id") else None
                if choices is not None:
                    if not choices:
                        feasible = False
                    else:
                        spec["enum"] = choices
            if feasible:
                branches.append(self._branch(tool["name"], args))
        branches.append(self._branch(DONE, {
            "type": "object", "properties": {"unmet_or_unknown": {"type": "string"}},
            "required": ["unmet_or_unknown"], "additionalProperties": False}))
        self.schema = {"anyOf": branches}
        Draft202012Validator.check_schema(self.schema)
        self.validator = Draft202012Validator(self.schema)

    @staticmethod
    def _branch(name, args):
        return {"type": "object", "properties": {
            "name": {"type": "string", "enum": [name]}, "arguments": args},
            "required": ["name", "arguments"], "additionalProperties": False}

    def validate(self, raw_json, current_context):
        if digest(current_context) != self.context_sha256:
            raise ValueError("Stale action contract; regenerate from current observation")
        value = strict_loads(raw_json)
        self.validator.validate(value)
        return {"kind": "done" if value["name"] == DONE else "tool", "call": value,
                "context_sha256": self.context_sha256,
                "schema_sha256": digest(self.schema)}

    def prefix_function(self, tokenizer_data):
        """A new parser per generation; tokenizer trie may be cached across turns."""
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(self.schema))
