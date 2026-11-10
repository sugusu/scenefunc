"""Shared policy contract for native floor/furniture/wall/ceiling observations.

No model loader, network client, tool executor or success oracle. A supplied
backend owns generation and its separately authorized cumulative budget.
All conditions use the same ordered-batch parser. Surface-object workflows
remain the existing separate native path, not silently relabeled supported.
"""
import copy
import hashlib
import json
import re
import time

from decision_field_routes import MODES, route_decision_text
from phase_policy import classify_stage, strict_json
from routed_multimodal import frame_completed_decision


NATIVE_STAGES = ('floor_plan', 'furniture', 'wall', 'ceiling')
CONDITIONS = ('native_joint', 'action_direct', 'decision_independent', 'decision_conditioned')
PHASE_SUFFIX = {
    'decision': ('\n\nCURRENT OUTPUT TASK: DECISION_RECORD. Produce one proposed decision '
                 'inside <think>...</think>, then stop at </think>. Do not issue a tool call. '
                 'When applying a supplied source design principle, use a JSON object '
                 'with exactly four keys: current_condition, '
                 'principle_applicability, proposed_decision, expected_cost_and_unknowns. '
                 'principle_applicability and proposed_decision must be objects. '
                 'Use only available evidence; do not invent a source principle when none '
                 'is supplied. Other evidence-limited or stopping records need not claim '
                 'that schema. This instruction is shared by all decision conditions. '
                 'Distinguish intended edits from observed results and keep unknowns explicit.'),
    'action': ('\n\nCURRENT OUTPUT TASK: NATIVE_ACTION_BATCH. Emit zero or more permitted native '
               '<tool_call> JSON blocks, in execution order. Every block must contain name and '
               'arguments. Use the original task, current state and actual tool feedback. '
               'Do not generate another <think> block. End the assistant turn without a tool '
               'call to propose stopping. A prior decision is an unverified proposal; '
               'stopping is not proof of stage or complete-room success.'),
}


class NativePhaseFailure(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt['status'])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def validate_native_observation(obs):
    stage = obs.get('stage')
    if stage not in NATIVE_STAGES:
        raise ValueError('Only four declared native policy stages are supported')
    messages = obs.get('messages')
    if (not isinstance(messages, list) or len(messages) != 2
            or messages[0].get('role') != 'system' or messages[1].get('role') != 'user'
            or not isinstance(messages[0].get('content'), str)):
        raise ValueError('Expected original system + current user observation')
    refs = []
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for item in content:
                if item.get('type') == 'image':
                    refs.append(str(item.get('image', item.get('path'))))
    expected = 0 if stage == 'floor_plan' else 5
    if len(refs) != expected or refs != list(map(str, obs.get('image_paths', []))):
        raise ValueError('Native stage view count/order differs from observation')
    # This is a message/reference contract. Actual image bytes and decoding
    # are checked by the real backend, not inferred from path existence here.
    if type(obs.get('step_idx')) is not int or obs['step_idx']<0:
        raise ValueError('Original native step counter required')
    return dict(stage=stage, image_references=refs, expected_views=expected,
                native_step_idx=obs['step_idx'],
                observation_sha256=fingerprint(messages), full_simulator_state_snapshot=False)


def prepare_phase_messages(messages, phase):
    result = copy.deepcopy(messages)
    if phase == 'joint':
        return result
    if phase not in PHASE_SUFFIX:
        raise ValueError('Unknown output phase')
    result[0]['content'] += PHASE_SUFFIX[phase]
    return result


def classify_native_batch(tokenizer, ids, *, phase, eos_ids, native_parser):
    """Verify natural boundaries, then preserve all original calls in order.

Malformed batches fail as a whole before dispatch. This common preflight is
stricter than the original regex parser; it is an explicit engineering change
shared by every condition, not candidate-only repair or paper reproduction.
"""
    if phase == 'decision':
        return classify_stage(tokenizer, ids, stage='decision', eos_ids=eos_ids)
    if phase not in ('joint', 'action') or not ids or any(type(t) is not int for t in ids):
        return dict(kind='invalid', reason='Empty/noninteger output or invalid phase')
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    if ids[-1] not in eos_ids or any(t in eos_ids for t in ids[:-1]):
        return dict(kind='invalid', reason='No unique natural native EOS at end', raw=raw)
    body = tokenizer.decode(ids[:-1], skip_special_tokens=False).strip()
    action_body = body
    if phase == 'joint' and ('<think>' in body or '</think>' in body):
        match = re.match(r'^<think>.*?</think>', body, re.S)
        if not match or body.count('<think>') != 1 or body.count('</think>') != 1:
            return dict(kind='invalid', reason='Malformed joint decision', raw=raw)
        action_body = body[match.end():].strip()
    elif '<think' in body or '</think' in body:
        return dict(kind='invalid', reason='Unexpected decision in action phase', raw=raw)
    if any(x in action_body for x in ('<|', '<done', '</done')):
        return dict(kind='invalid', reason='Unexpected role/completion control tag', raw=raw)
    # A truncated tag must not become a successful-looking no-call STOP.
    without_blocks = re.sub(r'<tool_call>.*?</tool_call>', '', action_body, flags=re.S)
    if '<tool_call' in without_blocks or '</tool_call' in without_blocks:
        return dict(kind='invalid', reason='Incomplete or noncanonical tool tag', raw=raw)
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', action_body, re.S)
    if len(blocks) != action_body.count('<tool_call>') or len(blocks) != action_body.count('</tool_call>'):
        return dict(kind='invalid', reason='Unclosed or unmatched tool block', raw=raw)
    decoded = []
    try:
        for block in blocks:
            call = strict_json(block)
            if (not isinstance(call, dict) or set(call) != {'name', 'arguments'}
                    or not isinstance(call['name'], str) or not call['name']
                    or not isinstance(call['arguments'], dict)):
                raise ValueError('Invalid native call envelope')
            decoded.append(call)
        think, native_calls = native_parser(body)
        if native_calls != decoded:
            raise ValueError('Native parser and guarded complete batch differ')
    except (ValueError, TypeError, AttributeError) as exc:
        return dict(kind='invalid', reason=str(exc), raw=raw)
    return dict(kind='batch_proposed' if decoded else 'stop_proposed', raw=raw,
                think=think, tool_calls=decoded, native_schema_validation_required=True,
                full_scene_success=None)


class NativeMultiStagePolicy:
    """Drop-in generate(obs) proposal shape, with actual D/A budget accounting.

Backend.generate(messages, assistant_prefix, phase, token_limit, seconds_limit)
returns token IDs and input-token cost. No corpus/gold data enters this API.
The caller is responsible for whole-run budget, live native schema validation,
hard subprocess timeout and stage handover. Per-proposal time checks are soft.
"""
    def __init__(self, backend, *, native_parser, condition, total_output_tokens,
                 decision_tokens, action_tokens, seconds, max_calls_per_batch,
                 field_route='full_record'):
        if condition not in CONDITIONS or field_route not in MODES:
            raise ValueError('Unknown condition/field route')
        if getattr(backend,'execution_kind',None) not in ('actual_model_generation','archived_sequence_replay'):
            raise ValueError('Backend must declare real generation versus archive replay')
        if condition != 'decision_conditioned' and field_route != 'full_record':
            raise ValueError('Field routing only applies to conditioned decisions')
        if any(type(v) is not int or v <= 0 for v in
               (total_output_tokens, decision_tokens, action_tokens, max_calls_per_batch)):
            raise ValueError('Explicit positive integer limits required')
        if not isinstance(seconds, (int, float)) or not 0 < seconds < float('inf'):
            raise ValueError('Explicit finite time limit required')
        self.backend=backend; self.parser=native_parser; self.condition=condition
        self.total=total_output_tokens; self.d_cap=decision_tokens; self.a_cap=action_tokens
        self.seconds=seconds; self.max_batch=max_calls_per_batch; self.field_route=field_route

    def generate(self, obs):
        observed=validate_native_observation(obs)
        tick=time.monotonic(); used=0; draft=None; final=None
        evidence=dict(condition=self.condition, field_route=self.field_route,
                      native_stage=observed['stage'], observation=observed,
                      phases=[], backend_requests=0, actual_model_generate_requests=0,
                      model_accounting_complete=True,
                      generated_or_replayed_output_tokens=0, input_tokens=0,
                      full_scene_success=None, paid_API_calls=0)
        phases=(['joint'] if self.condition=='native_joint' else ['action']
                if self.condition=='action_direct' else ['decision','action'])
        try:
            for phase in phases:
                elapsed=time.monotonic()-tick
                cap=min(self.total-used,self.total if phase=='joint' else self.d_cap if phase=='decision' else self.a_cap)
                if elapsed >= self.seconds or cap < 1:
                    raise ValueError('Shared D+A budget exhausted before '+phase)
                prefix=''; route=dict(status='not_consumed')
                if phase=='action' and self.condition=='decision_conditioned':
                    routed,route=route_decision_text(draft,self.field_route)
                    prefix=frame_completed_decision(routed)
                messages=prepare_phase_messages(obs['messages'],phase)
                entry=dict(phase=phase, status='started', output_token_limit=cap,
                           phase_messages_sha256=fingerprint(messages),
                           action_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                           action_reads_generated_D=bool(prefix), field_routing=route)
                evidence['phases'].append(entry); evidence['backend_requests']+=1
                response=self.backend.generate(messages, assistant_prefix=prefix, phase=phase,
                    token_limit=cap, seconds_limit=self.seconds-elapsed)
                ids=response['output_ids']
                if response.get('execution_kind') != self.backend.execution_kind:
                    raise ValueError('Response cannot change the backend execution kind')
                if type(response.get('input_tokens')) is not int or response['input_tokens']<0:
                    raise ValueError('Explicit nonnegative encoded input-token count required')
                entry.update(response, status='returned'); used+=len(ids)
                evidence['generated_or_replayed_output_tokens']=used
                evidence['input_tokens']+=response['input_tokens']
                evidence['actual_model_generate_requests']+=int(response['execution_kind']=='actual_model_generation')
                if len(ids)>cap:
                    raise ValueError('Backend exceeded cumulative output limit')
                if time.monotonic()-tick>self.seconds:
                    raise ValueError('Soft time limit exceeded after '+phase)
                parsed=classify_native_batch(self.backend.tokenizer,ids,phase=phase,
                                             eos_ids=self.backend.eos_ids,native_parser=self.parser)
                entry['classification']=parsed
                if parsed['kind']=='invalid':
                    raise ValueError('Invalid '+phase+': '+parsed['reason'])
                if phase=='decision':
                    draft=parsed['raw']
                else:
                    final=parsed
            if len(final['tool_calls'])>self.max_batch:
                raise ValueError('Whole batch exceeds remaining admitted native-call capacity')
            evidence.update(status=final['kind'], generated_decision_for_audit_only=draft,
                            proposed_native_calls=len(final['tool_calls']),
                            elapsed_seconds=time.monotonic()-tick)
            return dict(raw=final['raw'], think=(draft[len('<think>'):].rsplit('</think>',1)[0].strip()
                        if draft else final['think']), tool_calls=final['tool_calls'],
                        remote_receipt=None, full_scene_success=None, transport_receipt=evidence)
        except NativePhaseFailure:
            raise
        except BaseException as exc:
            if self.backend.execution_kind=='actual_model_generation':
                last=getattr(self.backend,'last_attempt',None)
                evidence['backend_last_attempt']=copy.deepcopy(last)
                # An interrupted backend cannot be reported as zero forwards.
                if last is None or (last.get('generation_started') is True and last.get('status')!='returned'):
                    evidence['model_accounting_complete']=False
                    evidence['actual_model_generate_requests']=None
            evidence.update(status='failed_proposal',error=type(exc).__name__+': '+str(exc),
                            elapsed_seconds=time.monotonic()-tick,
                            proposed_native_calls_not_dispatched=True)
            if isinstance(exc, Exception):
                raise NativePhaseFailure(evidence) from exc
            # Preserve interruption semantics; do not turn Ctrl-C/SystemExit into
            # a retryable policy failure or lose already incurred model work.
            exc.native_phase_receipt=evidence
            raise


async def execute_native_batch(env, proposal, *, remaining_native_calls):
    """Dispatch ONCE to original env.step(batch), preserving memory step order.

On interruption, reserved calls are reported separately from actual attempts;
there is no fake claim that all reserved calls completed. An empty proposal is
only stopping intent; it does not call finalizers or certify scene success.
"""
    calls=proposal['tool_calls']
    observed=proposal['transport_receipt']['observation']
    if env.stage!=observed['stage'] or env.step_idx!=observed['native_step_idx']:
        raise ValueError('Stage/counter changed after proposal; no dispatch')
    if type(remaining_native_calls) is not int or remaining_native_calls<0:
        raise ValueError('Nonnegative remaining cumulative tool budget required')
    if len(calls)>remaining_native_calls:
        raise ValueError('Cumulative native-call budget would be exceeded')
    receipt=dict(reserved_calls=len(calls),actual_attempted_calls=None,
                 results_received=0,status='reserved',full_scene_success=None)
    if not calls:
        receipt.update(status='stop_proposed',actual_attempted_calls=0)
        return [],receipt
    try:
        results=await env.step(copy.deepcopy(calls))
        if len(results)!=len(calls) or any(r['name']!=c['name'] for r,c in zip(results,calls)):
            raise ValueError('Original native batch returned an inconsistent result list')
        receipt.update(status='native_batch_returned',actual_attempted_calls=len(results),
                       results_received=len(results),tool_success_is_task_success=False)
        return results,receipt
    except BaseException as exc:
        receipt.update(status='native_batch_interrupted',error=type(exc).__name__,
                       actual_attempted_calls=None)
        # Preserve accounting even when the caller catches the original error.
        try:exc.native_batch_receipt=receipt
        except Exception:pass
        raise
