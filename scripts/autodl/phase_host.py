"""Host for the phase policy in an existing native furniture ToolEnvironment.

No model/service construction, new solver, or completion oracle is provided.
Native tool execution, observation rendering and stage handover stay upstream.
This host admits native additions/removals: it never freezes the object set.
An outer process must enforce a HARD timeout; synchronous inference/rendering
cannot be interrupted safely by this loop's monotonic budget checks.
"""
import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from phase_policy import PhaseOutcomeError
from scene_evidence import append_host_evidence, canonical_host_evidence, visible_scene_state


class ActionRejected(ValueError):
    """Invalid proposed action; infrastructure/schema faults are not this type."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


@dataclass(frozen=True)
class StageBudget:
    turns: int
    generation_attempts: int
    tool_attempts: int
    output_tokens: int
    wall_seconds: float

    def __post_init__(self):
        for key in ('turns', 'generation_attempts', 'tool_attempts', 'output_tokens'):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError('Positive integer budget required: ' + key)
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError('Positive finite wall budget required')


class NativeRoomPort:
    """Wrap an already configured native environment; do not authorize services.

    allowed_tools and acceptance are experiment-owned and identical across arms.
    Tool schema validation uses the LIVE SDK FunctionTool schemas. ID validity,
    native permission checks and execution remain the original tool's job.
    acceptance must return measured evidence, not model claims.
    """
    def __init__(self, env, *, allowed_tools, load_images, acceptance, evidence_format='append_v1'):
        if env.stage != 'furniture' or env.scene is None:
            raise ValueError('This adapter is qualified for the furniture stage only')
        self.env = env
        self.allowed = tuple(allowed_tools)
        if not self.allowed or len(set(self.allowed)) != len(self.allowed):
            raise ValueError('Explicit nonempty unique tool allowlist required')
        self.load_images = load_images
        self.acceptance = acceptance
        if evidence_format not in ('append_v1', 'canonical_v2'):
            raise ValueError('Unknown observation format')
        self.evidence_format = evidence_format

    def schemas(self):
        schemas = {}
        for name in self.allowed:
            tool = self.env.tool_map[name]
            schema = tool.params_json_schema
            if not isinstance(schema, dict):
                raise ValueError('Missing live FunctionTool JSON schema: ' + name)
            schemas[name] = copy.deepcopy(schema)
        return schemas

    def state(self):
        return copy.deepcopy(self.env.scene.to_state_dict())

    def context_digest(self):
        return digest({'state': self.state(), 'memory': self.env.mem,
                       'schemas': self.schemas()})

    def observe(self, history):
        before = self.context_digest()
        observation = self.env.observation()
        if self.context_digest() != before:
            raise RuntimeError('Scene/memory/schema changed during observation')
        messages = copy.deepcopy(observation['messages'])
        schemas = self.schemas()
        # Both arms get identical executable tools and real host feedback. D is
        # deliberately absent here. This is a NEW native deployment prompt;
        # it is not claimed token-identical to the archived 920 edit requests.
        formatter = canonical_host_evidence if self.evidence_format == 'canonical_v2' else append_host_evidence
        messages = formatter(messages, schemas=schemas,
                             before_state=self.state(), host_history=history)
        images = self.load_images(messages)
        if len(images) != 5:
            raise ValueError('Five current native views required')
        return {'messages': messages, 'images': images,
                'evidence_format': self.evidence_format,
                'state_sha256': digest(self.state()), 'context_sha256': before,
                'native_step_idx': observation['step_idx']}

    def validate(self, call):
        from jsonschema import Draft202012Validator, ValidationError
        if set(call) != {'name', 'arguments'} or call['name'] not in self.allowed:
            raise ActionRejected('Tool is not in the common executable allowlist')
        schema = self.schemas()[call['name']]
        Draft202012Validator.check_schema(schema)
        try:
            Draft202012Validator(schema).validate(call['arguments'])
            canonical(call)  # reject nonfinite numerical proposals
        except (ValidationError, ValueError) as exc:
            raise ActionRejected(str(exc)) from exc

    async def execute(self, call):
        return await self.env.step([call])

    def accept(self):
        value = self.acceptance(self.env)
        if not isinstance(value, dict) or value.get('state_sha256') != digest(self.state()):
            raise ValueError('Acceptance must be tied to the current native state')
        if value.get('full_scene_success') is not None:
            raise ValueError('Furniture-stage host cannot certify complete-room success')
        return value


async def run_phase_stage(*, port, policy, budget, output):
    """Count failures and both D/A calls, use actual post-tool state next turn."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {'status': 'running', 'arm': policy.arm, 'rounds': [],
              'evidence_format': getattr(port, 'evidence_format', None),
              'generation_attempts': 0, 'input_tokens': 0, 'output_tokens': 0,
              'tool_attempts': 0, 'native_calls': 0, 'accounting_complete': True,
              'budget': vars(budget), 'full_scene_success': None,
              'initial_state_sha256': digest(port.state())}
    history = []

    def save():
        result['elapsed_seconds'] = time.monotonic() - start
        write(output / 'result.json', result)

    def account(receipt):
        for key in ('generation_attempts', 'input_tokens', 'output_tokens'):
            result[key] += receipt.get(key, 0)
        if receipt.get('output_token_accounting_complete') is False:
            result['accounting_complete'] = False

    save()
    try:
        for turn in range(budget.turns):
            calls_needed = 1 if policy.arm in ('separate_direct_v3','joint_same_material_v1') else 2
            seconds = budget.wall_seconds - (time.monotonic() - start)
            tokens = budget.output_tokens - result['output_tokens']
            if (seconds <= 0 or tokens <= 0 or
                    result['generation_attempts'] + calls_needed > budget.generation_attempts):
                result['status'] = 'cumulative_budget_exhausted'
                break
            obs = port.observe(copy.deepcopy(history))
            seconds = budget.wall_seconds - (time.monotonic() - start)
            if seconds <= 0:
                result['status'] = 'observation_exhausted_wall_budget'
                break
            turn_dir = output / f'turn_{turn:03d}'
            turn_dir.mkdir()
            write(turn_dir / 'request.json', obs['messages'])
            row = {'turn': turn, 'before_sha256': obs['state_sha256'],
                   'evidence_format': obs.get('evidence_format'),
                   'context_sha256': obs['context_sha256']}
            result['rounds'].append(row)
            bounded = copy.copy(policy)
            bounded.total_tokens = min(policy.total_tokens, tokens)
            bounded.seconds = min(policy.seconds, seconds)
            bounded.seed = policy.seed + 2 * turn
            try:
                proposal = bounded.generate(obs)
            except PhaseOutcomeError as exc:
                proposal = exc.receipt
                account(proposal)
                row['policy'] = proposal
                history.append({'turn': turn, 'kind': 'policy_failure',
                                'status': proposal['status'],
                                'before_sha256': obs['state_sha256'],
                                'after_sha256': digest(port.state())})
                row['execution'] = history[-1]
                save()
                if not result['accounting_complete']:
                    result['status'] = 'backend_failure_accounting_incomplete'
                    break
                continue
            account(proposal)
            row['policy'] = proposal
            if (port.context_digest() != obs['context_sha256'] or
                    proposal['state_sha256'] != obs['state_sha256']):
                result['status'] = 'stale_observation_no_dispatch'
                break
            if time.monotonic() - start >= budget.wall_seconds:
                result['status'] = 'policy_exhausted_wall_budget_no_dispatch'
                break
            if proposal['status'] == 'stop_proposed':
                result['acceptance'] = port.accept()
                result['status'] = 'model_stop_proposed'
                row['execution'] = {'kind': 'stop_proposed',
                                    'after_sha256': digest(port.state())}
                break
            if proposal['status'] != 'tool_proposed' or len(proposal['tool_calls']) != 1:
                raise ValueError('Unexpected policy output contract')
            if result['tool_attempts'] >= budget.tool_attempts:
                result['status'] = 'tool_budget_exhausted_no_dispatch'
                break
            result['tool_attempts'] += 1
            call = proposal['tool_calls'][0]
            execution = {'turn': turn, 'call': call,
                         'before_sha256': obs['state_sha256']}
            try:
                port.validate(call)
            except ActionRejected as exc:
                # Validation changes no scene and does not bypass the budget.
                execution.update(kind='validation_rejection', error=str(exc))
            else:
                result['native_calls'] += 1
                execution['kind'] = 'native_execution'
                tick = time.monotonic()
                try:
                    execution['tool_results'] = await port.execute(call)
                except Exception as exc:
                    execution.update(kind='native_exception', error=str(exc))
                    raise
                finally:
                    execution['elapsed_seconds'] = time.monotonic() - tick
                    execution['after_sha256'] = digest(port.state())
                    row['execution'] = execution
            execution['after_sha256'] = digest(port.state())
            row['execution'] = execution
            history.append(copy.deepcopy(execution))
            write(turn_dir / 'actual_after_state.json', port.state())
            save()
        else:
            result['status'] = 'turn_budget_exhausted'
    except BaseException as exc:
        result.update(status='host_failure', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        result['final_state_sha256'] = digest(port.state())
        write(output / 'final_state.json', port.state())
        # Preserve evidence at EVERY exit, including a malformed policy/timeout.
        if 'acceptance' not in result:
            try:
                result['acceptance'] = port.accept()
            except Exception as exc:
                result['acceptance_error'] = str(exc)
        save()
    return result


@contextmanager
def furniture_driver(native_module, *, policy, budget, allowed_tools,
                     load_images, acceptance, evidence_format):
    """Temporarily connect to native main_async's existing four-stage path.

    The caller owns session/model configuration and all resource authorization.
    Other stages use the original driver and supplied common base policy. This
    does not instantiate services or silently skip floor/wall/ceiling stages.
    The experiment must explicitly choose its observation representation; a
    canonical training corpus must not silently deploy with append formatting.
    """
    if evidence_format not in ('append_v1', 'canonical_v2'):
        raise ValueError('Unknown observation format')
    original = native_module.driver_loop

    async def driver(*, env, policy: object, max_steps):
        if env.stage != 'furniture':
            return await original(env=env, policy=policy, max_steps=max_steps)
        port = NativeRoomPort(env, allowed_tools=allowed_tools,
                              load_images=load_images, acceptance=acceptance,
                              evidence_format=evidence_format)
        limited = StageBudget(min(max_steps, budget.turns), budget.generation_attempts,
                              budget.tool_attempts, budget.output_tokens, budget.wall_seconds)
        result = await run_phase_stage(port=port, policy=phase_policy,
                                       budget=limited, output=env.steps_dir / 'phase_host')
        rows = [{'step': r['turn'], 'think': None,
                 'tool_calls': r.get('policy', {}).get('tool_calls', []),
                 'tool_results': r.get('execution', {}).get('tool_results', []),
                 'host_status': r.get('execution', {}).get('kind', 'not_dispatched')}
                for r in result['rounds']]
        write(env.steps_dir / 'rollout.json', rows)
        if result['status'] in ('stale_observation_no_dispatch',
                                'backend_failure_accounting_incomplete'):
            raise RuntimeError(result['status'])
        return rows

    phase_policy = policy
    native_module.driver_loop = driver
    try:
        yield
    finally:
        native_module.driver_loop = original
