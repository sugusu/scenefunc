"""Execute the frozen B0/B1 furnished development comparison, no retries.

Uses existing model, argument validator, direct native editing closure and mesh
checks. It is neither new room generation nor a physics-ready SDK reproduction.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cpu_continuation import dump, read, sha, execute
from bounded_trajectory import make_request

W = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
R = W / 'autodl_continuation_20261110/repository'
CODE = R / 'scripts/autodl'
NATIVE = '/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python'
RENDER = '/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python'
MODEL = str(W / 'v100_runtime_20261104/runtime/bin/python')
CPU = W / 'autodl_continuation_20261110/cpu_replay/keep'
RAM = Path('/dev/shm/FRP1-PARIS-B-20261111')
sys.path.insert(0, str(W / 'joint_use_feedback_20261104'))
from evaluate import strong_measure, changes, payload
from measure_joint_use import measure


def initial():
    return dict(checkpoint=str(CPU/'tools/checkpoint.json'), audit=str(CPU/'audit.json'),
                measured=str(CPU/'planar_measurements.json'), joint=str(CPU/'complete_finite_pool_joint.json'),
                views=str(CPU/'views_receipt.json'), history=[], turns=0, status='active')


def model_launch(root, folder):
    command = [sys.executable, str(CODE/'furnished_pilot_process.py'), '--root', str(root),
               '--folder', str(folder), '--kind', 'model', '--label', 'model_process', '--seconds', '200', '--',
               MODEL, str(CODE/'model_step.py'), '--phase', 'generate', '--work-base', str(W),
               '--request', str(folder/'request.json'), '--output', str(folder/'model')]
    # The runner stores authoritative receipts before and after child execution.
    return subprocess.run(command, check=False).returncode


def observed(root, arm, step, state, calls):
    turn = root/arm/f'turn_{step:02d}'
    ramturn = RAM/arm/f'turn_{step:02d}'
    ramturn.mkdir(parents=True, exist_ok=True)
    dump(turn/'calls.json', calls)
    execute([NATIVE, str(W/'paris_local_run_results_20261104/dispatch_corrected.py'),
             '--calls', str(turn/'calls.json'), '--resume', state['checkpoint'],
             '--output', str(ramturn/'tools')], turn, 'dispatch', 180)
    cp = ramturn/'tools/checkpoint.json'
    for name in ['checkpoint.json', 'receipt.json', 'tool_responses.json', 'asset_records.json']:
        p = cp.parent/name
        if p.exists(): shutil.copyfile(p, turn/name)
    execute([NATIVE, str(W/'paris_local_run_results_20261104/audit_scene.py'), '--checkpoint', str(cp),
             '--out', str(turn/'audit.json')], turn, 'mesh_audit', 120)
    measured = measure(cp)
    joint = strong_measure(measured)
    dump(turn/'measurements.json', measured)
    dump(turn/'joint.json', joint)
    diff = changes(payload(Path(state['checkpoint'])), payload(cp))
    assert not diff['added'] and not diff['removed'], 'Preserve all initial IDs'
    dump(turn/'state_changes.json', diff)
    execute([RENDER, str(W/'paris_local_run_results_20261104/render_state.py'), '--scene',
             str(cp.parent/'partial_scene.glb'), '--out', str(ramturn/'views'), '--transient-blend'],
            turn, 'render', 300)
    shutil.copyfile(ramturn/'views/receipt.json', turn/'views_receipt.json')
    return dict(checkpoint=str(cp), audit=str(turn/'audit.json'), measured=str(turn/'measurements.json'),
                joint=str(turn/'joint.json'), views=str(ramturn/'views/receipt.json'))


def consume(root, arm, step, state):
    turn = root/arm/f'turn_{step:02d}'
    receipt = read(turn/'model_process_receipt.json')
    while receipt['status'] != 'terminal':
        # An independently launched initial arm may overlap CPU rendering. Wait
        # only while its recorded runner is actually alive; never relaunch it.
        os.kill(receipt['runner_pid'], 0)
        if time.time()-receipt['start_unix'] > receipt['timeout_seconds']+15:
            raise RuntimeError('Recorded model runner exceeded accounting window')
        time.sleep(2)
        receipt = read(turn/'model_process_receipt.json')
    if receipt['exit_code'] != 0:
        state['status'] = 'model_failure_no_retry'
        return state
    calls = read(turn/'model/parsed_calls.json')
    raw = (turn/'model/raw_output.txt').read_text()
    if not calls:
        state['status'] = 'native_no_tool_termination_not_success'
        state['history'].append(dict(raw_model_output=raw, feedback='no tool requested'))
        return state
    valid = len(calls) == 1 and calls[0].get('name') == 'move_furniture_tool'
    args = calls[0].get('arguments', {}) if valid else {}
    valid = valid and isinstance(args, dict) and set(args) == {'object_id', 'x', 'y', 'yaw'}
    valid = valid and args['object_id'] in {o['object_id'] for o in payload(Path(state['checkpoint']))['placed_instances']}
    valid = valid and all(isinstance(args[k], (int, float)) and not isinstance(args[k], bool)
                          and math.isfinite(args[k]) for k in ['x','y','yaw'])
    if not valid:
        feedback = dict(success=False, error='Exactly one finite, existing-ID, advertised move required')
        dump(turn/'rejection.json', feedback)
    else:
        updates = observed(root, arm, step, state, calls)
        state.update(updates)
        feedback = read(turn/'tool_responses.json')
    state['history'].append(dict(raw_model_output=raw, feedback=feedback))
    state['turns'] = step+1
    if state['turns'] >= 4: state['status'] = 'four_attempt_cap'
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--arms', nargs='+', default=['B0','B1'])
    a = ap.parse_args()
    root = a.root.resolve()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      PYTHONPATH='/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps')
    assert read(root/'cold_restore_verified.json')['payload_equal']
    assert read(root/'cold_restore_native_result.json')['exit_code'] == 0
    states = {arm: read(root/arm/'state.json') if (root/arm/'state.json').exists() else initial() for arm in a.arms}
    for step in range(4):
        for arm in a.arms:
            state = states[arm]
            if state['status'] != 'active' or state['turns'] > step: continue
            turn = root/arm/f'turn_{step:02d}'
            turn.mkdir(parents=True, exist_ok=True)
            try:
                if not (turn/'request.json').exists():
                    q = make_request(W, Path(state['checkpoint']), read(state['audit']), read(state['measured']),
                                     read(state['joint']), Path(state['views']), state['history'])
                    q[0] = read(root/arm/'turn_00/request.json')[0]
                    q[-1]['content'][-1]['text'] += '\nCommon retrieved professional case (shared unchanged across model conditions):\n' + json.dumps(read(root/'common_case.json'),ensure_ascii=False,separators=(',',':'))
                    dump(turn/'request.json', q)
                if not (turn/'model_process_receipt.json').exists():
                    prior = [read(p) for p in root.glob('B*/turn_*/model_process_receipt.json')]
                    if sum(p.get('elapsed_seconds',200) for p in prior) >= 1200:
                        state['status'] = 'shared_model_time_cap'
                        continue
                    model_launch(root, turn)
                state = consume(root, arm, step, state)
            except Exception as exc:
                state.update(status='execution_failure_no_retry', error=f'{type(exc).__name__}: {exc}')
                dump(turn/'failure.json',dict(error=state['error']))
            finally:
                states[arm] = state
                dump(root/arm/'state.json', state)
                print(json.dumps(dict(arm=arm,step=step,status=state['status'],checkpoint=state['checkpoint'])),flush=True)
    dump(root/('model_arms_terminal_'+'_'.join(a.arms)+'.json'), states)


if __name__ == '__main__':
    main()
