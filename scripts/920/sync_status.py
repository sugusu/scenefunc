"""Snapshot bounded 920 evidence and atomically publish durable run status."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    local = json.loads((root / 'STATUS.json').read_text())
    run_id = local['run_id']
    run = repo / 'runs/920' / run_id
    status = json.loads((run / 'STATUS.json').read_text())
    now = datetime.now(timezone.utc).isoformat()
    sources = [
        'MODEL_DOWNLOAD_RECEIPT.json', 'MODEL_DOWNLOAD_LAUNCH.json',
        'MODEL_CPU_VALIDATION.json', 'MODEL_FORWARD_LAUNCH.json',
        'MODEL_FORWARD_PREFLIGHT.json', 'RUN_PROTOCOL.json',
        'MODEL_FORWARD_UTF8_LAUNCH.json', 'MODEL_FORWARD_UTF8_PREFLIGHT.json',
        'model_forward_preflight_utf8.log',
        'model_download.log', 'model_forward_preflight.log',
        'preflight/REPOSITORY_WEIGHTS_MATCH.json',
        'preflight/UPSTREAM_DOWNLOAD_920.json',
        'preflight/GEOMETRY_ENVIRONMENT.json',
        'preflight/cpu_render_smoke_receipt.json',
        'preflight/upstream_download_920.log',
        'preflight/geometry_install.log',
        'preflight/model_local/A_engineering_raw_output.txt',
        'preflight/model_local/A_engineering_raw_output_UTF8.txt',
        'preflight/model_local/A_engineering_token_ids_UTF8.json',
        'preflight/HSSD_AUTHENTICATED_ACCESS_CHECK.json',
    ]
    evidence = []
    for relative in sources:
        src = root / relative
        if src.is_file():
            assert src.stat().st_size < 2 * 1024**2
            dest = run / 'evidence' / src.name
            shutil.copyfile(src, dest)
            evidence.append({'path': str(dest.relative_to(run)), 'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})
    status.update(updated_at=now, stage=a.stage, status='partial', latest_verifiable_artifacts=evidence)
    status['configuration_hashes']['RUN_PROTOCOL.json'] = hashlib.sha256((root / 'RUN_PROTOCOL.json').read_bytes()).hexdigest()
    status['configuration_hashes']['upstream.920.lock.json'] = hashlib.sha256((run / 'upstream.920.lock.json').read_bytes()).hexdigest()
    status['failure_reasons'] = [x for x in status['failure_reasons'] if x['code'] != 'UPSTREAM_HASH_MISMATCH']
    status['source_difference'] = 'Official pinned source independently verified; see evidence/UPSTREAM_LOCK_CORRECTION.json. Historical executor byte equivalence is not claimed.'
    status['next_step'] = 'Finish selected asset hashes, reconstruct and audit both unrevised drafts, then validate five-view memory budget before four G/U revision calls.'
    forward = root / 'MODEL_FORWARD_PREFLIGHT.json'
    launch = root / 'MODEL_FORWARD_LAUNCH.json'
    attempts = [json.loads(p.read_text()) for p in [forward, root / 'MODEL_FORWARD_UTF8_PREFLIGHT.json'] if p.is_file()]
    if (root / 'MODEL_FORWARD_UTF8_LAUNCH.json').is_file():
        launch = root / 'MODEL_FORWARD_UTF8_LAUNCH.json'
        forward = root / 'MODEL_FORWARD_UTF8_PREFLIGHT.json'
    status['engineering_attempts'] = attempts
    if launch.is_file():
        receipt = json.loads(launch.read_text())
        status.update(model_started_at=receipt['started_at'], model_exit_code=receipt.get('exit_code'), model_launch=receipt)
        log = Path(receipt['log'])
        status['model_first_log'] = {'path': str(log), 'line': log.read_text().splitlines()[0] if log.exists() and log.stat().st_size else None}
        if receipt['status'] == 'running':
            status['status'] = 'running'
    if forward.is_file():
        result = json.loads(forward.read_text())
        status['completed_planned']['engineering_model_generations']['completed'] = int(result['status'] == 'pass')
        status['engineering_model_result'] = result['status']
        status['formal_capacity_validated'] = False
    status['activity'] = a.stage
    atomic(run / 'STATUS.json', status)
    atomic(repo / 'runs/920/latest.json', {'run_id': run_id, 'stage': a.stage, 'status': status['status'], 'updated_at': now, 'run_directory': str(root), 'status_path': str((run / 'STATUS.json').relative_to(repo))})
    local.update(updated_at=now, stage='preflight', milestone=a.stage, branch='runs/920', configuration_sha256=status['configuration_hashes']['RUN_PROTOCOL.json'],
                 repository_status=str(run / 'STATUS.json'), blockers=[x['effect'] for x in status['failure_reasons']], next_step=status['next_step'])
    local['resources']['executor'] = str(root / 'runtime' / run_id / 'work')
    local['resources']['assets'] = 'locked HSSD selected set; authorized access verified, asset integrity pending'
    local['shared_image'] = json.loads((root / 'RUN_PROTOCOL.json').read_text())['shared_image']
    if forward.is_file():
        result = json.loads(forward.read_text())
        local['preflight']['model_forward'] = result['status']
        local['actual_calls']['engineering_model_forward'] = {'successful': sum(x['status'] == 'pass' for x in attempts), 'failed': sum(x['status'] == 'failed' for x in attempts)}
    atomic(root / 'STATUS.json', local)
    print(json.dumps({'run_id': run_id, 'stage': a.stage, 'status': status['status'], 'evidence_files': len(evidence)}))


if __name__ == '__main__':
    main()
