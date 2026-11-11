"""Census the 144 saved edit-policy outputs using the original strict contract.

No new model, tokenizer, native simulator, optimizer or source-file mutation.
Post-hoc normalization is tested as an ordinary common control, not a rollout.
"""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

from zero_argument_query import canonicalize_scene_query

PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
PREFIX = 'runs/920/P920-EDITSTUDY-v1_20261106T074752Z/'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--git-object-url', help='Optional transport for fetching '
                    'missing promisor Git objects; fixed commit remains unchanged')
    ap.add_argument('--public-blob-fallback', action='store_true', help='For '
                    'missing local objects, get exact SHA-pinned blobs from '
                    'this public repository API, verifying Git object identity')
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Keep original census outputs')
    a.out.mkdir(parents=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = []
    local_objects = set(subprocess.check_output([
        'git', 'cat-file', '--batch-all-objects',
        '--batch-check=%(objectname)']).decode().split()) if a.public_blob_fallback else None
    git_args = ['git', '-c', 'gc.auto=0']
    if a.git_object_url:
        git_args += ['-c', 'remote.origin.url='+a.git_object_url]
    reader = subprocess.Popen(git_args + ['cat-file', '--batch'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def blob(path):
        oid = subprocess.check_output(['git', 'rev-parse', PIN+':'+path]).decode().strip()
        if local_objects is not None and oid not in local_objects:
            cache = a.scratch/oid
            url = 'https://api.github.com/repos/sugusu/scenefunc/git/blobs/'+oid
            if cache.exists():
                raw = cache.read_bytes()
                origin = 'SHA-verified RAM object cache'
            else:
                import requests
                response = requests.get(url, timeout=25)
                response.raise_for_status()
                raw = base64.b64decode(response.json()['content'])
                cache.write_bytes(raw)
                origin = url
            assert hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest() == oid
            sources.append(dict(path=path, bytes=len(raw), sha256=digest(raw),
                                git_blob=oid, acquisition=origin))
            return raw
        reader.stdin.write((oid + '\n').encode())
        reader.stdin.flush()
        header = reader.stdout.readline().decode().strip().split()
        if len(header) != 3 or header[1] != 'blob':
            raise FileNotFoundError(path + ': ' + ' '.join(header))
        raw = reader.stdout.read(int(header[2]))
        if reader.stdout.read(1) != b'\n':
            raise RuntimeError('Broken Git batch framing')
        assert hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest() == oid
        sources.append(dict(path=path, bytes=len(raw), sha256=digest(raw),
                            git_blob=oid, acquisition='existing local Git object'))
        return raw

    def read(path):
        return json.loads(blob(path))

    def write(name, value):
        (a.out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False) + '\n')

    try:
        source = blob('scripts/920/shared_action_contract.py')
        module_path = a.scratch/'shared_action_contract.py'
        module_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('original_edit_contract', module_path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        tree = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PIN,
                                        PREFIX+'evaluation/']).decode().splitlines()
        outputs = [p for p in tree if p.endswith('/model/raw_output.txt')]
        assert len(outputs) == 144
        cells = sorted({p.split('/evaluation/')[1].split('/')[0] for p in outputs})
        receipts = {c: read(PREFIX+'evaluation/'+c+'/recorded/RESULT.json') for c in cells}
        assert len(cells) == 12
        rows = []
        totals = Counter()
        arms = defaultdict(Counter)
        by_cell = defaultdict(Counter)
        for path in sorted(outputs):
            cell = path.split('/evaluation/')[1].split('/')[0]
            turn = int(re.search(r'/round_(\d+)/', path).group(1))
            base = path.rsplit('/model/', 1)[0] + '/'
            raw = blob(path).decode()
            request = read(base+'request.json')
            context = original.context_from_request(request)
            system = next(m['content'] for m in request if m['role'] == 'system')
            if not isinstance(system, str):
                system = '\n'.join(x.get('text', '') for x in system)
            tools = original.schemas_from_observation(dict(system=system))
            contract = original.ActionContract(tools, context)
            receipt = next(x for x in receipts[cell]['rounds'] if x['turn'] == turn)
            blocks = re.findall(r'<tool_call>(.*?)</tool_call>', raw, re.DOTALL)
            call = None
            parse_rejection = None
            if (receipt['model'].get('output_rejected_reason') or
                    len(blocks) != 1 or raw.count('<tool_call>') != 1):
                parse_rejection = 'incomplete_or_invalid_tool_block'
            else:
                try:
                    call = original.strict_loads(blocks[0])
                except ValueError:
                    parse_rejection = 'invalid_strict_JSON'
            rejection = None
            try:
                if parse_rejection:
                    raise ValueError(parse_rejection)
                contract.validate(blocks[0], context)
            except Exception as exc:
                # Only known schema validation errors are part of the census.
                from jsonschema.exceptions import ValidationError
                if not isinstance(exc, (ValidationError, ValueError)):
                    raise
                rejection = type(exc).__name__
            was_rejected = 'rejection' in receipt
            assert was_rejected == (rejection is not None), path
            if not was_rejected:
                assert receipt['call'] == call
            normalized, change = canonicalize_scene_query(call, tools)
            normalized_valid = False
            try:
                if parse_rejection:
                    raise ValueError(parse_rejection)
                contract.validate(json.dumps(normalized, allow_nan=False), context)
                normalized_valid = True
            except Exception as exc:
                from jsonschema.exceptions import ValidationError
                if not isinstance(exc, (ValidationError, ValueError)):
                    raise
            arm = cell.split('_', 1)[1].rsplit('_', 1)[0]
            row = dict(cell=cell, arm=arm, turn=turn, raw_git_path=path,
                       raw_sha256=digest(raw.encode()), call=call,
                       rejected_original=was_rejected,
                       parse_rejection=parse_rejection,
                       validation_error_type=rejection,
                       has_explicit_think='<think>' in raw,
                       normalization=change,
                       normalized_valid_under_original_contract=normalized_valid,
                       normalized_call=normalized if change else None,
                       old_native_call_recorded=not was_rejected,
                       this_census_dispatched=False)
            rows.append(row)
            for count in [totals, arms[arm], by_cell[cell]]:
                count['outputs'] += 1
                count['schema_rejections'] += was_rejected
                count['format_or_incomplete_output_rejections'] += parse_rejection is not None
                count['old_native_calls'] += not was_rejected
                count['explicit_think'] += row['has_explicit_think']
                count['normalized_queries'] += change is not None
                count['normalized_queries_accepted'] += bool(change and normalized_valid)
                count['remaining_rejections_after_normalization'] += not normalized_valid
        # Negative controls ensure the adapter does not silently repair edits,
        # IDs, missing values or nonempty semantic query schemas.
        query_tools = [dict(name='get_current_scene_state', parameters=dict(
            type='object', properties={}, additionalProperties=False))]
        edit = dict(name='move_furniture_tool', arguments=dict(object_id='unknown', x=1))
        assert canonicalize_scene_query(edit, query_tools) == (edit, None)
        invalid = dict(name='get_current_scene_state', arguments=None)
        assert canonicalize_scene_query(invalid, query_tools) == (invalid, None)
        semantic_tools = [dict(name='get_current_scene_state', parameters=dict(
            type='object', properties=dict(object_id=dict(type='string')),
            additionalProperties=False))]
        q = dict(name='get_current_scene_state', arguments=dict(object_id='keep'))
        assert canonicalize_scene_query(q, semantic_tools) == (q, None)
        assert totals['old_native_calls'] == sum(r['native_calls'] for r in receipts.values())
        write('rows.json', rows)
        write('source_receipts.json', dict(pin=PIN, files=sources,
              git_object_url=a.git_object_url,
              native_validator_executed=True, simulator_or_model_imported=False))
        write('result.json', dict(
            scope='Retrospective strict-contract replay of saved responses, '
                  'not corrected closed-loop model trajectories',
            counts=dict(totals), arms={k: dict(v) for k, v in arms.items()},
            cells={k: dict(v) for k, v in by_cell.items()},
            distinct_response_sha256=len({r['raw_sha256'] for r in rows}),
            first_observable_failure='See exact saved calls in rows.json; query '
              'normalization can change dispatch eligibility but cannot choose '
              'a design or predict how later policy outputs would change.',
            normalization_is_common_engineering_control_not_innovation=True,
            old_quality_scores_unchanged=True,
            counterfactual_model_outputs_observed=False,
            new_model_calls=0, new_native_tool_calls=0, new_training=0,
            elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(totals), ensure_ascii=False))
    except BaseException as exc:
        write('failure.json', dict(error_type=type(exc).__name__, error=str(exc),
              elapsed_seconds=time.monotonic()-started, pin=PIN,
              files_read_before_error=sources, git_object_url=a.git_object_url,
              new_model_calls=0, new_training=0, new_native_tool_calls=0))
        raise
    finally:
        reader.stdin.close()
        reader.wait()


if __name__ == '__main__':
    main()
