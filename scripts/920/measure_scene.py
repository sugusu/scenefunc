"""Recompute actual top-surface witnesses for one immutable checkpoint."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    source = a.work / 'joint_use_witness_20261104/measure_joint_use.py'
    spec = importlib.util.spec_from_file_location('geometric_witness', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.measure(a.checkpoint)
    result['measurement_source_sha256'] = module.digest(source)
    result['protocol_sha256'] = module.digest(source.parent / 'PROTOCOL.json')
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'scene_sha256': result['scene_sha256'], 'seconds': result['seconds'], 'simultaneous': result['results']['scene_section_free']['simultaneous_use']['status']}))


if __name__ == '__main__':
    main()
