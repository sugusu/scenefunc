"""Index source identities before any room derivation or train/test allocation.

Streams the geometry CSV, discarding WKT values; this is a hierarchy audit,
not a geometry deduplicator, semantic label inspection or native import.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = {'started_at': now(), 'status': 'running', 'rows_scanned': 0,
              'model_calls': 0, 'training_updates': 0, 'allocated_projects': 0,
              'plan_sha256': hashlib.sha256(a.plan.read_bytes()).hexdigest()}
    def save():
        tmp = a.out / 'RESULT.json.tmp'
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(a.out / 'RESULT.json')
    save()
    print(now(), 'begin source identity scan; no room geometry output or split allocation', flush=True)
    started = time.monotonic()
    try:
        digest = hashlib.sha256()
        archive = Path(plan['archive'])
        assert archive.stat().st_size == plan['archive_bytes']
        with archive.open('rb') as f:
            for block in iter(lambda: f.read(2**20), b''):
                digest.update(block)
        assert digest.hexdigest() == plan['archive_sha256']
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['script_sha256']
        sites, buildings, plans, floors = set(), set(), set(), set()
        apartment_sites = defaultdict(set)
        building_sites, global_plan_parents = defaultdict(set), defaultdict(set)
        plan_floors, units = defaultdict(set), set()
        missing, entity_types, usages = Counter(), Counter(), Counter()
        columns_required = ['apartment_id', 'site_id', 'building_id', 'plan_id', 'floor_id',
                            'unit_id', 'area_id', 'unit_usage', 'entity_type', 'entity_subtype',
                            'geometry', 'elevation', 'height']
        members = []
        csv.field_size_limit(4 * 1024**2)
        with zipfile.ZipFile(archive) as z:
            assert len(z.infolist()) <= 10
            for item in z.infolist():
                assert not item.is_dir() and item.filename.endswith('.csv')
                assert not Path(item.filename).is_absolute() and '..' not in Path(item.filename).parts
                with z.open(item) as f:
                    header = f.readline(65536)
                fields = next(csv.reader([header.decode('utf-8-sig').rstrip('\r\n')]))
                members.append({'member': item.filename, 'bytes': item.file_size,
                                'compressed_bytes': item.compress_size, 'crc32': f'{item.CRC:08x}',
                                'column_count': len(fields), 'columns': fields,
                                'header_sha256': hashlib.sha256(header).hexdigest()})
            member = plan['geometry_member']
            assert z.getinfo(member).file_size <= plan['budget']['max_uncompressed_geometry_bytes']
            with z.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))
                assert reader.fieldnames == columns_required
                for index, row in enumerate(reader, 1):
                    if index > plan['budget']['max_rows']:
                        raise ValueError('Frozen row limit exceeded')
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError('Malformed CSV row')
                    site, building, pid, floor, unit, apartment = (
                        row[k] for k in ['site_id', 'building_id', 'plan_id', 'floor_id', 'unit_id', 'apartment_id'])
                    for key in columns_required[:8]:
                        if not row[key]:
                            missing[key] += 1
                    sites.add(site)
                    buildings.add((site, building))
                    plans.add((site, building, pid))
                    floors.add((site, building, floor))
                    building_sites[building].add(site)
                    global_plan_parents[pid].add((site, building))
                    plan_floors[(site, building, pid)].add(floor)
                    entity_types[row['entity_type']] += 1
                    usages[row['unit_usage']] += 1
                    if apartment:
                        apartment_sites[apartment].add(site)
                    if unit:
                        units.add((site, building, pid, floor, unit, apartment, row['unit_usage']))
                    result['rows_scanned'] = index
                    if index % 100000 == 0:
                        if time.monotonic() - started > plan['budget']['seconds']:
                            raise TimeoutError('Frozen index deadline exceeded')
                        save()
                        print(now(), 'rows', index, flush=True)
        # Do not include geometry, outcome measures or source coordinates in these indexes.
        with (a.out / 'unit_identity_index.csv').open('x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['site_id', 'building_id', 'plan_id', 'floor_id', 'unit_id', 'apartment_id', 'unit_usage'])
            writer.writerows(sorted(units))
        with (a.out / 'plan_floor_index.csv').open('x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['site_id', 'building_id', 'plan_id', 'floor_ids', 'floor_count'])
            writer.writerows([*key, '|'.join(sorted(value)), len(value)] for key, value in sorted(plan_floors.items()))
        (a.out / 'MEMBERS.json').write_text(json.dumps(members, indent=2) + '\n')
        result.update(status='source_identity_index_complete_no_split', exit_code=0,
                      counts={'sites': len(sites), 'site_buildings': len(buildings), 'site_building_plans': len(plans),
                              'site_building_floors': len(floors), 'unit_identity_rows': len(units),
                              'apartment_ids': len(apartment_sites), 'site_apartment_pairs': sum(map(len, apartment_sites.values()))},
                      missing_identifier_rows=dict(missing), entity_type_counts=dict(entity_types),
                      unit_usage_row_counts=dict(usages),
                      repeated_plans_on_multiple_floors=sum(len(v) > 1 for v in plan_floors.values()),
                      building_ids_in_multiple_sites=sum(len(v) > 1 for v in building_sites.values()),
                      plan_ids_with_multiple_site_building_parents=sum(len(v) > 1 for v in global_plan_parents.values()),
                      apartment_ids_in_multiple_sites=sum(len(v) > 1 for v in apartment_sites.values()),
                      all_cases_allocation='quarantine_unallocated',
                      source_example_site127_present='127' in sites,
                      geometry_member_crc_verified_by_full_zip_stream=True,
                      other_members_crc_read_to_end=False,
                      geometry_values_parsed_or_rendered=False,
                      simulations_values_inspected=False,
                      limitations='Identifier relationships only. Missing values and repeated plans need explicit treatment; cross-site geometric similarity/project aliases not checked. No family independence, sealed split, professional decisions or native room fidelity established.')
    except BaseException as exc:
        result.update(status='failed', exit_code=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(ended_at=now(), elapsed_seconds=time.monotonic() - started)
        save()
        print(now(), result['status'], 'rows', result['rows_scanned'], flush=True)


if __name__ == '__main__':
    main()
