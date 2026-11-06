"""Record Pine's attributed desk-reuse decision before native derivation."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    repo = Path(__file__).resolve().parents[2]
    base = repo.parent
    now = datetime.now(timezone.utc)
    run = 'P920-PINE-v1_' + now.strftime('%Y%m%dT%H%M%SZ')
    public = repo / 'runs/920' / run
    runtime = base / 'runtime' / run
    public.mkdir()
    runtime.mkdir()
    family = 'sydney_pine_nicholas_gurney_2022'
    registry_path = repo / 'research/920/data_readiness_v1/PROJECT_REGISTRY.json'
    registry = json.loads(registry_path.read_text())
    assert not any(row['family_id'] == family for row in registry['projects'])
    record = {
        'project_id': family, 'family_id': family, 'allocation': 'train',
        'split_fixed_at': now.isoformat(),
        'allocation_reason': 'Full interview and identity page already read; allocated before geometry/action derivation. Not a sealed test.',
        'reported_name': 'Pine / Everything Room', 'reported_architect': 'Nicholas (Nick) Gurney',
        'source_url': 'https://www.nevertoosmall.com/post/two-creative-maximalists-living-minimally',
        'primary_verified_url': 'https://www.nicholasgurney.com.au/pine',
        'aliases': ['Pine', 'Everything Room', 'Vicki Papa', 'Vicki Papageorgopoulos', 'Kirthana Selvaraj'],
        'source_correlation_group': 'never_too_small_editorial',
        'identity_status': 'Designer page confirms Pine/Sydney/55sqm/2022. NTS Small Home: Pine search record links resident names; direct secondary-page open timed out.',
        'source_review_exposure': 'Search and full interview exposed; no source photos, floorplans or CAD downloaded.',
        'training_eligible': False, 'training_authorized': False, 'sealed_test_eligible': False,
        'derived_knowledge_records': 1, 'derived_native_episodes': 0,
        'original_project_research_rights': 'Not established; attributed factual paraphrases only, no source images/full article copied.',
        'knowledge_decision_ids': ['pine_existing_desk_with_window_workstation_for_two_people'],
        'knowledge_record': str((public / 'SOURCE_RECORD.json').relative_to(repo)),
    }
    source = {
        'decision_id': record['knowledge_decision_ids'][0], 'family_id': family,
        'source_urls': [record['source_url'], record['primary_verified_url']],
        'checked_at': now.isoformat(), 'editorial_author': 'Camilla Janse van Vuuren',
        'editorial_published': '2025-05-22',
        'project_facts': {'reported_apartment_area_m2': 55, 'designer': 'Nicholas Gurney',
                          'reported_project_year': 2022,
                          'room_uses': ['creative work', 'guest room', 'books and collected objects']},
        'evidence_locator': 'NTS section The Everything Room, paragraph on window desk, existing IKEA desk and concurrent work (web lines173–178). Official Pine page lines15–17 confirms identity only.',
        'source_supported_components': [
            {'actor': 'designer Gurney', 'fact_zh': '在窗周设计书架，并设置窗下定制书桌。',
             'attribution': 'Interview author report, not a verbatim designer explanation.'},
            {'actor': 'residents', 'fact_zh': '沿相邻墙再利用原有IKEA桌，使两人能同时在房间工作。',
             'attribution': 'Reported resident adaptation; do not relabel as designer action.'},
        ],
        'condition_summary_zh': '创作、收藏、客用的同一房间需要两人同时工作，并继续使用已有桌子。',
        'rationale_summary_zh': '项目把设计师的窗下工作面与住户沿相邻墙再利用的原桌结合，让两个工作位共存。',
        'rationale_scope': 'Source-supported co-produced arrangement, not an expert ranking or historical software trajectory.',
        'raw_resident_brief_available': False, 'original_dimensions_or_ikea_model': None,
        'whole_source_room_reconstruction': False,
        'planned_reconstruction_scope': 'Research transfer of an existing-instance/two-workstation decision into a declared study/guest room; other room uses retained in evaluation, no folding or built-in shelving claim.',
        'unknowns': ['Original desk/room dimensions and historical initial state',
                     'Construction chronology and individual authorship of each step',
                     'Guest sleeping mechanism, load, comfort, storage capacity and lighting performance',
                     'Independent professional judgment of our reconstruction'],
        'reuse_scope': 'Short attributed factual paraphrases only; no copied project images/full article and no training authorization.',
        'identity_evidence_note': 'Existing desk reuse is source-reported; native ID represents a substituted initial instance, not the historical IKEA object.',
        'alias_crosscheck': 'NTS Small Home: Pine search record links resident names/55sqm/designer; direct open timed out. Official Pine page confirms project identity but not desk action.',
    }
    write(public / 'ALLOCATION.json', record)
    write(public / 'SOURCE_RECORD.json', source)
    registry['projects'].append(record)
    registry['updated_at'] = now.isoformat()
    write(registry_path, registry)
    metadata = base / 'resources/research_support/decision_asset_cache/datasets--hssd--hssd-hab/snapshots/4369cb9876214c7fbebcf552eb532380e4d287e4/semantics/objects.csv'
    candidates = []
    for row in csv.DictReader(metadata.open()):
        if row['wnsynsetkey'] != 'desk.n.01' or row['hasMultipleObjects'].lower() not in ('false', '0', ''):
            continue
        try:
            dims = [float(x) for x in row['aligned.dims'].split(',')]
        except ValueError:
            continue
        if len(dims) == 3 and 1.3 <= dims[0] <= 1.8 and .68 <= dims[1] <= .8 and .55 <= dims[2] <= .85:
            candidates.append({key: row[key] for key in ['id', 'name', 'aligned.dims']})
    candidates.sort(key=lambda row: row['id'])
    chosen = candidates[0]
    write(public / 'ASSET_SELECTION.json', {
        'metadata_path': str(metadata), 'metadata_sha256': sha(metadata),
        'filter': 'Single desk; metadata width1.3–1.8m,height.68–.8m,depth.55–.85m; lexical ID first.',
        'candidates': candidates, 'selected': chosen, 'planned_meshes': 1,
        'native_frame_pending': True, 'not_original_ikea_model': True,
    })
    asset_plan = {'run_id': run, 'frozen_at': now.isoformat(), 'asset_id': chosen['id'],
                  'asset_name': chosen['name'],
                  'budget': {'download_files': 2, 'maximum_download_bytes': 16*1024**2,
                             'resolve_seconds': 120, 'download_seconds': 180,
                             'inspect_seconds': 120, 'previews': 2, 'retries': 0},
                  'selection_plan_sha256': sha(public / 'ASSET_SELECTION.json'),
                  'purpose': 'One unscaled desk substitute for old and new workstations; not original IKEA/bespoke geometry.'}
    write(public / 'ASSET_PLAN.json', asset_plan)
    write(public / 'STATUS.json', {
        'run_id': run, 'question': 'Retain source-reported old desk while supporting two workstations and accounting for other room uses.',
        'status': 'source_allocated_asset_prepared', 'updated_at': now.isoformat(),
        'run_directory': str(runtime), 'completed_planned': {'sources': [1, 1], 'mesh_downloads': [0, 1], 'training_updates': [0, 0]},
        'configuration_hashes': {'asset_plan': sha(public / 'ASSET_PLAN.json')},
        'input_hashes': {'source': sha(public / 'SOURCE_RECORD.json'), 'metadata': sha(metadata)},
        'latest_verifiable_artifact': str(public / 'SOURCE_RECORD.json'), 'failure_reason': None,
        'next_step': 'Inspect one desk, then freeze finite reconstruction geometry/actions before execution.',
    })
    write(public.parent / 'latest.json', {'run_id': run, 'stage': 'pine_source_and_asset_prepared',
                                        'updated_at': now.isoformat(), 'run_directory': str(runtime)})
    write(public / 'PREPARATION_NOTES.json', {
        'earlier_shell_snippet': 'An inline source-preparation snippet had a Python syntax error (unclosed source_urls list). It never executed or created a run. Replaced with this persisted script.',
        'source_search_limit': 'Local source registry/research owner-name and URL checks found no prior family; not global duplicate detection.',
    })
    print(run, chosen)


if __name__ == '__main__':
    main()
