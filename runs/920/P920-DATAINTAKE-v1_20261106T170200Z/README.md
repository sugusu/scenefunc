# Public data intake: actual availability and usable scope

The fixed M3DLayout layout archive has been acquired and its SHA256 matches the official LFS object. Its three training files were inspected; 36 deterministic source-layout records are staged locally. This supplies layout coverage for later asset binding, **not new professional decisions or executable Scene-Designer demonstrations**. No model inference, scene edits or training occurred.

| Official training source | Records | Distinct scene IDs | Objects | Nonempty description |
|---|---:|---:|---:|---:|
| 3D-FRONT | 4,603 | 4,233 | 31,744 | 4,603 |
| Infinigen | 11,141 | 11,141 | 300,173 | 6,112 |
| Matterport3D | 1,347 | 1,347 | 16,796 | 1,347 |
| Total | 17,091 | 16,721 within-source IDs | 348,713 | 12,062 |

Scene IDs are not independent project counts. 3D-FRONT has 370 repeated-ID groups: 60 contain identical records, 310 contain differing records. All repeated-ID groups were excluded from the staged subset. Selection is the first 12 valid, single-ID records per source ordered by SHA256(source/scene_id), without selecting for task success or descriptive quality.

All objects have finite, positive numerical geometry under the inspected schema. This does not establish collision freedom, usability, survey accuracy or full-room quality. No inspected object has an asset reference; no inspected scene has explicit room-shell fields. The upstream loader uses the xz ground plane and **half extents**, not full furniture dimensions. Top-level `bounds` are dataset statistics, not each room's walls. Source layouts remain unchanged in the staged records.

The original archive includes validation/test files. Only their archive-directory names and sizes were read; their payloads were not opened or checked with a full-archive CRC scan. No claim of project-isolated evaluation is made. The 36 inspected train samples are development-exposed and unadmitted for training.

The [dataset card](https://huggingface.co/datasets/Metaverse-AI-Lab/M3DLayout/blob/72a2a80c3b96788635b9192b64853e1428d59cb4/README.md) declares CC-BY-NC-4.0. Source-specific asset terms still apply. The [official curation code](https://github.com/Graphic-Kiliani/M3DLayout-code/tree/b83d584e9d1e348c70299f380d1e903c6228ef22/data_curation/infinigen) generates descriptions of existing layouts with a VLM. These descriptions must not become pre-design resident requirements or professional rationale labels.

## VIDA availability

The [official repository](https://github.com/Loewen-Hob/VIDA/tree/550c85009e59b7495be795ecb3d63fb552685b06) and dataset card describe InteriorClarify consultation clarification. The complete checked tree contains code/cards but no consultation annotation files; the two cards contain no payload download link or verified dataset license. Consequently, **zero consultation records were acquired**. This is a requirements-layer reference, not a professional-action dataset. Availability elsewhere remains unverified; do not wait on it or manufacture samples.

## Evidence and recovery

- `PROTOCOL.json`, `INSPECTION_LOCK*.json`: scope and script hashes frozen before the respective executions.
- `RECOVERY.json`: first archive transfer ended at 27,751,276 bytes; official size is 32,459,127. `acquisition_receipt` preserves exit 1. A single byte-range continuation completed in 17.134 seconds; official SHA256 is `10c0b11d43f980cd03f214cbd6bf377192257c7da58b56c94125d20375ef39e2`.
- `inspection_receipt`: initial audit rejected duplicate IDs; exit 1 retained with `inspection_script_v1.py`. V2 reports duplicates and excludes them from staging, rather than silently deduplicating. It completed in 5.534 seconds, exit 0.
- `inspection_v2/RESULT.json`: source statistics, archive member access list, 36 local paths and hashes.
- `VALIDATION.json`: all 36 staged records checked for source preservation and non-fabricated labels. No official Scene-Designer processor or native replay was run.
- `SOURCES.json`: fixed revisions, source links and local text snapshot hashes.

Raw archive and staged records stay at `/data/wyh/scene_cvpr_20261020/design_decisions_920/external_data/m3dlayout_72a2a80/`. Commands, first logs and real start/exit times are in the receipts. Restore from the fixed dataset URL, verify SHA256, then run `inspect_m3dlayout_training.py` with fresh output directories and the recorded arguments. The intermediate representation is deliberately unbound; it cannot be submitted as native tool actions.

## Research decision

Do not expand this into a multi-terabyte scene download or count 17,091 layouts as professional supervision. Preserve this resource for basic layout/action coverage. The next core experiment still needs source-grounded conditional decisions, a working common execution interface, and whole-room evaluation. Any mesh/asset integration must first demonstrate utility on a bounded source sample; any additional training needs a concrete separately reviewed plan. The previous EDITSTUDY result remains negative and unchanged.
