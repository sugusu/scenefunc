"""Inspect the fixed author release; preserve payload privately and publish metadata only."""
import csv, hashlib, io, json, stat, zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from PIL import Image
P=Path(__file__).resolve().parent.parent;R=P.parents[2];D=R.parent;T=D/'runtime'/P.name
read=lambda f:json.loads(f.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def save(f,v):
 with f.open('x') as out:json.dump(v,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')
print(datetime.now(timezone.utc).isoformat(),'inspect one fixed professional-review mini archive',flush=True)
plan=read(P/'PROTOCOL.json');archive=T/'ArchiMetricsNet-Mini.zip';data=archive.read_bytes()
assert len(data)==plan['archive']['bytes']
blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==plan['archive']['git_blob_sha1']
root=T/'extracted';root.mkdir();members=[];records=[]
with zipfile.ZipFile(archive) as z:
 infos=z.infolist();assert len(infos)<=plan['budget']['max_archive_members']
 assert sum(x.file_size for x in infos)<=plan['budget']['max_uncompressed_bytes']
 assert len(set(x.filename for x in infos))==len(infos)
 for entry in infos:
  path=PurePosixPath(entry.filename);assert not path.is_absolute() and '..' not in path.parts and '\\' not in entry.filename
  assert not stat.S_ISLNK(entry.external_attr>>16)
  if entry.is_dir():continue
  payload=z.read(entry);assert len(payload)==entry.file_size
  dest=root/entry.filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
  members.append({'path':entry.filename,'bytes':len(payload),'sha256':sha(payload)})
 for entry in infos:
  if not entry.filename.endswith('.csv'):continue
  path=Path(entry.filename);stem=str(path.with_suffix(''));names={'scores':entry.filename,'plan':stem+'.png','analysis':stem+'_analysis.txt'}
  assert all(n in z.namelist() for n in names.values())
  reader=csv.DictReader(io.StringIO(z.read(names['scores']).decode('utf-8-sig')));assert reader.fieldnames==['Flow','Functionality','Overall']
  rows=list(reader);assert len(rows)==1;scores={k:float(v) for k,v in rows[0].items()};assert all(1<=v<=10 for v in scores.values())
  analysis=z.read(names['analysis']).decode('utf-8-sig');assert analysis.strip()
  with Image.open(io.BytesIO(z.read(names['plan']))) as im:
   size=list(im.size);mode=im.mode;im.verify()
  records.append({'record_id':path.stem,'category':path.parent.name,'scores_raw':scores,'scores_scale':'Observed raw 1..10; paper mentions normalized 0..1, no silent rescaling.',
   'files':{k:{'path':str((root/v).relative_to(D)),'sha256':sha(z.read(v))} for k,v in names.items()},'image_size':size,'image_mode':mode,'analysis_characters':len(analysis),
   'allocation':'development_source_review','professional_provenance':'Paper-level author report, no per-record annotator or QC identifiers in released files.',
   'original_project_family':None,'original_brief_available':False,'native_3d_actions':False,'training_eligible':False})
assert len(records)==165 and len(members)==496
save(T/'RECORD_INDEX.json',{'records':records,'source_commit':plan['commit'],'training_admitted':False})
image_hashes=[x['files']['plan']['sha256'] for x in records];text_hashes=[x['files']['analysis']['sha256'] for x in records]
metrics={'recorded_at':datetime.now(timezone.utc).isoformat(),'status':'released_professional_review_samples_inspected','source_commit':plan['commit'],
 'archive':{'path':str(archive),'bytes':len(data),'sha256':sha(data),'git_blob_sha1':blob},'zip_entries':len(infos),'payload_files':len(members),'uncompressed_bytes':sum(x['bytes'] for x in members),
 'complete_image_score_text_triples':len(records),'categories':len(set(x['category'] for x in records)),'unique_image_hashes':len(set(image_hashes)),
 'unique_record_ids':len(set(x['record_id'] for x in records)),'unique_analysis_hashes':len(set(text_hashes)),
 'image_shapes':dict(Counter(str(x['image_size'])+' '+x['image_mode'] for x in records)),
 'score_ranges':{k:{'minimum':min(x['scores_raw'][k] for x in records),'maximum':max(x['scores_raw'][k] for x in records),'distinct':len(set(x['scores_raw'][k] for x in records))} for k in ['Flow','Functionality','Overall']},
 'score_normalization_discrepancy':'Released CSV stores raw values on 1..10; paper Appendix A mentions normalization to 0..1. Preserve raw values and declare any later transform.',
 'paper_report':{'section':'3.1 and Appendix A','unaugmented_ratings_and_analyses':24000,'augmented_rating_pairs':120000,'annotators':25,'quality_checkers':4,'ratings_per_image':5,'source':'https://aclanthology.org/2025.acl-long.331.pdf','scope':'Author-reported full-dataset method; do not equate this Mini with the entire dataset or individually verified annotation history.'},
 'available_supervision':'Published-plan review: functionality, flow, overall scores and short analyses. No same-project 3D edit targets or critiques of our generated rooms.',
 'permission_scope':'No license found in complete fixed repository tree or archive member names; no public payload redistribution or training admission in this intake.',
 'counts_not_conflated':{'our_candidates_professionally_reviewed':0,'independent_source_projects_verified':0,'native_scene_action_pairs':0,'training_admitted':0},
 'server_record_index':str(T/'RECORD_INDEX.json'),'next_step':'Build a bounded source-review calibration set with original category/provenance and inspect case-to-critique alignment. Keep this entire Mini in development; no training or sealed-test claim.'}
save(P/'RESULT.json',metrics);save(P/'MEMBER_INDEX.json',{'files':members,'scope':'Paths, sizes and hashes only; annotation payload retained on server.'})
print(json.dumps({k:metrics[k] for k in ['complete_image_score_text_triples','categories','unique_image_hashes','unique_record_ids','unique_analysis_hashes','image_shapes','score_ranges']}),flush=True)
