"""Distinguish MOMENT checkpoint names from their recorded Flan-T5 backbone."""
from pathlib import Path
import csv
import json
import shutil
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as native
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

out=ROOT/'projects/flow_matching_research/results/2026-10-16/MOMENT_full22_union_and_three_industrial_matched_controls_v1'
previous_integrity=sha(out/'packet_integrity.json')
for pin in read(out/'packet_integrity.json')['files']:assert sha(out/pin['path'])==pin['sha256']
runtime=read(ROOT/'configs/runtime/fm_moment_full78_checkpoint_replay.v1.json');cases=native.verify(runtime)
bindings=read(out/'all22_complete_source_native_review_bindings.json')['records']
for cid,b in bindings.items():
    n=read(b['native_path']);assert sha(b['native_path'])==b['native_sha256']
    case=cases[cid];manifest=read(ROOT/case['job']['dataset_manifest_path']);data=next(d for d in manifest['datasets'] if d['id']==case['job']['dataset'])
    assert len(n['native_coverage'])==len(native.groups(case['job'],data))
p=out/'all18_fixed_validation_metrics_and_PA_controls.csv'
with p.open(encoding='utf-8-sig',newline='') as f:before=list(csv.DictReader(f))
after=[]
for row in before:
    c=cases[row['canonical_id']]['config'];model=Path(c['parameters']['checkpoint_directory']).name
    assert model in ['MOMENT-1-small','MOMENT-1-base']
    after.append(dict(row,size=model,backbone=row['size']))
assert len(before)==len(after)==18
assert all({k:v for k,v in a.items() if k not in ['size','backbone']}=={k:v for k,v in b.items() if k not in ['size','backbone']} for a,b in zip(before,after))
with p.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(after[0]));w.writeheader();w.writerows(after)
p=out/'README.md';text=p.read_text(encoding='utf-8')
assert text.count('google/flan-t5-base')==text.count('google/flan-t5-small')==3
text=text.replace('google/flan-t5-base','MOMENT-1-base').replace('google/flan-t5-small','MOMENT-1-small')
p.write_text(text,encoding='utf-8',newline='\n')
shutil.copyfile(ROOT/'tmp/close_MOMENT_full22_union_and_industrial_controls_20261016_v1.py',out/'closure_source.py')
shutil.copyfile(__file__,out/'publication_source.py')
write(out/'checkpoint_label_and_full_group_coverage_validation.json',dict(previous_unpublished_packet_integrity_sha256=previous_integrity,
    actual_official_checkpoint_names_and_recorded_backbone_distinguished=True,all18_numerical_rows_unchanged=True,
    all22_registered_group_counts_reverified=True,numerical_source_guards_arrays_models_unchanged=True,
    neural_forwards_and_training_updates_added=0))
write(out/'packet_integrity.json',dict(files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='packet_integrity.json']))
manifest=ROOT/'tmp/MOMENT_full22_union_industrial_controls_manifest_20261016_v1.json';m=read(manifest)
m['paths']=sorted(set(m['paths'])|{p.relative_to(ROOT).as_posix() for p in out.rglob('*') if p.is_file()}|{'projects/flow_matching_research/README.rst'})
page=ROOT/'projects/flow_matching_research/README.rst';page.write_text(page.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
write(manifest,m)
print(json.dumps(dict(complete_unique_cases=22,publication_files=len(m['paths']),numerical_rows_preserved=18)))
