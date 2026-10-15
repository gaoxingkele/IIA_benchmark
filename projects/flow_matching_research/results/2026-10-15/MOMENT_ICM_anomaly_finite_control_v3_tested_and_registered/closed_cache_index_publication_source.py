"""Bind independently closed source caches into configs, retaining322 explicit gaps."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
from scripts.flow_matching import prepare_moment_icm_anomaly_finite_control_v3 as p

ROOT,read,sha,write=p.ROOT,p.read,p.sha,p.write
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_anomaly_finite_control_v3_tested_and_registered'
proof_path=folder/'actual_full1627_cache_closure_and322_missingness_audit.json';proof=read(proof_path)
assert proof['passed'] and proof['original_successful_caches_independently_closed']==1627 and proof['complete_original_cache_arrays']==14643
source=read(ROOT/p.source.producer.CONFIG);registry={j['id']:j for j in source['jobs']};entries=[]
for row in proof['original_cache_records']:
    job=registry[row['id']];path=Path(row['cache_directory'])/'pretraining_preparation.json';original=read(path)
    assert original['passed'] and original['job']==job and sha(path)==row['source_proof_sha256']
    entries.append(dict(id=job['id'],family=job['family'],raw=job['raw'],NASA_paired_train_raw=job['train_raw'],
        directory=row['cache_directory'],source_proof_sha256=sha(path),all3split_arrays_independently_closed=True,
        source_environment=source['environment'],
        splits={split:dict(time_rows=s['time_rows'],source_length_gate_selected=s['source_length_gate_selected'],
            complete_stride1_windows=s['eligible_stride1_window_count'],source_dunder_len=s['source_dunder_len'],
            source_scaler_fit_extent_rows=s['source_scaler_fit_rows'],source_CSV_path=s['source_CSV_path'],files=s['files'])
            for split,s in original['splits'].items()}))
assert len(entries)==1627
index='configs/reproducibility/fm_moment_ICM_anomaly_original1627_closed_cache_index.2026-10-15_v2.json';assert not (ROOT/index).exists()
write(ROOT/index,dict(schema_version=2,paper_id='moment_long_context',paper_citation=source['paper_citation'],
    source_configuration=p.source.producer.CONFIG,actual_full_audit_configuration=p.audit.CONFIG,
    independent_full_audit_proof=proof_path.relative_to(ROOT).as_posix(),independent_full_audit_proof_sha256=sha(proof_path),
    original_source_cache_count=1627,original_cache_arrays=14643,original_cache_signal_coordinates=133589346,
    original_training_gate_selected=1627,original_training_windows=72417674,entries=entries,
    original_universe_source_count=1949,preserved_failed_source_count=322,
    separate_finite_control_configuration=p.CONFIG,complete1949_finite_cache_ready=False,
    source_header_and_split_behavior_unmodified=True,model_random_target_mask_not_cached=True,
    strict_temporal_independence_certified=False,historical_ICM_pipeline_certified=False,
    neural_training_performance_results=0,all_original_experiments_complete=False))
shutil.copyfile(ROOT/index,folder/'frozen_original1627_independently_closed_cache_index.json')
shutil.copyfile(__file__,folder/'closed_cache_index_publication_source.py')
manifest=ROOT/'tmp/MOMENT_ICM_anomaly_finite_control_v3_publication_manifest_20261015_v1.json'
m=read(manifest);paths=set(m['paths']);paths.add(index)
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',folder.relative_to(ROOT).as_posix()],cwd=ROOT,text=True).splitlines())
m['paths']=sorted(paths);write(manifest,m)
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=13,
    original1627_full_cache_closed=True,actual_full322_preparation_complete=False,new_neural_results=0,
    files_sha256={f.name:sha(f) for f in sorted(folder.iterdir()) if f.is_file() and f.name!='archive_validation.json'},
    all_original_experiments_complete=False))
print(json.dumps(dict(original_cache_index=index,closed_source_caches=1627,training_windows=72417674,exact_publication_files=len(paths))))
