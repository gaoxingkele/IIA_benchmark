"""Record the genuine author-width gap and the live, full-budget new lane."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching.run_anomaly_transformer_author_full_v1 import ROOT, read, sha, verify

relative='configs/experiments/fm_anomaly_transformer_author_full.v1.json'
config=read(ROOT/relative);verify(config)
xml=ROOT/'tmp/AnomalyTransformer_author_full_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==18
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
statepath=ROOT/config['state_root']/'status.json'
state=read(statepath);p=psutil.Process(26492)
assert abs(p.create_time()-1792040566.1641924)<.001
assert 'scripts.flow_matching.run_anomaly_transformer_author_full_v1' in p.cmdline()
assert state['pid']==p.pid
source=ROOT/config['source_directory']
model_source=source/'model/AnomalyTransformer.py'
tree=ast.parse(model_source.read_text(encoding='utf-8'))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AnomalyTransformer')
init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
defaults={arg.arg:ast.literal_eval(val) for arg,val in zip(init.args.args[-len(init.args.defaults):],init.args.defaults)}
assert defaults['d_model']==defaults['d_ff']==512
localpath=ROOT/'configs/models/fm_strict_baselines/fm_strict_anomaly_transformer__smd__registered.json'
local=read(localpath);localparams=local['parameters']['parameters']
assert localparams['d_model']==localparams['d_ff']==32
swatconfig=read(ROOT/'configs/datasets/mtsad_swat.json')
swatfiles=[]
for file in swatconfig['files'].values():
    path=ROOT/config['data_root']/swatconfig['directory']/file
    swatfiles.append(dict(path=path.relative_to(ROOT).as_posix(),exists=path.is_file(),
                          bytes=path.stat().st_size,sha256=sha(path)))
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/anomaly_transformer_author_full_width_registration_v1'
folder.mkdir(exist_ok=False)
launch=dict(captured_utc=datetime.now(timezone.utc).isoformat(),pid=p.pid,create_time=p.create_time(),
            command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]),actual_live=True,
            state_at_capture=state,configuration=relative,configuration_sha256=sha(ROOT/relative),
            running_numerical_worker=state.get('active_pid') is not None,
            full_budget_jobs_registered=25,runnable_source_loader_jobs=20,source_loader_unavailable_jobs=5,
            all_original_experiments_complete=False)
audit=dict(captured_utc=launch['captured_utc'],original_scope=config['original_scope'],
           inspected_local_configuration=localpath.relative_to(ROOT).as_posix(),
           inspected_local_configuration_sha256=sha(localpath),local_d_model=32,local_d_ff=32,
           existing_local_completed_SMD_seed1106_parameter_count=22750,
           source_model_file=model_source.relative_to(ROOT).as_posix(),source_model_sha256=sha(model_source),
           source_model_constructor_defaults=defaults,full_width_source_pipeline_required=True,
           new_full_width_registration=relative,prior_local_results_promoted_to_author_equivalence=False,
           raw_SWaT_files=swatfiles,SWaT_missing_material_is_source_loader_not_local_CSV=True,
           SWaT_rows_discrepancy=swatconfig['reported_shape_discrepancy'],
           unresolved=config['outstanding_author_fidelity_gaps'],
           scientific_benchmark_results_added=0,small_fixture_tests_are_plumbing_only=True,
           new_controller_live=True,full_author_training_models_completed_by_this_registration=0,
           all_original_experiments_complete=False)
write=lambda name,data:(folder/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
write('actual_live_launch_receipt.json',launch);write('source_width_and_remaining_fidelity_gaps.json',audit)
for name,path in [('frozen_registration.json',ROOT/relative),('validation.xml',xml),
                 ('author_runtime_source.py',ROOT/'scripts/flow_matching/run_anomaly_transformer_author_full_v1.py'),
                 ('registration_source.py',ROOT/'scripts/flow_matching/register_anomaly_transformer_author_full_v1.py'),
                 ('fidelity_tests.py',ROOT/'tests/test_fm_anomaly_transformer_author_full_v1.py'),
                 ('model_config.json',ROOT/'configs/models/fm_anomaly_transformer_author_full.v1.json')]:
    shutil.copyfile(path,folder/name)
(folder/'archive_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'README.md').write_text(
    'Anomaly Transformer full-width author-runtime gap and live registration\n\n'
    'Existing local strict/reconstruction configs use d_model=32,d_ff=32; the inspected pinned author '
    'model defaults are512/512. The completed local SMD seed1106 model has22750parameters and remains '
    'a local reconstruction, not author-pipeline certification. No existing source, raw data, frozen '
    'configuration or numerical worker was changed.\n\n'
    'Registered all25 main dataset/seed slots: SMD,MSL,SMAP,PSM,SWaT x1103..1107. Twenty original-loader '
    'jobs have a callable CPU port of the actual author source. SMD retains10epochs and stride100; '
    'MSL/SMAP/PSM retain3epochs and stride1; window100,batch256,512-width three-layer model, both minimax '
    'backwards, Adam1e-4, original learning-rate schedule, patience3, temperature50, pooled calibration, '
    'PA and original omitted tails remain. Source test-based checkpoint selection is explicitly preserved. '
    'Only CUDA device/NumPy infinity compatibility and observation hooks are changed in isolated F-drive '
    'workspaces. Original author source remains byte-identical.\n\n'
    'The five SWaT slots remain explicitly unavailable because this original snapshot has no SWaT '
    'loader. Local CSVs do exist. Their configured495000train rows differ from the paper475200rows; '
    'the19800-row discrepancy remains unresolved and is not silently trimmed.\n\n'
    'The controller is actually live; the capture records its exact PID/create time and admission state. '
    'It requires32GiB physical and40GiB commit headroom and retains16GiB emergency protection. Waiting '
    'does not own the shared heavy lock. Failed/interrupted attempts are preserved, never blindly retried. '
    'Registration is not completed training.18 validation checks include a small full-width source '
    'training/saved-score fixture; fixture results are plumbing only and add zero benchmark results.\n\n'
    'Independent checkpoint replay, multi-seed uncertainty, SWaT source/paper resolution, full-width '
    'paper ablations and hyperparameter studies, strict TAB and the full45/681/273 project scope all '
    'remain required. This new main-method lane does not redefine the overall objective.\n',encoding='utf-8')
files={p.relative_to(folder).as_posix():sha(p) for p in folder.iterdir() if p.is_file()}
write('archive_validation.json',dict(exact_file_sha256=files,tests_passed=18,
      benchmark_performance_claim=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),relative,
         'configs/models/fm_anomaly_transformer_author_full.v1.json',
         'scripts/flow_matching/run_anomaly_transformer_author_full_v1.py',
         'scripts/flow_matching/register_anomaly_transformer_author_full_v1.py',
         'tests/test_fm_anomaly_transformer_author_full_v1.py']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),
              previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
target=ROOT/'tmp/AnomalyTransformer_author_full_registration_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),controller_pid=p.pid,registered_jobs=25,
                      runnable_jobs=20,not_yet_completed=True,tests_passed=18)))
