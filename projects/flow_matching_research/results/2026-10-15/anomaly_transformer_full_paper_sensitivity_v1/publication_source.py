"""Archive the full reported axes and exact live400-job controller."""
import ast
from datetime import datetime,timezone
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import run_anomaly_transformer_paper_sensitivity_v1 as runner
ROOT,read,sha,write=runner.ROOT,runner.read,runner.sha,runner.write
relative='configs/experiments/fm_anomaly_transformer_paper_sensitivity.v1.json'
config=read(ROOT/relative);contract=runner.verify(config)
xml=ROOT/'tmp/AnomalyTransformer_paper_sensitivity_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==30
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
p=psutil.Process(18296);assert abs(p.create_time()-1792042673.5456176)<.001
assert 'scripts.flow_matching.run_anomaly_transformer_paper_sensitivity_v1' in p.cmdline()
state=read(ROOT/config['state_root']/'status.json');assert state['pid']==p.pid
assert state['selected_sensitivity_training_jobs']==400
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/anomaly_transformer_full_paper_sensitivity_v1'
folder.mkdir(exist_ok=False)
sources={
 'frozen_full400_registration.json':ROOT/relative,
 'runtime_source.py':ROOT/'scripts/flow_matching/run_anomaly_transformer_paper_sensitivity_v1.py',
 'registration_source.py':ROOT/'scripts/flow_matching/register_anomaly_transformer_paper_sensitivity_v1.py',
 'model_config.json':ROOT/'configs/models/fm_anomaly_transformer_paper_sensitivity.v1.json',
 'fidelity_tests.py':ROOT/'tests/test_fm_anomaly_transformer_paper_sensitivity_v1.py',
 'validation.xml':xml}
for name,path in sources.items():shutil.copyfile(path,folder/name)
tree=ast.parse(inspect.getsource(runner.parent.run_worker));binding=runner.WorkerBinding();tree=binding.visit(tree)
assert (binding.windows,binding.model,binding.reshape)==(7,1,1)
ast.fix_missing_locations(tree)
(folder/'exact_bound_worker_template.py').write_text(ast.unparse(tree)+'\n',encoding='utf-8')
write(folder/'actual_live_launch_receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
 pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]),
 actual_live=True,state_at_capture=state,configuration=relative,configuration_sha256=sha(ROOT/relative),
 new_training_jobs=400,reused_main_models=25,unique_main_and_sensitivity_models=425,
 benchmark_completions_added_by_registration=0,all_original_experiments_complete=False))
write(folder/'visually_checked_paper_axes.json',dict(paper=contract['paper'],paper_sha256=contract['paper_sha256'],
 rendered_and_visually_inspected_pdf_pages=[13,18],
 rendering_tool='Poppler25.07.0 pdftoppm PNG at130dpi; original PDF preserved',
 axis_evidence=config['axis_evidence'],
 interpretation='Reported single-factor sweeps, not a Cartesian product across simultaneous factors',
 primary_score_stream='all.multiplication',baseline_frozen_reference=runner.PARENT,
 baseline_sha256=config['parent_registration_sha256'],data_cache_reused_without_raw_overwrite=True,
 hidden_width_FFN_convention='FFN512 retained from the explicit source-based reconstruction; not paper-certified',
 GPU_efficiency_comparison_certified=False,
 storage_at_capture={d:dict(free_bytes=shutil.disk_usage(d).free,total_bytes=shutil.disk_usage(d).total) for d in ['D:/','F:/']}))
bindings=[]
for axis,value in runner.variants():
    selected=dict(runner.BASE);selected[axis]=value
    worker=runner.bound_worker(selected)
    bindings.append(dict(axis=axis,value=value,parameters=selected,full_score_streams=worker.__globals__['STREAMS'],
        loss_weight=worker.__globals__['minimax_losses'].keywords['weight'],
        complete_dataset_seed_slots=25,additional_training_updates_by_this_archive=0))
write(folder/'all_sixteen_exact_variant_bindings.json',dict(variants=bindings,
    fixed_batch_size=32,maximum_epochs=10,unchanged_original_resource_guards=config['settings'],
    independent_validation_retained=True,all_original_experiments_complete=False))
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'README.md').write_text('Complete reported Anomaly Transformer sensitivity reconstruction\n\n'
 'Primary PDF Figure7,page13 and Tables8-9,page18 were rendered with Poppler and visually inspected. '
 'Reported axes: window50,100,150,200,250,300; loss weight0..6; depth1..4; hidden256,512,1024. '
 'The16nonbase points require400new full training jobs: five real datasets xfive local seeds per point. '
 'The25existing main identities are reused across all four axes, yielding425unique models and500axis '
 'display slots. No baseline is retrained four times and no reported endpoint or seed was removed.\n\n'
 'The exact frozen main worker is bound only at seven window/index/coverage/tail sites,the model '
 'hidden/depth arguments,the complete per-layer score streams/component reshape and loss-weight '
 'argument. Source bytes are unchanged. Batch32,max10epochs,inherited early stopping and learning-rate '
 'rules,both detached minimax phases,all full covered points,independent validation,all scoring '
 'components,CPU-only dispatch and32GiBphysical/40GiBcommit/16GiBemergency protection remain. '
 'Every depth saves all four criteria plus every individual layer. Lambda0keeps association-based '
 'scoring while training only with reconstruction as the figure caption specifies.\n\n'
 'All variants share the immutable parent train-only normalization/80-20data cache on F:,and all new '
 'checkpoints/scores go to F:. FFN512and other ambiguous executable conventions remain explicitly '
 'inherited from the paper-contract reconstruction; CPU timings do not reproduce paper GPU '
 'memory/latency. Primary multiplication score was frozen before any test result; do not pick a '
 'best sensitivity result on test labels for a strict leaderboard.\n\n'
 'Thirty checks passed,including every16parameter binding,a real isolated CPU training/scoring '
 'fixture,zero-weight gradient behavior,scope/resource/protocol rejection and a cold controller '
 'import. Fixtures validate plumbing only. The400-job controller was actually live at capture, '
 'waiting for its unchanged resource admission; zero benchmark completions are claimed. All '
 'other training ablations,NeurIPS-TS,additional baselines,independent replay,uncertainty and full '
 '45/681/273minimum project scope remain. The old inventory count is not an exhaustive completion '
 'certificate for paper tables or additional known experiments.\n',encoding='utf-8')
files={p.relative_to(folder).as_posix():sha(p) for p in folder.iterdir() if p.is_file()}
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
 tests_passed=30,artifact_files_sha256=files,new_benchmark_completions_claimed=0,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),relative,
 'scripts/flow_matching/run_anomaly_transformer_paper_sensitivity_v1.py',
 'scripts/flow_matching/register_anomaly_transformer_paper_sensitivity_v1.py',
 'tests/test_fm_anomaly_transformer_paper_sensitivity_v1.py',
 'configs/models/fm_anomaly_transformer_paper_sensitivity.v1.json']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
target=ROOT/'tmp/AnomalyTransformer_full_paper_sensitivity_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),live_controller=p.pid,new_full_training_jobs=400,
                      reused_main_models=25,tests_passed=30)))
