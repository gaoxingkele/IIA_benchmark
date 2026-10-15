"""Publish actual Table2 implementations and live50-job training registration."""
import ast
from datetime import datetime,timezone
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import run_anomaly_transformer_paper_max_ablation_v1 as runner
ROOT,read,sha,write=runner.ROOT,runner.read,runner.sha,runner.write
relative='configs/experiments/fm_anomaly_transformer_paper_max_ablation.v1.json'
config=read(ROOT/relative);contract=runner.verify(config)
xml=ROOT/'tmp/AnomalyTransformer_paper_Max_ablation_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==16
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
p=psutil.Process(31592);assert abs(p.create_time()-1792044236.1758213)<.001
assert 'scripts.flow_matching.run_anomaly_transformer_paper_max_ablation_v1' in p.cmdline()
state=read(ROOT/config['state_root']/'status.json');assert state['pid']==p.pid
assert state['selected_direct_Max_training_jobs']==50
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/anomaly_transformer_full_Table2_Max_ablations_v1'
folder.mkdir(exist_ok=False)
sources={
 'frozen_full50_registration.json':ROOT/relative,
 'runtime_source.py':ROOT/'scripts/flow_matching/run_anomaly_transformer_paper_max_ablation_v1.py',
 'registration_source.py':ROOT/'scripts/flow_matching/register_anomaly_transformer_paper_max_ablation_v1.py',
 'model_config.json':ROOT/'configs/models/fm_anomaly_transformer_paper_max_ablation.v1.json',
 'gradient_prior_and_training_tests.py':ROOT/'tests/test_fm_anomaly_transformer_paper_max_ablation_v1.py',
 'validation.xml':xml}
for name,path in sources.items():shutil.copyfile(path,folder/name)
source=(ROOT/config['source_directory']/'model/attn.py').read_text(encoding='utf-8')
for variant in runner.VARIANTS:
    (folder/(variant+'_actual_attention_source.py')).write_text(
        runner.patched_attention('model/attn.py',source,variant),encoding='utf-8')
    tree=ast.parse(inspect.getsource(runner.parent.run_worker));binding=runner.MaxWorkerBinding(variant);tree=binding.visit(tree)
    assert (binding.backward,binding.calls,binding.patch_note)==(1,1,1)
    ast.fix_missing_locations(tree)
    (folder/(variant+'_exact_bound_training_template.py')).write_text(ast.unparse(tree)+'\n',encoding='utf-8')
write(folder/'actual_live_launch_receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    pid=p.pid,create_time=p.create_time(),command=p.cmdline(),private_bytes=p.memory_info().private,
    actual_live=True,state_at_capture=state,configuration=relative,configuration_sha256=sha(ROOT/relative),
    complete_training_jobs=50,variants=runner.VARIANTS,new_benchmark_completions_claimed=0,
    all_original_experiments_complete=False))
write(folder/'reported_Table2_targets_not_reproduced_results.json',dict(
    paper=contract['paper'],paper_sha256=contract['paper_sha256'],pdf_page=8,table=2,
    source='Rendered and visually verified primary PDF; only reported values, not measured outcomes',
    metric='F1_percent',fixed_sigma1_prior_max=dict(SMD=83.95,MSL=82.17,SMAP=70.65,SWaT=79.46,PSM=79.04,reported_dataset_average=79.05),
    learnable_prior_max=dict(SMD=88.88,MSL=85.20,SMAP=87.84,SWaT=81.65,PSM=93.83,reported_dataset_average=87.48),
    current_actual_reproduced_metrics=None,main_Table1_Table2_SMAP_discrepancy='96.69 versus96.90 already recorded by primary contract',
    selection_of_hyperparameters_using_these_targets=False))
versions={}
for package in ['torch','numpy','pandas','scikit-learn','psutil']:
    versions[package]=importlib.metadata.version(package)
write(folder/'registration_environment_snapshot.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    python=platform.python_version(),platform=platform.platform(),package_versions=versions,
    purpose='Observed at registration; not proof of an author-era environment or future training-time package immutability'))
write(folder/'autograd_counter_diagnostic_not_performance.json',dict(
    observed_fixture_issue='With torch.set_default_device(cpu), Tensor.backward public dispatch was entered twice while autograd.backward executed once',
    analytical_probe_tensor_dispatch_entries=2,analytical_probe_autograd_executions=1,
    analytical_gradient_of_sum_of_squared_ones=[2,2],
    corrected_measurement_layer='torch.autograd.backward, independently counted in both full-width integration fixtures',
    actual_required_backward_executions_per_optimizer_step=1,
    real_data_numerical_jobs_launched_before_tests_passed=0,benchmark_performance_claim=False))
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(folder/'README.md').write_text('Full Table2 direct-Max training ablations\n\n'
 'Primary PDF Table2/page8 was rendered and visually inspected: Assoc/Fix/Max and Assoc/Learnable/Max '
 'are distinct training experiments. Registered50full jobs:two variants xfive real datasets xfive seeds. '
 'Both use the parent512-width/three-layer/batch32/window100/max10epoch reconstruction with identical '
 'train-only normalization,independent validation and seven complete criterion/layer streams.\n\n'
 'Equation4direct Max differentiates the joint prior/series discrepancy without either detach,and '
 'performs exactly one autograd backward and one Adam update per minibatch. Fixed prior bypasses '
 'source sigma transforms to actualGaussian sigma1 and removes sigma-projection trainables. Its '
 'discarded projection is initialized before removal to consume the same RNG:all shared initial '
 'parameters match the learnable branch for the same seed. No minimax checkpoint or score-only '
 'substitution is accepted for these two training ablations.\n\n'
 'Sixteen checks passed,including analytical joint gradients,exactGaussian diagonal/neighbor values, '
 'shared-parameter initialization equality,sigma-parameter removal,two real full-width isolated CPU '
 'training/scoring fixtures and explicit scope/budget/guard rejection. A first development probe '
 'counted Tensor.backward dispatch entries; a minimal derivative witness showed CPU device context '
 'dispatches twice while autograd executes once. The corrected probe counts autograd.backward and '
 'verifies exactly one execution per actual optimizer update in both fixtures. This was a development '
 'instrumentation issue before real-data registration,not a failed benchmark or reduced budget.\n\n'
 'The actual50-job controller was live at capture,with about15MiBprivate commit and unchanged '
 '32GiBphysical/40GiBcommit/16GiBemergency guards. It uses the tested cold resource interface; '
 'all numerical children run original fresh CPU-1interpreters. All checkpoints/scores and shared '
 'derived data stay on F:. Current state is resource waiting,not completed training. No benchmark '
 'results are promoted. Reported Table2targets are separate from actual outcomes.\n\n'
 'This is an explicit mathematical reconstruction; inherited normalization,loss reduction,FFN, '
 'epsilon,early-stopping,LR and partition conventions remain uncertified as exact historical '
 'paper choices. Vanilla Transformer and other training ablations/distance/prior experiments, '
 'NeurIPS-TS,baselines,independent replays,uncertainty,strict TAB and all45/681/273minimum project '
 'requirements remain. Exact paper equivalence and complete execution are not claimed.\n',encoding='utf-8')
files={p.relative_to(folder).as_posix():sha(p) for p in folder.iterdir() if p.is_file()}
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=16,
    artifact_files_sha256=files,new_benchmark_completions_claimed=0,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),relative,
 'scripts/flow_matching/run_anomaly_transformer_paper_max_ablation_v1.py',
 'scripts/flow_matching/register_anomaly_transformer_paper_max_ablation_v1.py',
 'tests/test_fm_anomaly_transformer_paper_max_ablation_v1.py',
 'configs/models/fm_anomaly_transformer_paper_max_ablation.v1.json']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
target=ROOT/'tmp/AnomalyTransformer_Table2_Max_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),tests_passed=16,live_controller=p.pid,full_training_jobs=50,
                      measured_benchmark_completions=0)))
