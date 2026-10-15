"""Publish pure Transformer implementation and complete Table2 row bindings."""
import ast
from datetime import datetime,timezone
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil

from scripts.flow_matching import run_plain_transformer_paper_baseline_v1 as runner
ROOT,read,sha,write=runner.ROOT,runner.read,runner.sha,runner.write
relative='configs/experiments/fm_plain_transformer_paper_baseline.v1.json'
config=read(ROOT/relative);contract=runner.verify(config)
xml=ROOT/'tmp/PlainTransformer_Table2_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==13
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
p=psutil.Process(26496);assert abs(p.create_time()-1792045266.5138068)<.001
assert 'scripts.flow_matching.run_plain_transformer_paper_baseline_v1' in p.cmdline()
state=read(ROOT/config['state_root']/'status.json');assert state['pid']==p.pid
assert state['selected_plain_Transformer_training_jobs']==25
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/plain_Transformer_full_Table2_baseline_v1'
folder.mkdir(exist_ok=False)
sources={
 'frozen_full25_registration.json':ROOT/relative,
 'runtime_source.py':ROOT/'scripts/flow_matching/run_plain_transformer_paper_baseline_v1.py',
 'registration_source.py':ROOT/'scripts/flow_matching/register_plain_transformer_paper_baseline_v1.py',
 'model_config.json':ROOT/'configs/models/fm_plain_transformer_paper_baseline.v1.json',
 'attention_graph_and_training_tests.py':ROOT/'tests/test_fm_plain_transformer_paper_baseline_v1.py',
 'validation.xml':xml,
 'AUTHOR_CODE_LICENSE.txt':ROOT/config['source_directory']/'LICENSE'}
for name,path in sources.items():shutil.copyfile(path,folder/name)
raw=(ROOT/config['source_directory']/'model/attn.py').read_text(encoding='utf-8')
(folder/'actual_prior_free_attention_source.py').write_text(runner.patched_attention('model/attn.py',raw),encoding='utf-8')
tree=ast.parse(inspect.getsource(runner.parent.run_worker));binding=runner.PlainWorkerBinding();tree=binding.visit(tree)
assert (binding.backward,binding.calls,binding.patch_note,binding.coverage,binding.diagnostic)==(1,1,1,2,1)
ast.fix_missing_locations(tree)
(folder/'exact_bound_reconstruction_training_template.py').write_text(ast.unparse(tree)+'\n',encoding='utf-8')
write(folder/'actual_live_launch_receipt.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    pid=p.pid,create_time=p.create_time(),command=p.cmdline(),private_bytes=p.memory_info().private,
    actual_live=True,state_at_capture=state,configuration=relative,configuration_sha256=sha(ROOT/relative),
    full_training_jobs=25,prior_path_present=False,new_benchmark_completions_claimed=0,
    all_original_experiments_complete=False))
mainpath=runner.PARENT;main=read(ROOT/mainpath)
maxpath='configs/experiments/fm_anomaly_transformer_paper_max_ablation.v1.json';maxconfig=read(ROOT/maxpath)
rows=[]
for name,path,group,stream in [
 ('Transformer/Recon',relative,config['jobs'],'all.reconstruction'),
 ('AnomalyTransformer/Recon/Learnable/Minimax',mainpath,main['jobs'],'all.reconstruction'),
 ('AnomalyTransformer/AssDis/Learnable/Minimax',mainpath,main['jobs'],'all.association'),
 ('AnomalyTransformer/Assoc/Fix/Max',maxpath,[j for j in maxconfig['jobs'] if j['variant']=='fixed_sigma1_prior_max'],'all.multiplication'),
 ('AnomalyTransformer/Assoc/Learnable/Max',maxpath,[j for j in maxconfig['jobs'] if j['variant']=='learnable_prior_max'],'all.multiplication'),
 ('Final/Assoc/Learnable/Minimax',mainpath,main['jobs'],'all.multiplication')]:
    assert len(group)==25
    rows.append(dict(paper_row=name,training_configuration=path,training_configuration_sha256=sha(ROOT/path),
                     required_training_ids=[j['id'] for j in group],score_stream=stream,
                     implementation_status='callable_explicit_reconstruction_not_historical_author_equivalence'))
ids={i for row in rows for i in row['required_training_ids']};assert len(ids)==100
completed=[]
for cfg in [config,main,maxconfig]:
    for job in cfg['jobs']:
        result=Path(cfg['artifact_root'])/job['id']/'result.json'
        receipt=ROOT/cfg['state_root']/'receipts'/(job['id']+'.json')
        if result.exists() and receipt.exists() and read(receipt).get('passed'):
            r=read(result)
            assert r['status']=='completed' and r['job']==job
            for record in r['artifacts']:assert sha(record['path'])==record['sha256']
            completed.append(job['id'])
assert not completed  # These new full-width reconstructed tracks have not trained yet.
write(folder/'all_six_Table2_rows_implementation_and_execution_map.json',dict(
    paper=contract['paper'],paper_sha256=contract['paper_sha256'],pdf_page=8,table=2,rows=rows,
    callable_explicit_reconstruction_rows=6,required_row_evaluation_slots=150,unique_required_training_models=100,
    unique_minimax_main_models_reused_across_three_criteria=25,completed_guard_validated_training_models=completed,
    independent_checkpoint_replays_complete=0,paper_equivalence_certified=False,
    other_paper_experiments_and_full_project_complete=False))
write(folder/'reported_Transformer_row_not_reproduced_metrics.json',dict(
    paper=contract['paper'],paper_sha256=contract['paper_sha256'],pdf_page=8,table=2,
    metric='F1_percent',reported_values=dict(SMD=79.72,MSL=76.64,SMAP=73.74,SWaT=74.56,PSM=78.43),
    reported_dataset_average=76.62,actual_reproduced_metrics=None,
    target_numbers_used_to_select_hyperparameters=False))
(folder/'publication_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.txt -text\n',encoding='utf-8')
(folder/'README.md').write_text('Full plain Transformer reconstruction baseline and Table2 row map\n\n'
 'Registered25new full training jobs for the Table2Transformer/Recon row:five real datasets xfive seeds. '
 'The prior distance matrices,Gaussian prior computations and sigma-projection trainables are removed. '
 'The original attention/encoder ABI names are retained for source compatibility,but the computational '
 'graph is ordinary scaled-dot-product self-attention with reconstruction-only training and scoring. '
 'Discarded sigma projections consume initialization RNG before removal,so all shared original '
 'embedding/attention/FFN weights are paired exactly to the source encoder.\n\n'
 'All512width/FFN512/three layers/eight heads/batch32/window100/max10epochs and original parent '
 'early-stop/LR conventions remain. Exactly one autograd execution and Adam update per minibatch '
 'is independently counted in the full-width CPU fixture. Train-only normalization,independent '
 'validation,and every original covered timestamp remain. Only reconstruction error is a detection '
 'score. Saved layer attention entropy is explicitly diagnostic; no fake association discrepancy '
 'or prior arrays are fabricated. Parent fixed-delta controls are labeled diagnostics for this '
 'baseline; primary calibration is the frozen validation ratio. Historical baseline-specific '
 'architecture/tuning remains unknown,so exact paper equivalence is not certified.\n\n'
 'Thirteen checks passed:prior graph removal,absence of sigma trainables,standard attention formula, '
 'bitwise shared initialization/latent outputs,reconstruction loss/entropy diagnostics,actual full-width '
 'training and full-array coverage,scope/budget/guard rejection and a cold resource interface. A development '
 'reference initially divided by sqrt(dk) while the preserved source multiplies by its reciprocal; '
 'bitwise recomputation now follows the exact source operation order without changing the model or '
 'relaxing numerical tolerances. Fixtures are plumbing only,not benchmark results.\n\n'
 'The exact live controller and waiting state are archived. Original32GiBphysical/40GiBcommit '
 'admission and16GiBemergency protection remain; metadata uses the tested cold resource API. '
 'Checkpoints/scores and shared derived data stay on F:. No numerical worker or frozen existing '
 'configuration was changed. The upstream2022THUML MIT notice accompanies the derived attention code.\n\n'
 'A complete six-row Table2implementation map now points to100unique registered training models '
 'for150row/dataset/seed evaluation slots:25plain Transformer,50direct-Max,and25minimax main models '
 'reused for Recon,AssDis and final multiplication criteria. Callable explicit reconstruction is '
 'not completed training:zero of these new full-width reconstruction models are promoted as completed '
 'at this capture. Original source-runtime tracks remain separate. Table3statistics,other trained '
 'distance/prior studies,NeurIPS-TS,baselines,all independent replays,uncertainty,strictTAB and '
 'all45/681/273minimum project requirements remain.\n',encoding='utf-8')
files={p.relative_to(folder).as_posix():sha(p) for p in folder.iterdir() if p.is_file()}
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=13,
    artifact_files_sha256=files,new_benchmark_completions_claimed=0,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),relative,
 'scripts/flow_matching/run_plain_transformer_paper_baseline_v1.py',
 'scripts/flow_matching/register_plain_transformer_paper_baseline_v1.py',
 'tests/test_fm_plain_transformer_paper_baseline_v1.py',
 'configs/models/fm_plain_transformer_paper_baseline.v1.json']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
target=ROOT/'tmp/PlainTransformer_Table2_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),tests_passed=13,live_controller=p.pid,new_full_training_jobs=25,
    Table2_callable_reconstruction_rows=6,Table2_unique_training_models=100,actual_completed_new_models=0)))
