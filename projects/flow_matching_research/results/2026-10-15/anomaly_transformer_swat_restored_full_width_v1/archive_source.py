"""Archive the live SWaT restoration and verify the current paper contract."""
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from pypdf import PdfReader

from scripts.flow_matching import run_anomaly_transformer_swat_restored_v1 as restored
ROOT,read,sha=restored.ROOT,restored.read,restored.sha
config_path='configs/experiments/fm_anomaly_transformer_swat_restored.v1.json'
config=read(ROOT/config_path);restored.verify(config)
xml=ROOT/'tmp/AnomalyTransformer_swat_restored_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==15
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
pid=30512;ct=1792040981.7732399
p=psutil.Process(pid);assert abs(p.create_time()-ct)<.001
assert 'scripts.flow_matching.run_anomaly_transformer_swat_restored_v1' in p.cmdline()
state=read(ROOT/config['state_root']/'status.json');assert state['pid']==pid
paper=ROOT/'papers/literature/mtsad_pdfs/xu2022_anomaly_transformer.pdf'
paper_hash=sha(paper)
assert paper_hash=='ff8d3bb627fce9914eb8a9e78c4139e4852771dbee801da2c766dea028a17053'
reader=PdfReader(paper)
pages={str(n):reader.pages[n-1].extract_text() for n in [6,7,8,9,15,16,17,18,19]}
table13=' '.join(pages['19'].split())
assert 'SWaT Water 51 100 396,000 99,000 449,919' in table13
assert 'batch size of 32' in pages['6'] and 'non-overlapped sliding window' in pages['6']
assert 'Algorithm 2' in pages['15'] and 'Mean(P, dim=1)' in pages['15']
solver=ROOT/config['source_directory']/'solver.py'
solvertext=solver.read_text(encoding='utf-8')
assert 'self.vali(self.test_loader)' in solvertext
assert 'torch.mean(torch.sum(res, dim=-1), dim=1)' in solvertext
splits={
 'SMD':dict(train=566724,validation=141681,test=708420,features=38),
 'PSM':dict(train=105984,validation=26497,test=87841,features=25),
 'MSL':dict(train=46653,validation=11664,test=73729,features=55),
 'SMAP':dict(train=108146,validation=27037,test=427617,features=25),
 'SWaT':dict(train=396000,validation=99000,test=449919,features=51),
 'NeurIPS-TS':dict(train=20000,validation=10000,test=20000,features=1)}
for dataset,record in splits.items():
 if dataset=='NeurIPS-TS':continue
 d=read(ROOT/f'configs/datasets/mtsad_{dataset.lower()}.json')
 assert d['expected_shapes']['train'][0]==record['train']+record['validation']
 assert int(d['expected_shapes']['train'][0]*.8)==record['train']
 assert d['expected_shapes']['test']==[record['test'],record['features']]
# An analytical counterexample proves that the two aggregation orders differ.
kl=lambda a,b:sum(x*(math.log(x+.0001)-math.log(y+.0001)) for x,y in zip(a,b))
first=[.9,.1];second=[.1,.9];reference=[.5,.5]
mean_of_kl=(kl(first,reference)+kl(reference,first)+kl(second,reference)+kl(reference,second))/2
kl_of_means=kl(reference,reference)*2
assert mean_of_kl>0 and kl_of_means==0
contract=dict(schema_version=1,paper=paper.relative_to(ROOT).as_posix(),paper_sha256=paper_hash,
 evidence_pages={k:dict(pdf_page=int(k),text_sha256=__import__('hashlib').sha256(v.encode()).hexdigest()) for k,v in pages.items()},
 reported_table13_splits=splits,partition_order='First80percent/last20percent reproduces all five stated counts; exact source identity still requires audit',
 paper_training=dict(d_model=512,heads=8,layers=3,window=100,batch_size=32,
                     learning_rate=.0001,lambda_discrepancy=3,maximum_epochs=10,
                     train_window_stride=100,early_stopping_patience='not specified by paper text'),
 paper_evaluation=dict(calibration='independent validation subset',
                     r_percent=dict(SMD=.5,MSL=1,SMAP=1,SWaT=.1,PSM=1),
                     fixed_delta=dict(SMD=.1,MSL=.1,SMAP=.01,SWaT=.1,PSM=.01),
                     reported_main_F1_uses_PA=True,ROC_predefined_r_percent=[.5,1,1.5,2,10,20,30]),
 author_source_differences=[
   'Author scripts batch256 versus paper batch32',
   'Author scripts3epochs on MSL/SMAP/PSM/SWaT versus paper early stop within10; preserve source branch separately',
   'MSL/SMAP/PSM source training stride1 versus paper non-overlapped window100',
   'Source fits scaler/trains on full concatenated training pool; paper has80/20 train/validation counts',
   'Source checkpoint selection uses self.test_loader; paper describes independent validation',
   'Source percentile calibration combines train/test scores; paper describes validation scores',
   'Source averages KL over heads; Algorithm2 averages distributions over heads before KL',
   'Source uses temperature50 and sums layer contributions during scoring; Algorithm2/3 states layer mean and no temperature50'],
 head_aggregation_counterexample=dict(mean_of_symmetric_head_KL=mean_of_kl,symmetric_KL_of_head_means=kl_of_means,
                                     diagnostic_only_not_benchmark_result=True),
 paper_internal_discrepancies=['SMAP final F1 is96.69 in Table1 but96.90 in Table2',
    'Equation6 uses feature squared-L2 sum while Algorithm3 specifies feature mean; fixed-delta interpretation needs explicit choice'],
 required_experiment_groups=[
  dict(evidence='Table1/page7',requirement='five real datasets main method and15 baselines',status='full author-runtime main jobs queued; paper-equivalent protocol not certified'),
  dict(evidence='Figure3/page7',requirement='ROC/AUC at seven stated threshold proportions',status='must not equate to continuous unadjusted-score AUROC without protocol verification'),
  dict(evidence='Figure4/page7 and Table13/page19',requirement='NeurIPS-TS six-benchmark coverage',status='original data/loader experiment not established by the new five-dataset main queues'),
  dict(evidence='Table2/page8',requirement='vanilla Transformer Recon; Anomaly Transformer Recon/AssDis; fixed-prior Max; learnable-prior Max; learnable Minimax',status='full paper-equivalent implementations/runs remain required'),
  dict(evidence='Table3/page9',requirement='normal/abnormal adjacent-association contrasts for Recon/Max/Minimax',status='not certified by saved anomaly-score metrics'),
  dict(evidence='Table4/page15',requirement='single layer1/2/3 versus multilayer scoring',status='posthoc full checkpoint inference needed; not extra training seeds'),
  dict(evidence='Table5/page15',requirement='L2/CE/Wasserstein/JSD versus symmetricKL discrepancy',status='full training and exact distance conventions remain required'),
  dict(evidence='Table6/page16',requirement='learnable power-law versus Gaussian prior',status='distance-zero handling and source implementation still require resolution'),
  dict(evidence='Table7/pages16-17',requirement='Recon/AssDis/Addition/Multiplication criteria',status='posthoc inference and complete saved component arrays required'),
  dict(evidence='Tables8-9/page18',requirement='layers1/2/3/4 and hidden256/512/1024',status='separate full training sensitivity runs and source-dependent ffn conventions required'),
  dict(evidence='Tables10-11/pages18-19',requirement='validation distribution and fixed-delta versus ratio thresholds',status='source pooled threshold results do not certify this experiment'),
  dict(evidence='Table12/page19',requirement='additional BOCPD, TS-CP2 and U-Time baselines',status='source presence or smaller local adapters do not certify paper-equivalent runs')],
 original_scope=config['original_scope'],supersedes_only_stale_metadata='The475200SWaT row attribution in earlier local config/prose is not supported by this primary PDF. It does not change any frozen scientific artifact.',
 all_original_experiments_complete=False)
contract_path=ROOT/'configs/reproducibility/fm_anomaly_transformer_primary_paper_protocol.2026-10-15_v1.json'
with contract_path.open('x',encoding='utf-8',newline='\n') as stream:
 json.dump(contract,stream,indent=2);stream.write('\n')
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/anomaly_transformer_swat_restored_full_width_v1'
folder.mkdir(exist_ok=False)
launch=dict(captured_utc=datetime.now(timezone.utc).isoformat(),pid=pid,create_time=ct,
 command=p.cmdline(),actual_live=True,cpu_seconds=sum(p.cpu_times()[:2]),state_at_capture=state,
 selected_loader_restoration_slots=5,original_main_matrix_slots=25,
 configuration=config_path,configuration_sha256=sha(ROOT/config_path),
 original_loader_equivalence_certified=False,benchmark_results_added=0,
 all_original_experiments_complete=False)
(folder/'actual_live_launch_receipt.json').write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
for name,path in [('frozen_registration.json',ROOT/config_path),('validation.xml',xml),
 ('primary_paper_protocol.json',contract_path),('runtime_source.py',ROOT/'scripts/flow_matching/run_anomaly_transformer_swat_restored_v1.py'),
 ('registration_source.py',ROOT/'scripts/flow_matching/register_anomaly_transformer_swat_restored_v1.py'),
 ('fidelity_tests.py',ROOT/'tests/test_fm_anomaly_transformer_swat_restored_v1.py'),
 ('model_config.json',ROOT/'configs/models/fm_anomaly_transformer_swat_restored.v1.json')]:
 shutil.copyfile(path,folder/name)
(folder/'extracted_original_CSV_loader.py').write_text(restored.loader_code(),encoding='utf-8')
(folder/'archive_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
correction=('Primary-PDF SWaT/protocol correction,2026-10-15\n\n'
 'The exact primary PDF SHA256 '+paper_hash+' Table13,page19, states396000training/99000validation/449919test rows. '
 'Training plus validation is495000, matching the local CSV pool. Earlier local metadata attributing475200rows '
 'to this paper is stale and is corrected here. The frozen dataset/model/runtime configurations and their '
 'historical raw proof snapshots remain unchanged; this note supersedes only that metadata claim.\n\n'
 'The real unresolved split difference is full495000pool training/test-set checkpoint selection in the author '
 'scripts versus independent396000/99000training/validation in the paper. Further differences are batch256 '
 'versus32, source stride1 versus paper non-overlapped100 windows for several datasets, source3-epoch scripts '
 'versus paper up-to10, source pooled train/test calibration versus validation, and head/layer discrepancy '
 'aggregation and temperature. The new contract records these directly from PDF pages6,15,16,18,19. '
 'Full author-runtime reproducibility does not certify paper-algorithm equivalence.\n\n'
 'The already-frozen SWaT restoration config inherited the earlier475200metadata; this is an explicit '
 'metadata discrepancy, not an instruction to trim rows. The actual code uses the full495000pool, with '
 'all model/training/scoring/resource settings preserved. No numerical worker, raw file or frozen config '
 'was changed by this correction.\n')
(folder/'PRIMARY_PAPER_PROTOCOL_CORRECTION.md').write_text(correction,encoding='utf-8')
prior=ROOT/'projects/flow_matching_research/results/2026-10-15/anomaly_transformer_author_full_width_registration_v1/PRIMARY_PAPER_PROTOCOL_CORRECTION.md'
with prior.open('x',encoding='utf-8',newline='\n') as stream:stream.write(correction)
(folder/'README.md').write_text('Full-width Anomaly Transformer SWaT loader-restoration lane\n\n'
 'The exact SWATSegLoader class extracted from the pinned MaelNet author source is connected to the '
 'unchanged full-width Anomaly Transformer model and training/scoring solver. Five seed slots1103..1107 '
 'are registered. Their exact live controller handle, full source/data hashes, resource admission state '
 'and15 passing fidelity/integration checks are archived. Small fixture computations validate the '
 'interface only; no benchmark model has completed by this registration.\n\n'
 'The restored implementation retains512/512width,three layers,eight heads,window100,batch256,3epochs, '
 'original minimax updates,source early stopping,pooled threshold and PA. It uses every495000training-pool '
 'row and449919test rows; no guessed trimming or replacement of raw data. All32GiBphysical/40GiBcommit '
 'admission and16GiBemergency protection remain. The lane waits without owning the shared heavy lock.\n\n'
 'This is a separately named source-loader reconstruction. Original SWaT source equivalence remains '
 'uncertified. Read PRIMARY_PAPER_PROTOCOL_CORRECTION.md: the primary PDF actually has396000train '
 '+99000validation=495000pool rows, so the earlier475200attribution is corrected. The source model/solver '
 'still differs from the paper protocol and Algorithm2/3. All paper tables/ablations, sixth NeurIPS-TS '
 'benchmark, independent checkpoint replay,uncertainty and strict TAB remain required. Full45/681/273 '
 'scope remains; the old19-record Anomaly Transformer source inventory is not an exhaustive ablation '
 'completion certificate.\n',encoding='utf-8')
files={p.relative_to(folder).as_posix():sha(p) for p in folder.iterdir() if p.is_file()}
(folder/'archive_validation.json').write_text(json.dumps(dict(tests_passed=15,source_bytes_preserved=True,
 artifact_files_sha256=files,paper_contract_page_assertions_passed=True,all_original_experiments_complete=False),indent=2)+'\n',encoding='utf-8')
targets=[folder.relative_to(ROOT).as_posix(),prior.relative_to(ROOT).as_posix(),contract_path.relative_to(ROOT).as_posix(),
 config_path,'configs/models/fm_anomaly_transformer_swat_restored.v1.json',
 'scripts/flow_matching/run_anomaly_transformer_swat_restored_v1.py',
 'scripts/flow_matching/register_anomaly_transformer_swat_restored_v1.py',
 'tests/test_fm_anomaly_transformer_swat_restored_v1.py']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
 paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
manifest=dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths))
target=ROOT/'tmp/AnomalyTransformer_swat_restored_publication_manifest_20261015_v1.json'
target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(exact_files=len(paths),live_restored_controller=pid,tests_passed=15,
 primary_SWaT_train=396000,primary_SWaT_validation=99000,source_reproduction_is_not_paper_equivalence=True)))
