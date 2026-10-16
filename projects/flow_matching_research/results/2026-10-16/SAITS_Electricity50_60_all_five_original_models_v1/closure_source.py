"""Close two actual five-model cohorts using immutable completed full replay evidence."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import io
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scipy.stats import t as student_t
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.summarize_saits_full_five_seed_cohort_v1 import summarize,METRICS
from scripts.flow_matching import run_saits_completion_replay_queue_v1 as collector

runtime=read(ROOT/'configs/runtime/fm_saits_completion_replay_queue.v1.json')
jobs,template=collector.verify(runtime)
state=read(ROOT/runtime['state_root']/'status.json')
process=psutil.Process(state['pid']);assert process.create_time()==state['create_time'] and process.is_running()
required={f'saits_paper_saits_electricity_author_missing{rate}_seed{seed}' for rate in ('0.5','0.6') for seed in range(26,31)}
assert required.issubset(state['verified_original_ids'])
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/SAITS_Electricity50_60_all_five_original_models_v1'
frozen='configs/reproducibility/fm_saits_electricity50_60_all_five_original_models.2026-10-16_v1.json'
assert not packet.exists() and not (ROOT/frozen).exists()
archives=[]; rows=[]; pins=set(); summaries=[]
for rate,targets in [(.5,4221734),(.6,4842788)]:
    cohort=[]
    for seed in range(26,31):
        identity=f'saits_paper_saits_electricity_author_missing{rate}_seed{seed}'
        paths={phase:ROOT/runtime['generated_config_root']/(identity+'.'+suffix+'.json')
            for phase,suffix in [('native_config','native'),('review_config','review')]}
        nc,rc=read(paths['native_config']),read(paths['review_config'])
        assert rc['replay_configuration']==paths['native_config'].relative_to(ROOT).as_posix()
        for cfg in (nc,rc):
            for pin in cfg['source_receipts']:
                p=ROOT/pin['path'];assert sha(p)==pin['sha256'],str(p);pins.add(str(p))
        paths.update(native_report=ROOT/nc['artifact_directory']/'checkpoint_replay_audit.json',
            independent_proof=ROOT/rc['output_directory']/'independent_full_replay_batch_verification.json',
            native_guard=ROOT/nc['state_root']/'receipt.json',review_guard=ROOT/rc['state_root']/'receipt.json')
        report,proof=read(paths['native_report']),read(paths['independent_proof'])
        assert report['full_checkpoint_contents_and_re_inference_passed'] and not report['cuda_context_initialized']
        assert proof['passed'] and proof['all_original_grid_coordinates_independently_checked']
        assert proof['native_replay_report_sha256']==sha(paths['native_report'])
        assert report['metric_tolerance']==proof['metric_tolerance']==dict(rtol=2e-6,atol=2e-6)
        assert report['full_models_replayed']==len(report['records'])==proof['full_native_models_verified']==1
        record,checked=report['records'][0],proof['records'][0]
        assert record['id']==checked['id']==identity
        model=read(ROOT/record['config']);result=read(ROOT/record['result_path'])
        collector.check_completed_result(model,result)
        assert model==jobs[identity]['native']==result['config']
        assert model['method']=='saits' and model['dataset']=='electricity' and model['variant']=='author'
        assert model['missing_ratio']==rate and model['seed']==seed
        assert not model['corrected_masking'] and not model['diagnostic'] and not model.get('limit_test_cases')
        assert result['parameters']==record['model_parameters']==11511000
        assert result['test_windows']==290 and result['target_count']==targets
        assert record['seed']==checked['seed']==seed
        assert record['full_test_shape']==checked['full_test_shape']==[290,100,370]
        assert record['held_out_coordinates']==checked['held_out_coordinates']==targets
        assert record['full_original_checkpoint_and_data_identity_verified'] and record['full_test_grid_and_grouped_statistics_replayed']
        assert checked['actual_full_checkpoint_reinference_completed'] and checked['full_truth_masks_units_groups_equal']
        assert checked['full_group_statistics_independently_recomputed'] and checked['months']==result['bootstrap_groups']==10
        assert result['bootstrap_group']=='month' and record['original_metrics']==result['metrics']
        finish=checked['full_training_completion']
        assert finish==record['full_training_completion'] and finish['full_training_finished'] and not finish['benchmark_smoke']
        assert finish['original_maximum_epochs']==10000 and (finish['next_epoch']>=10000 or finish['author_patience_remaining']==0)
        for metric in METRICS:
            for metrics in (record['replayed_metrics'],record['independently_recomputed_metrics'],checked['independently_recomputed_metrics']):
                assert math.isfinite(metrics[metric]) and math.isclose(metrics[metric],result['metrics'][metric],rel_tol=2e-6,abs_tol=2e-6)
        dataset=ROOT/model['dataset_root'];output=ROOT/model['output_root']
        artifacts=dict(config=ROOT/record['config'],result=ROOT/record['result_path'],
            checkpoint=ROOT/record['checkpoint_path'],arrays=ROOT/record['replay_arrays'],dataset=dataset/'datasets.h5',
            resume=output/'resume.pt',original_group_statistics=output/'group_statistics.npz',data_audit=dataset/'data_audit.json')
        hashes={name:sha(path) for name,path in artifacts.items()}
        assert hashes==checked['hashes'] and sha(paths['native_config'])==report['configuration_sha256']
        for label,flag in [('native_guard','full_native_model_replay_passed'),('review_guard','full_native_batch_independent_review_passed')]:
            guard=read(paths[label]);assert guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard[flag]
        paths.update(original_result=artifacts['result'],
            source_guard=Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/receipts')/(identity+'.json'))
        guard=read(paths['source_guard']);assert guard['exit_code']==0 and guard['full_result_validated'] and not guard['resource_guard_aborted']
        row=dict(id=identity,seed=seed,method='saits',dataset='electricity',nominal_missing_ratio=rate,
            dataset_sha256=hashes['dataset'],original_metrics=result['metrics'],independent_metrics=checked['independently_recomputed_metrics'],
            native_replayed_metrics=record['replayed_metrics'],full_training_completion=finish,
            original_month_block_bootstrap_95_ci=result['month_block_bootstrap_95_ci'],validation_best=result['validation_best'],
            full_test_shape=[290,100,370],held_out_coordinates=targets,months=10,model_parameters=11511000,
            actual_replacement_mask_target_fraction=targets/10730000,full_artifact_hashes=hashes,
            citation=model['citation'],data_citation=model['data_citation'],protocol_boundaries=result['protocol_boundaries'])
        cohort.append(row);rows.append(row);archives.append((identity,paths))
        pins.update(str(p) for p in artifacts.values());pins.update(str(p) for p in paths.values())
    summary=summarize(cohort)
    independent={}
    for metric in METRICS:
        values=[r['original_metrics'][metric] for r in cohort]
        mean=math.fsum(values)/5;std=math.sqrt(math.fsum((v-mean)**2 for v in values)/4)
        low,high=student_t.interval(.95,4,loc=mean,scale=std/math.sqrt(5))
        check=dict(mean=mean,std=std,ci95_low=float(low),ci95_high=float(high))
        assert all(math.isclose(summary['summaries'][metric][k],v,rel_tol=1e-12,abs_tol=1e-12) for k,v in check.items())
        independent[metric]=check
    summaries.append(dict(nominal_missing_ratio=rate,held_out_coordinates=targets,
        actual_replacement_mask_target_fraction=targets/10730000,**summary,independently_reduced_with_SciPy_df4_t95=independent))

test=Path('F:/aicoding/IIA_Data/experiments/SAITS_Electricity50_60_all_five_model_closure_20261016_v1_tests.xml')
suites=list(ET.parse(test).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==56
assert not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
packet.mkdir(parents=True,exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.xml -text\n*.csv -text\n*.py -text\n',encoding='utf-8',newline='\n')
for identity,paths in archives:
    target=packet/'models'/identity;target.mkdir(parents=True)
    for label,path in paths.items():shutil.copyfile(path,target/(label+'.json'))
shutil.copyfile(__file__,packet/'closure_source.py');shutil.copyfile(test,packet/'validation_56_checks.xml')
for p in ['scripts/flow_matching/summarize_saits_full_five_seed_cohort_v1.py','tests/test_fm_saits_full_five_seed_cohort_v1.py',
          'tests/test_fm_saits_two_seed_cohort_v52.py','tests/test_fm_saits_completed_full_replay_batch_v1.py']:
    pins.add(str(ROOT/p))
pins.update(str(packet/n) for n in ['closure_source.py','validation_56_checks.xml'])
write(ROOT/frozen,dict(method='saits',dataset='electricity',nominal_missing_ratios=[.5,.6],original_training_seeds=[26,27,28,29,30],
    required_original_models=210,full_models_in_this_packet=10,all_original_model_ids=sorted(required),
    source_receipts=[dict(path=p,sha256=sha(p)) for p in sorted(pins)],original_scope=runtime['original_scope'],all_original_experiments_complete=False))
for pin in read(ROOT/frozen)['source_receipts']:assert sha(pin['path'])==pin['sha256']
write(packet/'full_five_model_cohort_summaries.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    records=summaries,per_original_model=rows,frozen_configuration=frozen,frozen_configuration_sha256=sha(ROOT/frozen),
    current_collector_full_verified_models=len(state['verified_original_ids']),required_original_models=210,
    additional_training_updates=0,additional_native_reinference=0,author_historical_pipeline_certified=False,
    strict_TAB_result=False,original_scope=runtime['original_scope'],all_original_experiments_complete=False))
live={}
for name,path in [('collector',ROOT/runtime['state_root']/'status.json'),
    ('actual_replacement_training',Path('F:/aicoding/IIA_Data/experiments/saits_guarded_epoch271_full210_recovery_v2/status.json'))]:
    s=read(path);p=psutil.Process(s['pid']);assert p.create_time()==s['create_time'] and p.is_running()
    live[name]=dict(state=s,exact_process_command=p.cmdline(),observed_cpu_seconds=sum(p.cpu_times()[:2]))
write(packet/'actual_full210_live_handles.json',live)
with (packet/'all10_actual_model_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['nominal_missing_ratio','actual_target_fraction','seed','completed_epoch_count','MAE','RMSE','MRE'])
    for r in rows:writer.writerow([r['nominal_missing_ratio'],r['actual_replacement_mask_target_fraction'],r['seed'],
        r['full_training_completion']['next_epoch'],*[r['original_metrics'][m] for m in METRICS]])
lines=['SAITS / Electricity：名义50%与60%缺失的两组完整原五种子结果','',
    '两组各包含26–30五个原训练种子，全部10个模型已经完成原最大10000epoch/patience30停止规则，且通过完整源检查点重推理和独立全网格/分组统计复核。每个模型11511000参数，测试290×100×370共10730000个网格坐标，10个月份；原生容差rtol=atol=2e-6未改。当前配置、结果、最佳检查点、resume/优化器/RNG、全部预测数组、源HDF5、原分组统计和审计文件SHA256再次核验，三阶段退出0资源回执全部保存。此次归档不重复训练或模型推理。','',
    '|名义缺失率|实际留出比例|指标|五模型均值|样本标准差|模型均值95% t区间(df4)|',
    '|---|---:|---|---:|---:|---|']
for r in summaries:
    for metric in METRICS:
        x=r['summaries'][metric]
        lines.append(f"|{r['nominal_missing_ratio']:.0%}|{r['actual_replacement_mask_target_fraction']:.6%}|{metric.upper()}|{x['mean']:.9g}|{x['std']:.9g}|[{x['ci95_low']:.9g}, {x['ci95_high']:.9g}]|")
lines+=['','以上区间描述固定划分上五次训练的模型均值不确定性，n=5，不代表跨数据集或跨划分稳健性。每个模型原有月份块bootstrap95%区间另存于JSON，与训练种子区间分开；月份块之间的时间独立性不作保证。未丢弃任一种子或按测试表现选取模型。','',
    '作者有放回mask导致名义50%和60%分别只留出4221734与4842788个测试坐标，实际比例见表。原月份划分较早test/中间validation/较晚train、1397与论文1400序列差异、未披露预处理种子、Windows loader随机流差异等边界均保持在逐模型记录中。这些是插补结果，不能作为异常检测F1、严格因果TAB结果或原论文历史流水线等价认证。','',
    f"本轮56项相关校验通过，原SAITS/BRITS队列已独立复核{len(state['verified_original_ids'])}/210项，其他原实验继续。45论文/681方法基线消融/273数据任务/702论文标量目标没有缩小且尚未全部完成。"]
(packet/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
paths=[frozen,*[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]]
paths += [str(path.relative_to(ROOT)) .replace('\\','/') for _,items in archives for name,path in items.items() if name.endswith('_config')]
candidate=set(paths);changed=set()
targets=[frozen,packet.relative_to(ROOT).as_posix(),runtime['generated_config_root']]
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    changed.update(subprocess.check_output(['git',*args,'--',*targets],text=True).splitlines())
paths=sorted(candidate & changed)
write(ROOT/'tmp/SAITS_Electricity50_60_all_five_publication_manifest_20261016_v1.json',dict(
    previous_root=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    previous_parent=subprocess.check_output(['git','-C',str(ROOT.parent/'flow-matching-research'),'rev-parse','HEAD'],text=True).strip(),paths=sorted(set(paths))))
print(json.dumps(dict(full_original_models=10,complete_five_model_cohorts=2,verified_full_queue=len(state['verified_original_ids']),
    cohort_summary=[dict(nominal_rate=r['nominal_missing_ratio'],summaries=r['summaries']) for r in summaries],exact_files=len(set(paths))),ensure_ascii=False))
