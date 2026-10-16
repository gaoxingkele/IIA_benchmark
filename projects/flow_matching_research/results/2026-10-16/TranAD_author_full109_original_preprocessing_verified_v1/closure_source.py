"""Independently reconstruct all prepared inputs from immutable original raw data."""
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import psutil
from scripts.flow_matching import run_tranad_author_all_entities_v1 as s

ROOT=s.ROOT;q=s.read(ROOT/s.CONFIG);s.verify(q)
proof=s.preparation_report(q);assert proof is not None
source=ROOT/q['source_root'];out=Path(q['artifact_root'])/'preparation';processed=out/'processed'
path=out/'complete_author_preprocessing_and_real_preflight.json'
base=ROOT/q['state_root'];guard_path=base/'attempts/preparation/guard_receipt.json';guard=s.read(guard_path)
assert guard['exact_child_handle']['command']==[q['python'],'-B','-X','utf8','-u','-m',s.MODULE,'--phase','prepare']
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
assert guard['proof_sha256']==s.sha(path)
assert guard['peak_tree_rss_bytes']<=q['preparation_settings']['maximum_worker_rss_bytes']
assert guard['peak_tree_private_bytes']<=q['preparation_settings']['maximum_worker_private_bytes']
try:
    previous=psutil.Process(guard['exact_child_handle']['pid'])
    assert previous.create_time()!=guard['exact_child_handle']['create_time']
except psutil.NoSuchProcess:
    pass
with (source/'data/SMAP_MSL/labeled_anomalies.csv').open(encoding='utf-8',newline='') as f:
    metadata=list(csv.DictReader(f))
file_bindings={x['path']:x for x in proof['files']}
records=[];counts={k:Counter() for k in ['SMAP','MSL','SMD']}
def csv_matrix(path):
    with path.open(encoding='utf-8',newline='') as f:
        return np.asarray([[float(x) for x in row] for row in csv.reader(f)],dtype=np.float64)

for entity in q['entities']:
    dataset,name=entity['dataset'],entity['entity']
    if dataset in ['SMAP','MSL']:
        raw=source/'data/SMAP_MSL'
        train=np.load(raw/'train'/(name+'.npy'),allow_pickle=False)
        test=np.load(raw/'test'/(name+'.npy'),allow_pickle=False)
        lower,upper=train.min(axis=0),train.max(axis=0)
        expected_train=(train-lower)/(upper-lower+0.0001)
        expected_test=(test-lower)/(upper-lower+0.0001)
        expected_labels=np.zeros(test.shape)
        selected=next(x for x in metadata if x['spacecraft']==dataset and x['chan_id']==name)
        ranges=ast.literal_eval(selected['anomaly_sequences'])
        for begin,end in ranges:
            expected_labels[int(begin):int(end),:]=1
        raw_paths=[raw/'train'/(name+'.npy'),raw/'test'/(name+'.npy'),raw/'labeled_anomalies.csv']
    else:
        raw=source/'data/SMD'
        expected_train=csv_matrix(raw/'train'/(name+'.txt'))
        expected_test=csv_matrix(raw/'test'/(name+'.txt'))
        expected_labels=np.zeros(expected_test.shape)
        with (raw/'interpretation_label'/(name+'.txt')).open(encoding='utf-8') as f:
            for line in f:
                interval,channels=line.strip().split(':')
                begin,end=map(int,interval.split('-'))
                for channel in map(int,channels.split(',')):
                    expected_labels[begin-1:end-1,channel-1]=1
        raw_paths=[raw/'train'/(name+'.txt'),raw/'test'/(name+'.txt'),raw/'interpretation_label'/(name+'.txt')]
    digests={};coordinates=0
    for role,expected in [('train',expected_train),('test',expected_test),('labels',expected_labels)]:
        actual_path=processed/dataset/(name+'_'+role+'.npy')
        actual=np.load(actual_path,allow_pickle=False)
        assert actual.dtype==expected.dtype and actual.shape==expected.shape
        assert actual.tobytes()==expected.tobytes(),(dataset,name,role)
        assert file_bindings[str(actual_path)]['sha256']==s.sha(actual_path)
        coordinates+=actual.size
        digests[role+'_sha256']=s.sha(actual_path)
    counts[dataset].update(entities=1,train_rows=len(expected_train),test_rows=len(expected_test),
        train_feature_coordinates=expected_train.size,test_feature_coordinates=expected_test.size,
        feature_label_coordinates=expected_labels.size,positive_feature_labels=int(expected_labels.sum()))
    records.append(dict(dataset=dataset,entity=name,train_rows=len(expected_train),test_rows=len(expected_test),features=expected_test.shape[1],
        train_dtype=expected_train.dtype.str,test_dtype=expected_test.dtype.str,label_dtype=expected_labels.dtype.str,
        every_raw_to_prepared_coordinate_byte_identical=True,compared_feature_coordinates=coordinates,
        **digests,original_raw_input_bindings=[dict(path=str(p),sha256=s.sha(p)) for p in raw_paths]))
assert len(records)==109 and len(file_bindings)==327
assert {k:v['entities'] for k,v in counts.items()}==dict(SMAP=54,MSL=27,SMD=28)
assert proof['configuration_sha256']==s.sha(ROOT/s.CONFIG)
assert not proof['raw_files_overwritten'] and not proof['old_local_tracks_modified'] and proof['complete_trained_models']==0
preflight=proof['actual_real_input_preflights']
assert len(preflight)==5 and {p['variant'] for p in preflight}==set(s.VARIANTS)
assert all(p['finite_forward_and_backward'] and p['optimizer_updates']==0 and p['full_original_training_batch']==128 and not p['benchmark_performance'] for p in preflight)
state=s.read(base/'status.json');owner=psutil.Process(state['pid'])
assert owner.create_time()==state['create_time'] and owner.is_running()
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/TranAD_author_full109_original_preprocessing_verified_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n',encoding='utf-8')
shutil.copyfile(path,packet/'actual_original_preprocessing_and_real_preflight.json')
shutil.copyfile(guard_path,packet/'actual_preparation_guard.json')
shutil.copyfile(base/'status.json',packet/'actual_training_continuation_state.json')
s.write(packet/'independent_complete_raw_to_prepared_verification.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    all327_full_prepared_arrays_independently_reconstructed_and_byte_equal=True,
    compared_coordinates=sum(x['compared_feature_coordinates'] for x in records),records=records,dataset_totals=counts,
    independent_NASA_first_metadata_exclusive_labels_and_train_fitted_normalization=True,
    independent_SMD_stdlib_CSV_float_parser_and_original_feature_label_slices=True,
    all_original_raw_pins_unchanged=s.verify(q)==q['jobs'],five_real_input_preflights=preflight,
    preflight_is_complete_training_or_benchmark_performance=False,new_optimizer_updates=0,
    all_original_experiments_complete=False))
with (packet/'all109_complete_original_input_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
    headers=[k for k in records[0] if k!='original_raw_input_bindings']
    writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows({k:r[k] for k in headers} for r in records)
s.write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],original_entities_prepared_and_verified=109,
    original_prepared_arrays=327,real_original_input_preflights=5,complete_author_source_training_conditions=len(state['completed_ids']),
    required_source_conditions=3270,required_primary_five_seed_cohorts=545,new_optimizer_updates_from_preflight=0,
    continuation_actual_pid=owner.pid,continuation_actual_create_time=owner.create_time(),all_original_experiments_complete=False))
lines=['# TranAD作者原始预处理：109实体和327完整数组已核验','',
    '原preprocess.py已在独立进程处理全部54SMAP、27MSL、28SMD实体；实际准备守卫退出0且未资源中止。新数组写入F盘独立目录，全部原始数据和旧本地轨道哈希保持。','',
    '| 数据集 | 实体 | 完整训练行 | 完整测试行 |','|---|---:|---:|---:|']
for dataset,item in counts.items():lines.append(f"| {dataset} | {item['entities']} | {item['train_rows']} | {item['test_rows']} |")
lines += ['',f"独立从原始输入重建所有327个训练、测试和逐特征标签数组，逐字节核对{sum(r['compared_feature_coordinates'] for r in records)}个坐标。NASA只用各实体完整训练集拟合归一化，遵循原首条metadata、右端不含标签并广播至各特征；SMD使用独立标准库CSV解析器和原解释标签区间，没有改用另一标签版本。",
    '五个源码分支在真实SMAP/P-1的原始128窗口训练批次上完成全量前向/反向有限性预检，优化器更新0。预检没有执行完整五轮训练，也没有新的基准性能；所有3270训练/新进程重放/原指标条件和545主要五种子组继续要求完成。',
    '预处理完成后的控制器现等待原完整训练40GiB提交内存余量与既有完整源CPU锁，不是仍等待预处理；状态中的上一次active_job可能保留preparation，此处以实际守卫和报告识别完成。没有降低门槛、缩小批次/实体或停止数值进程。',
    '原始45篇/681方法记录/273数据任务/702指标及原论文、TAB、新数据全部未完成项保留；本次完整数据准备与预检不冒充完整训练或历史作者成绩。']
(packet/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
s.write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=s.sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/TranAD_author_full109_original_preprocessing_verified.2026-10-16_v1.json'
assert not ara.exists()
s.write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),original_entities_prepared_and_verified=109,
    complete_trained_models=len(state['completed_ids']),required_models=3270,all_original_experiments_complete=False))
head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
s.write(ROOT/'tmp/TranAD_author_full109_original_preprocessing_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),
    paths=sorted([ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()])))
print(json.dumps(dict(packet=str(packet),entities=109,arrays=327,compared_coordinates=sum(x['compared_feature_coordinates'] for x in records),dataset_totals=counts,
    full_real_input_preflights=5,complete_author_training_conditions=len(state['completed_ids']),actual_controller=owner.pid)))
