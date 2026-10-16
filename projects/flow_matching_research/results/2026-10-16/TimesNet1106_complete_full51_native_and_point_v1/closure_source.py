"""Close actual full51 source/native/point proofs; never launch numerical work."""
from datetime import datetime,timezone
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from scripts.flow_matching import continue_timesnet_remaining4_cold_v3 as s

ROOT=s.ROOT;c=s.read(ROOT/s.original.CONFIG);jobs=s.original.verify(c);job=jobs[s.original.identity(1106)]
assert s.original.source_ready(c,job)
bindings={mode:s.checked_stage(c,job,mode) for mode in ['point','native']}
stages={mode:s.original.stage_config(c,job,mode) for mode in bindings}
proofs={mode:s.read(bindings[mode]['proof']) for mode in bindings}
for field in ['full_entities','full_test_points','full_validation_points','full_train_points','coverage',
    'independently_recomputed_primary_metrics','conditional_entity_cluster_f1_ci95','original_event_metrics','original_test_calibrated_TAB_core_diagnostic']:
    assert proofs['point'][field]==proofs['native'][field]
assert proofs['native']['full_native_checkpoint_reinference_passed'] and proofs['native']['all_replayed_timestamp_scores_bitwise_equal']
source=ROOT/job['output_directory'];result=s.read(source/'result.json')
replay=Path(stages['native']['output_directory'])/'replayed_scores.npz'
assert s.sha(source/'scores.npz')==s.sha(replay)==result['scores_sha256']
manifest=s.read(ROOT/job['dataset_manifest_path']);dataset=next(d for d in manifest['datasets'] if d['id']=='smap')
assert len(dataset['entities'])==51
array_proofs=[];points=dict(train=0,validation=0,test=0,labels=0)
def shape(payload):
    assert payload[:6]==b'\x93NUMPY';version=payload[6:8]
    if version==b'\x01\x00':n=struct.unpack('<H',payload[8:10])[0];start=10
    else:assert version in [b'\x02\x00',b'\x03\x00'];n=struct.unpack('<I',payload[8:12])[0];start=12
    h=ast.literal_eval(payload[start:start+n].decode('utf-8' if version==b'\x03\x00' else 'latin1'))
    assert not h['fortran_order'] and len(h['shape'])==1
    return h
coverage={(r['entity'],r['split']):r['points'] for r in proofs['native']['coverage']}
assert len(coverage)==153
expected=set()
with zipfile.ZipFile(source/'scores.npz') as before,zipfile.ZipFile(replay) as after:
    for e in dataset['entities']:
        prefix=e['array_prefix']
        for split in ['train','validation','test','labels']:
            name=prefix+'_'+split+'.npy';expected.add(name);a,b=before.read(name),after.read(name)
            assert a==b;header=shape(a)
            required=coverage[(e['entity_id'],'test' if split=='labels' else split)]
            assert header['shape']==(required,);points[split]+=required
            array_proofs.append(dict(entity=e['entity_id'],split=split,npz_member=name,points=required,dtype=header['descr'],
                complete_NPY_bytes=len(a),complete_NPY_sha256=hashlib.sha256(a).hexdigest(),bytes_identical=True))
    assert set(before.namelist())==set(after.namelist())==expected and len(expected)==204
assert points==dict(train=101565,validation=26684,test=408160,labels=408160)
oldfailed=jobs[s.original.identity(1105)];oldstage=s.original.stage_config(c,oldfailed,'native')
oldguard=ROOT/oldstage['state_root']/'receipt.json';failure=s.read(oldguard)
assert not failure['passed'] and failure['exit_code']!=0 and not failure['resource_guard_aborted']
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/TimesNet1106_complete_full51_native_and_point_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
for mode in ['point','native']:
    for path,name in [(Path(bindings[mode]['proof']),mode+'_complete_proof.json'),
        (ROOT/stages[mode]['state_root']/'receipt.json',mode+'_successful_guard.json'),
        (ROOT/stages[mode]['state_root']/'frozen_stage.json',mode+'_frozen_configuration.json')]:shutil.copyfile(path,packet/name)
shutil.copyfile(source/'result.json',packet/'original_source_result.json')
shutil.copyfile(ROOT/stages['native']['terminal_receipt'],packet/'original_successful_training_guard.json')
shutil.copyfile(ROOT/job['model_config'],packet/'original_frozen_model_configuration.json')
shutil.copyfile(oldguard,packet/'preserved_other_seed1105_strict_failure.json')
(packet/'all204_full_NPY_member_byte_comparisons.json').write_text(json.dumps(dict(complete_NPY_members=array_proofs,point_totals=points,
    source_archive_sha256=result['scores_sha256'],replay_archive_sha256=s.sha(replay)),indent=2)+'\n',encoding='utf-8')
metric_rows=[]
for ratio,record in result['metrics']['strict'].items():
    metric_rows.append(dict(training_seed=1106,validation_alarm_percent=ratio,threshold=record['calibration']['threshold'],
        **record['micro'],conditional_entity_cluster_F1_CI95=json.dumps(record['f1_ci95'])))
with (packet/'all3_validation_only_no_PA_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
xml=ROOT/'tmp/TimesNet_remaining4_full_audits_validation_20261016_v2.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)>0
assert all(int(x.get('errors',0))==int(x.get('failures',0))==int(x.get('skipped',0))==0 for x in suites)
shutil.copyfile(xml,packet/'original_unchanged_auditor_validation.xml')
windows=sum(r['windows'] for r in proofs['native']['coverage'] if r['split']=='train')
closure=dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_id=job['id'],training_seed=1106,
    complete_registered_native_and_point_stages=True,full_registered_entities=51,complete_score_arrays=153,complete_label_arrays=51,
    complete_score_points=536409,complete_test_points=408160,all204_original_NPY_members_compared_byte_for_byte=True,
    complete_compressed_archives_byte_identical=True,parameter_count=28132633,configured_epochs=3,
    original_training_windows=windows,configured_updates_derived_from_code=3*((windows+127)//128),
    actual_persisted_epoch_update_trace_available=False,additional_optimizer_updates=0,additional_training_seeds=0,
    source_checkpoint_sha256=result['checkpoint_sha256'],source_scores_sha256=result['scores_sha256'],
    primary_validation1_no_PA=proofs['native']['independently_recomputed_primary_metrics'],
    conditional_entity_cluster_F1_CI95=proofs['native']['conditional_entity_cluster_f1_ci95'],
    complete_five_seed_group=False,other_seed1105_strict_failure_preserved=True,
    canonical_NASA_full54_entities=False,historical_author_equivalence_certified=False,full_TAB_training_harness_certified=False,
    full_TAB_range_completion_claimed=False,original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
    source_stage_bindings=bindings,all_original_experiments_complete=False)
(packet/'complete_two_stage_closure.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf-8')
m=closure['primary_validation1_no_PA'];ci=closure['conditional_entity_cluster_F1_CI95']
(packet/'README.md').write_text(f'''# TimesNet SMAP／seed1106：完整51实体原生重放与分数复核

源训练成功守卫、独立保存分数复核、完整检查点重放均已通过。全51个登记实体、101565训练点、26684验证点、408160测试点保持；153个评分数组和51个标签数组共204个完整NPY成员，独立标准库再次逐字节比较一致，整个压缩评分归档SHA256也一致。原训练、验证、测试的536409个评分坐标均覆盖。

验证集1%阈值、无PA：F1={m['f1']:.12f}，precision={m['precision']:.12f}，recall={m['recall']:.12f}，AUROC={m['auroc']:.12f}，AP={m['average_precision']:.12f}。条件实体簇F1区间为{ci}；这是固定模型的实体重采样区间，不是五训练种子区间。三个原始验证阈值全部单列；测试校准和测试标签选择诊断保留原报告，未混入这些严格验证阈值点值。

本地冻结模型28132633参数、CPU float32/batch128/CPU2及原逐位一致性要求保持。配置训练3轮，{windows}个完整本地非重叠/尾部对齐训练窗口，按冻结代码推导{closure['configured_updates_derived_from_code']}次配置更新；没有保存实际逐轮/更新计数，不能把推导值称作实测。51实体子集不冒充完整NASA54实体，历史作者训练/预处理/模型协议和完整TAB训练策略仍未认证。

另一种子1105的原逐位失败完整保留，本例成功不覆盖其失败，也不将不同认证状态的模型混成完整五种子结论。当前只闭合1106两个阶段，没有新增训练或推理。本包复用冻结验证器既有检查XML，未把旧检查重复算作新增实验；新完整数组对比在本次真实执行。

原35基线、291数据控制、其余种子完整范围、全部TAB范围及45篇/681方法/273数据任务/702目标保持。整体实验未全部完成。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'closure_source.py')
(packet/'packet_integrity.json').write_text(json.dumps(dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=s.sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]),indent=2)+'\n',encoding='utf-8')
ara=ROOT/'projects/flow_matching_research/ara/execution/TimesNet1106_complete_full51.2026-10-16_v1.json';assert not ara.exists()
ara.write_text(json.dumps(dict(packet=packet.relative_to(ROOT).as_posix(),full_entities=51,full_test_points=408160,
    complete_point_and_native_stages=True,complete_five_seed_group=False,all_original_experiments_complete=False),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(packet=str(packet),primary_metrics=m,all204_arrays_byte_identical=True)))
