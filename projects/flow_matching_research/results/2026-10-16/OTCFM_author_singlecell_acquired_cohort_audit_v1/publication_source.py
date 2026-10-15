"""Verify all actual author payloads and distinguish paper/release cohorts."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from scripts.flow_matching import acquire_otcfm_author_singlecell_mendeley_v1 as source

root=source.ROOT
out=root/'projects/flow_matching_research/results/2026-10-16/OTCFM_author_singlecell_acquired_cohort_audit_v1'
out.mkdir(parents=True,exist_ok=False)
read,sha=source.read,source.sha
mapping=[]


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def snapshot(path,name):
    path=root/path
    shutil.copy2(path,out/name)
    assert path.read_bytes()==(out/name).read_bytes()
    mapping.append(dict(source=str(path),snapshot=name,sha256=sha(path)))


def column(group,key):
    node=group[key]
    if isinstance(node,h5py.Group):
        cats=node['categories']
        values=cats.asstr()[:] if cats.dtype.kind in 'SO' else cats[:]
        codes=node['codes'][:]
        assert codes.ndim==1 and (codes>=0).all() and (codes<len(values)).all()
        return values[codes]
    return node.asstr()[:] if node.dtype.kind in 'SO' else node[:]


def identity_digest(values):
    return hashlib.sha256(b''.join(str(v).encode('utf-8')+b'\0' for v in values)).hexdigest()


config=read(root/source.CONFIG)
terminal=read(root/source.BASE/'all_author_payloads_verified.json')
assert terminal['verified_payloads']==3 and len(terminal['records'])==3
assert terminal['configuration_sha256']==sha(root/source.CONFIG)
records=[]
for item in config['sources']:
    path=root/item['path']
    # All newly acquired full files are independently rehashed, not just opened.
    integrity=source.validate(path,item)
    receipt=read(path.with_name(path.name+'.verified.json'))
    assert receipt['sha256']==integrity['sha256'] and receipt['publisher_checksum_verified']
    with h5py.File(path,'r') as f:
        obs=f['obs'];ids=column(obs,obs.attrs['_index'])
        pcs=f['obsm/X_pca'][:,:5]
        assert len(ids)==len(pcs) and len(set(ids))==len(ids)
        assert pcs.shape[1]==5 and np.isfinite(pcs).all()
        record=dict(id=item['id'],path=item['path'],resolved_path=str(path.resolve()),**integrity,
            observed_cells=len(ids),cell_id_order_sha256=identity_digest(ids),
            full_pca_shape=list(f['obsm/X_pca'].shape),first_five_pc_dtype=str(pcs.dtype),
            first_five_pc_bytes_sha256=hashlib.sha256(pcs.tobytes()).hexdigest(),
            all_first_five_pc_values_finite=True,
            X_encoding=f['X'].attrs.get('encoding-type'),
            X_shape=[int(v) for v in f['X'].attrs['shape']] if isinstance(f['X'],h5py.Group) else list(f['X'].shape),
            obs_columns=list(obs),obsm_keys=list(f['obsm']),
            data_modified_or_preprocessed=False)
        if 'donor' in obs:
            donors=column(obs,'donor');days=column(obs,'day')
            record.update(donor_counts=dict(Counter(str(v) for v in donors)),
                day_counts=dict(Counter(str(v) for v in days)),paper_requested_donor='13176',
                paper_donor_present=bool(np.any(donors.astype(str)=='13176')),
                released_file_cannot_supply_paper_donor=not bool(np.any(donors.astype(str)=='13176')))
            assert set(record['donor_counts'])=={'32606'}
            assert set(record['day_counts'])=={'2','3','4','7'}
            record['paper_original_input_ready']=False
            record['actual_released_cohort']='32606'
        else:
            labels=column(obs,'sample_labels')
            record.update(timepoint_counts=dict(Counter(str(v) for v in labels)),
                paper_original_input_ready=False,
                boundary='EB tutorial v3 PCA is explicitly different from paper TrajectoryNet eb_velocity_v5 PCA; do not replace it.')
        if 'pca' in f['uns'] and 'params' in f['uns/pca']:
            record['published_pca_parameters']={key:bool(v[()]) for key,v in f['uns/pca/params'].items()}
        records.append(record)
    snapshot(path.with_name(path.name+'.verified.json'),item['filename']+'.verified.json')

# The preexisting organizer files are a distinct public source, with different
# cell identifiers. Inspect actual metadata; do not pretend to rehash40GB here.
catalog=read(root/'projects/flow_matching_research/materials/catalog.json')
public=[]
for item in catalog['artifacts']:
    if (item.get('id','').startswith('neurips2022_public_') and str(item.get('path','')).endswith('.h5ad')):
        with h5py.File(root/item['path'],'r') as f:
            donors=column(f['obs'],'donor');days=column(f['obs'],'day')
            selected=donors.astype(str)=='13176'
            matching=selected & np.isin(days.astype(str),['2','3','4','7'])
            index=column(f['obs'],f['obs'].attrs['_index'])
            public.append(dict(id=item['id'],path=item['path'],actual_cells=len(index),
                recorded_acquisition_sha256=item['sha256'],full_payload_hash_rechecked_this_audit=False,
                actual_donor13176_cells=int(selected.sum()),paper_days_donor13176_cells=int(matching.sum()),
                donor13176_day_counts=dict(Counter(str(v) for v in days[selected])),
                paper_days_cell_id_order_sha256=identity_digest(index[matching]),
                counts_encoding=f['layers/counts'].attrs['encoding-type'],
                counts_shape=[int(v) for v in f['layers/counts'].attrs['shape']],
                paper_preprocessed_PC_artifact_equivalence_certified=False))
assert len(public)==4
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),author_dataset_DOI=config['dataset_doi'],
    publisher_license=config['license'],publisher_api=source.API,
    total_publisher_verified_bytes=sum(r['bytes'] for r in records),verified_author_payloads=records,
    actual_public_organizer_metadata=public,
    actual_author_CITE_and_Multiome_donor='32606',paper_E5_requested_donor='13176',
    author_released_assets_present=True,paper_cohort_preprocessing_complete=False,
    raw_Kaggle_access_restrictions_resolved=False,wrong_cohort_substituted_for_original_experiment=False,
    no_model_training_or_checkpoint_inference_executed=True,all_original_experiments_complete=False,
    next_required_work=['Resolve cell-ID/feature/normalization mapping between original Kaggle/author and public organizer counts.',
        'Regenerate donor13176 days2,3,4,7 first5PCs and original whitening with a frozen, qualified preprocessing protocol.',
        'Preserve original EB TrajectoryNet artifact instead of replacing it with tutorial v3 PCs.',
        'Run all paper methods/baselines/ablations and leave-timepoint experiments with original budgets after input/environment acceptance.'],
    conclusion='Public author version1 CITE and Multiome processed files both contain only donor32606. '
        'Paper E5 specifies13176. Their presence does not close the original-cohort reproduction requirement. '
        'The public organizer payloads contain donor13176, but regenerated PCs still require protocol/alignment verification.')
write(out/'complete_payload_integrity_and_actual_cohort_audit.json',report)
snapshot(root/source.CONFIG,'acquisition_configuration.json')
snapshot(root/source.BASE/'publisher_version1_files.json','publisher_version1_files.json')
snapshot(root/source.BASE/'all_author_payloads_verified.json','all_author_payloads_verified.json')
for index,path in enumerate(['scripts/flow_matching/acquire_otcfm_author_singlecell_mendeley_v1.py',
                             'tests/test_fm_otcfm_author_singlecell_acquisition_v1.py']):
    snapshot(path,'source%02d.py'%index)
xml=Path('F:/aicoding/IIA_Data/experiments/otcfm_author_singlecell_acquisition_tests_v2.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==11
assert not sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
snapshot(xml,'validation11.xml')
write(out/'source_bindings.json',dict(files=mapping,tests_passed=11,
    acquisition_pre_registration_failed_fixture_preserved_locally='F:/aicoding/IIA_Data/experiments/otcfm_author_singlecell_acquisition_tests_v1.xml',
    boundary='Initial invalid fixture omitted required id; fixed before registration/download. Raw files never overwritten.'))
addendum=root/'projects/flow_matching_research/materials/otcfm_author_singlecell_addendum.v1.json'
write(addendum,dict(acquisition_configuration=source.CONFIG,evidence=str((out/'complete_payload_integrity_and_actual_cohort_audit.json').relative_to(root)),
    sources=[dict(id=r['id'],path=r['path'],bytes=r['bytes'],sha256=r['sha256'],publisher_checksum_verified=True) for r in records],
    paper_donor13176_inputs_ready=False,raw_Kaggle_authentication_gaps_remain=True,
    existing_catalog_and_live_frozen_registries_modified=False))
(out/'README.md').write_text('''# OT-CFM 作者单细胞文件已下载，实际 cohort 与论文不一致

通过作者仓库单细胞 notebook 指向的 [Mendeley version1](https://data.mendeley.com/datasets/hhny5ff7yj/1)，补充下载三个作者公开处理文件，共 2,384,324,025 字节。使用 aria2c 和既有代理，逐文件通过发布方大小、SHA256、HDF5 签名校验；之后再次独立读取完整文件计算 SHA256，并打开全部 HDF5 检查实际元数据与前五 PC。大文件保存在 F:，具体解析路径与哈希见报告。11 项下载保护检查通过，旧数据没有覆盖。

实际内容发现：op_cite_inputs_0.h5ad 包含 31,240 个细胞，op_train_multi_targets_0.h5ad 包含 33,701 个细胞，二者都只有 donor 32606，均覆盖 day 2/3/4/7。论文 PDF 第 32 页 E.5 明确写 donor 13176。因此不能把下载成功直接称为论文原队列的数据已齐全，更不能用这些不同 cohort 的成绩去宣称论文指标复现。作者已发布的 PCA 为 100 维，中心化开启，不使用高变基因；尚不能据此认证全部历史归一化/PCA 设置。

本地已有组织方公开四个 H5AD 确实包含 donor 13176：CITE 的四天合计 29,394 个细胞，Multiome 的四天合计 35,384 个细胞，另有 day10 必须单独识别。它们使用不同的细胞标识；CITE 组织方 RNA 特征数 22,085，作者文件为 22,050，Multiome 均为 23,418。因此还需要核对特征、原始计数/归一化、ID 与 PCA 拟合范围，再生成论文指定 cohort 的五 PC 和标准化输入，不能静默替换已有作者 PC。本次只读取这些既有组织方文件的真实元数据，没有重新对约 40 GB 计算哈希，报告中明确区分已有获取哈希与本次验证。

ebdata_v3.h5ad 有 18,203 个细胞、50 个 PC，是作者 notebook 的另一个教程版本。作者 notebook 明确指出论文使用 TrajectoryNet 的 eb_velocity_v5.npz，预处理不同；新下载的 EB v3 不替代已经注册的论文 EB 输入。

这是已完成的素材获取及真实输入审计，未训练模型、未重新推理检查点，也没有新增排行榜结果。原 Kaggle 文件访问限制、论文 cohort 预处理、全部 Table4 原生方法/基线与留时间点实验、原 45/681/273/702 研究范围仍保留。新素材使用独立 acquisition 配置与材料补充清单，未修改任何正在运行的冻结注册表。
''',encoding='utf-8',newline='\n')
snapshot(Path(__file__),'publication_source.py')
git=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),
    paths=[source.CONFIG,'scripts/flow_matching/acquire_otcfm_author_singlecell_mendeley_v1.py',
           'tests/test_fm_otcfm_author_singlecell_acquisition_v1.py',addendum.relative_to(root).as_posix()]
          +[p.relative_to(root).as_posix() for p in out.iterdir()])
(root/'tmp/OTCFM_author_singlecell_exact_publication_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(payloads=len(records),verified_bytes=sum(r['bytes'] for r in records),
    publisher_checksums_verified=True,actual_CITE_and_Multiome_donor32606=True,paper_donor13176_inputs_complete=False,
    tests=11,files=len(manifest['paths']))))
