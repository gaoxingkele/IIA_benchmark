from dataclasses import dataclass
import json
import numpy as np
import pytest
from scripts.flow_matching import moment_icm_disk_epoch_plan_v1 as p
from scripts.flow_matching.moment_icm_channel_data_v1 import paired_epoch_batches


@dataclass
class Dataset:
    dataset_key:str
    width:int
    n:int
    split:str='TRAIN'
    def __len__(self):return self.n


@pytest.mark.parametrize('seed,batch',[(0,1),(13,3),(1103,8),(2**32-1,64)])
def test_every_batch_exactly_matches_original_eager_planner_with_partial_batches(tmp_path,seed,batch):
    datasets=[Dataset('a',1,11),Dataset('b',8,15),Dataset('empty',8,0),Dataset('c',1,17),Dataset('d',8,4)]
    descriptors=[dict(dataset_key=d.dataset_key,width=d.width,items=len(d),split=d.split) for d in datasets]
    original=paired_epoch_batches(datasets,batch,seed)
    p.build(tmp_path/'plan',descriptors,batch,seed);plan=p.DiskEpochPlan(tmp_path/'plan')
    assert len(plan)==len(original)
    actual=[plan.reference(i) for i in range(len(plan))]
    assert actual==list(original)
    assert sorted(item for ref in actual for item in ref.items)==[(i,j) for i,d in enumerate(datasets) for j in range(len(d))]
    assert plan.validate_complete(block_items=7)['all_original_source_items_exactly_once']


def test_prefix_mapping_retains_large_int64_counts_without_per_item_objects():
    sources=[dict(dataset_key='huge',split='train',width=1,items=2**33+7),dict(dataset_key='other',split='train',width=8,items=25)]
    x=p.layout(sources,2048)
    assert x['total_items']==2**33+32 and len(x['buckets'])==2
    assert x['buckets'][0]['items']==2**33+7


def test_saved_permutation_is_int64_and_chunked_without_overwrite(tmp_path):
    a=np.arange(1000,dtype=np.int32);file=tmp_path/'a.npy'
    record=p.save_int64_blocks(file,a,block_bytes=64)
    assert record['write_blocks']>1 and np.load(file).dtype==np.dtype('int64')
    assert np.array_equal(p.read_indices(tmp_path,record,313,17),a[313:330])
    with pytest.raises(FileExistsError):p.save_int64_blocks(file,a)


def test_mixed_splits_and_changed_plan_scope_are_rejected(tmp_path):
    sources=[dict(dataset_key='a',split='TRAIN',width=1,items=7),dict(dataset_key='b',split='TEST',width=8,items=5)]
    with pytest.raises(ValueError):p.layout(sources,4)
    sources[1]['split']='TRAIN';p.build(tmp_path/'plan',sources,4,13)
    path=tmp_path/'plan/epoch_plan.json';x=json.loads(path.read_text());x['total_items']-=1;path.write_text(json.dumps(x))
    with pytest.raises(ValueError):p.DiskEpochPlan(tmp_path/'plan')


def test_duplicate_indices_are_rejected_even_after_rehashing(tmp_path):
    source=[dict(dataset_key='a',split='train',width=1,items=17)]
    p.build(tmp_path/'plan',source,4,13);file=tmp_path/'plan/width1_permutation.npy'
    values=np.load(file);values[3]=values[0];np.save(file,values,allow_pickle=False)
    path=tmp_path/'plan/epoch_plan.json';x=json.loads(path.read_text());x['permutation_files']['1']['sha256']=p.sha(file);path.write_text(json.dumps(x))
    plan=p.DiskEpochPlan(tmp_path/'plan')
    with pytest.raises(ValueError):plan.validate_complete(block_items=3)


def test_native_environment_real_three_source_groups_match_every_original_batch(tmp_path):
    import os,subprocess
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    code=r'''
import json,sys,numpy as np
from pathlib import Path
from scripts.flow_matching.moment_icm_channel_data_v1 import GroupedCases,paired_epoch_batches,materialize_batch
from scripts.flow_matching import moment_icm_disk_epoch_plan_v1 as p
c=json.loads(Path('configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json').read_text())
data=[GroupedCases(Path(c['artifact_root'])/name/'TRAIN',dataset_key=name,split='TRAIN',seed=13,
 policy='appendix_D_partition8_remainder_replacement') for name in ['Adiac','BasicMotions','ArticularyWordRecognition']]
try:
 desc=[dict(dataset_key=d.dataset_key,split=d.split,width=d.width,items=len(d)) for d in data]
 old=paired_epoch_batches(data,32,13);p.build(Path(sys.argv[1])/'native',desc,32,13)
 plan=p.DiskEpochPlan(Path(sys.argv[1])/'native');assert len(plan)==len(old)
 for i,reference in enumerate(old):assert plan.reference(i)==reference
 proof=plan.validate_complete(block_items=17);assert proof['all_original_source_items_exactly_once']
 for i in sorted({0,len(plan)//2,len(plan)-1}):
  a=materialize_batch(data,old[i]);b=materialize_batch(data,plan.reference(i))
  assert a['values'].tobytes()==b['values'].tobytes() and a['input_mask'].tobytes()==b['input_mask'].tobytes()
  assert a['provenance']==b['provenance']
 assert np.__version__=='1.25.2' and 'torch' not in sys.modules
finally:
 for d in data:d.close()
'''
    result=subprocess.run([str(root/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8','-c',code,str(tmp_path)],cwd=root,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-1000:]+result.stderr[-4000:]
