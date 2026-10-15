import os
import subprocess
import pytest
from scripts.flow_matching import recover_moment_icm_insect_chunked_export_v2 as s


@pytest.mark.parametrize('dataset', ['ArticularyWordRecognition', 'AtrialFibrillation', 'JapaneseVowels'])
def test_chunked_files_are_byte_identical_to_original_pretraining_export_for_real_sources(tmp_path, dataset):
    code = r'''
from pathlib import Path
import sys
from scripts.flow_matching import recover_moment_icm_insect_chunked_export_v2 as s
from scripts.flow_matching import prepare_moment_icm_classification_v1 as components
root=s.ROOT
cls,_,_=components.source_components(root/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original')
path=root/'data/public_datasets/flow_matching/project_additions/moment_timeseries_pile/classification/UCR'/sys.argv[1]/(sys.argv[1]+'_TEST.ts')
dataset=s.original.source_pretraining_dataset(cls,path)
cases=len(dataset)//dataset.n_channels
old=s.original.export_pretraining_split(dataset,cases,Path(sys.argv[2])/'original')
new=s.chunked_export(dataset,cases,Path(sys.argv[2])/'chunked',max_mapping_bytes=1024)
assert old==new,(old,new)
assert set(new['files_sha256'])=={'values.npy','input_mask.npy','original_instance_rows.npy'}
assert 'torch' not in sys.modules
'''
    result = subprocess.run([str(s.ROOT / '.venv/moment-author-py310/Scripts/python.exe'), '-B', '-X', 'utf8',
        '-c', code, dataset, str(tmp_path)], cwd=s.ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-3000:]


def test_preserved_prefix_hash_rejects_truncated_input(tmp_path):
    p=tmp_path/'short';p.write_bytes(b'abc')
    with pytest.raises(ValueError):s.prefix_sha(p,4)
