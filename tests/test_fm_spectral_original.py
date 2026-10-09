import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from scripts.flow_matching.prepare_spectral_original_data import ROOT,extract_member,paper_sines
from scripts.flow_matching.register_spectral_original import table1_claims
from scripts.flow_matching.prepare_spectral_original_data import sha
from scripts.flow_matching import run_spectral_original_job as worker
from iia_benchmark.models.fm_spectral_author import build_author_model,chunked_field,FrozenAuthorGenerationDataset,bind_chunked_sampling


def test_raw_extraction_preserves_existing_and_rejects_escape(tmp_path):
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('datasets/stock.csv',b'original bytes');z.writestr('../outside',b'escape')
    with zipfile.ZipFile(archive) as z:
        path=extract_member(z,'datasets/stock.csv',tmp_path/'raw')
        assert extract_member(z,'datasets/stock.csv',tmp_path/'raw').read_bytes()==b'original bytes'
        path.write_bytes(b'user supplied different data')
        with pytest.raises(ValueError):extract_member(z,'datasets/stock.csv',tmp_path/'raw')
        assert path.read_bytes()==b'user supplied different data'
        with pytest.raises(ValueError):extract_member(z,'../outside',tmp_path/'raw')


def test_paper_sines_formula_and_paired_data_identity(tmp_path):
    values=paper_sines(123,num=7,dim=5,window=24)
    np.testing.assert_array_equal(values,paper_sines(123,num=7,dim=5,window=24))
    assert not np.array_equal(values,paper_sines(124,num=7,dim=5,window=24))
    rng=np.random.RandomState(123);frequency=rng.uniform();phase=(rng.uniform()*2-1)*np.pi
    np.testing.assert_allclose(values[0,:,0],np.sin(2*np.pi*frequency*np.arange(24)+phase),rtol=0,atol=0)
    path=tmp_path/'data.npy';np.save(path,values)
    dataset=FrozenAuthorGenerationDataset(path)
    assert (len(dataset),dataset.window,dataset.var_num)==(7,24,5)
    torch.testing.assert_close(dataset[0],torch.from_numpy(values[0]).float(),rtol=0,atol=0)


def source_model(**overrides):
    policy=json.loads((ROOT/'configs/reproducibility/spectral_mean_flow_original_protocol.v1.json').read_text(encoding='utf-8'))
    params=dict(seq_length=24,feature_size=6,n_layers=2,d_model=8,sampling_timesteps=8)
    params.update(overrides)
    return build_author_model(ROOT/policy['author_source'],'models.spectral_flow.model.SpectralFlow',params)


def test_actual_author_field_is_sample_independent_and_noise_order_is_preserved():
    torch.set_num_threads(1);torch.manual_seed(42)
    model=source_model(sampling_timesteps=8);x=torch.randn(5,24,6);times=torch.linspace(.1,.9,5)[:,None]
    full,aux=model.compute_flow(x,times)
    chunks,chunk_aux=chunked_field(model.compute_flow,x,times,2)
    torch.testing.assert_close(full,chunks,rtol=1e-4,atol=1e-5)
    assert aux.keys()==chunk_aux.keys()
    torch.manual_seed(123);original=model.generate_mts(batch_size=5)
    torch.manual_seed(123);repeated=model.generate_mts(batch_size=5)
    torch.testing.assert_close(original,repeated,rtol=0,atol=0)
    bind_chunked_sampling(model,2)
    torch.manual_seed(123);chunked=model.generate_mts(batch_size=5)
    # Tiny field errors can amplify during integration. This optimization must
    # remain diagnostic and must never be bound to the formal original runner.
    assert not torch.allclose(original,chunked,rtol=1e-4,atol=1e-5)
    assert 'bind_chunked_sampling(' not in Path(worker.__file__).read_text(encoding='utf-8')


def test_full_original_stocks_parameter_count_matches_primary_table():
    model=source_model(n_layers=10,d_model=64,sampling_timesteps=500)
    assert sum(p.numel() for p in model.parameters())==356800


def test_primary_table1_transcription_covers_both_methods_and_all_datasets():
    policy=json.loads((ROOT/'configs/reproducibility/spectral_mean_flow_original_protocol.v1.json').read_text(encoding='utf-8'))
    cache=ROOT/'papers/literature/flow_matching/ara_text_cache/spectral_mean_flow'/f"{policy['source_pdf_sha256']}.json"
    rows=table1_claims(json.loads(cache.read_text(encoding='utf-8')),policy)
    assert len(rows)==48 and len({(r['method'],r['dataset'],r['metric']) for r in rows})==48
    assert next(r for r in rows if (r['method'],r['dataset'],r['metric'])==('spectral_flow','stocks','context_fid'))['paper_mean']==.008
    assert all('half-width' in r['spread_type'] and r['source_page']==9 for r in rows)


def valid_result_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    reference=tmp_path/'registered_reference.npy';np.save(reference,np.zeros((3,24,6)))
    copied=tmp_path/'copied_reference.npy';np.save(copied,np.zeros((3,24,6)))
    generated=tmp_path/'generated.npy';np.save(generated,np.ones((4,24,6)))
    model=torch.nn.Linear(6,6);optimizer=torch.optim.Adam(model.parameters())
    for _ in range(2):
        optimizer.zero_grad();model(torch.ones(1,6)).square().sum().backward();optimizer.step()
    checkpoint=tmp_path/'complete.pt'
    torch.save({'step':2,'model':model.state_dict(),'ema':model.state_dict(),'opt':optimizer.state_dict()},checkpoint)
    job={'output_directory':'job','training_updates':2,'data_shape':[3,24,6],'generation_batch_size':2,
         'reference_path':'registered_reference.npy','reference_sha256':sha(reference)}
    record={'job_sha256':worker.fingerprint(job),'status':'completed','optimizer_updates':2,
        'evaluated_reference_samples':3,'evaluation_repeats':5,'benchmark_smoke':False,
        'metrics':{k:{'raw_repeats':[.2]*5,'mean':.2} for k in ['context_fid','correlational','discriminative','predictive']},
        'artifacts':[{'role':role,'path':p.name,'sha256':sha(p)} for role,p in [('checkpoint',checkpoint),('generated',generated),('reference',copied)]]}
    target=tmp_path/'job/result.json';target.parent.mkdir()
    target.write_text(json.dumps(record),encoding='utf-8')
    return job,record,target,checkpoint,generated,copied


def test_partial_model_checkpoint_and_incomplete_evaluators_cannot_count_as_complete(tmp_path,monkeypatch):
    job,record,target,checkpoint,_,_=valid_result_fixture(tmp_path,monkeypatch)
    assert worker.complete(job)
    record['metrics']['predictive']['raw_repeats']=[.2]*4;target.write_text(json.dumps(record))
    assert not worker.complete(job)
    record['metrics']['predictive']['raw_repeats']=[.2]*5
    saved=torch.load(checkpoint,weights_only=True);saved['step']=1;torch.save(saved,checkpoint)
    record['artifacts'][0]['sha256']=sha(checkpoint);target.write_text(json.dumps(record))
    assert not worker.complete(job)


def test_truncated_generation_and_changed_reference_cannot_count_as_complete(tmp_path,monkeypatch):
    job,record,target,_,generated,copied=valid_result_fixture(tmp_path,monkeypatch)
    np.save(generated,np.ones((2,24,6)));record['artifacts'][1]['sha256']=sha(generated);target.write_text(json.dumps(record))
    assert not worker.complete(job)
    np.save(generated,np.ones((4,24,6)));record['artifacts'][1]['sha256']=sha(generated)
    np.save(copied,np.ones((3,24,6)));record['artifacts'][2]['sha256']=sha(copied);target.write_text(json.dumps(record))
    assert not worker.complete(job)
