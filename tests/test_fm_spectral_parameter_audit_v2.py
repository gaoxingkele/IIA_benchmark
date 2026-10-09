from scripts.flow_matching.prepare_spectral_original_data import ROOT,read
from scripts.flow_matching.register_spectral_parameter_audit_v2 import parameter_audit
from iia_benchmark.models.fm_spectral_author import build_author_model


def test_released_baseline_keeps_learned_positions_and_reports_paper_count_separately():
    config=read(ROOT/'configs/models/fm_spectral_author__diffusion_ts__stocks__released_author_data.json')
    parameters=config['parameters']
    model=build_author_model(ROOT/parameters['source_root'],parameters['model_target'],parameters['parameters'])
    audit=parameter_audit(model,291318)
    assert audit['all_parameters']==294390 and audit['trainable_parameters']==294390
    assert audit['difference']==3072 and audit['difference_matches_position_parameter_count']
    assert len(audit['learned_position_parameters'])==2
    assert all(p['trainable'] for p in audit['learned_position_parameters'])
    assert not audit['author_model_altered']
