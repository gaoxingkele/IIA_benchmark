"""Register all five missing SWaT slots as a separately named loader restoration."""
import json
from pathlib import Path

from scripts.flow_matching import run_anomaly_transformer_author_full_v1 as original
from scripts.flow_matching import run_anomaly_transformer_swat_restored_v1 as restored

ROOT, read, sha = original.ROOT, original.read, original.sha
CONFIG = 'configs/experiments/fm_anomaly_transformer_swat_restored.v1.json'
PARENT = 'configs/experiments/fm_anomaly_transformer_author_full.v1.json'


def main():
    target=ROOT/CONFIG
    if target.exists():raise FileExistsError('Keep frozen registrations')
    parent=read(ROOT/PARENT);original.verify(parent)
    config=dict(parent)
    config['jobs']=[dict(id=f'anomaly_transformer_author__restored_swat_loader__swat__seed{s}',
        dataset='SWaT',seed=s,epochs=3,anomaly_ratio=0.1,dataset_config='configs/datasets/mtsad_swat.json',
        status='registered_full_author_runtime') for s in parent['seeds']]
    config.update(parent_original_matrix=PARENT,parent_original_matrix_sha256=sha(ROOT/PARENT),
        loader_source=restored.LOADER_SOURCE,
        loader_source_provenance='MaelNet registered author snapshot a5f5bb5a23b90773fd15a5059a6f0c0ef2853c56',
        loader_restoration_author_equivalence_certified=False,
        training_rows_policy='all_495000_payload_rows_no_trim',
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_swat_restored_v1',
        state_root='experiments/runs/fm_anomaly_transformer_swat_restored_v1',
        outstanding_author_fidelity_gaps=parent['outstanding_author_fidelity_gaps']+[
            'This separately named loader restoration does not establish the unavailable original SWaT loader',
            '495000 payload rows versus475200 paper rows remains unresolved; no guessed trim performed'],
        boundary='SWaT loader-restoration track: exact MaelNet CSV loader class around the pinned full-width '
            'Anomaly Transformer model/solver. Full495000train/449919test rows, stride-one training, '
            'original3epochs/batch256/512width/minimax/early-stopping/pooled-calibration/PA preserved. '
            'Local seed repetitions1103..1107. This is an explicit recovered implementation, not the '
            'missing original SWaT loader, paper-era data-equivalence proof, or strict TAB certificate. '
            'No other original dataset jobs are restarted or duplicated; full45/681/273scope remains.')
    tracked=[ROOT/PARENT,ROOT/restored.LOADER_SOURCE,Path(__file__),
             ROOT/'scripts/flow_matching/run_anomaly_transformer_swat_restored_v1.py',
             ROOT/'tests/test_fm_anomaly_transformer_swat_restored_v1.py',
             ROOT/'configs/models/fm_anomaly_transformer_swat_restored.v1.json',
             ROOT/'configs/datasets/mtsad_swat.json']
    dataset=read(ROOT/'configs/datasets/mtsad_swat.json')
    tracked += [ROOT/parent['data_root']/dataset['directory']/name for name in dataset['files'].values()]
    config['source_receipts']=parent['source_receipts']+[
        dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in tracked]
    restored.verify(config)
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(config,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(config=CONFIG,selected_restored_loader_slots=5,
        parent_original_slots=25,full_width=512,original_loader_equivalence_certified=False)))


if __name__=='__main__':main()
