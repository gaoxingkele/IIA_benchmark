"""Inspect all numeric arrays in the original code-referenced EB100 asset."""
import argparse
from pathlib import Path
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching.acquire_TrajectoryNet_EB100_official_v1 import verify_blob,CONFIG


def audit(config):
    import numpy as np
    path=Path(config['output_directory'])/config['filename']
    proof=verify_blob(path,config['expected_bytes'],config['expected_git_blob_sha1'])
    official=read(Path(config['output_directory'])/'verified_acquisition.json')
    if proof['sha256']!=official['sha256']:raise ValueError('Verified complete original public asset required')
    header=[]
    with np.load(path,allow_pickle=False) as data:
        for name in data.files:
            a=data[name]
            if a.dtype.hasobject or not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all():
                raise ValueError('Every original array must load without pickle and contain finite numeric data')
            header.append(dict(name=name,shape=list(a.shape),dtype=str(a.dtype),all_finite=True))
        pcs=data['pcs'];labels=data['sample_labels']
        levels,counts=np.unique(labels,return_counts=True)
        if pcs.shape!=(16819,100) or labels.shape!=(16819,) or levels.tolist()!=[0,1,2,3,4]:
            raise ValueError('Actual complete16819-cell/100-PC/five-timepoint original input required')
        if data['pcs_delta'].shape!=pcs.shape:raise ValueError('Complete original RNA-velocity PC array required')
        days={str(k):int(v) for k,v in zip(levels,counts)}
        record=dict(id='trajectorynet_EB100_code_referenced',path=str(path),sha256=proof['sha256'],bytes=proof['bytes'],
            cells=16819,PC_dimensions=100,native_max_dim=5,native_whiten=True,PCs_already_whitened=False,
            model_timepoint_counts=days,timepoint_names=[str(i) for i in levels],
            required_intermediate_leaveout_indices=[1,2,3],all_arrays_allow_pickle_false=True,
            original_code_referenced_public_asset_verified=True,historical_training_recipe_equivalence_certified=False,
            full_gene_or_barcode_identity_metadata_available=False,
            first5_raw_means=pcs[:,:5].mean(axis=0).tolist(),first5_raw_std_ddof0=pcs[:,:5].std(axis=0).tolist(),
            boundary='Exact original2020 public TrajectoryNet Git blob named by the frozen OT-CFM EBData(version5) code. All16819 cells/100 original PC columns retained, native first5/whiten once. This is a separate source artifact from released18203-cell/50-PC H5. Original code-reference identity is verified; unreported historical OT-CFM training hyperparameters remain uncertified.')
    source=read(ROOT/'projects/flow_matching_research/results/2026-10-16/OTCFM_first_complete_native_model_and_full120_replay_v1/EB_CITE_complete_released_input_audit.json')['records'][0]
    report=dict(status='completed',records=[record],all_original_array_headers=header,full_raw_data_preserved=True,
        original_public_git_blob_proof=proof,original2020_identity=read(ROOT/'experiments/runs/fm_TrajectoryNet_EB100_official_acquisition_v1/initial2020_blob_identity.json'),
        released_H5_comparison=dict(cells=source['cells'],PC_dimensions=source['released_pca_shape'][1],
            timepoint_counts=source['timepoint_counts'],raw_sha256=source['source_sha256'],
            absolute_cell_count_difference=source['cells']-record['cells'],same_row_population_and_PC_artifact=False),
        no_new_PCA_or_whitening_performed=True,model_optimizer_updates=0,new_benchmark_performance_claim=False,
        original_scope=config['original_scope'],all_original_experiments_complete=False)
    exclusive(Path(config['output_directory'])/'native_input_audit.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args()
    import json
    print(json.dumps(audit(read(ROOT/args.config))))


if __name__=='__main__':main()
