"""Audit every published QC cell/count, then prepare a distinct full-raw-RNA track.

No intersection-only gene axis or missing-gene zero filling is permitted. The
published QC cell roster is preserved; actual raw counts supply all58676 RNA
genes. This is independent GEO preprocessing, not a historical Kaggle replica.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v3 as pca_source

MODULE='scripts.flow_matching.prepare_CITE13176_GEO_raw_backed_QC_v1'
CONFIG='configs/data/fm_CITE13176_GEO_raw_backed_QC_pca5.v1.json'
TEST='tests/test_fm_CITE13176_GEO_raw_backed_QC_v1.py'
REPORT='full_raw_backed_QC_preprocessing.json'
POLICY='configs/data/fm_otcfm_multiome13176_public_independent.v1.json'
ACQUISITION='configs/acquisition/fm_CITE13176_GEO_official.v1.json'


def registration():
    old=read(ROOT/POLICY);acq=read(ROOT/ACQUISITION)
    report=Path(acq['output_directory'])/'all8_payload_integrity.json';r=read(report)
    if r['status']!='all8official_payloads_complete_and_fully_read' or r['files']!=8:
        raise ValueError('Actual complete original public raw and processed files required')
    files=[MODULE.replace('.','/')+'.py',TEST,POLICY,ACQUISITION,str(report),
        'scripts/flow_matching/reconstruct_otcfm_multiome13176_public_v3.py',
        pca_source.AUTHOR_ROOT+'/runner/src/datamodules/components/time_dataset.py',
        pca_source.AUTHOR_ROOT+'/runner/src/datamodules/distribution_datamodule.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py']
    return dict(input_records=r['records'],complete_acquisition_report=str(report),complete_acquisition_sha256=sha(report),
        donor='13176',days=[2,3,4,7],expected_QC_cells={'2':6071,'3':7643,'4':8485,'7':7195},
        expected_QC_total_cells=29394,expected_raw_total_cells=35261,expected_RNA_features=58676,
        data_track='independent_GEO13176_published_QC_cells_full_raw_RNA',
        raw_feature_selection='all exact Gene Expression feature IDs;no gene intersection,HVG restriction,or missing-feature zero imputation',
        counts_identity='every published raw.X coordinate including implicit zeros equals exact raw10x count at mapped cell/gene IDs',
        normalization='float64 log1p(1e6*RNA_count/sum_of_all58676_raw_RNA_gene_counts_per_cell)',
        normalization_candidates_audited=['published_gene_library','all_raw_RNA_library','all_raw_feature_library'],
        candidate_audit_tolerance=dict(rtol=2e-6,atol=2e-6),candidate_choice_changes_preparation=False,
        independent_normalization_library='all_raw_RNA_library',pca_dimensions=5,pca_initial_vector_seed=0,pca_arpack_tolerance=0.,
        author_loader_whiten_flag=True,all_timepoints_in_preprocessing_fit=True,
        settings=old['settings'],resource_lock=old['resource_lock'],stable_samples=old['stable_samples'],
        poll_seconds=old['poll_seconds'],numerical_threads=old['numerical_threads'],python=sys.executable,
        packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','h5py','scikit-learn']},
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(files)],
        output_directory='F:/aicoding/IIA_Data/experiments/CITE13176_GEO_raw_backed_QC_pca5_v1',
        state_root='experiments/runs/fm_CITE13176_GEO_raw_backed_QC_pca5_v1',
        source_citation=acq['source_citation'],paper_citation=acq['paper_citation'],original_scope=acq['original_scope'],
        historical_Kaggle_or_paper_processed_identity_certified=False,
        boundary='All29394publisherQCcells and all58676actual rawRNA genes;all35261raw cells remain in untouched source files. '
            'Every published raw-count coordinate is checked including zeros,with exact cell/gene IDs and no gene-axis intersection. '
            'Different per-day published gene filters are audited,not replaced by fabricated zero measurements. '
            'Fixed independent full-raw-RNA CP1M/log1p representation,centeredARPACK PCA5 seed0 and original once-only whitening. '
            'Normalization candidates diagnose the published transform only;no candidate is selected using expression or benchmark agreement. '
            'No historical Kaggle matrix,PCA,weights,full reference identity,or strictTAB certification. '
            'Existing360singlecell cases and all45/681/273/702original obligations remain;no raw/source overwrite.')


def verify(config):
    if config!=registration():raise ValueError('Entire original CITE QC cohort/raw genes and unchanged resource/data protocol required')


def text_values(values):
    return [v.decode('utf-8') if isinstance(v,bytes) else str(v) for v in values]


def unique_map(source,target,kind):
    if len(set(source))!=len(source) or len(set(target))!=len(target):raise ValueError('Ambiguous duplicate '+kind+' identity')
    lookup={key:i for i,key in enumerate(source)}
    if any(key not in lookup for key in target):raise ValueError('Missing actual raw '+kind+' identity')
    return [lookup[key] for key in target]


def csr_block(group,start,end):
    import numpy as np
    from scipy.sparse import csr_matrix
    if group.attrs.get('encoding-type')!='csr_matrix':raise ValueError('Explicit published CSR encoding required')
    shape=tuple(map(int,group.attrs['shape']));ptr=group['indptr'][start:end+1]
    lo,hi=int(ptr[0]),int(ptr[-1])
    return csr_matrix((group['data'][lo:hi],group['indices'][lo:hi],ptr-ptr[0]),shape=(end-start,shape[1]))


def compare_counts(expected,observed):
    import numpy as np
    if expected.shape!=observed.shape or not np.array_equal(expected,observed):
        raise ValueError('Every published raw-count coordinate including implicit zeros must exactly match original10x')
    return int(expected.size)


def normalization_audit(counts,expression,libraries,tolerance):
    import numpy as np
    if counts.shape!=expression.shape or not np.isfinite(expression).all():raise ValueError('Full finite published log expression required')
    rows={}
    for name,totals in libraries.items():
        totals=np.asarray(totals,dtype=np.float64)
        if totals.shape!=(len(counts),) or not np.isfinite(totals).all() or (totals<=0).any():raise ValueError('Complete positive source library counts required')
        expected=np.log1p(np.asarray(counts,dtype=np.float64)*(1e6/totals)[:,None])
        error=np.abs(expected-expression)
        rows[name]=dict(coordinates=int(counts.size),maximum_absolute_difference=float(np.max(error)),
            mismatch_coordinates=int(np.count_nonzero(error>tolerance['atol']+tolerance['rtol']*np.abs(expression))))
    return rows


def normalized_raw_RNA_row(counts,indices,feature_to_RNA):
    import numpy as np
    values=np.asarray(counts,dtype=np.float64);indices=np.asarray(indices)
    if values.shape!=indices.shape or not np.isfinite(values).all() or (values<0).any() or not np.equal(values,np.floor(values)).all():
        raise ValueError('Actual finite nonnegative integer raw10x counts required')
    mapped=feature_to_RNA[indices];keep=(mapped>=0)&(values>0)
    genes=mapped[keep];values=values[keep]
    order=np.argsort(genes,kind='stable');genes=genes[order];values=values[order]
    if len(genes)!=len(np.unique(genes)) or not len(values) or values.sum()<=0:raise ValueError('Unique RNA gene coordinates and positive RNA library required')
    total=float(values.sum(dtype=np.float64))
    return genes.astype(np.int32),np.log1p(values*(1e6/total)),total


def build_raw_backed_input(config,out):
    import h5py
    import numpy as np
    from scipy.sparse import csc_matrix
    records=config['input_records'];cells=[];days=[];source_rows=[];RNA_totals=[];ptr=[0];audits=[];canonical_genes=None
    with (out/'normalized_RNA_data.f64').open('xb') as datafile,(out/'normalized_RNA_indices.i32').open('xb') as indexfile:
        for day in config['days']:
            roles={r['kind']:r for r in records if r['day']==day}
            raw_path=Path(roles['raw_10x_filtered_counts']['path']);processed_path=Path(roles['GEO2025_processed_multimodal']['path'])
            with h5py.File(raw_path,'r') as raw,h5py.File(processed_path,'r') as published:
                m=raw['matrix'];shape=tuple(map(int,m['shape'][:]))
                raw_data=m['data'][:];raw_indices=m['indices'][:];raw_ptr=m['indptr'][:]
                raw_counts=csc_matrix((raw_data,raw_indices,raw_ptr),shape=shape)
                if not raw_counts.has_canonical_format or (raw_data<0).any():raise ValueError('Complete canonical nonnegative raw10x CSC required')
                ids=text_values(m['features']['id'][:]);barcodes=text_values(m['barcodes'][:]);types=text_values(m['features']['feature_type'][:])
                gene_features=np.flatnonzero(np.asarray(types)=='Gene Expression');gene_ids=[ids[i] for i in gene_features]
                if len(gene_ids)!=config['expected_RNA_features'] or len(set(gene_ids))!=len(gene_ids):raise ValueError('All unique rawRNA gene IDs required')
                if canonical_genes is None:canonical_genes=gene_ids
                elif gene_ids!=canonical_genes:raise ValueError('All four original raw RNA gene axes must agree exactly')
                rna=published['mod/rna'];qc_ids=text_values(rna['obs']['_index'][:]);published_ids=text_values(rna['var']['gene_ids'][:])
                if len(qc_ids)!=config['expected_QC_cells'][str(day)]:raise ValueError('Every published QC cell is required')
                qc_positions=np.asarray(unique_map(barcodes,qc_ids,'cell'),dtype=np.int64)
                positions=np.asarray(unique_map(ids,published_ids,'gene'),dtype=np.int32)
                if not np.isin(positions,gene_features).all():raise ValueError('Published RNA genes cannot map to protein features')
                if text_values(rna['raw']['var']['gene_ids'][:])!=published_ids:raise ValueError('Published counts/expression gene axes must align exactly')
                expected_shape=(len(qc_ids),len(published_ids))
                if tuple(map(int,rna['raw']['X'].attrs['shape']))!=expected_shape or tuple(map(int,rna['X'].attrs['shape']))!=expected_shape:
                    raise ValueError('Complete published expression/count geometry required')
                feature_to_RNA=np.full(shape[0],-1,dtype=np.int32);feature_to_RNA[gene_features]=np.arange(len(gene_ids),dtype=np.int32)
                all_RNA=[];all_features=[]
                for barcode,source_index in zip(qc_ids,qc_positions):
                    start,end=map(int,raw_ptr[source_index:source_index+2]);v=raw_data[start:end];ix=raw_indices[start:end]
                    g,x,total=normalized_raw_RNA_row(v,ix,feature_to_RNA)
                    datafile.write(x.astype('<f8',copy=False).tobytes());indexfile.write(g.astype('<i4',copy=False).tobytes());ptr.append(ptr[-1]+len(g))
                    cells.append(f'day{day}:{barcode}');days.append(day);source_rows.append(int(source_index));RNA_totals.append(total)
                    all_RNA.append(total);all_features.append(float(np.sum(v,dtype=np.float64)))
                checked=0;candidate={name:dict(coordinates=0,maximum_absolute_difference=0.,mismatch_coordinates=0) for name in config['normalization_candidates_audited']}
                for start in range(0,len(qc_ids),128):
                    end=min(start+128,len(qc_ids));selected=raw_counts.T[qc_positions[start:end]][:,positions].toarray()
                    published_counts=csr_block(rna['raw']['X'],start,end).toarray()
                    checked+=compare_counts(selected,published_counts)
                    expression=csr_block(rna['X'],start,end).toarray()
                    libraries=dict(published_gene_library=selected.sum(axis=1,dtype=np.float64),
                        all_raw_RNA_library=all_RNA[start:end],all_raw_feature_library=all_features[start:end])
                    observation=normalization_audit(selected,expression,libraries,config['candidate_audit_tolerance'])
                    for name,row in observation.items():
                        candidate[name]['coordinates']+=row['coordinates'];candidate[name]['mismatch_coordinates']+=row['mismatch_coordinates']
                        candidate[name]['maximum_absolute_difference']=max(candidate[name]['maximum_absolute_difference'],row['maximum_absolute_difference'])
                audits.append(dict(day=day,raw_cells=shape[1],published_QC_cells=len(qc_ids),published_genes=len(published_ids),
                    complete_raw_RNA_features=len(gene_ids),all_counts_equal=True,full_count_coordinates_including_zeros=checked,
                    all_published_cell_and_gene_ids_uniquely_mapped=True,published_normalization_candidates=candidate,
                    preparation_normalizer_not_selected_by_candidate_agreement=True))
            del raw_counts,raw_data,raw_indices,raw_ptr,selected,published_counts,expression;gc.collect()
            write(out/'progress.json',dict(phase='full_raw_processed_count_audit',completed_days=day,QC_cells=len(cells)))
    if len(cells)!=config['expected_QC_total_cells'] or len(set(cells))!=len(cells) or sum(a['raw_cells'] for a in audits)!=config['expected_raw_total_cells']:
        raise ValueError('Every complete QC cell and raw source cell count must remain accounted for')
    np.save(out/'normalized_RNA_indptr.npy',np.asarray(ptr,dtype=np.int32),allow_pickle=False)
    np.savez_compressed(out/'raw_source_cell_and_gene_identity.npz',cell_ids=np.asarray(cells),sample_labels=np.asarray(days,dtype=np.int64),
        source_rows=np.asarray(source_rows,dtype=np.int64),gene_ids=np.asarray(canonical_genes),full_RNA_UMI_totals=np.asarray(RNA_totals,dtype=np.float64))
    exclusive(out/'complete_raw_processed_alignment.json',dict(passed=True,records=audits,
        QC_cells=len(cells),raw_source_cells=sum(a['raw_cells'] for a in audits),RNA_features=len(canonical_genes),
        no_gene_intersection_or_fabricated_missing_gene_zeros=True,normalization=config['normalization'],
        historical_Kaggle_or_paper_processed_identity_certified=False))
    return np.asarray(days,dtype=np.int64),np.asarray(cells),np.asarray(source_rows,dtype=np.int64)


def worker(config):
    import numpy as np
    from scipy.sparse import csr_matrix
    if {name:importlib.metadata.version(name) for name in config['packages']}!=config['packages']:
        raise ValueError('Frozen preprocessing environment required')
    for item in config['input_records']:
        p=Path(item['path'])
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Every entire original raw/processed input must remain immutable')
    out=Path(config['output_directory']);out.mkdir(parents=True,exist_ok=False)
    labels,cell_ids,source_rows=build_raw_backed_input(config,out)
    values=np.memmap(out/'normalized_RNA_data.f64',dtype='<f8',mode='r');indices=np.memmap(out/'normalized_RNA_indices.i32',dtype='<i4',mode='r')
    ptr=np.load(out/'normalized_RNA_indptr.npy',allow_pickle=False)
    matrix=csr_matrix((values,indices,ptr),shape=(len(labels),config['expected_RNA_features']),copy=False)
    if not np.shares_memory(matrix.data,values) or not np.shares_memory(matrix.indices,indices) or not matrix.has_canonical_format:
        raise ValueError('Complete memmapped sparse input without full dense/copy materialization required')
    write(out/'progress.json',dict(phase='full_centered_PCA5',cells=matrix.shape[0],genes=matrix.shape[1],nonzeros=matrix.nnz))
    pcs,pca=pca_source.centered_pca(matrix,5,config['pca_initial_vector_seed'],config['pca_arpack_tolerance'])
    whitened,scaler=pca_source.author_whitening(pcs)
    independent=(pcs.astype(np.float64)-pcs.mean(axis=0,dtype=np.float64))/pcs.std(axis=0,dtype=np.float64)
    if not np.allclose(whitened,independent,rtol=2e-6,atol=2e-6):raise ValueError('Every original whitening coordinate must independently agree')
    path=out/'CITE13176_GEO_QC_full_RNA_author_loader.npz'
    np.savez_compressed(path,pcs=pcs,sample_labels=labels,cell_ids=cell_ids,source_rows=source_rows)
    np.savez_compressed(out/'PCA_and_original_whitening_parameters.npz',
        **{k:v for k,v in pca.items() if hasattr(v,'shape')},whitening_mean=scaler.mean_,whitening_scale=scaler.scale_)
    if not np.isfinite(pcs).all() or pcs.shape!=(29394,5):raise ValueError('Every original QC cell and all five native PCs required')
    tree=ast.parse((ROOT/pca_source.AUTHOR_ROOT/'runner/src/datamodules/components/time_dataset.py').read_text(encoding='utf-8'))
    fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tnet_dataset']
    if len(fn)!=1:raise ValueError('One exact original author NPZ loader required')
    namespace=dict(np=np);exec(compile(ast.fix_missing_locations(ast.Module(body=fn,type_ignores=[])),'<unchanged-author-CITE-loader>','exec'),namespace)
    loaded,loaded_labels,levels=namespace['tnet_dataset'](str(path))
    if not np.array_equal(loaded,pcs) or not np.array_equal(loaded_labels,labels) or levels.tolist()!=config['days']:
        raise ValueError('Actual unchanged author loader must retain every cell/PC/timepoint')
    artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file() and p.name!='progress.json']
    exclusive(out/REPORT,dict(passed=True,data_track=config['data_track'],configuration_sha256=sha(ROOT/CONFIG),
        cells=29394,raw_RNA_features=58676,PCA_dimensions=5,days=config['expected_QC_cells'],
        target_input_for_author_loader=str(path),target_input_sha256=sha(path),native_author_whiten_flag_must_remain_true=True,
        complete_raw_processed_count_alignment_sha256=sha(out/'complete_raw_processed_alignment.json'),
        full_PCA_projection_relative_residual=pca['relative_projection_residual'],full_PCA_orthogonality_error=pca['orthogonality_error'],
        original_full_whitening_block_and_all_coordinates_verified=True,original_author_NPZ_loader_full_identity_verified=True,
        all_timepoints_in_preprocessing_fit=True,
        historical_Kaggle_or_paper_processed_identity_certified=False,training_or_model_inference_executed=False,
        original_inputs_overwritten=False,artifacts=artifacts,original_scope=config['original_scope'],all_original_experiments_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if args.worker:worker(c);return
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda value:None)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=resource_api();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Original complete CITE audit/preprocessing owner already live')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve every previous audit/preprocessing attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,all_original_experiments_complete=False)
    def update(v):state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        stable=0
        while stable<c['stable_samples']:
            memory=api['resource_snapshot']();stable=stable+1 if api['enough'](memory,c['settings']) else 0
            update(dict(state='waiting_original_full_preprocessing_admission',stable_samples=stable,**memory));time.sleep(c['poll_seconds'])
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
            command=[c['python'],'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                update(dict(state='running_full_CITE_raw_alignment_and_PCA',active_pid=child.pid,exact_child_handle=handle))
                code,usage=capped_wait(child,c,api,update)
            p=Path(c['output_directory'])/REPORT
            passed=code==0 and not usage['resource_guard_aborted'] and p.exists() and read(p)['passed']
            exclusive(base/'guard_receipt.json',dict(passed=passed,exit_code=code,exact_child_handle=handle,**usage,report_sha256=sha(p) if p.exists() else None))
            update(dict(state='full_CITE_preprocessing_passed' if passed else 'failed_preserved_no_retry',active_pid=None))
    finally:owner.close()


if __name__=='__main__':main()
