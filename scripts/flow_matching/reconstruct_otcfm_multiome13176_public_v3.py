"""Full author/public matrix reconciliation, then paper-described donor PCA5.

No training or benchmark score is produced. Original files stay read-only.
"""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE='scripts.flow_matching.reconstruct_otcfm_multiome13176_public_v3'
CONFIG='configs/reproducibility/fm_otcfm_multiome13176_public_reconstruction.v3.json'
TEST='tests/test_fm_otcfm_multiome13176_reconstruction_v3.py'
AUTHOR_CONFIG='configs/acquisition/fm_otcfm_author_singlecell_mendeley.v1.json'
AUTHOR_ROOT='experiments/runs/flow_matching_campaign/sources/ot_cfm/original'


def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def registration():
    assets=read(ROOT/AUTHOR_CONFIG)
    author=next(s for s in assets['sources'] if s['filename']=='op_train_multi_targets_0.h5ad')
    catalog_path='projects/flow_matching_research/materials/catalog.json'
    public=next(s for s in read(ROOT/catalog_path)['artifacts'] if s['id']=='neurips2022_public_multiome_dataset_mod1_h5ad')
    paths=[MODULE.replace('.','/')+'.py',TEST,AUTHOR_CONFIG,catalog_path,
        'scripts/flow_matching/reconstruct_otcfm_multiome13176_public_v2.py',
        'configs/reproducibility/fm_otcfm_multiome13176_public_reconstruction.v2.json',
        'experiments/runs/fm_otcfm_multiome13176_public_reconstruction_v2/guard_receipt.json',
        'experiments/runs/fm_otcfm_multiome13176_public_reconstruction_v2/worker.stderr.log',
        'scripts/flow_matching/reconstruct_otcfm_multiome13176_public_v1.py',
        'configs/reproducibility/fm_otcfm_multiome13176_public_reconstruction.v1.json',
        'experiments/runs/fm_otcfm_multiome13176_public_reconstruction_v1/guard_receipt.json',
        'experiments/runs/fm_otcfm_multiome13176_public_reconstruction_v1/worker.stderr.log',
        AUTHOR_ROOT+'/runner/src/datamodules/components/time_dataset.py',
        AUTHOR_ROOT+'/runner/src/datamodules/distribution_datamodule.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py']
    return dict(inputs=dict(author=dict(path=author['path'],bytes=author['expected_bytes'],sha256=author['checksum'].split(':')[1]),
            public=dict(path=public['path'],bytes=public['bytes'],sha256=public['sha256'])),
        reference_donor='32606',paper_donor='13176',days=[2,3,4,7],
        expected_reference_cells=33701,expected_public_reference_candidates=33704,expected_paper_cells=35384,expected_features=23418,
        public_layer='normalized',encoding_transform='log1p(expm1(public_float64)*100)',encoding_scale=100.,
        encoding_tolerance=dict(rtol=2e-6,atol=2e-6),
        reference_match='Unique positive-gene support and fixed transformed expression-value anchors; full33701-to33704 injection and every author matrix coordinate verified. No positive-support-only identity claim.',
        pca_dimensions=5,pca_initial_vector_seed=0,pca_arpack_tolerance=0.,
        pca_fit_scope='All donor13176 cells at days2,3,4,7, before leave-timepoint training as paper E5 describes.',
        whitening='Original CustomTrajectoryDataModule StandardScaler block, ddof0, on all selected donor PC coordinates.',
        numerical_threads=2,
        settings=dict(minimum_available_memory_bytes=16*2**30,minimum_commit_headroom_bytes=32*2**30,
            emergency_commit_headroom_bytes=16*2**30,maximum_worker_rss_bytes=3*2**30,maximum_worker_private_bytes=6*2**30),
        resource_lock='experiments/runs/fm_otcfm_singlecell_input_audit_cpu.lock',stable_samples=6,poll_seconds=5,
        state_root='experiments/runs/fm_otcfm_multiome13176_public_reconstruction_v3',
        output_directory='data/public_datasets/flow_matching/processed_otcfm_multiome13176_public_pca5_v3',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
        citation='https://arxiv.org/abs/2302.00482',paper_pdf_page=32,paper_section='E.5',
        author_data='https://doi.org/10.17632/hhny5ff7yj.1',
        public_data='https://openproblems-data.s3.amazonaws.com/resources/datasets/openproblems_neurips2022/pbmc_multiome/log_cp10k/dataset_mod1.h5ad',
        boundary='Complete original cells/features, no subsampling or replacement. Full donor32606 matrix identity gate before '
            'applying fixed encoding to donor13176. Public reference candidates contain3 more day2 cells than the released author subset; their IDs and metadata are recorded, omission cause unknown. This discrepancy is not a zero-cell/QC claim. Full paper13176 cohort remains35384.  First5 centered sparse PCs and original whitening, no organizer embedding reuse. '
            'Reconstructed paper-described input, not a claim of historical artifact bit identity; original PCA solver/seed history '
            'unavailable. CITE feature mapping and all original methods/baselines/ablations/full budgets remain pending.',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        all_original_experiments_complete=False)


def verify(config):
    if config!=registration():raise ValueError('All original data coordinates, encoding gate, PCA and resources must stay frozen')


def column(group,key):
    import h5py
    node=group[key]
    if isinstance(node,h5py.Group):
        category=node['categories'];codes=node['codes'][:]
        values=category.asstr()[:] if category.dtype.kind in 'SO' else category[:]
        if codes.ndim!=1 or (codes<0).any() or (codes>=len(values)).any():raise ValueError('Complete actual categorical identities required')
        return values[codes]
    return node.asstr()[:] if node.dtype.kind in 'SO' else node[:]


def transformed(values,scale):
    import numpy as np
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or not np.isfinite(values).all() or (values<0).any():raise ValueError('Finite nonnegative full expression coordinates required')
    if not isinstance(scale,(int,float)) or not np.isfinite(scale) or scale<=0:raise ValueError('Positive finite frozen expression scale required')
    with np.errstate(over='raise',invalid='raise'):
        result=np.log1p(np.expm1(values)*scale)
    if not np.isfinite(result).all():raise ValueError('Finite transformed coordinates required')
    return result


def unique_anchor_candidate(columns,frequencies,selected,positive_rows):
    import numpy as np
    candidates=set(selected)
    for feature in columns[np.argsort(frequencies[columns],kind='stable')]:
        candidates.intersection_update(positive_rows(int(feature)))
        if len(candidates)<=1:break
    if len(candidates)!=1:raise ValueError('Full author cell must have a unique actual positive-gene support match')
    return next(iter(candidates))


def check_column_identity(author_indices,author_values,public_indices,public_values,tolerance):
    import numpy as np
    if (not np.array_equal(author_indices,public_indices)
            or not np.isfinite(author_values).all() or not np.isfinite(public_values).all()
            or not np.allclose(author_values,public_values,**tolerance)):
        raise ValueError('Full original reference gene coordinates or expression values differ')
    return float(np.max(np.abs(author_values-public_values))) if len(author_values) else 0.


def centered_pca(matrix,dimensions,seed,tolerance):
    import numpy as np
    from scipy.sparse.linalg import LinearOperator,svds
    from sklearn.utils.extmath import svd_flip
    if dimensions<=0 or dimensions>=min(matrix.shape):raise ValueError('All cells/features and a valid component count required')
    n,p=matrix.shape;mean=np.asarray(matrix.sum(axis=0,dtype=np.float64)).ravel()/n
    def mv(v):return np.asarray(matrix@np.asarray(v).reshape(-1)).ravel()-mean@np.asarray(v).reshape(-1)
    def rmv(v):return np.asarray(matrix.T@np.asarray(v).reshape(-1)).ravel()-mean*np.sum(v)
    def mm(v):return matrix@v-(mean@v)[None,:]
    def rmm(v):return matrix.T@v-mean[:,None]*np.sum(v,axis=0)[None,:]
    operator=LinearOperator((n,p),matvec=mv,rmatvec=rmv,matmat=mm,rmatmat=rmm,dtype=np.float64)
    initial=np.random.RandomState(seed).normal(size=min(matrix.shape))
    u,s,vt=svds(operator,k=dimensions,which='LM',tol=tolerance,v0=initial)
    order=np.argsort(s)[::-1];u,s,vt=u[:,order],s[order],vt[order,:]
    u,vt=svd_flip(u,vt)
    coordinates=u*s
    residual=np.linalg.norm(operator@vt.T-coordinates)/max(np.linalg.norm(coordinates),1.)
    orthogonality=float(np.max(np.abs(vt@vt.T-np.eye(dimensions))))
    if residual>1e-6 or orthogonality>1e-8:raise ValueError('Full centered PCA projection and orthogonality validation failed')
    return coordinates.astype(np.float32),dict(mean=mean,components=vt,singular_values=s,
        explained_variance=s*s/(n-1),relative_projection_residual=float(residual),orthogonality_error=orthogonality)


def author_whitening(coordinates):
    import ast
    from types import SimpleNamespace
    from sklearn.preprocessing import StandardScaler
    tree=ast.parse((ROOT/AUTHOR_ROOT/'runner/src/datamodules/distribution_datamodule.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CustomTrajectoryDataModule')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    block=next(n for n in init.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='whiten')
    expected=ast.parse('self.scaler=StandardScaler();self.scaler.fit(self.data);self.data=self.scaler.transform(self.data)').body
    if ast.dump(ast.Module(body=block.body,type_ignores=[]))!=ast.dump(ast.Module(body=expected,type_ignores=[])):
        raise ValueError('Exact original complete whitening block required')
    holder=SimpleNamespace(data=coordinates.copy())
    exec(compile(ast.fix_missing_locations(ast.Module(body=block.body,type_ignores=[])),'<exact-author-whitening>','exec'),dict(self=holder,StandardScaler=StandardScaler))
    return holder.data,holder.scaler



def validate_cohort_roster(config, author_donors, author_days, reference_candidates, target):
    import numpy as np
    if (len(reference_candidates) != config['expected_public_reference_candidates']
            or len(target) != config['expected_paper_cells']
            or len(author_donors) != config['expected_reference_cells']
            or len(author_days) != config['expected_reference_cells']
            or set(author_donors) != {config['reference_donor']}
            or set(author_days) != set(config['days'])
            or len(np.unique(reference_candidates)) != len(reference_candidates)
            or len(np.unique(target)) != len(target)
            or np.intersect1d(reference_candidates, target).size):
        raise ValueError('All author reference cells, public candidates and complete paper cohort required')


def validate_reference_injection(mapping, candidates, author_days, public_days, author_cells):
    import numpy as np
    if (len(mapping) != author_cells or len(np.unique(mapping)) != author_cells
            or not np.isin(mapping, candidates).all()
            or not np.array_equal(author_days, public_days[mapping])):
        raise ValueError('Every actual author cell needs a unique value-verifiable public identity and day')
    extra = np.setdiff1d(candidates, mapping)
    if len(extra) != len(candidates) - author_cells:
        raise ValueError('All public-only reference candidates must be explicitly accounted for')
    return extra



def unique_expression_candidate(columns, values, frequencies, selected, public_column, tolerance, diagnostic=lambda record:None):
    import numpy as np
    if len(columns) != len(values) or not len(columns):
        raise ValueError('Complete positive author expression anchors required')
    candidates = set(selected)
    for index in np.argsort(frequencies[columns], kind='stable'):
        feature, expected = int(columns[index]), values[index]
        rows, actual = public_column(feature)
        eligible = rows[np.isclose(actual, expected, **tolerance)]
        candidates.intersection_update(eligible.tolist())
        if len(candidates) <= 1:
            break
    if len(candidates) != 1:
        diagnostic(dict(expression_candidates=len(candidates), author_nonzeros=len(columns),
            author_feature=feature, author_expression=float(expected),
            public_gene_rows=len(rows), closest_public_expression_absolute_error=
                float(np.min(np.abs(actual - expected))) if len(actual) else None,
            candidate_public_rows=sorted(candidates)))
        raise ValueError('Complete fixed expression anchors did not identify a unique author cell')
    return next(iter(candidates))


def worker(config):
    import gc
    import h5py
    import numpy as np
    from scipy import sparse
    from collections import Counter
    import sklearn,scipy
    verify(config)
    for entry in config['inputs'].values():
        path=ROOT/entry['path']
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('Every original input byte must match the frozen payload')
    out=ROOT/config['output_directory'];out.mkdir(parents=True,exist_ok=False)
    progress=lambda values:write(out/'progress.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),**values))
    progress(dict(phase='loading_full_author_sparse_matrix'))
    with h5py.File(ROOT/config['inputs']['author']['path'],'r') as af,h5py.File(ROOT/config['inputs']['public']['path'],'r') as pf:
        features=column(af['var'],af['var'].attrs['_index']);public_features=column(pf['var'],pf['var'].attrs['_index'])
        if not np.array_equal(features,public_features) or len(features)!=config['expected_features']:raise ValueError('All actual original gene identifiers and order must match')
        adonor=column(af['obs'],'donor').astype(str);aday=column(af['obs'],'day').astype(int)
        pdonor=column(pf['obs'],'donor').astype(str);pday=column(pf['obs'],'day').astype(int)
        selected=np.flatnonzero((pdonor==config['reference_donor']) & np.isin(pday,config['days']))
        target=np.flatnonzero((pdonor==config['paper_donor']) & np.isin(pday,config['days']))
        validate_cohort_roster(config, adonor, aday, selected, target)
        author_cells = config['expected_reference_cells']
        raw=af['X']
        if raw.attrs['encoding-type']!='csr_matrix':raise ValueError('Exact author CSR encoding required')
        author=sparse.csr_matrix((raw['data'][:],raw['indices'][:],raw['indptr'][:]),shape=tuple(raw.attrs['shape']))
        author.sum_duplicates();author.eliminate_zeros();author.sort_indices()
        layer=pf['layers'][config['public_layer']]
        if layer.attrs['encoding-type']!='csc_matrix':raise ValueError('Actual complete organizer CSC input required')
        pointer=layer['indptr'][:];frequencies=np.diff(pointer);selected_set=set(selected.tolist())
        @lru_cache(maxsize=2048)
        def reference_column(feature):
            lo,hi=int(pointer[feature]),int(pointer[feature+1]);rows=layer['indices'][lo:hi];values=layer['data'][lo:hi]
            keep=np.isin(rows,selected)&(values>0)
            return rows[keep],transformed(values[keep],config['encoding_scale'])
        mapping=np.empty(author_cells,dtype=np.int64)
        for row in range(author_cells):
            lo,hi=author.indptr[row:row+2]
            def diagnostics(record):
                progress(dict(phase='reference_expression_anchor_diagnostic',author_row=row,**record))
            mapping[row]=unique_expression_candidate(author.indices[lo:hi],author.data[lo:hi],frequencies,
                selected_set,reference_column,config['encoding_tolerance'],diagnostics)
            if row%1000==0:progress(dict(phase='matching_every_actual_author_cell',completed_cells=row,total_cells=author_cells))
        public_only = validate_reference_injection(mapping, selected, aday, pday, author_cells)
        public_ids = column(pf['obs'], pf['obs'].attrs['_index'])
        exclusive(out/'public_reference_cohort_difference.json', dict(
            author_reference_cells=author_cells, public_reference_candidates=len(selected),
            full_author_reference_injected=True, public_only_rows=public_only.tolist(),
            public_only_cell_ids=public_ids[public_only].tolist(),
            public_only_days=pday[public_only].tolist(), public_only_cells=len(public_only),
            omission_cause='unknown; no zero-expression or historical QC claim',
            paper_target_cells=len(target), paper_target_subsampling_performed=False))
        reference_column.cache_clear()
        np.save(out/'author_to_public_rows.npy',mapping,allow_pickle=False)
        reference_rows=np.full(len(pdonor),-1,dtype=np.int32);reference_rows[mapping]=np.arange(len(mapping),dtype=np.int32)
        target_rows=np.full(len(pdonor),-1,dtype=np.int32);target_rows[target]=np.arange(len(target),dtype=np.int32)
        progress(dict(phase='converting_full_author_matrix_to_columns',matched_cells=len(mapping)))
        original_nnz=int(author.nnz);author_columns=author.tocsc();del author;gc.collect()
        target_pointer=[0];checked_nnz=0;max_error=0.
        data_path=out/'paper13176_normalized_data.f32';index_path=out/'paper13176_normalized_indices.i32'
        with data_path.open('xb') as data_file,index_path.open('xb') as index_file:
            for feature in range(config['expected_features']):
                lo,hi=int(pointer[feature]),int(pointer[feature+1]);rows=layer['indices'][lo:hi];values=layer['data'][lo:hi]
                if not np.isfinite(values).all() or (values<0).any():raise ValueError('Finite nonnegative original expression values required')
                ref=reference_rows[rows];keep=(ref>=0)&(values>0);idx=ref[keep];order=np.argsort(idx);idx=idx[order]
                reconstructed=transformed(values[keep][order],config['encoding_scale'])
                a,b=author_columns.indptr[feature:feature+2];truth=author_columns.data[a:b]
                max_error=max(max_error,check_column_identity(author_columns.indices[a:b],truth,idx,reconstructed,config['encoding_tolerance']))
                checked_nnz+=len(truth)
                target_idx=target_rows[rows];keep=(target_idx>=0)&(values>0);idx=target_idx[keep];order=np.argsort(idx);idx=idx[order]
                if len(idx)>1 and (np.diff(idx)<=0).any():raise ValueError('Unique canonical target cell/gene coordinates required')
                reconstructed=transformed(values[keep][order],config['encoding_scale']).astype('<f4')
                data_file.write(reconstructed.tobytes());index_file.write(idx.astype('<i4').tobytes());target_pointer.append(target_pointer[-1]+len(idx))
                if feature%500==0:progress(dict(phase='checking_all_reference_matrix_values_and_exporting_paper_cohort',completed_features=feature,total_features=len(features),verified_reference_nonzeros=checked_nnz))
        if checked_nnz!=original_nnz:raise ValueError('Every original explicit and implicit-zero coordinate must match')
        del author_columns;gc.collect()
        np.save(out/'paper13176_normalized_indptr.npy',np.asarray(target_pointer,dtype=np.int64),allow_pickle=False)
        target_days=pday[target].copy();target_ids=column(pf['obs'],pf['obs'].attrs['_index'])[target]
        np.savez_compressed(out/'paper13176_cell_identity.npz',source_rows=target,cell_ids=target_ids,day=target_days,feature_ids=features)
    exclusive(out/'full_reference_matrix_reconciliation.json',dict(passed=True,author_cells=len(mapping),features=config['expected_features'],
        all_matrix_coordinates=len(mapping)*config['expected_features'],all_reference_nonzeros_checked=checked_nnz,
        implicit_zero_coordinates_verified=True,author_to_public_mapping_sha256=sha(out/'author_to_public_rows.npy'),
        fixed_encoding_transform=config['encoding_transform'],maximum_absolute_value_error=max_error,
        tolerance=config['encoding_tolerance'],full_author_reference_injection_and_days_match=True,public_reference_candidates=len(selected),public_only_reference_cells=len(public_only),subset_spot_check_only=False))
    nnz=target_pointer[-1]
    # Keep source-compatible float32 coordinates, and promote them once on disk.
    # Avoid a hidden float64 copy of the complete sparse matrix on each matvec.
    raw_data=np.memmap(data_path,dtype='<f4',mode='r',shape=(nnz,))
    pca_data_path=out/'centered_PCA_operator_data.f64'
    if pca_data_path.exists():raise FileExistsError('Existing PCA operator cache preserved')
    with pca_data_path.open('xb') as stream:
        for start in range(0,nnz,1024*1024):stream.write(raw_data[start:start+1024*1024].astype('<f8').tobytes())
    raw_data._mmap.close();del raw_data
    data=np.memmap(pca_data_path,dtype='<f8',mode='r',shape=(nnz,));indices=np.memmap(index_path,dtype='<i4',mode='r',shape=(nnz,))
    matrix=sparse.csc_matrix((data,indices,np.asarray(target_pointer,dtype=np.int64)),shape=(len(target),len(features)),copy=False)
    progress(dict(phase='computing_full_cohort_centered_PCA5',cells=matrix.shape[0],features=matrix.shape[1],nonzeros=matrix.nnz))
    pcs,pca=centered_pca(matrix,config['pca_dimensions'],config['pca_initial_vector_seed'],config['pca_arpack_tolerance'])
    whitened,scaler=author_whitening(pcs)
    independent=(pcs.astype(np.float64)-pcs.mean(axis=0,dtype=np.float64))/pcs.std(axis=0,dtype=np.float64)
    if not np.allclose(whitened,independent,rtol=2e-6,atol=2e-6):raise ValueError('Independent original ddof0 whitening check failed')
    np.savez_compressed(out/'paper13176_author_loader_input.npz',pcs=pcs,sample_labels=target_days,cell_ids=target_ids,source_rows=target)
    np.savez_compressed(out/'paper13176_whitened_diagnostic.npz',pcs=whitened,sample_labels=target_days)
    np.savez_compressed(out/'paper13176_PCA_and_whitening_parameters.npz',**{k:v for k,v in pca.items() if hasattr(v,'shape')},whitening_mean=scaler.mean_,whitening_scale=scaler.scale_)
    import ast
    tree=ast.parse((ROOT/AUTHOR_ROOT/'runner/src/datamodules/components/time_dataset.py').read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tnet_dataset']
    if len(functions)!=1:raise ValueError('Complete original NPZ author loader required')
    namespace=dict(np=np);exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'<unchanged-author-NPZ-loader>','exec'),namespace)
    actual,labels,levels=namespace['tnet_dataset'](str(out/'paper13176_author_loader_input.npz'))
    if not np.array_equal(actual,pcs) or not np.array_equal(labels,target_days) or levels.tolist()!=config['days']:
        raise ValueError('Original full NPZ loader must retain every point/time/coordinate')
    artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='progress.json']
    exclusive(out/'paper_input_reconstruction_report.json',dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=CONFIG,configuration_sha256=sha(ROOT/CONFIG),full_reference_matrix_reconciliation_passed=True,public_reference_candidates=len(selected),public_only_reference_cells=len(public_only),
        cells=len(target),features=len(features),PCA_dimensions=pcs.shape[1],donor=config['paper_donor'],
        days=dict(Counter(str(x) for x in target_days)),first_five_centered_PCs_freshly_computed=True,
        original_author_loader_exact=True,original_author_whitening_block_exact=True,
        pca_projection_relative_residual=pca['relative_projection_residual'],pca_orthogonality_error=pca['orthogonality_error'],
        target_input_for_author_loader=str(out/'paper13176_author_loader_input.npz'),author_whiten_flag_must_remain_true=True,
        historical_PCA_artifact_bit_identity_certified=False,source_cohort_replaced_with_wrong_donor=False,
        PCA_and_whitening_fit_includes_all_selected_timepoints=True,strict_inductive_TSAD_result=False,
        original_inputs_overwritten=False,training_or_model_checkpoint_inference_executed=False,
        CITE_feature_mapping_complete=False,all_original_experiments_complete=False,artifacts=artifacts,
        environment=dict(python=sys.version,numpy=np.__version__,h5py=h5py.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        boundary=config['boundary']))
    print(json.dumps(dict(passed=True,full_reference_cells=len(mapping),paper_cells=len(target),PC_dimensions=5)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:exclusive(ROOT/CONFIG,registration());print(json.dumps(dict(registered=True,full_reference_cells=33701,paper_cohort_cells=35384)));return
    config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,full_paper_cells=config['expected_paper_cells'],features=config['expected_features'],original_scope=config['original_scope'])));return
    if args.worker:worker(config);return
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda value:None)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing full input reconstruction controller preserved')
    if (ROOT/config['output_directory']).exists():owner.close();raise FileExistsError('Existing full/partial reconstruction preserved')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,all_original_experiments_complete=False)
    def update(value):state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        stable=0
        while stable<config['stable_samples']:
            snapshot=api['resource_snapshot']();stable=stable+1 if api['has_headroom'](snapshot,config['settings']) else 0
            update(dict(state='waiting_original_full_memory_admission',stable_samples=stable,**snapshot));time.sleep(config['poll_seconds'])
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=psutil.Process(child.pid).cmdline())
                update(dict(state='running_full_original_matrix_and_paper_input_reconstruction',active_pid=child.pid,exact_child_handle=handle))
                code,usage=capped_wait(child,config,api,update)
        report=ROOT/config['output_directory']/'paper_input_reconstruction_report.json'
        passed=code==0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
        exclusive(base/'guard_receipt.json',dict(exit_code=code,passed=passed,exact_child_handle=handle,**usage,report_sha256=sha(report) if report.exists() else None))
        update(dict(state='completed' if passed else 'failed_attempt_preserved',active_pid=None,full_original_input_reconstruction_passed=passed))
    finally:owner.close()


if __name__=='__main__':main()
