"""Verify acquired source facts and remaining target image reproduction gaps."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
from pathlib import Path

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha,verify_receipts
from scripts.flow_matching.run_flow_mismatching_full_image_evaluation_v1 import write_new


def region_receipt(path,text):
    source=(ROOT/path).read_text(encoding='utf-8')
    start=source.find(text)
    if start<0:raise ValueError('Required complete original source evidence missing: '+path)
    return dict(path=path,sha256=sha(ROOT/path),line=source[:start].count('\n')+1,evidence=text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report_path='projects/flow_matching_research/results/2026-10-11/flow_mismatching_reflect_wt_sources_v1/full_original_reflect_wt_material_audit.json'
    report=read(ROOT/report_path);cfg=read(ROOT/report['configuration']);verify_receipts(cfg['source_receipts'])
    if not report['passed'] or report['configuration_sha256']!=sha(ROOT/report['configuration']):
        raise ValueError('Exact complete original material audit required')
    for receipt in report['downloads']+report['original_papers']:
        if sha(receipt['path'])!=receipt['sha256']:raise ValueError('Original archive/PDF bytes changed')
    count=0
    for repository in report['repositories']:
        paths=set()
        for file in repository['files']:
            path=Path(repository['original_code_root'])/file['path'];payload=path.read_bytes()
            if (len(payload)!=file['bytes'] or sha(path)!=file['sha256']
                    or hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()!=file['verified_git_blob']):
                raise ValueError('Every original source Git blob must remain identical')
            if file['path'] in paths:raise ValueError('Duplicate source file receipt')
            paths.add(file['path']);count+=1
        if len(paths)!=repository['all_original_git_files_verified']:
            raise ValueError('Complete source file count differs')
    if count!=75 or len(report['original_papers'])!=2:raise ValueError('Complete acquired source and paper scope required')
    base='data/public_datasets/flow_matching/flow_mismatching_reflect_wt_sources_v1'
    loader=base+'/reflect_author_original/MedicalDataLoader.py'
    tree=ast.parse((ROOT/loader).read_text(encoding='utf-8'))
    functions=[]
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in ('BraTS2021Dataset','ATLASDataset'):
            method=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='generate_random_walk_contour')
            functions.append(dict(dataset=node.name,start_line=method.lineno,end_line=method.end_lineno,
                canonical_AST_sha256=hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest()))
    if len(functions)!=2 or functions[0]['canonical_AST_sha256']!=functions[1]['canonical_AST_sha256']:
        raise ValueError('Both native mask definitions must be inspected and equivalent')
    train=base+'/fmad_author_original/train.py';infer=base+'/fmad_author_original/infer.py'
    findings=[
        region_receipt(loader,'max_objects=4'),
        region_receipt(loader,'steps_adj = (self.embedding_size/32)**2'),
        region_receipt(loader,'weights = [10 - 9/steps * x for x in np.arange(steps)]'),
        region_receipt(loader,'cv2.fillPoly(mask, [contour], 255)'),
        region_receipt(loader,'final_mask = np.where(final_mask == 0, obj_mask * i, final_mask)'),
        region_receipt(loader,'if random.random()<0.9:'),
        region_receipt(train,'LNWT = nn.LayerNorm(x_0[0].shape).to(c.device)'),
        region_receipt(train,'test_loader = torch.utils.data.DataLoader(test_dataset'),
        region_receipt(train,'if epoch == 0 or det_auroc >= det_auroc_best:'),
        region_receipt(train,'if epoch == 0 or loc_auroc >= loc_auroc_best:'),
        region_receipt(infer,'dt = 1.0 / step'),
        region_receipt(infer,'t = j * dt'),
        region_receipt(infer,'logp = -0.5*torch.mean(x_t**2, 1)'),
        region_receipt(base+'/fmad_author_original/train_config.py','LN_enable = True'),
        region_receipt(base+'/fmad_author_original/requirements.txt','opencv-python==4.11.0.86')]
    gaps=[
        'Original REFLECT is medical VAE latent/brain ROI/DTD replacement; target uses image RGB Gaussian corruption and same common U-Net.',
        'Native mask count/weighted length/connectivity/fill/overlap/90percent random rules are now sourced; target full-image ROI/geometry/draw sharing still uncertified.',
        'Related WT uses per-sample fresh identity-affine LayerNorm; default epsilon1e-5/population variance verified against installed Torch, not target-author implementation.',
        'Related native WT uses left-grid Euler; target WT image integrator/steps remain unreported and this does not identify ReconFlow solver.',
        'Original feature-space100epoch/WRN50/AdamW+schedule differs from target image-domain2000epoch/full U-Net/StableAdamW controlled comparison.',
        'Related WT author training code selects best-det/best-loc using per-epoch test AUROC. Original-author reproduction and strict TAB need distinct checkpoint protocols.',
        'Source-selected best test checkpoints do not by themselves prove any specific published number used that selection.',
        'Full original model training/checkpoints/data-specific GPU capacity and performance remain unexecuted for these two image adapters.',
        'Related medical feature datasets, VAE/DTD weights and author exact dependency environment are not made complete by downloading these component sources.']
    output=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
        original_material_audit=dict(path=report_path,sha256=sha(ROOT/report_path)),
        all75_original_source_files_rehashed_and_Git_blobs_verified=True,all_original_papers_unchanged=True,
        native_mask_methods=functions,complete_original_source_evidence=findings,
        component_model_configuration='configs/models/fm_reflect_wt_native_components.v1.json',
        related_native_test_AUROC_checkpoint_selection_verified=True,
        related_native_test_checkpoint_selection_is_strict_TAB=False,
        source_facts_do_not_certify_target_original_adapter=True,
        remaining_original_reproduction_gaps=gaps,original_scope=report['original_scope'],
        benchmark_performance=False,new_measured_seed_slots=0,all_original_experiments_complete=False)
    write_new(ROOT/args.output,output)
    print(dict(all_original_files=count,native_mask_definitions=2,source_evidence=len(findings),native_model_performance=False))


if __name__=='__main__':main()
