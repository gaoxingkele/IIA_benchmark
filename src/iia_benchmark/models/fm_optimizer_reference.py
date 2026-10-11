"""Immutable related-author optimizer; target-paper implementation remains unknown."""
import hashlib
import importlib.util
from pathlib import Path


def load_verified_dinomaly_stableadamw(source_path, expected_sha256, *, asset_root=None):
    path=Path(source_path)
    if not path.is_absolute():
        if asset_root is None:raise ValueError('An explicit registered asset root is required')
        path=Path(asset_root)/path
    path=path.resolve()
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected_sha256:raise ValueError('Original StableAdamW source SHA256 differs')
    spec=importlib.util.spec_from_file_location('iia_immutable_stableadamw_'+actual[:16],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    optimizer=module.StableAdamW
    optimizer.__iia_original_source_sha256__=actual
    optimizer.__iia_original_source_path__=str(path)
    optimizer.__iia_target_author_equivalence_certified__=False
    return optimizer


def build_verified_dinomaly_stableadamw(params, *, source_path, expected_sha256,
        learning_rate, betas, epsilon, weight_decay, amsgrad, update_rms_clip_threshold,
        asset_root=None):
    if update_rms_clip_threshold<=0 or epsilon<=0:
        raise ValueError('Positive registered numerical guard and RMS threshold are required')
    optimizer=load_verified_dinomaly_stableadamw(source_path,expected_sha256,asset_root=asset_root)
    return optimizer(params,lr=learning_rate,betas=tuple(betas),eps=epsilon,
        weight_decay=weight_decay,amsgrad=amsgrad,clip_threshold=update_rms_clip_threshold)
