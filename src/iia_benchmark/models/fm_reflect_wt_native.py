"""SHA-verified original REFLECT mask and WT reversed-flow component bodies.

Refs: github.com/farzad-bz/REFLECT and github.com/lil-wayne-0319/fmad.
Their medical/feature settings are not the missing Flow Mismatching adapter.
"""
import ast
import hashlib
import random
import types
from pathlib import Path

import numpy as np
import torch
from .fm_supplemental_author import _resolve_asset


def _tree(path, expected_sha256, asset_root):
    path=_resolve_asset(path,asset_root)
    source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=expected_sha256:
        raise ValueError('Original immutable REFLECT/WT source SHA256 differs')
    return path,ast.parse(source.decode('utf-8'),filename=str(path))


def load_verified_reflect_masks(source_path: str|Path, expected_sha256: str, *, asset_root=None):
    """Select both original complete contour functions; never import data loader."""
    import cv2
    path,tree=_tree(source_path,expected_sha256,asset_root)
    selected={}
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in ('BraTS2021Dataset','ATLASDataset'):
            methods=[x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='generate_random_walk_contour']
            if len(methods)!=1:raise ValueError('Every original complete contour method required')
            namespace=dict(np=np,random=random,cv2=cv2,__name__='_iia_native_reflect_mask')
            exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(path),'exec'),namespace)
            selected[node.name]=namespace['generate_random_walk_contour']
    if set(selected)!={'BraTS2021Dataset','ATLASDataset'}:
        raise ValueError('Both native medical contour implementations required')
    return types.SimpleNamespace(**selected,source_path=str(path),source_sha256=expected_sha256)


def reflect_native_mask(native_core, initial_mask, *, channels: int, max_objects: int, source_dataset: str):
    """Explicit ROI and geometry; native global Python/NumPy RNG draws unchanged."""
    if (initial_mask.ndim!=2 or initial_mask.shape[0]!=initial_mask.shape[1]
            or not np.isin(initial_mask,(0,1)).all() or not initial_mask.any()
            or channels<1 or max_objects<1 or source_dataset not in ('BraTS2021Dataset','ATLASDataset')):
        raise ValueError('Explicit nonempty square native ROI/channels/max_objects/source required')
    owner=types.SimpleNamespace(embedding_size=initial_mask.shape[0],embedding_dim=channels)
    return getattr(native_core,source_dataset)(owner,initial_mask,max_objects)


def load_verified_wt_flow(source_path: str|Path, expected_sha256: str, *, asset_root=None):
    """Complete original ReversedFlow class; no feature extractor import."""
    path,tree=_tree(source_path,expected_sha256,asset_root)
    nodes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ReversedFlow']
    if len(nodes)!=1:raise ValueError('One original complete ReversedFlow definition required')
    namespace=dict(torch=torch,np=np,F=torch.nn.functional,__name__='_iia_native_wt_flow')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
    return types.SimpleNamespace(flow=namespace['ReversedFlow'](),source_path=str(path),source_sha256=expected_sha256)


def wt_native_standardize(images, *, epsilon: float):
    """Native fresh LayerNorm identity-affine behavior; all C/H/W population moments.

    Epsilon is explicit in config. The related native nn.LayerNorm default is1e-5;
    this does not certify the target image adaptation's epsilon/variance choice.
    """
    if images.ndim!=4 or not images.is_floating_point() or epsilon<=0:
        raise ValueError('Complete float images and positive explicit epsilon required')
    return torch.nn.functional.layer_norm(images,images.shape[1:],eps=epsilon)


def wt_native_objective(velocity, images, time, noise, *, native_core, epsilon, labels=None):
    """Native reverse interpolant/target; draws supplied for independent auditing."""
    if time.shape!=(images.shape[0],) or noise.shape!=images.shape:
        raise ValueError('Complete native image/noise/time geometry required')
    mapped=wt_native_standardize(images,epsilon=epsilon)
    state,endpoint=native_core.flow.create_flow_rev(mapped,time,x_1=noise)
    predicted=velocity(state,time,labels) if labels is not None else velocity(state,time)
    if predicted.shape!=images.shape:raise ValueError('Complete native WT velocity shape required')
    return native_core.flow.mse_loss_rev(predicted,mapped,endpoint)


@torch.no_grad()
def wt_native_endpoint(velocity, mapped_images, *, native_core, steps: int, labels=None):
    """Related WT native left-grid Euler; explicit steps, not ReconFlow solver.

    Native generate_pdf uses j/steps and full endpoint0->1. Preserve this
    algorithm; target Flow Mismatching author solver identity remains unknown.
    """
    if not isinstance(steps,int) or isinstance(steps,bool) or steps<1:
        raise ValueError('Positive explicit original WT integration budget required')
    state=mapped_images
    for j in range(steps):
        time=state.new_full((state.shape[0],),j/steps)
        predicted=velocity(state,time,labels) if labels is not None else velocity(state,time)
        if predicted.shape!=state.shape:raise ValueError('Complete native WT endpoint shape required')
        state=native_core.flow.euler(state,predicted,1./steps)
    return state
