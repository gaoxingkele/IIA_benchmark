"""Pinned TAB range evaluators with a verified rank-prefix acceleration.

Preserves all 250 threshold samples, inclusive ties, every integer buffer,
TAB's asymmetric range extension, existence reward and trapezoidal integration.
No subsampling of timestamps or buffer cap is used. The original downloaded
files are untouched and usable as an independent executable reference.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import List

import numpy as np


def load_reference(source_root):
    source_root = Path(source_root)
    namespace = {'np': np, 'List': List}
    utils = ast.parse((source_root / 'utils.py').read_text(encoding='utf-8'))
    body = [node for node in utils.body if isinstance(node, ast.FunctionDef) and node.name == 'get_list_anomaly']
    if len(body) != 1:
        raise ValueError('Pinned TAB anomaly length helper changed')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_root / 'utils.py'), 'exec'), namespace)
    suffix = hashlib.sha256(str(source_root.resolve()).encode()).hexdigest()[:12]
    name = '_fm_tab_vus_' + suffix
    spec = importlib.util.spec_from_file_location(name, source_root / 'vus_metrics.py')
    vus = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vus)
    package_name = '_fm_tab_affiliation_' + suffix
    if package_name not in sys.modules:
        package_path = source_root / 'affiliation'
        spec = importlib.util.spec_from_file_location(package_name, package_path / '__init__.py', submodule_search_locations=[str(package_path)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    generics = importlib.import_module(package_name + '.generics')
    affiliation = importlib.import_module(package_name + '.metrics')
    namespace.update(convert_vector_to_events=generics.convert_vector_to_events, pr_from_events=affiliation.pr_from_events)
    label_tree = ast.parse((source_root / 'classification_metrics_label.py').read_text(encoding='utf-8'))
    body = [node for node in label_tree.body if isinstance(node, ast.FunctionDef) and node.name in ('affiliation_f', 'affiliation_precision', 'affiliation_recall')]
    if len(body) != 3:
        raise ValueError('Pinned TAB affiliation functions changed')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_root / 'classification_metrics_label.py'), 'exec'), namespace)
    return {'vus': vus, 'events': generics.convert_vector_to_events, 'pr_from_events': affiliation.pr_from_events,
            **{key: namespace[key] for key in ('get_list_anomaly', 'affiliation_f', 'affiliation_precision', 'affiliation_recall')}}


def positive_ranges(values):
    flags = np.asarray(values) > 0
    transitions = np.diff(np.r_[False, flags, False].astype(int))
    return np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1) - 1


def accelerated_vus(labels, scores, max_buffer):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels) or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Require complete aligned binary labels and finite scores')
    if np.unique(labels).size != 2:
        raise ValueError('TAB VUS is undefined for single-class labels')
    if max_buffer < 0 or int(max_buffer) != max_buffer:
        raise ValueError('Invalid TAB buffer')
    order = np.argsort(-scores, kind='stable')
    ranked = scores[order]
    thresholds = ranked[np.linspace(0, len(scores) - 1, 250).astype(int)]
    counts = np.searchsorted(-ranked, -thresholds, side='right')
    starts, ends = positive_ranges(labels)
    original_positive = labels.sum()
    roc_values, pr_values = [], []
    for window in range(int(max_buffer) + 1):
        extended = labels.astype(float).copy()
        # Exact original extension, including window//2 and right-end exclusion.
        if window:
            for start, end in zip(starts, ends):
                right = np.arange(end, min(end + window // 2, len(labels)))
                extended[right] += np.sqrt(1 - (right - end) / window)
                left = np.arange(max(start - window // 2, 0), start)
                extended[left] += np.sqrt(1 - (start - left) / window)
        extended = np.minimum(1., extended)
        seg_starts, seg_ends = positive_ranges(extended)
        maxima = np.sort([scores[a:b + 1].max() for a, b in zip(seg_starts, seg_ends)])
        existence = (len(maxima) - np.searchsorted(maxima, thresholds, side='left')) / len(maxima)
        prefix = np.r_[0., np.cumsum(extended[order])]
        tp = prefix[counts]
        p_new = (original_positive + extended.sum()) / 2
        tpr_middle = np.minimum(tp / p_new, 1.) * existence
        fpr_middle = (counts - tp) / (len(labels) - p_new)
        precision = np.r_[1., tp / counts]
        tpr, fpr = np.r_[0., tpr_middle, 1.], np.r_[0., fpr_middle, 1.]
        roc_values.append(float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)))
        pr_values.append(float(np.sum(np.diff(tpr[:-1]) * (precision[1:] + precision[:-1]) / 2)))
    return {'VUS_ROC': float(np.mean(roc_values)), 'VUS_PR': float(np.mean(pr_values)),
            'max_buffer': int(max_buffer), 'buffers_evaluated': int(max_buffer) + 1, 'threshold_samples': 250}
