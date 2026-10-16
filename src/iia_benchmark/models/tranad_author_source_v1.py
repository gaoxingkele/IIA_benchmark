"""Execute frozen TranAD classes and training functions from their source AST.

The source imports unrelated DGL models. Only explicitly selected, unmodified
definitions are compiled. Old torch Transformer containers are expressed as
their sequential layer loops because current torch passes new causal keywords
and assumes standard-layer attributes absent from these author custom layers.
No normalization, activation, positional encoding or batch policy is replaced.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace


VARIANTS = ('TranAD', 'TranAD_Basic', 'TranAD_Transformer',
            'TranAD_Adversarial', 'TranAD_SelfConditioning')


def definition(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(matches) != 1:
        raise ValueError('Exactly one frozen author definition required: '+name)
    return matches[0]


def compile_definition(path, name, namespace):
    node = definition(path, name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def literal_assignment(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    matches = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(matches) != 1:
        raise ValueError('Unique original literal assignment required: '+name)
    return ast.literal_eval(matches[0].value)


def load_author_runtime(config, dataset):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    root = Path(config['source_root'])
    for pin in config['source_receipts']:
        if hashlib.sha256((root/pin['path']).read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError('Frozen original author source changed: '+pin['path'])
    lr_d = literal_assignment(root/'src/constants.py', 'lr_d')
    lm_d = literal_assignment(root/'src/constants.py', 'lm_d')
    if dataset not in lr_d or dataset not in lm_d:
        raise ValueError('An explicit original dataset learning rate and POT policy is required')

    class SequentialEncoder(nn.Module):
        def __init__(self, encoder_layer, num_layers, norm=None):
            super().__init__()
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
            self.num_layers, self.norm = num_layers, norm

        def forward(self, src, mask=None, src_key_padding_mask=None):
            for layer in self.layers:
                src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            return self.norm(src) if self.norm is not None else src

    class SequentialDecoder(nn.Module):
        def __init__(self, decoder_layer, num_layers, norm=None):
            super().__init__()
            self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
            self.num_layers, self.norm = num_layers, norm

        def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
            for layer in self.layers:
                tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            return self.norm(tgt) if self.norm is not None else tgt

    ns = dict(__name__='_frozen_tranad_author_source_v1', np=np, torch=torch, nn=nn,
        math=math, lr=lr_d[dataset], lm=lm_d[dataset][1], DataLoader=DataLoader,
        TensorDataset=TensorDataset, tqdm=tqdm, TransformerEncoder=SequentialEncoder,
        TransformerDecoder=SequentialDecoder, args=SimpleNamespace(model='TranAD'))
    for name in ['PositionalEncoding', 'TransformerEncoderLayer', 'TransformerDecoderLayer']:
        compile_definition(root/'src/dlutils.py', name, ns)
    for name in VARIANTS:
        compile_definition(root/'src/models.py', name, ns)
    for name in ['convert_to_windows', 'backprop']:
        compile_definition(root/'main.py', name, ns)
    return ns


def author_model(config, dataset, variant, features):
    if variant not in VARIANTS:
        raise ValueError('Only the five exact released TranAD branches are registered')
    ns = load_author_runtime(config, dataset)
    ns['args'].model = variant
    return ns[variant](features).double(), ns


def load_author_metrics(config, dataset):
    import importlib.util
    from sklearn.metrics import roc_auc_score, ndcg_score
    import numpy as np
    root = Path(config['source_root'])
    # The SPOT file has no src.* imports or execution-time CLI side effects.
    spec = importlib.util.spec_from_file_location('_frozen_tranad_spot_v1', root/'src/spot.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    lm = literal_assignment(root/'src/constants.py', 'lm_d')[dataset][1]
    ns = dict(np=np, SPOT=module.SPOT, lm=lm, roc_auc_score=roc_auc_score, ndcg_score=ndcg_score)
    for name in ['calc_point2point', 'adjust_predicts', 'pot_eval']:
        compile_definition(root/'src/pot.py', name, ns)
    for name in ['hit_att', 'ndcg']:
        compile_definition(root/'src/diagnosis.py', name, ns)
    return ns
