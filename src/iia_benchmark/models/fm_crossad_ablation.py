"""Paper-described Table 3 reconstructions, distinct from unprovided author variants."""
from __future__ import annotations
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from .fm_crossad_author_runtime import import_author

COMPONENTS = {
    1: (False, False, False, False), 2: (True, False, False, False),
    3: (True, True, False, False), 4: (True, True, False, True),
    5: (False, False, True, True), 6: (True, True, True, True),
}


def build_ablation(runtime, model_parameters, row):
    if row not in COMPONENTS:
        raise ValueError('Unknown visually verified CrossAD Table 3 row')
    import_author(runtime)
    from models.CrossAD.Basic_CrossAD import Basic_CrossAD
    from models.CrossAD.Context_Blocks import Extractor
    if row == 6:
        return Basic_CrossAD(SimpleNamespace(**model_parameters))
    multi, cross, subseries, global_context = COMPONENTS[row]
    parameters = dict(model_parameters)
    if not multi:
        parameters['ms_kernels'] = [1]
    elif not cross:
        # Same-scale reconstruction includes finest raw scale instead of next-scale raw target.
        parameters['ms_kernels'] = list(parameters['ms_kernels']) + [1]

    class ReconstructedCrossAD(Basic_CrossAD):
        def __init__(self):
            super().__init__(SimpleNamespace(**parameters))
            self.table3_row = row
            self.components_ = COMPONENTS[row]
            self.source_equivalence_ = False
            self.target_lengths_ = self.ms_t_lens[1:] if cross else self.ms_t_lens[:-1]
            if not subseries:
                self.context_net = None  # No unused query/router/extractor parameters masquerading as removed components.
            if global_context and not subseries:
                self.direct_bank = Extractor([], context_size=parameters['bank_size'],
                    query_len=sum(self.ms_p_lens[:-1]), d_model=parameters['d_model'],
                    decay=parameters['decay'], epsilon=parameters['epsilon'])
                self.reconstruction_boundary_ = 'Row 4 stores complete encoder multi-scale token sequences directly; bank token shape is an explicit reconstruction choice.'
            else:
                self.direct_bank = None
                self.reconstruction_boundary_ = 'Source encoder/decoder blocks reused; each-scale local reconstruction for rows 1/2, local cross-scale reconstruction for row 3, single-scale query/global bank for row 5. Exact unpublished author architectures unknown.'

        def _forward(self, x_enc, *args):
            if row == 5:
                return super()._forward(x_enc, *args)
            bs, length, channels = x_enc.shape
            ci = x_enc.permute(0, 2, 1).reshape(bs*channels, length, 1)
            scales = self.ms_utils(ci)
            target_scales = scales[1:] + [ci] if cross else list(scales)
            target = torch.cat(target_scales, dim=1).reshape(bs, channels, -1).permute(0, 2, 1)
            embeddings = [self.patch_embedding(values.permute(0, 2, 1))[0] for values in scales]
            positions = self.ms_utils(self.pos_embedding(self.ms_p_lens[-1]))
            encoded, _ = self.encoder(torch.cat(embeddings, dim=1)+torch.cat(positions, dim=1), self.scale_ind_mask)
            if global_context:
                if self.training:
                    distances, q_hat = self.direct_bank.update_context(encoded)
                    memory = self.direct_bank.concat_context(q_hat+self.direct_bank.context.detach()-q_hat.detach())
                    q_distance = distances.reshape(bs, channels, 1).permute(0, 2, 1)
                else:
                    memory = self.direct_bank.concat_context(self.direct_bank.context)
                    q_distance = encoded.new_zeros(bs, 1, channels)
                memory = memory.unsqueeze(0).expand(bs*channels, -1, -1)
            else:
                memory = encoded
                q_distance = encoded.new_zeros(bs, 1, channels)
            if cross:
                pieces = self.ms_utils.split_2_list(encoded, self.ms_p_lens, 'encoder')
                queries = torch.cat(self.ms_utils.up(pieces, self.ms_p_lens), dim=1)
                self_mask, cross_mask = self.next_scale_mask, None
            else:
                queries = encoded
                # Block both attention paths between scales: a genuine same-scale control.
                self_mask = cross_mask = self.scale_ind_mask
            decoded, _, _ = self.decoder(queries, memory, self_mask, cross_mask)
            patch_lengths = self.ms_p_lens[1:] if cross else self.ms_p_lens[:-1]
            padded_lengths = [p*parameters['patch_len'] for p in patch_lengths]
            pieces = torch.split(decoded.reshape(bs*channels, -1, 1), padded_lengths, dim=1)
            reconstructed = torch.cat([v[:, :n] for v, n in zip(pieces, self.target_lengths_)], dim=1)
            reconstructed = reconstructed.reshape(bs, channels, -1).permute(0, 2, 1)
            return target, reconstructed, q_distance

        def _ms_anomaly_score(self, decoded, target):
            if row == 5:
                return super()._ms_anomaly_score(decoded, target)
            losses = torch.split(F.mse_loss(decoded, target, reduction='none'), self.target_lengths_, dim=1)
            fine_length = self.target_lengths_[-1]
            return sum(loss if loss.shape[1] == fine_length else
                       F.interpolate(loss.permute(0, 2, 1), size=fine_length, mode='linear').permute(0, 2, 1)
                       for loss in losses)

    return ReconstructedCrossAD()
