"""One AdamW update after every retained full-batch CNF endpoint gradient.

The fixed FP32/CPU path has no AMP, clipping, scheduler, accumulation or DDP.
Endpoint likelihood equations are supplied by the pinned source adapter.
"""
from types import MethodType


def training_step(self, batch, batch_idx):
    import torch
    left = self.hparams.leaveout_timepoint
    if (not self.is_trajectory or len(batch) not in (4,5) or not 0 < left < len(batch)-1
            or any(len(x)!=1024 for x in batch)):
        raise ValueError('Every original full1024 batch and interior leaveout required')
    optimizer = self.optimizers()
    optimizer.zero_grad()
    indices = [i for i in reversed(range(len(batch))) if i!=left]
    losses, regs = [], []
    for index in indices:
        reg, loss = self.step(batch, training=True, endpoint_index=index)
        contribution = (reg+loss)/len(indices)
        if not torch.isfinite(contribution):
            raise ValueError('Full original endpoint likelihood must stay finite')
        self.manual_backward(contribution)
        losses.append(loss.detach()/len(indices));regs.append(reg.detach()/len(indices))
        del reg,loss,contribution
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.parameters()):
        raise ValueError('Every full original parameter gradient must stay finite')
    optimizer.step()
    loss,reg = torch.stack(losses).sum(),torch.stack(regs).sum()
    self.log_dict({'train/loss':loss+reg,'train/mse':loss,'train/reg':reg},
        on_step=True,on_epoch=False,prog_bar=True,batch_size=1024)
    return (loss+reg).detach()


def install(model):
    if not getattr(model,'_iia_endpoint_backward_installed',False) or not model.automatic_optimization:
        raise ValueError('Exactly one installation on the original endpoint-adapted CNF required')
    model.automatic_optimization=False
    model.training_step=MethodType(training_step,model)
    return model
