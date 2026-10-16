"""Give an otherwise empty validation loader the largest available uniform batch.

Only validation receives a readonly hparams view. Original train/test loader
objects, full training batch, optimizer and complete test populations stay exact.
"""
from types import MethodType


class ValidationParameters:
    def __init__(self, parameters, size):
        self._parameters=parameters;self.batch_size=size
    def __getattr__(self,name):return getattr(self._parameters,name)


class ValidationView:
    def __init__(self,datamodule,size):
        self._datamodule=datamodule;self.hparams=ValidationParameters(datamodule.hparams,size)
    def __getattr__(self,name):return getattr(self._datamodule,name)


def install(datamodule,expected_batch_size):
    original_batch=datamodule.hparams.batch_size
    available=min(len(parts[1]) for parts in datamodule.split_timepoint_data)
    effective=min(original_batch,available)
    if effective<=0 or effective!=expected_batch_size:
        raise ValueError('Every original validation split and registered validation batch must remain exact')
    if effective<original_batch:
        view=ValidationView(datamodule,effective)
        combined=type(datamodule).combined_loader
        def validation_only(self):return combined(view,1,shuffle=False,load_full=False)
        datamodule.val_dataloader=MethodType(validation_only,datamodule)
    return dict(original_training_batch=original_batch,effective_validation_batch=effective,
        minimum_available_validation_samples=available,training_and_test_loaders_changed=False,
        validation_only_empty_loader_repair=effective<original_batch)
