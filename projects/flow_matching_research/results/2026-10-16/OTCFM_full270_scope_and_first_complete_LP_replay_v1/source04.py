from types import SimpleNamespace
import pytest
from scripts.flow_matching.otcfm_validation_batch_compat_v1 import install


class LoaderFixture:
    def __init__(self,batch,minimum):
        self.hparams=SimpleNamespace(batch_size=batch,num_workers=0,pin_memory=False)
        self.split_timepoint_data=[(range(1000),range(minimum),range(100)),(range(1100),range(minimum+10),range(110))]
    def combined_loader(self,index,shuffle=False,load_full=False):
        return dict(batch=self.hparams.batch_size,index=index,shuffle=shuffle,load_full=load_full)
    def train_dataloader(self):return self.combined_loader(0,shuffle=True)
    def val_dataloader(self):return self.combined_loader(1)
    def test_dataloader(self):return self.combined_loader(2,load_full=True)


def test_validation_only_view_repairs_empty_batch_without_changing_train_test_or_hparams():
    dm=LoaderFixture(512,266);train=dm.train_dataloader();test=dm.test_dataloader();params=dm.hparams
    actual=install(dm,266)
    assert actual['validation_only_empty_loader_repair'] and dm.val_dataloader()['batch']==266
    assert dm.hparams is params and params.batch_size==512
    assert dm.train_dataloader()==train and dm.test_dataloader()==test


@pytest.mark.parametrize('batch,minimum',[(128,266),(512,699)])
def test_already_viable_original_validation_loader_is_untouched(batch,minimum):
    dm=LoaderFixture(batch,minimum);before=dm.val_dataloader.__func__
    actual=install(dm,batch)
    assert not actual['validation_only_empty_loader_repair'] and dm.val_dataloader.__func__ is before


@pytest.mark.parametrize('expected',[128,265,512])
def test_no_unregistered_validation_size_can_silently_change_protocol(expected):
    with pytest.raises(ValueError):install(LoaderFixture(512,266),expected)
