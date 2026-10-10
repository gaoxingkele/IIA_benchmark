from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT
from scripts.flow_matching.register_gtgan_native_commands_v1 import native_arguments,recipes


def test_exact_author_training_budgets_and_all_source_supported_missing_flags():
    source=ROOT/'data/public_datasets/flow_matching/gtgan_native_source_v1/original'
    cases=list(recipes());assert [r[0] for r in cases]==[0,.3,.5,.7]
    for rate,entry,args,boundary in cases:
        actual=native_arguments(source,entry,args)
        assert actual['first_epoch']==10000 and actual['batch_size']==128 and actual['seq_len']==24
        assert actual['max_steps']==(8500 if rate==0 else 2000)
        assert actual['max_steps_metric']==10 and actual['dims']=='32-64-64-32'
        assert actual['atol']==.01 and actual['rtol']==.001 and actual['train']
        assert actual['last_activation_r']=='softplus'
        if rate:assert actual['missing_value']==rate
        if rate in [.3,.5]:assert 'equivalence unproved' in boundary
