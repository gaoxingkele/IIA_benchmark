from collections import Counter
from scripts.flow_matching import audit_moment_icm_primary_scope_v1 as detail
from scripts.flow_matching import moment_icm_main_table_targets_v1 as main

PAPER='papers/literature/flow_matching/user_supplied/2026-10-07/moment_long_context/3e67405b691183dab729d6f2e56421280d8bb1268761f190468d8bce9e326413.pdf'


def test_all_main_aggregate_cells_remain_separate_from_detailed_settings():
    rows=main.targets(detail.ROOT/PAPER)
    assert dict(Counter(r['table'] for r in rows))=={1:42,2:18,3:12}
    combined=rows+detail.targets(detail.ROOT/PAPER)
    assert len(combined)==702
    assert len({(r['table'],r['model'],r['dataset'],r.get('horizon'),r['metric']) for r in combined})==702


def test_primary_beta_switches_are_not_renamed_into_4fold_and8fold_models():
    rows=main.targets(detail.ROOT/PAPER)
    assert {r['model'] for r in rows if r['table']==2}=={'MOMENT-Tiny pretrained','ICM pretrained beta frozen','ICM pretrained beta finetuned'}
    assert not any('fold' in r['model'] for r in rows)
    assert next(r['reported'] for r in rows if r['table']==2 and r['model']=='ICM pretrained beta frozen' and r['dataset']=='ETTh1')==.439


def test_aggregate_classification_statistics_are_dataset_not_training_seed_uncertainty():
    rows=main.targets(detail.ROOT/PAPER)
    assert next(r['reported'] for r in rows if r['table']==3 and r['model']=='MOMENT-Tiny + ICM' and r['metric']=='std_dataset_accuracy')==.268
    counts=[r for r in rows if r['table']==3 and r['metric'] in ['dataset_wins','dataset_ties','dataset_losses']]
    assert len(counts)==6 and sum(r['reported'] for r in counts)==52
    assert all(r['dataset']=='26_multivariate_datasets' for r in counts)
