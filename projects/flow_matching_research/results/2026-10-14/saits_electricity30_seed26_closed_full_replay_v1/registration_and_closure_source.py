import json
from pathlib import Path

base=Path(__file__).with_name('archive_brits_seed30_and_cfm_iteration_20261014.py').read_text()
source=base.split('for name,text in [\n',1)[0]
assert 'for name,text in [' not in source
source=source.replace('brits_electricity20_seed30_closed_full_replay_v1','saits_electricity30_seed26_closed_full_replay_v1')
source=source.replace('saits_paper_brits_electricity_author_missing0.2_seed30','saits_paper_saits_electricity_author_missing0.3_seed26')
source=source.replace('1944331','2781282').replace('1001','201').replace('best_epoch=969','best_epoch=169')
start=source.index("(folder/'README.md').write_text(")
source=source[:start]
values=dict(__file__=__file__)
exec(compile(source,__file__,'exec'),values)
folder=values['folder'];row=values['row'];result=values['result']
assert result['config']['method']=='saits' and result['config']['missing_ratio']==.3 and result['config']['seed']==26
assert result['validation_best']['epoch']==169 and result['parameters']==11511000
(folder/'closure_source.py').write_text(source)
(folder/'registration_and_closure_source.py').write_bytes(Path(__file__).read_bytes())
closurepath=folder/'actual_original_seed30_complete_closure.json'
closure=json.loads(closurepath.read_text())
closure.update(method='saits',dataset='electricity',nominal_missing_ratio=.3,seed=26,
    original_model_parameters=result['parameters'],effective_target_fraction_of_all_full_test_cells=2781282/10730000,
    actual_completed_and_independently_reviewed_canonical_job='saits_paper_saits_electricity_author_missing0.3_seed26')
closurepath.unlink()
with (folder/'actual_original_saits_electricity30_seed26_complete_closure.json').open('x') as stream:
    json.dump(closure,stream,indent=2);stream.write('\n')
(folder/'README.md').write_text('SAITS / Electricity / nominal30% missing / seed26 completed the frozen10000 maximum-epoch, patience30 author protocol at next_epoch201 (best validation epoch169). The11511000-parameter model and all290x100x370 grid cells,2781282 held-out target coordinates and10 original month groups passed native CPU replay and independent full-array review at the original2e-6 tolerance. Every current config/model/result/full-data/array/group-statistic/proof hash was rechecked. Exact historical author per-rate tuning and preprocessing seeds remain unavailable; the released best configuration was reused and the original replacement masking is distinct from corrected masking. Actual targets cover25.9206% of full test-grid cells. Original prediction arrays were unavailable, so no bitwise prediction claim. This is one seed, not a five-seed aggregate, strict TAB or completion of all210 SAITS/BRITS tasks or the broader45/681/273 scope.\n')
print(json.dumps(dict(method='saits',nominal_missing_ratio=.3,seed=26,next_epoch=201,best_epoch=169,
    metrics=row['independently_recomputed_metrics'],held_out_coordinates=row['held_out_coordinates'])))
