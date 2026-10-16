from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
from scripts.flow_matching import continue_timesnet_remaining4_cold_v3 as s

ROOT=s.ROOT;c=s.read(ROOT/s.original.CONFIG);jobs=s.original.verify(c)
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/TimesNet1106_complete_full51_native_and_point_v1'
closure=s.read(packet/'complete_two_stage_closure.json');assert closure['all204_original_NPY_members_compared_byte_for_byte']
rows=[];binds=[]
for seed in range(1103,1108):
    row=dict(training_seed=seed,source_complete=False,complete_point_audit=False,complete_bitwise_native_audit=False,
        native_failure_preserved=False,precision=None,recall=None,F1=None,AUROC=None,AP=None)
    point=None
    for mode in ['point','native']:
        if seed==1103:
            path=ROOT/c['existing_seed1103_configuration'][mode];stage=s.read(path)
            proof=Path(stage['output_directory'])/'independent_full_saved_score_verification.json';guard=ROOT/stage['state_root']/'receipt.json'
            p,g=s.read(proof),s.read(guard)
            assert p['passed'] and p['mode']==mode and p['configuration_sha256']==s.sha(path)
            assert g['exit_code']==0 and not g['resource_guard_aborted'] and g['full_saved_score_review_passed']
            source_result=next(j for j in s.read(ROOT/s.original.QUEUE)['jobs'] if j['id']==stage['job_id'])
            source=s.read(ROOT/source_result['output_directory']/'result.json')
            assert source['scores_sha256']==p['original_scores_sha256']==s.sha(ROOT/source_result['output_directory']/'scores.npz')
            assert source['checkpoint_sha256']==p['original_checkpoint_sha256']==s.sha(ROOT/source_result['output_directory']/'model.pt')
            if mode=='native':assert p['full_native_checkpoint_reinference_passed'] and p['replay_scores_sha256']==p['original_scores_sha256']
            binding=dict(proof=str(proof),proof_sha256=s.sha(proof),guard_sha256=s.sha(guard),configuration_sha256=s.sha(path))
            row['source_complete']=True
        else:
            job=jobs[s.original.identity(seed)];row['source_complete']=s.original.source_ready(c,job)
            stage=s.original.stage_config(c,job,mode);guard=ROOT/stage['state_root']/'receipt.json'
            if not guard.exists():continue
            g=s.read(guard)
            if not g['passed']:
                assert mode=='native' and seed==1105 and g['exit_code']!=0 and not g['resource_guard_aborted']
                row['native_failure_preserved']=True
                binds.append(dict(seed=seed,mode=mode,status='original_failed_preserved',guard=str(guard),guard_sha256=s.sha(guard)))
                continue
            binding=s.checked_stage(c,job,mode);p=s.read(binding['proof'])
        assert p['full_entities']==51 and p['full_test_points']==408160
        binds.append(dict(binding,seed=seed,mode=mode,status='complete_original_guarded_stage'))
        row['complete_point_audit' if mode=='point' else 'complete_bitwise_native_audit']=True
        if mode=='point':point=p
    if point:
        m=point['independently_recomputed_primary_metrics'];row.update(precision=m['precision'],recall=m['recall'],F1=m['f1'],AUROC=m['auroc'],AP=m['average_precision'])
    rows.append(row)
assert sum(r['source_complete'] for r in rows)==4 and sum(r['complete_point_audit'] for r in rows)==4
assert sum(r['complete_bitwise_native_audit'] for r in rows)==3
assert rows[2]['native_failure_preserved'] and not rows[4]['source_complete']
with (packet/'all5_original_training_seed_slots_validation1_no_PA.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(packet/'all_existing_success_and_failure_stage_bindings.json').write_text(json.dumps(dict(stages=binds,
    existing_completed_models_are_not_new_training_executions=True,complete_five_seed_aggregate=False),indent=2)+'\n',encoding='utf-8')
with (packet/'README.md').open('a',encoding='utf-8') as f:
    f.write('\n随附原五种子位置表：1103、1104、1106原生逐位重放完成，1105原失败保留，1107源训练待实际准入；源结果4/5，分数复核4/5，严格原生重放3/5。表只列各自验证1%阈值点值，不汇总部分种子均值或区间，旧成功模型不算本次新增训练。\n')
shutil.copyfile(__file__,packet/'packaging_source.py')
(packet/'packet_integrity.json').write_text(json.dumps(dict(files=[dict(path=x.relative_to(packet).as_posix(),sha256=s.sha(x)) for x in sorted(packet.rglob('*')) if x.is_file() and x.name!='packet_integrity.json']),indent=2)+'\n',encoding='utf-8')
ara='projects/flow_matching_research/ara/execution/TimesNet1106_complete_full51.2026-10-16_v1.json'
paths=[ara,'projects/flow_matching_research/README.rst']+[x.relative_to(ROOT).as_posix() for x in packet.rglob('*') if x.is_file()]
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/TimesNet1106_complete_full51_manifest_20261016_v1.json';assert not manifest.exists()
manifest.write_text(json.dumps(dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(paths),full_bitwise_native_cases=3,point_cases=4,full_five_seed_aggregate=False)))
