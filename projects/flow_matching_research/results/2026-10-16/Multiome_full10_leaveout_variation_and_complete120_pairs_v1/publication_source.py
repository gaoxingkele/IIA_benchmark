"""Publish complete actual heldout diagnostics and full120future comparisons."""
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scipy.stats import t as student_t
from scripts.flow_matching import analyze_otcfm_Multiome_complete120_comparisons_v1 as s


def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def copy(p,t):
    with t.open('xb') as f:f.write(p.read_bytes())
    assert s.sha(p)==s.sha(t)


c=s.read(s.ROOT/s.CONFIG);s.verify(c)
identity='within__released_author_donor32606__released_yaml__cfm'
p=Path(c['artifact_root'])/identity/'comparison_diagnostic.json';gp=s.ROOT/c['state_root']/'attempts'/identity/'guard_receipt.json'
r,g=s.read(p),s.read(gp)
assert r['passed'] and r['id']==identity and r['configuration_sha256']==s.sha(s.ROOT/s.CONFIG)
assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['report_sha256']==s.sha(p)
cfg=s.read(s.ROOT/s.rows.CONFIG);gid='released_author_donor32606__released_yaml__cfm';group=next(v for v in cfg['groups'] if v['id']==gid)
records=r['source_rows'][gid];assert s.balanced(records,group)==r['metrics'] and len(records)==10
q=s.read(s.ROOT/s.rows.native.QUEUE);registered_pairs=0
for task in c['groups']:
    if task['kind']!='paired':continue
    ga,gb=[next(v for v in cfg['groups'] if v['id']==identity) for identity in task['upstream_row_ids']]
    for seed in range(42,47):
        for left in [1,2]:
            a,b=[next(j for j in q['jobs'] if j['id'] in g['required_case_ids'] and j['seed']==seed and j['leaveout_timepoint']==left) for g in [ga,gb]]
            assert all(a[k]==b[k] for k in ['data','scientific_protocol','net','expected_model_parameters','native_dimensions'])
            registered_pairs+=1
assert registered_pairs==120
up,ug=s.upstream(gid);ev=r['upstream_evidence'][0]
assert ev['report_sha256']==s.sha(up) and ev['guard_sha256']==s.sha(ug)
# Independently use scalar arithmetic and SciPy t4;the first completed diagnostic
# must not merely assert that its own serialized output equals itself.
checks={}
for metric in s.METRICS:
    matrix={(v['seed'],v['leaveout_timepoint']):v['heldout'][metric] for v in records}
    values=[matrix[k] for k in sorted(matrix)];grand=statistics.mean(values)
    left={l:statistics.mean(matrix[seed,l] for seed in range(42,47)) for l in [1,2]}
    seeds={seed:statistics.mean([matrix[seed,1],matrix[seed,2]]) for seed in range(42,47)}
    delta=[matrix[seed,2]-matrix[seed,1] for seed in range(42,47)]
    mean,std=statistics.mean(delta),statistics.stdev(delta)
    ci=student_t.interval(.95,4,loc=mean,scale=std/math.sqrt(5)) if std else (mean,mean)
    actual=r['metrics'][metric]['paired_leaveout2_minus1']
    assert math.isclose(actual['mean'],mean,rel_tol=1e-12,abs_tol=1e-12)
    assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(actual['ci95'],ci))
    total=sum((v-grand)**2 for v in values);ssleft=5*sum((v-grand)**2 for v in left.values())
    ssseed=2*sum((v-grand)**2 for v in seeds.values());residual=sum((matrix[seed,l]-left[l]-seeds[seed]+grand)**2 for seed,l in matrix)
    d=r['metrics'][metric]['descriptive_variance_decomposition']
    assert all(math.isclose(d[k],value,rel_tol=1e-12,abs_tol=1e-12) for k,value in
        [('total_SS',total),('leaveout_SS',ssleft),('seed_SS',ssseed),('residual_SS',residual)])
    assert math.isclose(total,ssleft+ssseed+residual,rel_tol=1e-12,abs_tol=1e-12)
    checks[metric]=dict(paired_df4_t95_and_scalar_variance_identity_verified=True)
folder=s.ROOT/'projects/flow_matching_research/results/2026-10-16/Multiome_full10_leaveout_variation_and_complete120_pairs_v1'
folder.mkdir(parents=True,exist_ok=False)
copy(p,folder/'actual_complete_ten_model_comparison_diagnostic.json');copy(gp,folder/'actual_diagnostic_guard.json')
copy(up,folder/'actual_required_complete_upstream_row.json');copy(ug,folder/'actual_required_upstream_row_guard.json')
save(folder/'independent_scalar_SciPy_statistics_verification.json',dict(passed=True,checks=checks,
    original_model_conditions=10,training_seeds=5,independent_leaveout_models_per_seed=2,
    future_registered_matched_method_condition_pairs_checked=registered_pairs,
    no_new_model_training_or_inference=True,causal_or_hypothesis_test_claim=False,all_original_experiments_complete=False))
with (folder/'all8_heldout_metric_uncertainty_and_leaveout_differences.csv').open('x',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['metric','five_seed_mean','five_seed_STD','all10model_STD','leaveout1_mean','leaveout2_mean',
        'paired_leaveout2_minus1_mean','paired_t95_low','paired_t95_high','descriptive_leaveout_SS_fraction','descriptive_seed_SS_fraction','descriptive_residual_SS_fraction'])
    for metric,v in r['metrics'].items():
        d=v['descriptive_variance_decomposition'];a=v['paired_leaveout2_minus1']
        w.writerow([metric,v['primary_five_seed_summary']['mean'],v['primary_five_seed_summary']['sample_STD'],v['all10_model_sample_STD'],
            v['per_leaveout_five_seed_summary']['1']['mean'],v['per_leaveout_five_seed_summary']['2']['mean'],a['mean'],*a['ci95'],
            d['leaveout_fraction'],d['seed_fraction'],d['residual_fraction']])
base=s.ROOT/c['state_root'];state=s.read(base/'status.json');launch=s.read(base/'launch.json');actor=psutil.Process(launch['pid'])
assert actor.is_running() and actor.create_time()==launch['create_time'] and identity in state['completed_diagnostics']
assert not state['failed_diagnostics_preserved']
copy(base/'launch.json',folder/'all24_diagnostic_controller_launch.json');copy(base/'status.json',folder/'all24_diagnostic_state_snapshot.json')
save(folder/'actual_live_remaining_diagnostic_handle.json',dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
    distinct_original_models=120,required_within_row_diagnostics=12,required_matched_method_comparisons=12,
    model_condition_references=360,references_are_not_extra_training_seeds=True,
    completed_within_row_diagnostics=1,completed_method_pair_comparisons=0,all_original_experiments_complete=False))
new=[s.MODULE.replace('.','/')+'.py',s.TEST,s.CONFIG];pins=[]
for i,path in enumerate(new+[s.rows.CONFIG,s.rows.native.QUEUE,s.rows.replay.CONFIG]):
    p=s.ROOT/path;target=folder/f'source{i:02d}{p.suffix}';copy(p,target);pins.append(dict(path=path,sha256=s.sha(p),snapshot=target.name))
save(folder/'source_manifest.json',pins)
xml=Path('F:/aicoding/IIA_Data/experiments/Multiome_complete120_comparative_diagnostics_20261016_v1_tests.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'));assert sum(int(v.get('tests','0')) for v in suites)==14
assert not any(int(v.get(k,'0')) for v in suites for k in ['errors','failures','skipped'])
copy(xml,folder/'validation_14_checks.xml');copy(Path(__file__),folder/'publication_source.py')
(folder/'.gitattributes').write_text('*.json -text\n*.py -text\n*.xml -text\n*.csv -text\n',encoding='utf-8')
w=r['metrics']['1-Wasserstein'];d=w['descriptive_variance_decomposition'];a=w['paired_leaveout2_minus1']
(folder/'README.md').write_text('Complete ten-model Multiome variation diagnostic;all120model comparison plan\n\n'
    'For released donor32606 / released-YAML I-CFM,the five-seed leaveout1 W1 mean='+str(w['per_leaveout_five_seed_summary']['1']['mean'])+
    ',leaveout2 mean='+str(w['per_leaveout_five_seed_summary']['2']['mean'])+'. '
    'Five matched training-seed differences leaveout2-minus1 have mean='+str(a['mean'])+',descriptive df4t95='+str(a['ci95'])+'. '
    'The complete row remains1.0908240139484406,with five-seed two-leaveout-mean STD0.030187232020799152. '
    'The separate all10model STD is0.1166928055177199. Different aggregation units cannot be silently interchanged to match the paper SD0.099.\n\n'
    'The exact balanced sum-of-squares decomposition assigns '+str(100*d['leaveout_fraction'])+'%to the grouping by leaveout,'
    +str(100*d['seed_fraction'])+'%to training-seed mean differences,and'+str(100*d['residual_fraction'])+'%to the remaining interaction. '
    'These describe variation among ten observed model conditions;they are not causal fractions,an ANOVA significance test or a generalization guarantee. '
    'All8actual heldout metrics,not only W1,are retained and independently checked with scalar arithmetic and SciPy t4.\n\n'
    'The live plan covers all120original source conditions:12complete within-row diagnostics and12method-pair comparisons within identical input/recipe/architecture/seeds/leaveouts. '
    'Every paired comparison waits for both complete10model row guards and actual full native proofs. '
    'Currently one within-row diagnostic completed;no between-method performance comparison is complete. '
    'Declared budgets remain fixed and actual epoch/update/time differences are reported,not assumed compute-matched. '
    'All360references reuse120original models;they never become360independent training models. '
    'This is posthoc analysis with no model inference,training or LP rerun. '
    'An inherited capped-wait telemetry label may say replaying_full_checkpoints while this metadata worker only rehashes artifacts and computes summaries.\n\n'
    'Data donor32606differs from paper-described13176,whose historical matrix/PCA/weight identity remains uncertified. '
    'The original source uses all-timepoint whitening,leftout-timepoint validation and full-timepoint test distributions. '
    'Do not promote this generative single-cell diagnostic to strict inductive anomaly-detection/TAB performance. '
    '14relevant checks passed. Preserve all45papers/681method-baseline-ablation records/273data tasks/702scalar targets and all other original experiments.\n',encoding='utf-8')
paths=new+[p.relative_to(s.ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
save(s.ROOT/'tmp/Multiome_complete120_comparisons_publication_manifest_20261016_v1.json',dict(
    previous_root=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.ROOT,text=True).strip(),
    previous_parent=subprocess.check_output(['git','-C',str(s.ROOT.parent/'flow-matching-research'),'rev-parse','HEAD'],text=True).strip(),paths=sorted(paths)))
print(json.dumps(dict(complete_within_row_diagnostics=1,required_diagnostics=24,required_original_models=120,
    paired_leaveout_difference=a,descriptive_leaveout_fraction=d['leaveout_fraction'],files=len(paths))))
