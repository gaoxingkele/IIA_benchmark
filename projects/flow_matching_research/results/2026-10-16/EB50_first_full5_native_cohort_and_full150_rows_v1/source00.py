"""Require complete150 source/native proofs before twelve EB/CITE row summaries."""
import argparse
import ast
from copy import deepcopy
import inspect as pyinspect
import itertools
import math
from pathlib import Path
import statistics
import sys
import types

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import summarize_otcfm_EB100_full_paper_rows_v1 as template
from scripts.flow_matching import run_otcfm_EB_CITE_native150_v2 as native
from scripts.flow_matching import audit_otcfm_EB_CITE_full150_replay_v1 as replay
from scripts.flow_matching.otcfm_EB_CITE_result_integrity_v1 import inspect, full_metric_names

MODULE = 'scripts.flow_matching.summarize_otcfm_EB_CITE_full150_paper_rows_v1'
CONFIG = 'configs/reproducibility/fm_otcfm_EB_CITE_all150_full_paper_rows.v1.json'
TEST = 'tests/test_fm_otcfm_EB_CITE_full150_paper_rows_v1.py'
TARGETS = {
    'released_author_EB': template.TARGETS,
    'released_author_CITE': {
        'cfm': dict(paper_name='I-CFM', mean=.965, std=.111),
        'otcfm': dict(paper_name='OT-CFM', mean=.882, std=.058),
        'sbcfm': dict(paper_name='SB-CFM', mean=1.067, std=.107)}}


def registration():
    q = read(ROOT/native.QUEUE); native.verify(q)
    replay.verify(read(ROOT/replay.CONFIG))
    groups = []
    for dataset, protocol, method in itertools.product(TARGETS, native.contract.PROTOCOLS, native.contract.METHODS):
        jobs = [j for j in q['jobs'] if (j['dataset'],j['protocol'],j['method']) == (dataset,protocol,method)]
        leaveouts = [1,2,3] if dataset == 'released_author_EB' else [1,2]
        if len(jobs) != len(leaveouts)*5 or {(j['seed'],j['leaveout_timepoint']) for j in jobs} != set(itertools.product(range(42,47),leaveouts)):
            raise ValueError('All original seeds and separately trained leaveouts required')
        groups.append(dict(id=dataset+'__'+protocol+'__'+method, dataset=dataset, protocol=protocol,
            method=method, leaveout_indices=leaveouts, required_case_ids=[j['id'] for j in jobs],
            paper_target=TARGETS[dataset][method], historical_paper_input_identity_certified=False))
    policy = 'configs/reproducibility/fm_common105_matched577_all1726_labels.2026-10-16_v1.json'
    paths = [native.QUEUE,replay.CONFIG,MODULE.replace('.','/')+'.py',TEST,policy,
        template.MODULE.replace('.','/')+'.py','scripts/flow_matching/otcfm_EB_CITE_result_integrity_v1.py',
        'scripts/flow_matching/audit_cfm_full_arrays_v1.py',
        'papers/literature/flow_matching/pdfs/ot_cfm.pdf','papers/literature/flow_matching/extracted/ot_cfm.txt']
    return dict(groups=groups,required_rows=12,required_distinct_native_models=150,source_queue=native.QUEUE,
        native_replay_configuration=replay.CONFIG,source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
        paper_citation=q['citation'],paper_target_table=4,paper_PDF_page_one_based=11,
        mean_aggregation='equal mean over independently trained leaveouts within seed;equal mean over five seeds',
        uncertainty='sampleSTD and Student-t95df4 over five seed-level leaveout means;all-model spread separately',
        author_reported_SD_aggregation_certified=False,no_table_result_from_partial_rows=True,
        all_source_and_full_native_guards_required=True,no_test_based_recipe_selection=True,
        settings=read(ROOT/policy)['settings'],resource_lock=read(ROOT/policy)['resource_lock'],
        numerical_threads=2,python=sys.executable,
        state_root='experiments/runs/fm_otcfm_EB_CITE_all150_full_paper_rows_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/otcfm_EB_CITE_all150_full_paper_rows_v1',
        original_scope=q['original_scope'],all_original_experiments_complete=False,
        boundary='Metadata-only summaries of all150original source/full-native models,not new model training or inference. '
            'Each EB row requires all15models,each CITE row all10models;all36/44metrics and every full timepoint W1/W2 solve. '
            'Released EB18203cells/50PC is separate from original-code EB16819cells/100PC. '
            'Released CITE31240cells is donor32606,not paper-described13176. Numeric paper comparisons cannot certify historical reproduction. '
            'The paper SD convention is uncertified;both recipes and all-model spreads remain separate. '
            'Original270,EB10090,deferredGEO60 and entire45paper/681/273/702scope remain required. '
            'Reuse existing metadata-only scalar policy without changing any native32GiB gate or numerical tolerances.')


def verify(c):
    if c != registration(): raise ValueError('All12rows and150full source/native conditions must remain fixed')


def summarize(records, group):
    from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
    leaveouts = group['leaveout_indices']
    expected = set(itertools.product(range(42,47),leaveouts))
    if (len(records)!=len(expected) or len({r['id'] for r in records})!=len(expected)
            or {r['id'] for r in records}!=set(group['required_case_ids'])
            or {(r['seed'],r['leaveout_timepoint']) for r in records}!=expected
            or any((r['dataset'],r['protocol'],r['method'])!=(group['dataset'],group['protocol'],group['method']) for r in records)
            or not all(math.isfinite(r['heldout_W1']) and r['heldout_W1']>=0 for r in records)):
        raise ValueError('Missing,duplicated or mixed conditions cannot become a paper comparison row')
    means = [math.fsum(r['heldout_W1'] for r in records if r['seed']==seed)/len(leaveouts) for seed in range(42,47)]
    values = [r['heldout_W1'] for r in records]
    stats = independent_stats(means)
    if not math.isclose(stats['mean'],math.fsum(values)/len(values),rel_tol=1e-14,abs_tol=1e-14):
        raise ValueError('Every original leaveout and seed needs equal weight')
    return dict(dataset=group['dataset'],method=group['method'],protocol=group['protocol'],
        independently_trained_models=len(values),training_seeds=5,leaveouts_per_seed=len(leaveouts),
        seed_level_leaveout_means=means,seed_mean_summary=stats,degrees_of_freedom=4,
        all_model_sample_STD=statistics.stdev(values),paper_target=group['paper_target'],
        mean_difference_from_paper=stats['mean']-group['paper_target']['mean'],
        author_reported_SD_aggregation_certified=False,historical_paper_equivalence_certified=False,
        comparison='FULL_SOURCE_AND_NATIVE_COMPLETE_NUMERIC_COMPARISON_ONLY',all_original_experiments_complete=False)


def validate_native(job, source, provenance, report, guard, report_sha, config_sha):
    sizes = list(job['data']['days'].values()); count = len(sizes)-2
    geometry = [list(pair) for pair in replay.interior_geometry(sizes)]
    if (not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or guard['full_replay_report_sha256']!=report_sha
            or report['status']!='passed' or report['id']!=job['id'] or report['configuration_sha256']!=config_sha
            or report['source_provenance']['result_sha256']!=provenance['result_sha256']
            or report['full_training_epochs']!=source['actual_completed_epochs']
            or report['full_training_optimizer_updates']!=source['optimizer_updates']
            or report['original_posttest_selected_checkpoint_updates']!=source['posttest_selected_checkpoint_optimizer_updates']
            or not report['source_total_and_selected_optimizer_counters_separately_verified']
            or not report['all_learned_tensors_and_optimizer_moments_checked']
            or not report['full_native_test_reinference_passed'] or report['extra_model_training_updates']!=0
            or report['native_metric_tolerance']!=dict(rtol=2e-6,atol=2e-6)
            or set(report['native_metrics'])!=full_metric_names(len(sizes))
            or report['original_metrics']!=source['test_metrics']
            or report['complete_interior_timepoint_cost_shapes']!=geometry):
        raise ValueError('Every complete source/native/guard/weight/history/geometry proof required')
    if any(not math.isclose(v,source['test_metrics'][k],rel_tol=2e-6,abs_tol=2e-6) for k,v in report['native_metrics'].items()):
        raise ValueError('Every original native scalar tolerance remains fixed')
    distances = report['full_test_population_distances_independently_recomputed']
    if len(distances)!=1: raise ValueError('One complete native test invocation required')
    solves = distances[0]['solves']
    if (len(solves)!=2*count or {(s['timepoint'],s['power']) for s in solves}!=set(itertools.product(range(1,count+1),[1,2]))
            or any(s['native_LP_result_code']!=1 or s['full_cost_shape']!=geometry[s['timepoint']-1] for s in solves)):
        raise ValueError('Every full interior-timepoint W1/W2 OPTIMAL solve is required')
    for s in solves:
        if s['power']==1 and (not s['full_costs_independently_recomputed'] or s['independent_LP_result_code']!=1
                or not math.isfinite(s['objective_absolute_difference']) or not math.isfinite(s['LP_objective_perturbation_bound'])
                or not 0<=s['objective_absolute_difference']<=s['LP_objective_perturbation_bound']):
            raise ValueError('Every full independent W1 matrix/OPTIMAL/objective bound required')


availability = template.availability
guard_paths = template.guard_paths


def worker(c, group):
    q = read(ROOT/native.QUEUE); rc = read(ROOT/replay.CONFIG)
    records, proofs = [], []
    for identity in group['required_case_ids']:
        job = next(j for j in q['jobs'] if j['id']==identity)
        if availability(q,rc,job) is not True: raise ValueError('Wait for every actual source and native process guard')
        result, provenance = inspect(q,job)
        _, guard_path, report_path = guard_paths(q,rc,job)
        report, guard = read(report_path), read(guard_path)
        validate_native(job,result,provenance,report,guard,sha(report_path),sha(ROOT/replay.CONFIG))
        for item in report['all_full_test_arrays_and_artifacts']:
            p = Path(item['path'])
            if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']: raise ValueError('Every original replay artifact must remain immutable')
        records.append(dict(id=identity,dataset=job['dataset'],method=job['method'],protocol=job['protocol'],
            seed=job['seed'],leaveout_timepoint=job['leaveout_timepoint'],heldout_W1=result['primary_heldout_W1']))
        proofs.append(dict(id=identity,source_provenance=provenance,native_report_sha256=sha(report_path),native_guard_sha256=sha(guard_path)))
    summary = summarize(records,group)
    out = Path(c['artifact_root'])/group['id']; out.mkdir(parents=True,exist_ok=False)
    exclusive(out/'complete_paper_row.json',dict(passed=True,**summary,records=records,proofs=proofs,
        configuration_sha256=sha(ROOT/CONFIG),extra_training_or_inference=False,source_artifacts_modified=False))


def controller_tree():
    tree = ast.parse(pyinspect.getsource(template.controller))
    class Scope(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value==6: node.value=12
            elif node.value=='all6_paper_row_outcomes_terminal_other_original_work_remains':
                node.value='all12_paper_row_outcomes_terminal_other_original_work_remains'
            return node
    return ast.fix_missing_locations(Scope().visit(tree))


def controller(c):
    namespace = dict(template.controller.__globals__)
    namespace.update(MODULE=MODULE,CONFIG=CONFIG,native=native,replay=replay,worker=worker)
    exec(compile(controller_tree(),__file__,'exec'),namespace)
    namespace['controller'](c)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true');parser.add_argument('--group-id');args=parser.parse_args()
    if args.register: exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if args.group_id: worker(c,next(g for g in c['groups'] if g['id']==args.group_id))
    else: controller(c)


if __name__=='__main__': main()
