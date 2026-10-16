"""Complete120-model Multiome row aggregation with the original legacy counters."""
import argparse
import ast
import itertools
import math
from pathlib import Path
import sys
from types import FunctionType
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import run_otcfm_singlecell_native_v4 as native
from scripts.flow_matching import audit_otcfm_singlecell_full_replay_v2 as replay
from scripts.flow_matching.otcfm_singlecell_result_integrity_v2 import inspect
from scripts.flow_matching.otcfm_EB_CITE_result_integrity_v1 import full_metric_names
from scripts.flow_matching import summarize_otcfm_EB_CITE_full150_paper_rows_v1 as generic

MODULE='scripts.flow_matching.summarize_otcfm_Multiome_full120_paper_rows_v1'
CONFIG='configs/reproducibility/fm_otcfm_Multiome_all120_full_paper_rows.v1.json'
TEST='tests/test_fm_otcfm_Multiome_full120_paper_rows_v1.py'
TARGETS=dict(cfm=dict(paper_name='I-CFM',mean=1.085,std=.099),
    otcfm=dict(paper_name='OT-CFM',mean=.937,std=.054),sbcfm=dict(paper_name='SB-CFM',mean=1.129,std=.363))


def registration():
    q=read(ROOT/native.QUEUE);native.verify(q);replay.verify(read(ROOT/replay.CONFIG))
    datasets=sorted({j['dataset'] for j in q['jobs']});groups=[]
    if datasets!=['independent_public_donor13176','released_author_donor32606']:
        raise ValueError('Both complete original Multiome input tracks required')
    for dataset,protocol,method in itertools.product(datasets,q['protocol_tracks'],q['methods']):
        jobs=[j for j in q['jobs'] if (j['dataset'],j['protocol'],j['method'])==(dataset,protocol,method)]
        if len(jobs)!=10 or {(j['seed'],j['leaveout_timepoint']) for j in jobs}!=set(itertools.product(range(42,47),[1,2])):
            raise ValueError('All five seeds and both independently trained leaveouts required')
        groups.append(dict(id=dataset+'__'+protocol+'__'+method,dataset=dataset,protocol=protocol,method=method,
            leaveout_indices=[1,2],required_case_ids=[j['id'] for j in jobs],paper_target=TARGETS[method],
            historical_paper_input_identity_certified=False))
    policy='configs/reproducibility/fm_common105_matched577_all1726_labels.2026-10-16_v1.json'
    paths=[native.QUEUE,replay.CONFIG,MODULE.replace('.','/')+'.py',TEST,policy,
        generic.MODULE.replace('.','/')+'.py',generic.template.MODULE.replace('.','/')+'.py',
        'scripts/flow_matching/otcfm_singlecell_result_integrity_v2.py',
        'scripts/flow_matching/audit_cfm_full_arrays_v1.py','papers/literature/flow_matching/pdfs/ot_cfm.pdf',
        'papers/literature/flow_matching/extracted/ot_cfm.txt']
    return dict(groups=groups,required_rows=12,required_distinct_native_models=120,source_queue=native.QUEUE,
        native_replay_configuration=replay.CONFIG,source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
        paper_citation=q['citation'],paper_target_table=4,paper_PDF_page_one_based=11,
        mean_aggregation='equal mean of two separately trained leaveouts within seed,then equal mean over five seeds',
        uncertainty='sampleSTD/t95df4 across five seed-level leaveout means;all10model spread separately',
        author_reported_SD_aggregation_certified=False,no_table_result_from_partial_rows=True,
        all_source_and_full_native_guards_required=True,no_test_based_recipe_selection=True,
        settings=read(ROOT/policy)['settings'],resource_lock=read(ROOT/policy)['resource_lock'],numerical_threads=2,python=sys.executable,
        state_root='experiments/runs/fm_otcfm_Multiome_all120_full_paper_rows_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/otcfm_Multiome_all120_full_paper_rows_v1',
        original_scope=q['original_scope'],all_original_experiments_complete=False,
        boundary='All120original Multiome source/native conditions in12complete method/recipe/input rows. '
            'Require all10models per row,all36native metrics and every full two-timepoint W1/W2/independentW1 LP proof. '
            'The legacy source optimizer_updates is selected post-test checkpoint count;actual training updates remain in full epoch_history. '
            'Verify both against actual native checkpoint proof without rewriting original results or synthesizing newer-schema fields. '
            'Released33701cells/donor32606 and independent35384cells/donor13176 remain separate. '
            'Donor agreement alone does not certify historical matrix/PCA/model identity. '
            'Both recipes and spread definitions remain separate;Table4 SD convention is uncertified. '
            'Metadata-only aggregation,no extra training/inference or LP execution;all original guards/tolerances and420defined singlecell/all45paper obligations remain.')


def verify(c):
    if c!=registration():raise ValueError('Every original120condition and all12complete rows required')


def validate_native(job,source,provenance,r,g,report_sha,config_sha):
    sizes=list(job['data']['days'].values());geometry=[[sizes[i],sizes[i+1]] for i in range(2)]
    if len(sizes)!=4:raise ValueError('All four complete original timepoint populations required')
    if (not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['full_replay_report_sha256']!=report_sha
            or r['status']!='passed' or r['id']!=job['id'] or r['configuration_sha256']!=config_sha
            or r['source_provenance']['result_sha256']!=provenance['result_sha256']
            or r['full_training_epochs']!=source['actual_completed_epochs']
            or r['full_training_optimizer_updates']!=source['epoch_history'][-1]['optimizer_updates']
            or r['original_posttest_selected_checkpoint_updates']!=source['optimizer_updates']
            or r['selected_checkpoint_updates']!=source['optimizer_updates']
            or not r['original_result_update_counter_semantics_corrected_without_overwrite']
            or not r['all_learned_tensors_and_optimizer_moments_checked'] or not r['full_native_test_reinference_passed']
            or r['extra_model_training_updates']!=0 or r['native_metric_tolerance']!=dict(rtol=2e-6,atol=2e-6)
            or set(r['native_metrics'])!=full_metric_names(4) or r['original_metrics']!=source['test_metrics']):
        raise ValueError('Every original full native/guard/weight/history proof and both legacy counters required')
    if any(not math.isclose(v,source['test_metrics'][k],rel_tol=2e-6,abs_tol=2e-6) for k,v in r['native_metrics'].items()):
        raise ValueError('All original36native scalar tolerances remain fixed')
    distances=r['full_test_population_distances_independently_recomputed']
    if len(distances)!=1:raise ValueError('One complete original test invocation required')
    solves=distances[0]['solves']
    if (len(solves)!=4 or {(s['timepoint'],s['power']) for s in solves}!=set(itertools.product([1,2],[1,2]))
            or any(s['native_LP_result_code']!=1 or s['full_cost_shape']!=geometry[s['timepoint']-1] for s in solves)):
        raise ValueError('Both complete timepoint populations and all four OPTIMAL W1/W2 solves required')
    for s in solves:
        if s['power']==1 and (not s['full_costs_independently_recomputed'] or s['independent_LP_result_code']!=1
                or not math.isfinite(s['objective_absolute_difference']) or not math.isfinite(s['LP_objective_perturbation_bound'])
                or not 0<=s['objective_absolute_difference']<=s['LP_objective_perturbation_bound']):
            raise ValueError('Each full independent W1 matrix and objective perturbation bound required')


def worker(c,group):
    namespace=dict(generic.worker.__globals__,native=native,replay=replay,inspect=inspect,
        validate_native=validate_native,CONFIG=CONFIG,MODULE=MODULE)
    return FunctionType(generic.worker.__code__,namespace)(c,group)


def controller_tree():
    tree=generic.controller_tree()
    class Scope(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='waiting_all15_source_and_native_conditions_per_row':n.value='waiting_all10_source_and_native_conditions_per_row'
            return n
    return ast.fix_missing_locations(Scope().visit(tree))


def controller(c):
    namespace=dict(generic.template.controller.__globals__,MODULE=MODULE,CONFIG=CONFIG,native=native,replay=replay,worker=worker)
    exec(compile(controller_tree(),__file__,'exec'),namespace);namespace['controller'](c)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--group-id');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if a.group_id:worker(c,next(g for g in c['groups'] if g['id']==a.group_id))
    else:controller(c)


if __name__=='__main__':main()
