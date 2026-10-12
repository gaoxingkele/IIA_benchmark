"""Fully audit and replay the newly completed original Pi source PSM seed1103."""
import argparse
import ast
import sys
from pathlib import Path

from scripts.flow_matching import run_pi_causal_completed_pair_replay_v1 as original

ROOT,read,digest=original.ROOT,original.read,original.digest
CANONICAL='pi_author__source_runtime__psm__full__seed1103'
RECOVERY=CANONICAL+'__shared_commit_full_retry_v6'
SCOPE=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)


def controller_ast(source):
    """Keep the complete existing point/native/full-array review controller."""
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='controller']
    if len(functions)!=1:raise ValueError('One complete original replay controller required')
    return ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[]))


def verify(config):
    for pin in config['source_receipts']:
        if digest(pin['path'])!=pin['sha256']:raise ValueError('Frozen full original Pi source changed: '+pin['path'])
    queue=read(config['original_queue'])
    if (len(queue['jobs'])!=1140 or len({j['id'] for j in queue['jobs']})!=1140
            or queue['settings']['seeds']!=[42,1103,1104,1105,1106,1107]
            or config['required_original_jobs']!=1140 or config['original_scope']!=SCOPE
            or [c['canonical_id'] for c in config['cases']]!=[CANONICAL]
            or config['additional_training_seed_slots']!=0
            or config['required_original_training_updates']!=1554
            or config['required_full_train_score_points']!=13238200
            or config['required_full_test_score_points']!=87800
            or config['required_full_saved_score_streams']!=15):
        raise ValueError('Keep all1140 original jobs, six seeds and every full newly completed source-model score')
    native_template=read(config['native_template']);review_template=read(config['review_template'])
    recovery=read(config['recovery_queue'])
    actual=next(c for c in recovery['jobs'] if c['job']['id']==RECOVERY)
    case=config['cases'][0];point=read(case['point_configuration']);native=read(case['native_configuration'])
    if (any(native[k]!=native_template[k] for k in original.GUARDS)
            or any(point[k]!=review_template[k] for k in ('python','settings','resource_lock'))
            or point['cpu_threads']!=native_template['cpu_threads']
            or native['score_tolerance']!=native_template['score_tolerance']
            or native['independent_completion']!=point['output_directory']+'/independent_full_recovery_completion.json'
            or point['recovery_queue']!=config['recovery_queue'] or native['recovery_queue']!=config['recovery_queue']
            or point['recovery_job_id']!=RECOVERY or native['recovery_job_id']!=RECOVERY):
        raise ValueError('Exact original full point/native/review guards, recovered identity and tolerance required')
    bound,job,result=original.completion.binding(point)
    canonical=next(j for j in queue['jobs'] if j['id']==CANONICAL)
    if (bound!=actual or job!=canonical or bound['original_job']!=canonical or bound['kind']!='pi'
            or bound['original_queue']!=config['original_queue'] or bound['settings']!=queue['settings']
            or job['seed']!=1103 or job['branch']!='source_runtime' or job['dataset']!='PSM' or job['recipe']!='full'
            or result['parameter_count']!=1930289 or result['active_attention_layers']!=1
            or [e['epoch'] for e in result['epochs']]!=[1,2,3]
            or any(e['steps']!=518 or e['early_stop'] for e in result['epochs'])
            or result['native_train_points']!=132481 or result['training_windows']!=132382
            or result['native_test_points']!=87841 or result['scored_test_points']!=87800
            or result['dropped_test_tail_points']!=41
            or job['author_config']['data']['batch_size']!=256 or job['author_config']['data']['win_size']!=100
            or job['author_config']['model']['d_model']!=512 or job['author_config']['training']['n_epochs']!=3):
        raise ValueError('Only the exact completed full-size original source PSM1103 recovery may be replayed')
    usage=read(bound['job']['output_directory']+'/resource_usage.json')
    if usage.get('resource_guard_aborted') is not False:
        raise ValueError('Actual original full training must have a nonaborted resource receipt')
    return queue,native_template,review_template


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(args.config);queue,_,_=verify(config)
    tree=controller_ast(Path(original.__file__).read_text(encoding='utf-8'))
    if args.probe_only:
        compile(tree,'<complete-original-point-native-review-controller>','exec')
        import json
        print(json.dumps(dict(passed=True,required_original_jobs=len(queue['jobs']),
            selected_newly_completed_original_id=CANONICAL,original_full_training_updates=1554,
            full_train_score_points_required=13238200,full_test_score_points_required=87800,
            all_original_score_streams_required=15,additional_training_seed_slots=0,
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
            full_native_checkpoint_reinference_passed=False,all_original_experiments_complete=False)));return
    namespace=dict(vars(original),verify=verify,IDS=[CANONICAL])
    exec(compile(tree,original.__file__,'exec'),namespace)
    namespace['controller'](config,args.config)


if __name__=='__main__':main()
