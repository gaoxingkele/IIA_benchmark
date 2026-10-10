"""Keep author evaluation protocols and evaluator repeats separate from seed results."""
from collections import defaultdict
from datetime import datetime, timezone

from scripts.flow_matching.capture_result_listing_v4 import ROOT, read, sha


def author_rows(report, base):
    groups = defaultdict(list)
    for job in report['author_pipeline_jobs']:
        groups[job['algorithm'], job['dataset'], job['author_recipe']].append(job)
    rows = []
    for (algorithm, dataset, recipe), jobs in groups.items():
        values = defaultdict(list)
        for job in jobs:
            if job['status'] != 'completed':
                continue
            if sha(ROOT / job['result_path']) != job['result_sha256']:
                raise ValueError('Author result changed since capture')
            metrics = job['metrics']
            if algorithm == 'pi_transformer_historical_mirror':
                for protocol, source in [('author_threshold_PA', metrics),
                                         ('same_author_threshold_no_PA', metrics['pointwise_no_PA_same_author_threshold'])]:
                    for key in ('precision', 'recall', 'f1', 'auroc', 'average_precision'):
                        if source.get(key) is not None:
                            values[protocol, key].append(source[key])
            elif algorithm == 'crossad_complete':
                original = metrics['original_author']
                best = original['_best_f1'][0]
                for key, field in [('precision', 'P_best'), ('recall', 'R_best'), ('f1', 'F1_best')]:
                    values['release_checkpoint_test_label_oracle', key].append(best[field])
                for key, field in [('auroc', 'AUC_ROC'), ('average_precision', 'AUC_PR')]:
                    values['release_checkpoint_author_evaluation', key].append(original['_auc'][0][field])
                for source, keys in [(original['_vus'][0], ('VUS_ROC', 'VUS_PR')),
                                     (original['_affiliation'][0], ('Affiliation_F1',))]:
                    for key in keys:
                        values['release_checkpoint_author_evaluation', key].append(source[key])
                control = metrics['native_validation_control']['strict']['1']['micro']
                for key in ('precision', 'recall', 'f1', 'auroc', 'average_precision'):
                    if control.get(key) is not None:
                        values['release_checkpoint_validation99_no_PA', key].append(control[key])
            else:
                raise ValueError('New author method requires metric protocol review: ' + algorithm)
        for (protocol, metric), samples in values.items():
            # A second author seed requires a reviewed aggregation rather than
            # silently treating repeated published-weight evaluations as training seeds.
            if len(samples) != 1:
                raise ValueError('New author repetitions require uncertainty review')
            rows.append(dict(task='anomaly_detection', algorithm=algorithm + '/' + recipe,
                dataset=dataset, protocol=protocol, metric=metric, n=1,
                required_repeats=len(jobs), mean=samples[0], std=None, se=None,
                ci95_low=None, ci95_high=None, paper_mean=None,
                author_equivalence_certified=False, uncertainty_scope='single completed author seed or fixed checkpoint',
                comparison='separate_author_protocol_not_strict_leaderboard',
                captured_utc=report['tsad_capture_utc'], source=base + '/author_pipeline_jobs.csv'))
    return rows


def table45_rows(config):
    from scripts.flow_matching.run_spectral_table45_original_v1 import verify, verify_result
    path = config['table45_queue']
    queue = read(ROOT / path)
    verify(queue)
    rows, jobs = [], []
    stamp = datetime.now(timezone.utc).isoformat()
    for job in queue['jobs']:
        model = read(ROOT / job['model_config'])
        result_path = ROOT / job['output_directory'] / 'result.json'
        result = verify_result(queue, job) if result_path.exists() else None
        args = model['author_arguments']
        # Each released command trains ONE generator. Its five/ten evaluator
        # repeats must not create between-training-seed uncertainty.
        rows.append(dict(task=model['task'], algorithm='Spectral-author/' + model['method'],
            dataset=model['dataset'], protocol='table' + str(model['table']) + '_original/' + model['data_audit_key'],
            epochs=args['epochs'], missing_ratio=args['missing_value'],
            score_recipe='w_eig=' + str(args['w_eig']) if 'w_eig' in args else '', metric=model['metric'],
            n=1 if result else 0, required_repeats=1, mean=result['mean'] if result else None,
            std=None, se=None, ci95_low=None, ci95_high=None,
            evaluator_repeats=model['evaluator_repeats'],
            evaluator_std=result['std'] if result else None,
            evaluator_se=result['se'] if result else None,
            evaluator_ci95=result['ci95'] if result else None,
            uncertainty_scope='one generator seed10; evaluator spread is not training-seed uncertainty',
            paper_mean=model['paper_claim']['mean'],
            paper_spread=model['paper_claim']['reported_plus_minus'],
            comparison='released_full_pipeline_uncertified' if result else 'no_complete_result',
            captured_utc=stamp, source=path, author_equivalence_certified=False))
        jobs.append(dict(job, queue=path, method=model['method'], dataset=model['dataset'],
            status='completed' if result else 'no_complete_result',
            result_path=result_path.relative_to(ROOT).as_posix() if result else None,
            result_sha256=sha(result_path) if result else None))
    return rows, jobs


def overview(report, target):
    from scripts.flow_matching.publish_result_listing_v4 import table
    rows = report['strict_summary']
    models = sorted({r['algorithm_config'] for r in rows})
    index = {(r['algorithm_config'], r['dataset']): r for r in rows}
    lines = ['# 算法与数据集 F1 总表', '',
        '异常检测源快照 UTC：' + report['tsad_capture_utc'], '',
        'F1 使用百分数。验证分数第99百分位定阈值，无PA；非重叠窗口预算未获作者等价认证。', '',
        '单元格为均值 [完整种子/预定种子]。—表示已登记但暂无完整运行；未登记表示该配置没有对应实验槽。固定预训练权重1/1不表示重新训练。', '',
        '[全部技术指标](strict_algorithm_dataset.md)；[全任务全协议指标](results.html)；[全量CSV](all_algorithm_dataset_metrics.csv)。']
    columns = [('原论文常用数据集', ['PSM', 'SMAP', 'MSL', 'SMD', 'SWAT']),
               ('扩展数据集', ['SWAN', 'CICIDS']),
               ('工业与过程数据集', ['TEP_CLASSIC', 'SKAB', 'PRONTO_DAY2', 'PRONTO_DAY3', 'PRONTO_DAY4'])]
    covered = set()
    for title, datasets in columns:
        values = []
        for model in models:
            cells = []
            for dataset in datasets:
                row = index.get((model, dataset))
                if row is None:
                    cells.append('未登记')
                else:
                    covered.add((model, dataset))
                    cells.append('—' if not row['completed_seeds'] else
                        f"{100*row['f1_mean']:.2f} [{row['completed_seeds']}/{row['required_seeds']}]")
            values.append([model] + cells)
        lines += ['', '## ' + title, '', table(['算法配置'] + datasets, values)]
    if covered != set(index):
        raise ValueError('F1 overview lost a registered dataset or configuration')
    return '\n'.join(lines) + '\n'
