"""Publish passed full tau0.25 model replay without inventing new performance rows."""
import argparse
import copy
from datetime import datetime,timezone
import shutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=read(ROOT/args.config)
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:
            raise ValueError('Frozen actual full checkpoint proof changed')
    base=ROOT/config['base_publication'];previous=read(base/'publication_validation.json')
    for name,digest in previous['outputs'].items():
        if sha(base/name)!=digest:
            raise ValueError('Published numerical table changed')
    audit=read(ROOT/config['actual_full_model_replay']);record=audit['records'][0]
    if (not audit['full_checkpoint_contents_and_re_inference_passed'] or audit['full_job_count']!=1
            or record['canonical_id']!='grasp_protocol_v3__tau_0p25__swan__swan_sf__seed1103'
            or record['epochs_completed']!=1500 or record['optimizer_updates']!=93000
            or len(record['recipes'])!=1 or not record['full_checkpoint_restoration_and_reinference_passed']
            or audit['additional_training_seed_slots']!=0 or audit['author_equivalence_certified']):
        raise ValueError('Actual original canonical full checkpoint and budget required')
    recipe=record['recipes'][0]
    if (recipe['complete_validation_points']!=3935 or recipe['complete_test_points']!=72000
            or recipe['forward_evaluation_counts']['validation']!=dict(count=800,windows=194300)
            or recipe['forward_evaluation_counts']['test']!=dict(count=14100,windows=3597550)
            or recipe['prediction_disagreements']!=0 or not recipe['full_recipe_reinference_passed']):
        raise ValueError('Full original scoring range and classification agreement required')
    for path_key,digest_key in [('checkpoint_path','checkpoint_sha256'),('result_path','result_sha256')]:
        if sha(ROOT/record[path_key])!=record[digest_key]:raise ValueError('Actual trained model changed')
    for field in ['replay_score','original_score']:
        if sha(recipe[field+'_path'])!=recipe[field+'_sha256']:raise ValueError('Actual full inference arrays changed')
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Preserve every immutable publication')
    target.mkdir(parents=True)
    for name in previous['outputs']:
        if name!='README.md':shutil.copyfile(base/name,target/name)
    now=datetime.now(timezone.utc).isoformat()
    (target/'README.md').write_text(
        '# 完整算法—数据集实验指标\n\n'
        +f'发布：{now}。继承v11全部原始数值与采集时间，本次增加GRASP τ=0.25完整检查点重推断证据。\n\n'
        +'主表7117条指标记录：1786条有实测值，5331条缺结果。指标行数不等于训练次数；全部原论文实验仍未完成。\n\n'
        +'[可搜索完整表](results.html)；[完整CSV](all_algorithm_dataset_metrics.csv)；[完整可读表](all_metrics.md)；[异常检测全部指标](strict_algorithm_dataset.md)；[全部F1矩阵](overview.md)。\n\n'
        +'GRASP τ=0.25/SWAN/seed1103实际完成1500轮、93000次优化更新；原检查点恢复后，3935个验证点与72000个测试点全部重新推断。观察到800次验证模型调用与14100次测试模型调用，覆盖194300与3597550次窗口计算。\n\n'
        +'验证第99百分位阈值、无PA的Precision/Recall/F1分别为38.7339%/83.8308%/52.9858%，与原运行分类完全一致。回顾性测试标签Best-F1为53.0506%；原生PRC和AUROC重算差异分别约4.0e-9和1.9e-9。CPU/GPU分数满足预注册数值容差，不声称分数逐位一致。\n\n'
        +'[完整模型重推断证据](../grasp_tau025_checkpoint_replay_v1/checkpoint_replay_audit.json)；[后续完整范围执行记录](../checkpoint_full_scope_resume_v3/README.md)。\n\n'
        +'这仍是1/10训练种子，重推断不增加训练种子、样本量或新的性能记录。作者等价和完整TAB认证未取得。原先13个训练模型中的7个完整重推断已保留，剩余6个模型24个配方继续执行；GiFlow保留9个完整重推断批次并继续其余5个批次。尚未完成的范围不能算通过。\n\n'
        +'新接入的USHCN100次完整500轮任务另有120条待完成聚合指标，见[待完成清单](../ls4_native_ushcn_training_execution_v1/pending_native_metrics.csv)。它们不以数据审计或容量诊断代替训练结果。\n',encoding='utf-8')
    validation=copy.deepcopy(previous)
    validation.update(published_utc=now,base_publication_utc=previous['published_utc'],newly_measured_metric_rows=0,
        newly_completed_tau025_model_reinference_complete=True,performance_rows_changed=False,
        additional_training_seed_slots=0,all_original_experiments_complete=False)
    validation['source_receipts']+=config['source_receipts']+[dict(path=args.config,sha256=sha(ROOT/args.config))]
    validation['outputs']={p.name:sha(p) for p in target.iterdir() if p.is_file()}
    write(target/'publication_validation.json',validation)
    print(dict(rows=validation['aggregate_metric_rows'],measured=validation['measured_metric_rows'],
        full_tau025_model_replay=True,new_training_seeds=0,new_performance_rows=0))


if __name__=='__main__':main()
