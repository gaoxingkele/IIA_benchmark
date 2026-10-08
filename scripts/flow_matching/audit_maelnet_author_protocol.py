"""Anchor author-protocol differences to the frozen source and real preflight."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def main():
    config_path = ROOT / 'configs/experiments/fm_maelnet_author_execution.v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    preflight_path = ROOT / config['preflight_report']
    preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
    original = ROOT / config['source']
    evidence = []
    observations = [
        ('validation_overlap', 'data_provider/data_loader.py', 'self.val = self.train[(int)(data_len * 0.8):]',
         'Validation is the last20% of the same full array used for training, retained in all five actual native loader audits.'),
        ('flattened_overlapping_scores', 'exp/opt_rl2_anomaly.py', 'attens_energy_test = np.concatenate(attens_energy_test, axis=0).reshape(-1)',
         'Overlapping window scores are flattened without aggregation to one score per timestamp. Multiplicity and event geometry differ from strict pointwise evaluation.'),
        ('test_score_threshold', 'exp/opt_rl2_anomaly.py', 'combined_energy = np.concatenate([train_energy, threshold_energy], axis=0)',
         'Threshold calibration includes threshold-loader scores drawn from the test payload; this is not validation-only calibration.'),
        ('PA_label_dependent_state', 'utils/agentreward.py', 'new_gt,new_pred = adjustment(self.gtruth, pred_tmp)',
         'Ground-truth-dependent PA predictions are constructed before the environment states; their predictions become observation coordinate2.'),
        ('label_reward', 'utils/agentreward.py', 'if self.gtruth[self.time_step]==1:',
         'TP/FN/FP/TN rewards depend on test labels. The frozen reward values remain1,-1.5,-0.6,.01.'),
        ('train_evaluate_same_environment', 'exp/opt_rl2_anomaly.py', 'acc,prec, rec, f1, _, list_preds, reward =eval_model(model, env_off)',
         'DQN learns on env_off for the entire flattened test sequence and is evaluated on that same env_off.'),
        ('DQN_seed_not_explicit', 'exp/opt_rl2_anomaly.py', "model = DQN('MlpPolicy', env_off, verbose=1, tensorboard_log=path)",
         'The original DQN call does not pass seed. Top-level Python/NumPy/Torch seeds are retained, but do not establish deterministic SB3 initialization.'),
    ]
    for identifier, relative, token, observation in observations:
        path = original / relative
        lines = path.read_text(encoding='utf-8').splitlines()
        matches = [i for i, line in enumerate(lines, 1) if token in line]
        if not matches:
            raise ValueError('Author evidence anchor missing: ' + identifier)
        evidence.append({'id': identifier, 'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                         'lines': matches, 'observation': observation})
    decoder_reads = {}
    for model in ['AnomalyTransformer', 'DCDetector', 'MaelNetS2']:
        path = original / 'models' / (model + '.py')
        decoder_reads[model] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                                'd_layers_token_count_in_model_file': path.read_text(encoding='utf-8').count('d_layers')}
    report = {'author_repository': 'https://github.com/hpc-inspirasi/ModMaelNet', 'commit': config['source_commit'],
              'config_sha256': sha(config_path), 'preflight': config['preflight_report'], 'preflight_sha256': sha(preflight_path),
              'source_version_boundary': 'Pinned acquired author repository, publication-era commit and exact paper table-row mapping not established',
              'source_protocol_evidence': evidence, 'native_datasets': preflight['native_datasets'],
              'decoder_argument_audit': decoder_reads,
              'decoder_argument_inference': 'The three selected model files do not read d_layers. Matching parameter counts in preflight are consistent with repository decoder-depth recipes being naming changes rather than separate decoder architectures; paper architectural fidelity remains unproven.',
              'observed_label_dependence': {'diagnostic_inputs': '64-point interface probe, fixed scores and thresholds',
                                          'changed_PA_predictions_when_labels_change': preflight['test_label_dependent_PA_prediction_changes'],
                                          'benchmark_performance': False},
              'environment_deviations': preflight['environment'],
              'runtime_repairs': json.loads((ROOT / config['corrected_source'] / 'runtime_patch_manifest.json').read_text(encoding='utf-8'))['repairs'],
              'all_original_experiments_complete': False, 'paper_metrics_reproduced': False}
    path = ROOT / 'projects/flow_matching_research/results/2026-10-09/maelnet_author/protocol_audit.json'
    write_json(path, report)
    lines = ['# MaelNet官方流程执行与协议核查', '',
             '已冻结150个配方—数据集—种子任务、600个作者训练/RL阶段。完整运行尚未结束；18个真实输入、完整容量接口检查及64步DQN诊断不能作为性能成绩。', '',
             '## 保留的作者协议', '',
             '原作者代码将验证段放在完整训练数组内部；阈值包含测试数据分数；重叠窗口测试分数直接展开。环境先用测试标签做PA，状态包含PA预测；DQN以测试标签产生奖励，并在同一环境评价。',
             '该轨专门记录作者协议成绩，与验证集校准、无PA、每时间点一个分数的严格TAB轨分别解释。原始代码和数据均未覆盖。', '',
             '| 数据集 | 训练行×特征 | 测试行×特征 | stride | 训练窗口 | 测试窗口 | 作者展开测试点/DQN请求步数 |',
             '|---|---|---|---:|---:|---:|---:|']
    for dataset, data in preflight['native_datasets'].items():
        shape = lambda kind: '×'.join(str(v) for v in data['shapes'][kind])
        lines.append(f"| {dataset} | {shape('train')} | {shape('test')} | {data['stride']} | {data['window_counts']['train']} | {data['window_counts']['test']} | {data['flattened_test_points']} |")
    lines += ['', '## 修复与可比性边界', '',
              '- 修复chunk参数类型、非MSL加载器参数、CPU注意力张量、NumPy2的无穷常量及Gym观测dtype/范围，新增模型/分数/策略/指标留存。训练损失、容量、原stride、早停与RL奖励保持作者实现。',
              '- 实际环境为Python3.12与继承的Torch2.13.0+cu126，尚未对齐作者列出的Torch2.6.0。其他已装版本见protocol_audit.json。',
              '- 三个学习器模型文件没有读取d_layers；接口预检参数量也未随该参数改变。六套发布配方全部登记，仍需核查它们与原论文消融表的关系。',
              '- 64步RL测试只验证接口；改变诊断标签导致37个PA预测位置变化，证明这份代码的状态依赖标签，不能据此量化整篇论文性能偏高的幅度。',
              '- SWaT原生训练文件495000行与部分论文表的475200行差异保留；SMAP/MSL/SMD原生拼接数组不等同于严格轨的实体划分。', '',
              '[逐项源码行号和哈希](protocol_audit.json)，实际阶段日志、检查点与完整指标位于登记队列所引用的本地输出目录。', '']
    path.with_name('README.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps({'evidence_claims': len(evidence), 'native_datasets': len(preflight['native_datasets']), 'paper_metrics_reproduced': False}))


if __name__ == '__main__':
    main()
