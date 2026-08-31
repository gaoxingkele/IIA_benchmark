# Claims

## C01: Partial-protocol numerical proximity does not establish paper-exact reproduction
- **Statement**: A result that is numerically close to a paper under a reduced author-artifact default does not establish paper-exact reproduction when named evaluation conditions or result rows are omitted.
- **Conditions**: Applies to the P0 alarm-flood studies whose public artifact default is a strict subset of the published evaluation protocol; authoritative container execution and independent same-fold comparison remain separate gates.
- **Sources**: [trace/sessions/2026-08-30_001.yaml:key_context «要把每篇论文的实验条件闭合到 P3：同一数据、同一预处理、同一切分、同一超参数、同一指标、同一参考表格。» [input]]
- **Status**: testing
- **Provenance**: user
- **Falsification**: Show that an allegedly reduced default actually executes every named paper condition and yields a one-to-one result for every target row without an additional wrapper or rerun.
- **Proof**: [N03, N07, ara/evidence/tables/p0_checkpoint_2026-08-30.md]
- **Dependencies**: []
- **Tags**: paper-exact, protocol, reproducibility, P3

## C02: Complete author-code computation is not sufficient for P3 closure
- **Statement**: Completing every scheduled author-code task establishes computational coverage, but does not establish paper-exact closure while a numeric, authoritative-environment, or independent-implementation gate remains open.
- **Conditions**: Applies to the P0 alarm-flood reproductions governed by the repository's frozen paper-exact protocol cards and non-interchangeable transfer/paper-exact result lanes.
- **Sources**: [docs/reports/p0_paper_exact_checkpoint_2026-08-31.md:7 «当前均为 **P2**，还不能标记为 P3：ConE 已实现论文数值闭合；CASIM 和 BiP 保留了未通过容差的负结果；三项都还缺归档 Docker 镜像内复跑以及本仓库独立实现的同折对照。» [result]]
- **Status**: testing
- **Provenance**: ai-suggested
- **Falsification**: Demonstrate that the repository's P3 definition accepts an author-code grid with any named numeric, container-equivalence, or independent same-fold gate unresolved.
- **Proof**: [N13, N14, N15, N17, ara/evidence/tables/p0_paper_exact_checkpoint_2026-08-31.md]
- **Dependencies**: [C01]
- **Tags**: paper-exact, computational-coverage, P3, reproducibility

## C03: Calibration/training overlap explains only part of the public BiP discrepancy
- **Statement**: In the public BiP artifact, removing calibration/random-forest training overlap changes prediction error and interval width without a uniform benefit, so overlap alone does not explain the paper-to-artifact discrepancy.
- **Conditions**: Supported for paired TEP and synthetic runs of MBW, EAC, and ACM under fixed Python-level randomness, and for CASIM when both Python-level and Numba RNG states are reset immediately before the single-estimator, single-process fit.
- **Sources**: [docs/reports/p0_bip_casim_numba_control_2026-08-31.md:5 «CASIM 的 Numba 随机性混杂已经闭合。在真实 Code Ocean TEP 与 synthetic 数据、相同五折索引和相同作者模型参数下，同时重置 Python-level NumPy 与 Numba RNG 后，overlap/disjoint 两条通道的 **10/10 test bifurcation 完全一致，最大绝对差由 32 降为 0**。» [result], docs/reports/p0_bip_casim_numba_control_2026-08-31.md:7 «这使 RF 训练子集效应可以在当前 `n_estimators=1`、`n_jobs_multirocket=1` 条件下作受控归因。结果不支持“删除 calibration/RF 重叠会统一改善性能”：TEP 的 point MAE 和区间宽度小幅改善，synthetic 的两项均变差；两者 coverage MAE 均仅小幅改善。» [result]]
- **Status**: testing
- **Provenance**: ai-suggested
- **Falsification**: Under controlled paired folds and controlled model randomness, show that removing overlap uniformly closes the frozen BiP paper gaps across the scoped models and datasets.
- **Proof**: [N15, N16, N20, N21, experiments/paper_harness/p0_paper_exact/run_3/paper_grid/summary.json]
- **Dependencies**: [C02]
- **Tags**: BiP, split-overlap, conformal, reproducibility
- **Last revised**: 2026-08-31 (2026-08-31_001#2)

## C04: Score-controlled same-fold comparison localizes conformal-layer error
- **Statement**: Holding base-classifier scores and folds fixed permits an independent conformal-layer comparison; exact paired agreement then localizes remaining implementation risk to score generation or end-to-end integration rather than conformal calibration.
- **Conditions**: Demonstrated for the official ConE-AFC synthetic protocol with author MBW-LR scores and the repository ConE calibrator; independent base classifiers, the full wrapper, and the archived container remain outside this claim.
- **Sources**: [docs/reports/p0_cone_independent_same_fold_2026-08-31.md:5 «本仓库独立 `ConEAFCCalibrator` 已在官方 18,750 条 synthetic alarm-flood 数据的完整 50 折上完成同折验证。使用与作者网格完全相同的 MBW-LR 基础分数、51 个时间前缀、3 个 alpha、3 个每类校准量时，coverage、average set size、singleton rate 和 empty rate 共 **1,800/1,800 个配对指标逐项完全一致，最大绝对差为 0**。» [result], docs/reports/p0_cone_independent_same_fold_2026-08-31.md:7 «这闭合了独立 conformal 校准与集合生成子门槛，但不是完整 P3：基础 MBW-LR 分数仍由作者实现生成，其余四个基础分类器和端到端独立 wrapper 尚未完成；本机也没有 Docker CLI。» [result]]
- **Status**: testing
- **Provenance**: ai-suggested
- **Falsification**: With identical finite base scores, class order, calibration rows, folds, and metric definitions, observe a nonzero paired discrepancy attributable to the independent conformal threshold or set-generation layer.
- **Proof**: [N22, ara/evidence/tables/p0_validation_continuation_2026-08-31.md, experiments/paper_harness/p0_paper_exact/run_2/independent_same_fold/mbw_lr/summary.json]
- **Dependencies**: [C01, C02]
- **Tags**: ConE, conformal, same-fold, implementation-attribution
