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
- **Conditions**: Supported for paired TEP and synthetic runs of MBW, EAC, and ACM under fixed Python-level randomness; CASIM is excluded from causal attribution until its Numba feature-sampling RNG is controlled.
- **Sources**: [docs/reports/p0_bip_paper_grid_complete_2026-08-31.md:72 «因此，无重叠修正使 TEP MBW 的 point MAE 和区间宽度改善，并使 Table 4 MBW 进入容差，但会使 TEP EAC 的 point/coverage MAE 变差；不存在统一收益。» [result], docs/reports/p0_bip_paper_grid_complete_2026-08-31.md:74 «CASIM 的 10/10 受控配对仍出现 test bifurcation 差异，最大单折差为 32。根因证据位于 vendored `CASIM_multirocket.py`：`_fit_multi` 和 bias 采样调用 Numba `np.random`，而 `MultiRocketMultivariate` 构造器不接收 Arsenal 传入的 `random_state`。因此 CASIM 的 overlap/disjoint 数值只能作描述性比较，不能作纯切分因果估计。» [result]]
- **Status**: testing
- **Provenance**: ai-suggested
- **Falsification**: Under controlled paired folds and controlled model randomness, show that removing overlap uniformly closes the frozen BiP paper gaps across the scoped models and datasets.
- **Proof**: [N15, N16, experiments/paper_harness/p0_paper_exact/run_3/paper_grid/summary.json]
- **Dependencies**: [C02]
- **Tags**: BiP, split-overlap, conformal, reproducibility
