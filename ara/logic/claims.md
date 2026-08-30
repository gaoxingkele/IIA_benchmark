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
