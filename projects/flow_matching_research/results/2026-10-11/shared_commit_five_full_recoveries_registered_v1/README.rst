Five full original experiment recoveries
=======================================

Four terminal memory-aborted original cases now have isolated full fresh retry
outputs: Pi-Transformer PSM1103, CrossAD PSM2025, CFM NODE Pendulum1106,
and TimeGAN Sine1103. Complete registries of13,139,285,45 cases and unchanged
original model/loss/optimizer/data/split/batch/train/evaluator budgets are
retained. No incomplete checkpoint is certified for continuation for these
four cases. Failed outputs are preserved. Retries retain canonical seeds and
add no independent seed observations.

RCGAN MNIST1104 is qualified for exact full-state continuation: all84 native
model/optimizer checkpoints restore bitwise; all81 persisted float32 loss rows
reproduce bitwise; the three missing loss observations are recomputed from
actual state, with original missing elapsed times still null; all4984 latest
generator samples and the next D/G optimizer transition reproduce bitwise.
Full NumPy/Python RNG and cumulative36000-row order chains are verified. The
416 remaining epochs retain the original total500, full1284 D/G updates per
epoch and complete native MMD/evaluator procedure. Original seed1103 training
remains owned by its existing controller. The seed1104 controller waits on
the same original family lock and unchanged guards.

The initial full84 state audit v2 failed because it retained seed1103's231-row
trace assumption. Its actual terminal failure and stderr are preserved.
Separate immutable v3 binds the actual81-row seed1104 trace and correctly
labels all84 checkpoints and the three missing observations; no numerical
tolerance, graph, optimizer or full sampling scope was weakened.

Current controller identity evidence verifies PID, creation time and complete
actual command. Controller liveness and capacity qualification are not
training completion. All five experiments remain incomplete at this capture;
no benchmark score or new independent seed is claimed. The45-paper,
681-method/baseline/ablation and273-original-data/task objective remains open.
Relevant validation:54 tests passed in9.18seconds.
