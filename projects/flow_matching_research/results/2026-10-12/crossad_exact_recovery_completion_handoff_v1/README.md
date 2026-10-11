# CrossAD exact recovery completion handoff

The immutable original139-job completion controller reads canonical output
directories. Four preserved failed executions already have exact recovery
controllers writing new directories: released PSM and SMAP checkpoints, fresh
GECCO seed2025, and fresh PSM seed2025. Without this handoff those successful
future recovery outputs would remain outside its completion accounting.

The companion dispatches only these four recovery outputs after actual full
execution completes. It preserves all model/data/optimizer/batch/early-stop/
evaluator fields and numerical tolerances. Independent full raw-point audit,
native complete checkpoint/model/evaluator/SPOT re-execution, and independent
complete-array/label/row/metric review run through unchanged existing workers.
Only successful nonaborted proofs map back to the original canonical ID.
Released checkpoint evaluation contributes zero training seeds; a fresh retry
contributes the same original seed, never another independent repeat.

Its union retains all139 obligations, all7 fixed release evaluations,
all132 full fresh training runs and all90 reconstructed ablations. Original
controller status IDs alone do not certify completion: new entries must have
actual successful full native and independent post-review evidence. Existing
three full verified executions are preserved. Current coverage remains3/139;
all four recovery outputs are still pending. No new benchmark score is published.

113 relevant validation cases pass. The cold actual source/evidence probe
verifies frozen registration without importing NumPy or Torch in the waiting
parent. The companion and original recovery/training owners were checked by
actual PID, creation time and command. The45-paper,681-method/baseline/ablation,
273-original-data-task objective remains active and incomplete.

The original PowerShell cold-probe stdout is preserved as UTF16 text; the
parsed JSON copy is normalized to UTF8 without changing any captured value.
