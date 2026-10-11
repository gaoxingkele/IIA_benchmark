# All ten original PhysioNet models: strict patient discrepancy investigation

The original full CPU replay exited 1 after its first patient-statistic comparison failed. Its full native global metric checks had passed before that check. The failure is preserved; CPU/GPU drift or another cause has not been established.

Both methods retain all original seeds 26–30, all 2398 patients × 48 hours × 37 features and 82605 held-out coordinates per model, the selected full checkpoints, original training/optimizer/RNG completion checks, pinned numerical environment, 4 CPU threads, family lock, 16 GiB physical / 28 GiB commit gate and existing 2e-6 absolute/relative tolerance.

The new native controller records numeric patient discrepancies and continues all full models. Identity, coordinate, checkpoint, finite-array, global metric and resource failures remain fatal. A single patient discrepancy leaves the full reproduction flag and receipt failed. Complete prediction arrays are saved for a separate independent full HDF5/patient-coordinate audit, which also keeps any patient mismatch failed. Native worker exit 0 denotes finishing inference, not certification.

81 relevant tests pass, including full-source AST reversal and rejection of fabricated completion. Controllers were verified by actual PID/create time/command/CPU; registration or waiting is not execution completion. No leaderboard row, extra training seed, original BRITS2018, author-paper or TAB certification is added. All 45 papers / 681 method records / 273 original-data tasks remain in scope.
