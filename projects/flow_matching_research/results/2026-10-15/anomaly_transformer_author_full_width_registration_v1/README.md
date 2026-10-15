Anomaly Transformer full-width author-runtime gap and live registration

Existing local strict/reconstruction configs use d_model=32,d_ff=32; the inspected pinned author model defaults are512/512. The completed local SMD seed1106 model has22750parameters and remains a local reconstruction, not author-pipeline certification. No existing source, raw data, frozen configuration or numerical worker was changed.

Registered all25 main dataset/seed slots: SMD,MSL,SMAP,PSM,SWaT x1103..1107. Twenty original-loader jobs have a callable CPU port of the actual author source. SMD retains10epochs and stride100; MSL/SMAP/PSM retain3epochs and stride1; window100,batch256,512-width three-layer model, both minimax backwards, Adam1e-4, original learning-rate schedule, patience3, temperature50, pooled calibration, PA and original omitted tails remain. Source test-based checkpoint selection is explicitly preserved. Only CUDA device/NumPy infinity compatibility and observation hooks are changed in isolated F-drive workspaces. Original author source remains byte-identical.

The five SWaT slots remain explicitly unavailable because this original snapshot has no SWaT loader. Local CSVs do exist. Their configured495000train rows differ from the paper475200rows; the19800-row discrepancy remains unresolved and is not silently trimmed.

The controller is actually live; the capture records its exact PID/create time and admission state. It requires32GiB physical and40GiB commit headroom and retains16GiB emergency protection. Waiting does not own the shared heavy lock. Failed/interrupted attempts are preserved, never blindly retried. Registration is not completed training.18 validation checks include a small full-width source training/saved-score fixture; fixture results are plumbing only and add zero benchmark results.

Independent checkpoint replay, multi-seed uncertainty, SWaT source/paper resolution, full-width paper ablations and hyperparameter studies, strict TAB and the full45/681/273 project scope all remain required. This new main-method lane does not redefine the overall objective.
