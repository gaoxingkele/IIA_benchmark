# CPU replay lock clarification: 2026-10-15

The native BRITS seed27 replay used four CPU threads and did not initialize CUDA.
Its registered shared lock was `experiments/runs/fm_saits_readonly_checkpoint_cpu.lock`.
The earlier training-complete/pending-replay README incorrectly called this a GPU
lock. The captured `waiting_for_heavy_lock` state alone did not identify a GPU
lock or establish GRASP as its owner. That earlier attribution is withdrawn.

See the unchanged [native configuration](saits_paper_brits_electricity_author_missing0.4_seed27/native_configuration.json)
and [complete native report](saits_paper_brits_electricity_author_missing0.4_seed27/full_native_report.json).
The original training used its registered CUDA configuration; this correction
concerns the separate CPU checkpoint replay. Numerical settings, admission limits,
results, artifacts and all frozen raw evidence are unchanged.
