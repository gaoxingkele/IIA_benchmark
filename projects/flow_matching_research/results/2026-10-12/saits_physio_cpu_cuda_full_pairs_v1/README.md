Complete SAITS/PhysioNet CPU/CUDA prediction-pair comparison

All ten pairs of previously executed full original checkpoint replays were compared on every coordinate. Exactly 42,588,480 full prediction coordinates and 826,050 held-out coordinates were inspected. Both sides share the actual current source checkpoint, original result and configuration hashes. Full truths, masks, test indices, patient identities and ordering match exactly; each side own full per-patient statistics were independently recomputed from its complete predictions. Fifteen tests passed, including incomplete/misaligned input rejection, preservation of strict mismatches and byte-level signed-zero handling. The guarded full comparison exited0 without resource abort.

| Original model | CPU strict patient-statistic check | CUDA strict patient-statistic check | Maximum paired prediction difference on held-out coordinates |
|---|---|---|---:|
| saits_paper_saits_physio_seed26 | fail | fail | 7.62939453125e-06 |
| saits_paper_brits_physio_seed26 | fail | pass | 3.4868717193603516e-06 |
| saits_paper_saits_physio_seed27 | fail | pass | 6.794929504394531e-06 |
| saits_paper_brits_physio_seed27 | pass | pass | 1.9073486328125e-06 |
| saits_paper_saits_physio_seed28 | fail | pass | 5.841255187988281e-06 |
| saits_paper_brits_physio_seed28 | pass | pass | 1.9073486328125e-06 |
| saits_paper_saits_physio_seed29 | fail | pass | 6.9141387939453125e-06 |
| saits_paper_brits_physio_seed29 | pass | pass | 1.9073486328125e-06 |
| saits_paper_saits_physio_seed30 | pass | pass | 5.7220458984375e-06 |
| saits_paper_brits_physio_seed30 | pass | pass | 2.6226043701171875e-06 |

All ten CPU/CUDA prediction pairs differ at the byte level. The largest held-out absolute difference is 7.62939453125e-6. Five CPU model cases and one CUDA model case fail the unchanged original per-patient tolerance of rtol2e-6/atol2e-6. These are observations of the existing full replays, not new training seeds or tolerance changes. The one original-device CUDA failure remains the SAITS seed26 squared-error statistic for patient155135: original2.5207282105362423, replay2.5207380134909236; its9.802954681337184e-6difference exceeds the original7.041456421072484e-6allowance.

The comparison rules out a mismatch between these paired arrays in patient order, masks, truths or selected weights. It establishes small full-grid prediction differences and their downstream statistic differences. It does not establish the causal explanation of the original training-time mismatch: original training-time full predictions and numerical backend flags were not persisted. Do not attribute the failure solely to CPU versus CUDA, accept it because global metrics agree, or rerun until a favorable attempt appears.

This diagnostic uses the existing frozen full-array read-only CPU admission profile; it does not change any native model replay admission or acceptance gate. No model inference, optimizer step, native code edit or new measured leaderboard row was performed. Every original failed replay and all five currently active training handles are preserved. Full native reproduction, paper equivalence and strict TAB certification remain false or unproven. The 45-paper/681-method-baseline-ablation/273-original-data-task scope remains unfinished.
