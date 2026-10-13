# CrossAD GECCO seed1103 original full-budget commit recovery

The original scientific process PID36568/create_time1791780874.360876 reached
12 completed epochs of the original20 maximum and logged epoch13 iteration200.
Its controller recorded exit1 and resource_guard_aborted=true when actual system
commit headroom was7297990656 bytes, below the original8GiB emergency reserve.
5184 completed-epoch forward calls and all12 validation losses are retained.
Exact unsaved epoch13 optimizer updates were not recorded and are not invented.

Every original partial file and raw failure log remains untouched. Exact log,
failure, resource and trace bytes are archived here; the historical controller
snapshot may legitimately differ from its subsequently advancing live state.
The partial best checkpoint8106653 bytes is additionally archived on F: with
SHA256852c20da3c293caf6afca72e7353a8f8de3589620f130aa16905f5fb96b08428.
Author EarlyStopping.save_checkpoint saves only model.state_dict(); neither
optimizer nor RNG/loader/early-stop continuation state was persisted. The
partial model is not an exact resumable optimizer trajectory. No result/stage
receipt exists and no partial budget is credited toward the fresh retry.

The new immutable native queue retains all139 original main, release and
ablation slots. Only GECCO1103 output is redirected; the other138 bindings,
canonical IDs, six original seeds, all scientific/data/source fields and native
full-batch preflight proofs are unchanged. The lightweight retry dispatches
only this failed canonical slot. It uses the exact actual pre-incident Python
interpreter and unchanged original memory-v2 author runner. No added seed,
scientific source edit, shorter budget, batch reduction or tolerance change is
introduced. The original20 maximum, patience3, batch128, seq_len128, two CPU
threads, Adam, LR schedule, author SPOT/range/VUS and native-tail controls remain.

Only admission commit reserve is raised from16GiB to40GiB; the original24GiB
physical gate,8GiB emergency guard and shared heavy CPU lock remain unchanged.
The actual new controller is live and waiting for memory without owning the
shared lock. It has not started a training worker. Failed retries are preserved
and never automatically repeated. Every native stage and artifact plus actual
zero-exit/non-aborted guard receipt is needed for selected-slot completion.

21 relevant tests pass. The actual entry verifies all original scientific pins,
scope and complete-batch proof without importing Torch or NumPy. These checks
and historical training losses are not benchmark performance. Author flat-data
protocol does not certify entity-grouped TAB or paper equivalence. All45papers,
681method/baseline/ablation records,273original-data tasks and every other queue
remain in scope. Overall completion is still unproved.
