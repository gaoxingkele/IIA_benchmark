SpectralFlow Energy seed42: complete training, aborted original sampling
=====================================================================

The original released-data job completed all 25,000 optimizer updates with
its full source model and batch128. The native process17280 (creation time
1791730660.2965653) is missing and its terminal receipt records exit1 and a
resource guard abort during original unchunked batch2001 sampling. There is
no complete generated array, no full four-evaluator result, and no benchmark
score. The precise failing tensor operation is not established by this receipt.

CPU-only inspection verified every finite model/EMA/optimizer tensor, every
optimizer parameter's complete 25,000-step budget, and both complete finite
19,712 x24 x28 frozen data arrays and their original checksums. Checkpoint
SHA256: 726a43de864f56cf734381a9bc245328e98ad831afc58c14846b42ab192ae5bf.
The preserved checkpoint has119model tensors,240EMA tensors and117optimizer
parameter states. All25guard tests passed; these tests are not performance.

The abort snapshot reports265,289,728free GPU bytes and7,049,220,096free
commit bytes. Original checkpoint, logs, failure, raw data, source, guards,
full budgets and live serial controller are preserved. No smaller batches,
chunked fields, fewer iterations or replacement performance numbers were used.
The native checkpoint lacks post-training RNG and scheduler state; exact
continuation of the failed sampling trajectory is not certified.

Pending work includes full original sampling and all four full evaluators,
then independent full-array/checkpoint/metric verification. Existing Stocks
full-operator/trajectory reviews do not certify this wider Energy model.
No new training seed slot or completed generation experiment is counted.
All70generation jobs/14cohorts and45papers/681method-ablation/273data-task
obligations remain open as originally registered. The combined v49 leaderboard
is unchanged.
