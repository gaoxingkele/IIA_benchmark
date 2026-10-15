# MOMENT Base / SMAP / context512: full native closure and matched Small comparison

New complete109,641,608-parameter Base replay;Small is the previously closed35,341,512-parameter control and is not counted twice. Both use the identical input hash,all51entities and153TRAIN/validation/TEST roles:101565/26684/408160 native points. Each performs1140full-window forwards. All204source score/label/timestamp arrays have identical dtype,shape and content bytes. Both actual guarded stages exited successfully without resource abort;the second independent process checks every array,core/event metric,original200resample interval and all10pooled diagnostic ratios.

|Model|Frozen validation1% no-PA F1|Same-threshold PA F1|AUROC|AP|
|---|---:|---:|---:|---:|
|MOMENT-1-small|0.10560927|0.68665956|0.45465374|0.12634272|
|MOMENT-1-base|0.10029775|0.68633660|0.46291593|0.12845292|

All three prespecified validation quantiles0.1/1/5are retained;the primary1%is not chosen by test F1. No test-derived score inversion when AUROC is below0.5. Base and Small differ in model size,on the same fixed preprocessing;this single frozen-checkpoint comparison does not establish significance or broad model-scale effects. The grouped bootstrap intervals remain conditional on model and threshold,not training-seed uncertainty.

The original full-window float64-MSE cross-check retains frozen relative/absolute tolerance2e-6. Its maximum absolute difference is3.228915159456526e20for Base and5.813435295107488e20for Small,reflecting the extreme numerical scale of this unchanged source track;do not interpret the accepted relative check as near-zero absolute error. Full saved-score magnitude profiles are archived separately. No clipping,rescaling,preprocessing fix or tolerance change is made here. The distinct constant-fit normalization controls remain separately registered and are not certified by this packet.

Full51-entity grouped source splits are retained;source historical guard gaps,all preprocessing-value reconstruction,VUS/Affiliation low-level integrals,the complete TAB executor and historical-paper equivalence remain unproven. Windows include future context,so event delay is not online causal delay. These official release zero-shot tasks are separate from Tiny/ICM full-PILE training,classification and forecasting.

Reuse of15unchanged frozen validation checks is backed by rehashing source/config/data/weights/proofs and both exact successful guard commands. One additional complete checkpoint/dataset case is closed,no new training seed;queue snapshot14/78. Other cases and at least45papers/681method records/273data tasks plus added experiments remain required. The full research objective is not complete.

Sources:[MOMENT paper](https://arxiv.org/abs/2402.03885),[official Base weights](https://huggingface.co/AutonLab/MOMENT-1-base),[NASA source](https://github.com/khundman/telemanom). Large weights and arrays stay at their original registered disk paths.
