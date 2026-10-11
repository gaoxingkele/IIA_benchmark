C-RNN-GAN full native classical-data training
===========================================

The complete frozen 500-epoch released-data controls comprise FM one/three tones,
no-FM one/three tones, minibatch-FM one/three tones, and maximum-likelihood one
tone. Each has five canonical independent training seeds1103..1107, giving35
obligations. The original45-paper/681-method-baseline-ablation/273-original-task
scope plus adopted baselines is retained and remains incomplete.

The unchanged author loader selects registered classical music. It visits3029
released classical MIDI files;3014 parse,15 malformed originals remain preserved,
and10323257 parsed note events are retained. The previously audited3435 parsed
compositions/10661408notes across ALL released genres are a different inventory.
The first registration used that wrong all-genre expectation. Its native loader
attempt failed the explicit count gate before model training. Its terminal
receipt and original logs are preserved;v2 retains the same canonical seed IDs.

Released classical data have127 composers with files;the source registers154;
the paper reports3697 compositions/160composers. Historical paper corpus identity
is unverified. Hip-hop/jazz files do not replace missing classical compositions.

G and D retain the complete paper geometry:two recurrent layers,width350,
bidirectional D. Batch20,source500-epoch maximum and every complete source
minibatch remain. Six full pretraining epochs use lengths4,8,12,16,20,24;the
adversarial phase uses100. The ML control uses500 pretraining epochs. This500
budget is a source-default/local control,not an explicit paper budget. The
default1440-minute wall-clock exit is disabled to finish500 epochs. Original
loss-gated stopping remains;if it stops early,no complete result is published.

Reviewed changes run in memory:correct swapped G/D optimizer arguments,restore
symmetric70-percent freezing,use current epoch for curriculum membership,
normalize original held-out filename separators on Windows,and initialize only
the newly rebuilt nontrainable learning-rate scalar. Every other main/run_epoch
AST statement remains. MIDI compatibility restores the legacy velocity property.
Original source/raw hashes are checked before and after full loading/training.

Each full epoch records actual optimizer fetches,train/validation losses,all
native model variables/checkpoints,NumPy/Python RNG states,and original MIDI
sample/statistic outputs. Every live variable must be finite and bitwise equal
to its saved checkpoint tensor. Final source testing traverses every complete
held-out minibatch. Source incomplete-tail discarding remains and is recorded.

Remaining alignment gaps include historical data,source feature-matching sum
versus paper1/m normalization,three-tone shared timing encoding,and stateful
TensorFlow stochastic inference/RNG continuation. Source-literal algorithm and
historical paper tracks remain obligations;these controls do not replace them.
Checkpoint-variable integrity does not certify stochastic model replay.

launch.json identifies the actual controller. full_classical_registration_audit.json
freezes complete source/model/data obligations. preserved_first_registration_failure.json
records the proved first failure. Runtime and validation reports give timestamped
evidence only. No benchmark metric rows are added before complete training,
evaluation and appropriate audits pass.
