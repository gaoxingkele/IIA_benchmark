C-RNN-GAN full native memory recovery v3
=======================================

The original full v2 worker stopped under its explicitly registered10GiB RSS cap after completing4/500 epochs and520 original pretraining updates. Its peak RSS was10795196416 bytes and commit headroom at the abort was25544167424 bytes, above the original16GiB emergency margin. Both controller and native worker handles were independently confirmed missing; terminal receipt, final logs, all complete epochs/checkpoints and raw inputs remain preserved. Each completed epoch checkpoint is independently verified using the actual native TensorFlow reader and exact tensor/file hashes; this is not stochastic model-prediction replay or a full benchmark result.

The recovery queue preserves all35 original canonical method/seed slots and every scientific flag:3014 classical MIDI/10323257 notes, width350/depth2,batch20,original full curriculum100,original validation/test holdouts and500 epochs. A fresh v3 attempt directory is used for each job; no old raw or experiment directory is overwritten.

The dedicated worker RSS/private limits become22/24GiB; admission requires32GiB physical/40GiB commit, reserving24GiB worker private plus the unchanged16GiB emergency margin. Actual full-curriculum peak usage is still unproven; any future guard exit remains an incomplete preserved attempt. The verified live controller currently waits for resources.

Native TensorFlow random operation state cannot yet be exactly resumed from the saved v2 checkpoints. The same canonical seed restarts from initialization under unchanged full budget. No recovery seed is an additional independent repeat, no test-score selection is performed, and no partial epoch or synthetic capacity test is promoted as benchmark performance. Full45-paper/681-method/273-task scope remains incomplete.
