# Method

Encodes variable-length raw alarm floods as exponentially attenuated fixed histograms, learns the attenuation by gradient optimization, compresses correlated alarms in an autoencoder latent space, and classifies with a modified Transformer using separate pretraining followed by joint fine-tuning.
