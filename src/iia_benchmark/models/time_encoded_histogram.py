"""Optimal time-encoded histograms with an autoencoder/Transformer hybrid.

Najafi and Chen (2026) encode variable alarm floods with exponentially
attenuated histograms, pretrain an autoencoder and a modified Transformer
separately, and jointly fine-tune the attenuation and neural components.  The
publisher exposes the architecture but not every equation/hyperparameter, so
this module implements that complete trainable pipeline with explicit local
choices and keeps the reported TEP score as a separate reproduction gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class TimedAlarmSequence:
    """A variable-length alarm flood with absolute or relative timestamps."""

    timestamps: np.ndarray
    tags: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "tags", tuple(self.tags))
        if timestamps.ndim != 1 or timestamps.size == 0 or timestamps.size != len(self.tags):
            raise ValueError("timestamps and tags must be equal nonempty vectors")
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("timestamps must be finite")

    @classmethod
    def from_events(
        cls, events: Sequence[tuple[float, Hashable]]
    ) -> "TimedAlarmSequence":
        ordered = tuple(sorted(events, key=lambda item: item[0]))
        return cls(
            np.asarray([item[0] for item in ordered], dtype=float),
            tuple(item[1] for item in ordered),
        )

    def prefix(self, length: int) -> "TimedAlarmSequence":
        if length <= 0:
            raise ValueError("prefix length must be positive")
        end = min(int(length), len(self.tags))
        return TimedAlarmSequence(self.timestamps[:end], self.tags[:end])


def time_encoded_histogram(
    sequence: TimedAlarmSequence,
    tag_vocabulary: Sequence[Hashable],
    attenuation: float,
    normalize: bool = True,
) -> np.ndarray:
    """Return exponentially time-weighted alarm-tag frequencies."""

    if attenuation < 0:
        raise ValueError("attenuation must be nonnegative")
    vocabulary = tuple(tag_vocabulary)
    if not vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("tag_vocabulary must be nonempty and unique")
    order = np.argsort(sequence.timestamps, kind="stable")
    timestamps = sequence.timestamps[order]
    tags = tuple(sequence.tags[index] for index in order)
    span = float(timestamps[-1] - timestamps[0])
    relative = (timestamps - timestamps[0]) / span if span > 0 else np.zeros_like(timestamps)
    weights = np.exp(-attenuation * relative)
    result = np.zeros(len(vocabulary), dtype=float)
    index = {tag: position for position, tag in enumerate(vocabulary)}
    for tag, weight in zip(tags, weights):
        if tag in index:
            result[index[tag]] += weight
    if normalize and np.sum(result) > 0:
        result /= np.sum(result)
    return result


class OptimalTimeEncodedHistogramClassifier:
    """Three-phase autoencoder/Transformer alarm-flood classifier.

    Phase 1 minimizes autoencoder reconstruction loss.  Phase 2 freezes the
    encoder and trains an encoder-only Transformer classifier over latent
    dimensions.  Phase 3 jointly fine-tunes both networks and the positive
    attenuation factor with classification plus reconstruction loss.
    """

    def __init__(
        self,
        latent_size: int = 8,
        autoencoder_hidden_size: int = 16,
        transformer_width: int = 8,
        attention_heads: int = 2,
        transformer_layers: int = 1,
        pretrain_epochs: int = 30,
        classifier_epochs: int = 30,
        joint_epochs: int = 40,
        learning_rate: float = 0.02,
        initial_attenuation: float = 1.0,
        reconstruction_weight: float = 0.1,
        random_state: int = 0,
    ) -> None:
        integer_values = (
            latent_size,
            autoencoder_hidden_size,
            transformer_width,
            attention_heads,
            transformer_layers,
            pretrain_epochs,
            classifier_epochs,
            joint_epochs,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("network sizes, layers and epoch counts must be positive")
        if transformer_width % attention_heads:
            raise ValueError("transformer_width must be divisible by attention_heads")
        if learning_rate <= 0 or initial_attenuation <= 0 or reconstruction_weight < 0:
            raise ValueError("learning rate/attenuation must be positive and reconstruction weight nonnegative")
        self.latent_size = int(latent_size)
        self.autoencoder_hidden_size = int(autoencoder_hidden_size)
        self.transformer_width = int(transformer_width)
        self.attention_heads = int(attention_heads)
        self.transformer_layers = int(transformer_layers)
        self.pretrain_epochs = int(pretrain_epochs)
        self.classifier_epochs = int(classifier_epochs)
        self.joint_epochs = int(joint_epochs)
        self.learning_rate = float(learning_rate)
        self.initial_attenuation = float(initial_attenuation)
        self.reconstruction_weight = float(reconstruction_weight)
        self.random_state = int(random_state)

    def _prepare(self, sequences: Sequence[TimedAlarmSequence]):
        torch = self._torch
        if not sequences:
            raise ValueError("at least one timed alarm sequence is required")
        maximum = max(len(sequence.tags) for sequence in sequences)
        tag_indices = np.zeros((len(sequences), maximum), dtype=np.int64)
        relative_times = np.zeros((len(sequences), maximum), dtype=np.float32)
        mask = np.zeros((len(sequences), maximum), dtype=bool)
        vocabulary_index = {tag: index for index, tag in enumerate(self.tag_vocabulary_)}
        for row, sequence in enumerate(sequences):
            order = np.argsort(sequence.timestamps, kind="stable")
            timestamps = sequence.timestamps[order]
            span = float(timestamps[-1] - timestamps[0])
            relative = (
                (timestamps - timestamps[0]) / span
                if span > 0
                else np.zeros_like(timestamps)
            )
            column = 0
            for event_index, source_index in enumerate(order):
                tag = sequence.tags[int(source_index)]
                if tag not in vocabulary_index:
                    continue
                tag_indices[row, column] = vocabulary_index[tag]
                relative_times[row, column] = relative[event_index]
                mask[row, column] = True
                column += 1
        return (
            torch.tensor(tag_indices, dtype=torch.long),
            torch.tensor(relative_times, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
        )

    def fit(
        self, sequences: Sequence[TimedAlarmSequence], y: np.ndarray
    ) -> "OptimalTimeEncodedHistogramClassifier":
        try:
            import torch
            from torch import nn
            from torch.nn import functional as functional
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install iia-benchmark[deep] for the histogram hybrid") from exc
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.size != len(sequences):
            raise ValueError("y must contain one label per timed alarm sequence")
        self.classes_ = np.asarray(sorted(set(labels.tolist()), key=repr))
        if self.classes_.size < 2:
            raise ValueError("at least two classes are required")
        self.tag_vocabulary_ = tuple(
            sorted({tag for sequence in sequences for tag in sequence.tags}, key=repr)
        )
        self._torch = torch
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        vocabulary_size = len(self.tag_vocabulary_)
        latent_size = min(self.latent_size, vocabulary_size)

        class HybridNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.raw_attenuation = nn.Parameter(
                    torch.tensor(np.log(np.expm1(self_outer.initial_attenuation)), dtype=torch.float32)
                )
                self.encoder = nn.Sequential(
                    nn.Linear(vocabulary_size, self_outer.autoencoder_hidden_size),
                    nn.ReLU(),
                    nn.Linear(self_outer.autoencoder_hidden_size, latent_size),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_size, self_outer.autoencoder_hidden_size),
                    nn.ReLU(),
                    nn.Linear(self_outer.autoencoder_hidden_size, vocabulary_size),
                    nn.Sigmoid(),
                )
                self.token_projection = nn.Linear(1, self_outer.transformer_width)
                self.class_token = nn.Parameter(
                    torch.zeros(1, 1, self_outer.transformer_width)
                )
                self.position = nn.Parameter(
                    torch.zeros(1, latent_size + 1, self_outer.transformer_width)
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=self_outer.transformer_width,
                    nhead=self_outer.attention_heads,
                    dim_feedforward=2 * self_outer.transformer_width,
                    dropout=0.0,
                    batch_first=True,
                    activation="gelu",
                )
                self.transformer = nn.TransformerEncoder(
                    layer, num_layers=self_outer.transformer_layers
                )
                self.classifier = nn.Linear(
                    self_outer.transformer_width, len(self_outer.classes_)
                )

            def attenuation(self):
                return functional.softplus(self.raw_attenuation)

            def histogram(self, tag_indices, relative_times, mask):
                weights = torch.exp(-self.attenuation() * relative_times) * mask
                histogram = torch.zeros(
                    tag_indices.shape[0], vocabulary_size, device=tag_indices.device
                )
                histogram.scatter_add_(1, tag_indices, weights)
                total = histogram.sum(dim=1, keepdim=True).clamp_min(1e-12)
                return histogram / total

            def classify_latent(self, latent):
                tokens = self.token_projection(latent.unsqueeze(-1))
                class_tokens = self.class_token.expand(latent.shape[0], -1, -1)
                encoded = self.transformer(
                    torch.cat((class_tokens, tokens), dim=1) + self.position
                )
                return self.classifier(encoded[:, 0])

            def forward(self, tag_indices, relative_times, mask):
                histogram = self.histogram(tag_indices, relative_times, mask)
                latent = self.encoder(histogram)
                reconstruction = self.decoder(latent)
                return self.classify_latent(latent), histogram, reconstruction

        self_outer = self
        self.network_ = HybridNetwork()
        tag_indices, relative_times, mask = self._prepare(sequences)
        targets = torch.tensor(
            [int(np.flatnonzero(self.classes_ == label)[0]) for label in labels],
            dtype=torch.long,
        )
        mse = nn.MSELoss()
        cross_entropy = nn.CrossEntropyLoss()
        history: dict[str, list[float]] = {"autoencoder": [], "classifier": [], "joint": []}

        with torch.no_grad():
            fixed_histograms = self.network_.histogram(tag_indices, relative_times, mask)
        optimizer = torch.optim.Adam(
            list(self.network_.encoder.parameters()) + list(self.network_.decoder.parameters()),
            lr=self.learning_rate,
        )
        for _ in range(self.pretrain_epochs):
            optimizer.zero_grad()
            reconstruction = self.network_.decoder(self.network_.encoder(fixed_histograms))
            loss = mse(reconstruction, fixed_histograms)
            loss.backward()
            optimizer.step()
            history["autoencoder"].append(float(loss.detach()))

        classifier_parameters = (
            list(self.network_.token_projection.parameters())
            + list(self.network_.transformer.parameters())
            + list(self.network_.classifier.parameters())
            + [self.network_.class_token, self.network_.position]
        )
        optimizer = torch.optim.Adam(classifier_parameters, lr=self.learning_rate)
        for _ in range(self.classifier_epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                latent = self.network_.encoder(fixed_histograms)
            loss = cross_entropy(self.network_.classify_latent(latent), targets)
            loss.backward()
            optimizer.step()
            history["classifier"].append(float(loss.detach()))

        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
        for _ in range(self.joint_epochs):
            optimizer.zero_grad()
            logits, histogram, reconstruction = self.network_(
                tag_indices, relative_times, mask
            )
            loss = cross_entropy(logits, targets) + self.reconstruction_weight * mse(
                reconstruction, histogram
            )
            loss.backward()
            optimizer.step()
            history["joint"].append(float(loss.detach()))
        self.training_history_ = {
            key: np.asarray(values, dtype=float) for key, values in history.items()
        }
        return self

    @property
    def attenuation_(self) -> float:
        if not hasattr(self, "network_"):
            raise RuntimeError("fit must be called before accessing attenuation")
        return float(self.network_.attenuation().detach())

    def encode_histograms(self, sequences: Sequence[TimedAlarmSequence]) -> np.ndarray:
        if not hasattr(self, "network_"):
            raise RuntimeError("fit must be called before encoding")
        tag_indices, relative_times, mask = self._prepare(sequences)
        with self._torch.no_grad():
            values = self.network_.histogram(tag_indices, relative_times, mask)
        return values.cpu().numpy()

    def predict_proba(self, sequences: Sequence[TimedAlarmSequence]) -> np.ndarray:
        if not hasattr(self, "network_"):
            raise RuntimeError("fit must be called before prediction")
        tag_indices, relative_times, mask = self._prepare(sequences)
        self.network_.eval()
        with self._torch.no_grad():
            logits, _, _ = self.network_(tag_indices, relative_times, mask)
            probabilities = self._torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def predict(self, sequences: Sequence[TimedAlarmSequence]) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(sequences), axis=1)]

    def predict_evolution(
        self,
        sequences: Sequence[TimedAlarmSequence],
        prefix_lengths: Sequence[int],
    ) -> dict[int, np.ndarray]:
        lengths = tuple(sorted(set(int(item) for item in prefix_lengths)))
        if not lengths or lengths[0] <= 0:
            raise ValueError("prefix lengths must be positive")
        return {
            length: self.predict([sequence.prefix(length) for sequence in sequences])
            for length in lengths
        }
