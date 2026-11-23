"""Track memory: the previous frame's contrail map as an extra input to the encoder features.

Labellers follow each contrail through the frame sequence and keep drawing it as it ages and
fades. A model that sees one frame at a time loses such contrails. With ``track_memory`` the
network also receives a contrail probability map of the previous frame (its own previous
prediction at inference) as the last input channel; at every scale the map is pooled to the
feature size and projected onto the features by small layers whose last convolution starts at
zero, so before training the memory changes nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MEMORY_HIDDEN = 16


class TrackMemoryEncoder(nn.Module):
    """Add a projection of the memory map to every scale of the encoder.

    The input is ``(B, image_channels + 1, H, W)``: the memory map is the last channel. The first
    ``image_channels`` channels go through ``encoder``; the last one, a probability map in [0, 1],
    is pooled to each feature's size (average and max pooling, two channels) and passed through
    ``Conv3x3(2 -> 16) - ReLU - Conv3x3(16 -> C)``, whose last convolution starts at zero (weights
    and bias), then added to that feature. Scale 0 (the input itself, which the decoders do not use)
    and empty scales are returned unchanged.

    Parameters
    ----------
    encoder
        The model's encoder (any smp encoder or wrapper exposing ``out_channels`` and
        ``output_stride``).
    image_channels
        Number of image channels before the memory channel.
    """

    def __init__(self, encoder: nn.Module, image_channels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.image_channels = image_channels
        self.out_channels = list(encoder.out_channels)  # type: ignore[arg-type]
        self.output_stride = encoder.output_stride
        self.inject = nn.ModuleList(
            [
                _zero_projection(channels) if channels > 0 else nn.Identity()
                for channels in self.out_channels[1:]
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the encoder's features of the image channels, each plus its memory projection."""
        if x.shape[1] != self.image_channels + 1:
            msg = (
                f"expected {self.image_channels} image channels and 1 memory channel "
                f"({self.image_channels + 1} channels), got {x.shape[1]} channels"
            )
            raise ValueError(msg)
        image, memory = x[:, : self.image_channels], x[:, self.image_channels :]
        features = self.encoder(image)
        out = [features[0]]
        for feature, inject in zip(features[1:], self.inject, strict=True):
            if feature.shape[1] == 0:
                out.append(feature)
                continue
            size = feature.shape[-2:]
            pooled = torch.cat(
                [F.adaptive_avg_pool2d(memory, size), F.adaptive_max_pool2d(memory, size)], dim=1
            )
            out.append(feature + inject(pooled.to(feature.dtype)))
        return out


def _zero_projection(channels: int) -> nn.Sequential:
    """Return ``Conv3x3(2 -> 16) - ReLU - Conv3x3(16 -> channels)`` with the last conv at zero."""
    last = nn.Conv2d(MEMORY_HIDDEN, channels, kernel_size=3, padding=1)
    nn.init.zeros_(last.weight)
    bias = last.bias
    if bias is not None:
        nn.init.zeros_(bias)
    return nn.Sequential(
        nn.Conv2d(2, MEMORY_HIDDEN, kernel_size=3, padding=1), nn.ReLU(inplace=True), last
    )
