"""A convolutional 1/2-scale stem for encoders that start at 1/4 scale.

Transformer-style timm encoders (CAFormer, ConvNeXt, ...) have no feature map at 1/2 of the
input resolution; smp fills that level with an empty (0-channel) tensor. Unet++ sizes its
nested decoder nodes from the encoder's channels, so at that level they would be empty layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

STEM_CHANNELS = 64


class HalfScaleStemEncoder(nn.Module):
    """Wrap an smp encoder and replace its empty 1/2-scale feature with a conv stem.

    The stem is two 3x3 convolutions (the first with stride 2), each followed by
    BatchNorm and ReLU, computed from the input image.

    Parameters
    ----------
    encoder
        An smp encoder whose ``out_channels[1]`` is 0.
    stem_channels
        Channels of the 1/2-scale feature.
    """

    def __init__(self, encoder: nn.Module, stem_channels: int = STEM_CHANNELS) -> None:
        super().__init__()
        channels = list(encoder.out_channels)  # type: ignore[arg-type]
        if channels[1] != 0:
            raise ValueError(f"encoder already has a 1/2-scale feature: {channels}")
        self.encoder = encoder
        self.output_stride = encoder.output_stride
        self.stem = nn.Sequential(
            nn.Conv2d(channels[0], stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = [channels[0], stem_channels, *channels[2:]]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the encoder features with the stem output at 1/2 scale."""
        features = self.encoder(x)
        return [features[0], self.stem(x), *features[2:]]
