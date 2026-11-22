"""Late temporal fusion: the same encoder on every frame, the frames' features merged per scale.

A temporal input stacks ``F = 2k+1`` three-channel frames, the centre frame in the middle (the data
option ``context_frames``). Stacked as the channels of one image, the frames are mixed by the first
convolution, before any feature exists. Here every frame goes through the same encoder instead, and
the frames' features are merged at every scale, where a contrail that moved a few pixels between
two frames is still the same feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn

FRAME_CHANNELS = 3


class TemporalFusionEncoder(nn.Module):
    """Run one encoder on each of ``frames`` stacked frames and merge their features per scale.

    The input is ``(B, 3*F, H, W)``: ``F`` three-channel frames, frame-major, the centre frame in
    the middle. Every frame goes through ``encoder`` (shared weights); at every scale the ``F``
    feature maps are concatenated and a 3x3 convolution maps them back to the encoder's channels.
    Each convolution starts as the identity on the centre frame (every other weight and the bias
    zero), so before training the network gives exactly the single-frame model's output on the
    centre frame; the neighbour weights then learn from zero. Scale 0 (the input itself, which the
    decoders do not use) is the centre frame.

    Parameters
    ----------
    encoder
        A single-frame smp encoder (3 input channels).
    frames
        Number of stacked frames ``F``: odd and at least 3.

    Raises
    ------
    ValueError
        If ``frames`` is even or smaller than 3.
    """

    def __init__(self, encoder: nn.Module, frames: int) -> None:
        super().__init__()
        if frames < 3 or frames % 2 == 0:
            raise ValueError(f"frames must be odd and at least 3, got {frames}")
        self.encoder = encoder
        self.frames = frames
        self.out_channels = list(encoder.out_channels)  # type: ignore[arg-type]
        self.output_stride = encoder.output_stride
        self.fuse = nn.ModuleList(
            [
                _centre_identity_conv(channels, frames) if channels > 0 else nn.Identity()
                for channels in self.out_channels[1:]
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return one fused feature map per scale, shaped as the encoder's for a single frame."""
        batch, channels, height, width = x.shape
        if channels != self.frames * FRAME_CHANNELS:
            msg = (
                f"expected {self.frames} stacked frames ({self.frames * FRAME_CHANNELS} channels), "
                f"got {channels} channels"
            )
            raise ValueError(msg)
        centre = self.frames // 2
        features = self.encoder(x.reshape(batch * self.frames, FRAME_CHANNELS, height, width))
        fused = [x[:, centre * FRAME_CHANNELS : (centre + 1) * FRAME_CHANNELS]]
        for feature, fuse in zip(features[1:], self.fuse, strict=True):
            _, feature_channels, feature_height, feature_width = feature.shape
            if feature_channels == 0:
                # An empty scale (no feature at this resolution) stays empty.
                empty = feature.reshape(batch, self.frames, 0, feature_height, feature_width)
                fused.append(empty[:, centre])
                continue
            stacked = feature.reshape(
                batch, self.frames * feature_channels, feature_height, feature_width
            )
            fused.append(fuse(stacked))
        return fused


def _centre_identity_conv(channels: int, frames: int) -> nn.Conv2d:
    """Return a 3x3 conv ``frames*channels -> channels``: the identity on the centre frame."""
    conv = nn.Conv2d(frames * channels, channels, kernel_size=3, padding=1)
    with torch.no_grad():
        conv.weight.zero_()
        bias = conv.bias
        if bias is not None:
            bias.zero_()
        index = torch.arange(channels)
        conv.weight[index, (frames // 2) * channels + index, 1, 1] = 1.0
    return conv
