from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_mask(mask: Optional[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones((target.shape[0], 1, target.shape[2], target.shape[3]), device=target.device, dtype=target.dtype)
    if mask.dtype != target.dtype:
        mask = mask.to(dtype=target.dtype)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask


class RangeSparseConv2d(nn.Module):
    """Linear convolution that respects the sparsity mask in range view."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = False,
        norm_fn: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.norm = norm_fn(out_channels) if norm_fn is not None else None
        self.activation = activation() if callable(activation) else activation
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = _ensure_mask(mask, x)
        x = x * mask
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)

        if self.stride != (1, 1):
            mask = F.max_pool2d(
                mask,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            mask = (mask > 0.0).to(dtype=out.dtype)
        else:
            mask = mask[..., : out.shape[2], : out.shape[3]]

        out = out * mask
        return out, mask


class RangeSparseResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_fn: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.downsample = downsample or stride != 1 or in_channels != out_channels

        self.conv1 = RangeSparseConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_fn=norm_fn,
            activation=activation,
        )
        self.conv2 = RangeSparseConv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_fn=norm_fn,
            activation=None,
        )
        if self.downsample:
            self.shortcut = RangeSparseConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                norm_fn=norm_fn,
                activation=None,
                bias=False,
            )
        else:
            self.shortcut = None
        self.activation = activation()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = x
        residual_mask = mask

        out, mask = self.conv1(x, mask)
        out, mask = self.conv2(out, mask)

        if self.shortcut is not None:
            identity, residual_mask = self.shortcut(identity, residual_mask)

        out = out + identity * mask
        out = self.activation(out)
        out = out * mask
        return out, mask


class RangeSparseStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        norm_fn: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        blocks = []
        for idx in range(num_blocks):
            blocks.append(
                RangeSparseResidualBlock(
                    in_channels if idx == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride if idx == 0 else 1,
                    norm_fn=norm_fn,
                    activation=activation,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x, mask = block(x, mask)
        return x, mask


class RangeSparseBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_channels: List[int],
        stage_blocks: List[int],
        stage_strides: List[int],
        norm_fn: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        assert len(stage_channels) == len(stage_blocks) == len(stage_strides), (
            "stage_channels, stage_blocks, stage_strides must match in length"
        )

        self.stem = RangeSparseConv2d(
            in_channels,
            stage_channels[0],
            kernel_size=3,
            stride=1,
            norm_fn=norm_fn,
            activation=activation,
        )

        stages = []
        for idx, (out_ch, num_blocks, stride) in enumerate(zip(stage_channels, stage_blocks, stage_strides)):
            in_ch = stage_channels[idx - 1] if idx > 0 else stage_channels[0]
            stages.append(
                RangeSparseStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_blocks=num_blocks,
                    stride=stride,
                    norm_fn=norm_fn,
                    activation=activation,
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward(
        self, range_image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, mask = self.stem(range_image, mask)
        for stage in self.stages:
            feats, mask = stage(feats, mask)
        return feats, mask
