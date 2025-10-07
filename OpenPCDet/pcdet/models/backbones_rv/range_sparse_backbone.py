from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..model_utils.range_sparse_utils import RangeSparseBackbone


class RangeSparseBackboneModule(nn.Module):
    """Process range-view tensors with Range Sparse Net backbone."""

    def __init__(self, model_cfg, input_channels: int) -> None:
        super().__init__()
        self.model_cfg = model_cfg

        stage_channels = list(model_cfg.STAGE_CHANNELS)
        stage_blocks = list(model_cfg.STAGE_BLOCKS)
        stage_strides = list(model_cfg.STAGE_STRIDES)

        self.backbone = RangeSparseBackbone(
            in_channels=input_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            stage_strides=stage_strides,
            norm_fn=lambda num_features: nn.BatchNorm2d(num_features, eps=1e-3, momentum=0.01),
            activation=nn.ReLU,
        )
        self.output_key = model_cfg.get('OUTPUT_KEY', 'range_features')
        self.mask_output_key = model_cfg.get('MASK_OUTPUT_KEY', 'range_mask')

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'range_image' not in data_dict:
            raise KeyError("RangeSparseBackboneModule requires 'range_image' in batch data.")

        range_image = data_dict['range_image']  # (B, H, W, C) or (B, C, H, W)
        if range_image.dim() != 4:
            raise ValueError(f"range_image must be 4D tensor, got {range_image.shape}")

        expected_channels = self.backbone.stem.conv.in_channels
        if range_image.shape[1] != expected_channels:
            if range_image.shape[-1] != expected_channels:
                raise ValueError(
                    f"range_image has incompatible channel dimension: {range_image.shape},"
                    f" expected {expected_channels}"
                )
            range_image = range_image.permute(0, 3, 1, 2).contiguous()

        mask = data_dict.get('range_mask', None)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.to(dtype=range_image.dtype)
        features, mask = self.backbone(range_image, mask)

        data_dict[self.output_key] = features
        data_dict[self.mask_output_key] = mask
        return data_dict
