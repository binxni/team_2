from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class RangeViewScatter(nn.Module):
    """Project range-view features to a BEV grid via scatter pooling."""

    def __init__(self, model_cfg, grid_size):
        super().__init__()
        self.model_cfg = model_cfg

        self.voxel_size = torch.tensor(model_cfg.VOXEL_SIZE[:2], dtype=torch.float32)
        self.point_cloud_range = torch.tensor(model_cfg.POINT_CLOUD_RANGE, dtype=torch.float32)
        self.num_bev_features = model_cfg.get('NUM_OUTPUT_FEATURES', model_cfg.INPUT_CHANNELS)
        self.pooling_method = model_cfg.get('POOLING', 'mean').lower()

        pc_range = self.point_cloud_range
        voxel = self.voxel_size
        bev_width = int(round((pc_range[3] - pc_range[0]) / voxel[0]))
        bev_height = int(round((pc_range[4] - pc_range[1]) / voxel[1]))
        self.bev_shape = (bev_height, bev_width)
        if self.pooling_method not in {'mean', 'sum'}:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

        feature_indices = model_cfg.get('GEOMETRY_CHANNELS', {})
        self.x_channel = feature_indices.get('x', 0)
        self.y_channel = feature_indices.get('y', 1)

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'range_features' not in data_dict:
            raise KeyError("RangeViewScatter requires 'range_features' in batch data.")
        if 'range_image' not in data_dict:
            raise KeyError("RangeViewScatter requires original 'range_image' to read coordinates.")

        range_features = data_dict['range_features']  # (B, C, H, W)
        if range_features.dim() != 4:
            raise ValueError(f"range_features must be 4D tensor, got {range_features.shape}")

        range_image = data_dict['range_image']
        if range_image.dim() != 4:
            raise ValueError(f"range_image must be 4D tensor, got {range_image.shape}")

        if range_image.shape[1] != range_features.shape[1]:
            range_image = range_image.permute(0, 3, 1, 2).contiguous()

        range_mask = data_dict.get('range_mask', None)
        if range_mask is None:
            raise KeyError("RangeViewScatter requires 'range_mask' to locate valid range pixels.")
        if range_mask.dim() == 3:
            range_mask = range_mask.unsqueeze(1)
        range_mask = range_mask.to(dtype=range_features.dtype)

        batch_size, num_channels, height, width = range_features.shape
        bev_height, bev_width = self.bev_shape
        device = range_features.device

        bev_features = torch.zeros((batch_size, num_channels, bev_height, bev_width), device=device, dtype=range_features.dtype)
        accumulation = torch.zeros((batch_size, 1, bev_height, bev_width), device=device, dtype=range_features.dtype)

        voxel_size = self.voxel_size.to(device)
        pc_range = self.point_cloud_range.to(device)
        x_offset = pc_range[0]
        y_offset = pc_range[1]
        grid_x_limit = pc_range[3]
        grid_y_limit = pc_range[4]
        eps = 1e-6

        range_features_flat = range_features.view(batch_size, num_channels, -1)
        x_coords = range_image[:, self.x_channel, :, :].view(batch_size, -1)
        y_coords = range_image[:, self.y_channel, :, :].view(batch_size, -1)
        mask_flat = range_mask.view(batch_size, -1) > 0.5

        for b in range(batch_size):
            valid = mask_flat[b]
            if valid.sum() == 0:
                continue

            xs = x_coords[b, valid]
            ys = y_coords[b, valid]
            feats = range_features_flat[b, :, valid]

            grid_x = torch.floor((xs - x_offset) / (voxel_size[0] + eps)).long()
            grid_y = torch.floor((ys - y_offset) / (voxel_size[1] + eps)).long()

            within_bounds = (
                (grid_x >= 0)
                & (grid_x < bev_width)
                & (grid_y >= 0)
                & (grid_y < bev_height)
                & (xs >= x_offset)
                & (xs <= grid_x_limit)
                & (ys >= y_offset)
                & (ys <= grid_y_limit)
            )

            if within_bounds.sum() == 0:
                continue

            grid_x = grid_x[within_bounds]
            grid_y = grid_y[within_bounds]
            feats = feats[:, within_bounds]

            linear_index = grid_y * bev_width + grid_x
            bev_flat = bev_features[b].view(num_channels, -1)
            acc_flat = accumulation[b].view(-1)

            bev_flat.index_add_(1, linear_index, feats)
            acc_flat.index_add_(0, linear_index, torch.ones_like(linear_index, dtype=acc_flat.dtype))

        if self.pooling_method == 'mean':
            denom = accumulation.clone()
            denom[denom == 0] = 1.0
            bev_features = bev_features / denom
            bev_features = bev_features * (accumulation > 0)

        data_dict['spatial_features'] = bev_features
        data_dict['spatial_features_stride'] = torch.tensor([1, voxel_size[0], voxel_size[1]], device=device)
        self.num_bev_features = range_features.shape[1]
        return data_dict
