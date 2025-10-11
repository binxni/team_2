from __future__ import annotations

from .detector3d_template import Detector3DTemplate
from .. import backbones_2d, dense_heads
from ..backbones_2d import map_to_bev
from .. import backbones_rv


class RangeSparseNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = ['range_backbone', 'map_to_bev_module', 'backbone_2d', 'dense_head']
        self.module_list = self.build_networks()

    def build_range_backbone(self, model_info_dict):
        if self.model_cfg.get('RANGE_BACKBONE', None) is None:
            return None, model_info_dict

        cfg = self.model_cfg.RANGE_BACKBONE
        module = backbones_rv.__all__[cfg.NAME](
            model_cfg=cfg,
            input_channels=cfg.NUM_INPUT_CHANNELS
        )
        model_info_dict['module_list'].append(module)
        model_info_dict['num_range_features'] = cfg.STAGE_CHANNELS[-1]
        return module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_cfg = self.model_cfg.MAP_TO_BEV
        module = map_to_bev.__all__[map_cfg.NAME](
            model_cfg=map_cfg,
            grid_size=model_info_dict['grid_size'] if model_info_dict.get('grid_size') is not None else [0, 0, 0]
        )
        model_info_dict['module_list'].append(module)
        model_info_dict['num_bev_features'] = model_info_dict.get('num_range_features', map_cfg.INPUT_CHANNELS)
        bev_h, bev_w = module.bev_shape
        model_info_dict['grid_size'] = [1, bev_h, bev_w]
        model_info_dict['voxel_size'] = [map_cfg.VOXEL_SIZE[0], map_cfg.VOXEL_SIZE[1], 1.0]
        model_info_dict['point_cloud_range'] = map_cfg.POINT_CLOUD_RANGE
        return module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict.get('num_bev_features')
        )
        model_info_dict['module_list'].append(module)
        model_info_dict['num_bev_features'] = module.num_bev_features
        return module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict

        cfg = self.model_cfg.DENSE_HEAD
        module = dense_heads.__all__[cfg.NAME](
            model_cfg=cfg,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not cfg.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict.get('grid_size'),
            point_cloud_range=model_info_dict.get('point_cloud_range'),
            voxel_size=model_info_dict.get('voxel_size'),
            predict_boxes_when_training=cfg.get('PREDICT_BOXES_WHEN_TRAINING', False)
        )
        model_info_dict['module_list'].append(module)
        return module, model_info_dict

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict

        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {'loss_rpn': loss_rpn.item(), **tb_dict}
        return loss_rpn, tb_dict, disp_dict
