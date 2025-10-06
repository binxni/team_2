from functools import partial

import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    def double_flip(self, points):
        # y flip
        points_yflip = points.copy()
        points_yflip[:, 1] = -points_yflip[:, 1]

        # x flip
        points_xflip = points.copy()
        points_xflip[:, 0] = -points_xflip[:, 0]

        # x y flip
        points_xyflip = points.copy()
        points_xyflip[:, 0] = -points_xyflip[:, 0]
        points_xyflip[:, 1] = -points_xyflip[:, 1]

        return points_yflip, points_xflip, points_xyflip

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            data_dict['voxels'] = voxels_list
            data_dict['voxel_coords'] = voxel_coords_list
            data_dict['voxel_num_points'] = voxel_num_points_list
        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict
    
    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        return data_dict
    
    def image_calibrate(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        transforms = []
        for img_process_info in img_process_infos:
            resize, crop, flip, rotate = img_process_info

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict
    
    # pcdet/datasets/processor/data_processor.py에 추가

    def rain_noise_simulation(self, data_dict=None, config=None):
        """비 노이즈 시뮬레이션을 위한 데이터 프로세서"""
        if data_dict is None:
            return partial(self.rain_noise_simulation, config=config)
        
        # 학습 모드에서만 적용
        if not self.training:
            return data_dict
        
        # 설정된 확률로만 적용
        if np.random.random() > config.get('APPLY_PROBABILITY', 0.5):
            return data_dict
        
        points = data_dict['points']
        
        # 1. 비 강도 설정 (랜덤)
        rain_intensity = np.random.uniform(
            config.RAIN_INTENSITY_RANGE[0], 
            config.RAIN_INTENSITY_RANGE[1]
        )
        
        # 2. 허위 반사점 생성 (빗방울)
        noise_points = self._generate_rain_droplet_noise(points, rain_intensity, config)
        
        # 3. 거리별 포인트 감쇠 시뮬레이션
        points = self._apply_distance_attenuation(points, rain_intensity, config)
        
        # 4. Intensity 감소 시뮬레이션
        points = self._apply_intensity_attenuation(points, rain_intensity, config)
        
        # 5. 최종 포인트 결합
        if len(noise_points) > 0:
            data_dict['points'] = np.concatenate([points, noise_points], axis=0)
        else:
            data_dict['points'] = points
        
        return data_dict

    def _generate_rain_droplet_noise(self, points, rain_intensity, config):
        """빗방울로 인한 허위 반사점 생성 (원점 기준 반경 내에서만)"""
        # 노이즈 포인트 개수 계산
        base_noise_density = config.get('BASE_NOISE_DENSITY', 0.002)
        noise_density = base_noise_density * rain_intensity
        num_noise_points = int(len(points) * noise_density)
        
        if num_noise_points == 0:
            return np.array([]).reshape(0, points.shape[1])
        
        # 노이즈 생성 반경 설정 (기본값: 2.0m)
        noise_radius = config.get('NOISE_RADIUS', 2.0)
        
        # 원점 기준 반경 내에서만 노이즈 포인트 생성
        generated_points = []
        attempts = 0
        max_attempts = num_noise_points * 10  # 무한 루프 방지
        
        while len(generated_points) < num_noise_points and attempts < max_attempts:
            # 원점 기준 반경 내에서 랜덤 포인트 생성
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, noise_radius)
            
            noise_x = radius * np.cos(angle)
            noise_y = radius * np.sin(angle)
            
            # Z축은 주로 지상 위쪽에 집중 (비는 위에서 아래로)
            z_bias_range = config.get('Z_BIAS_RANGE', [0.5, 8.0])
            noise_z = np.random.uniform(z_bias_range[0], z_bias_range[1])
            
            # 빗방울의 낮은 intensity 시뮬레이션
            rain_intensity_range = config.get('RAIN_INTENSITY_VALUES', [0.05, 0.3])
            noise_intensity = np.random.uniform(
                rain_intensity_range[0], 
                rain_intensity_range[1]
            )
            
            generated_points.append([noise_x, noise_y, noise_z, noise_intensity])
            attempts += 1
        
        if len(generated_points) == 0:
            return np.array([]).reshape(0, points.shape[1])
        
        # 노이즈 포인트 구성 [x, y, z, intensity]
        noise_points = np.array(generated_points)
        
        return noise_points

    def _apply_distance_attenuation(self, points, rain_intensity, config):
        """거리별 포인트 감쇠 시뮬레이션"""
        # 원점으로부터의 거리 계산
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        
        # 거리별 감쇠 확률 계산
        max_distance = config.get('MAX_ATTENUATION_DISTANCE', 70.0)
        base_attenuation = config.get('BASE_ATTENUATION_RATE', 0.05)
        
        # 비 강도에 따른 감쇠율 조정
        attenuation_rate = base_attenuation * rain_intensity
        
        # 거리에 비례한 감쇠 확률 (멀수록 더 많이 제거)
        attenuation_probs = np.clip(
            attenuation_rate * (distances / max_distance), 
            0.0, 
            config.get('MAX_ATTENUATION_PROB', 0.3)
        )
        
        # 랜덤 샘플링으로 포인트 제거
        keep_mask = np.random.random(len(points)) > attenuation_probs
        
        return points[keep_mask]

    def _apply_intensity_attenuation(self, points, rain_intensity, config):
        """Intensity 감소 시뮬레이션"""
        if points.shape[1] < 4:  # intensity 채널이 없으면 스킵
            return points
        
        # 거리별 intensity 감소
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # 비 강도에 따른 intensity 감소율
        intensity_reduction = config.get('INTENSITY_REDUCTION_FACTOR', 0.1) * rain_intensity
        
        # 거리별 차등 적용
        reduction_factors = 1.0 - (intensity_reduction * distances / 50.0)
        reduction_factors = np.clip(reduction_factors, 0.3, 1.0)  # 최소 30%는 유지
        
        points[:, 3] *= reduction_factors
        
        return points

    def weather_point_dropout(self, data_dict=None, config=None):
        """날씨로 인한 포인트 손실 시뮬레이션"""
        if data_dict is None:
            return partial(self.weather_point_dropout, config=config)
        
        if not self.training or np.random.random() > config.get('APPLY_PROBABILITY', 0.3):
            return data_dict
        
        points = data_dict['points']
        
        # 균등 드롭아웃
        dropout_ratio = np.random.uniform(*config.DROPOUT_RATIO_RANGE)
        keep_indices = np.random.choice(
            len(points), 
            int(len(points) * (1 - dropout_ratio)), 
            replace=False
        )
        
        data_dict['points'] = points[keep_indices]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
