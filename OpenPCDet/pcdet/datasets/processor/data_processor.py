from functools import partial

import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - open3d is optional in some envs
    o3d = None

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
        self.voxel_mean_generator = None

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

    def transform_points_to_voxels_selective(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_selective, config=config)

        voxel_size = np.array(config.VOXEL_SIZE, dtype=np.float32)
        max_points_cfg = config.MAX_POINTS_PER_VOXEL
        if isinstance(max_points_cfg, dict):
            max_points_per_voxel = max_points_cfg[self.mode]
        else:
            max_points_per_voxel = int(max_points_cfg)

        max_voxels_cfg = config.MAX_NUMBER_OF_VOXELS
        if isinstance(max_voxels_cfg, dict):
            max_num_voxels = max_voxels_cfg[self.mode]
        else:
            max_num_voxels = int(max_voxels_cfg)

        point_cloud_min = self.point_cloud_range[:3]
        grid_size = self.grid_size

        def build_voxels(points):
            if points.shape[0] == 0:
                empty_voxels = np.zeros((0, max_points_per_voxel, points.shape[-1]), dtype=points.dtype)
                empty_coords = np.zeros((0, 3), dtype=np.int32)
                empty_num = np.zeros((0,), dtype=np.int32)
                return empty_voxels, empty_coords, empty_num

            coords = np.floor((points[:, :3] - point_cloud_min) / voxel_size).astype(np.int32)
            valid_mask = np.logical_and(coords >= 0, coords < grid_size).all(axis=1)
            points_valid = points[valid_mask]
            coords_valid = coords[valid_mask]

            voxel_points = {}
            voxel_order = []
            for pt, coord in zip(points_valid, coords_valid):
                coord_key = tuple(coord.tolist())
                if coord_key not in voxel_points:
                    if len(voxel_points) >= max_num_voxels:
                        continue
                    voxel_points[coord_key] = []
                    voxel_order.append(coord_key)
                voxel_points[coord_key].append(pt)

            num_voxels = len(voxel_order)
            feature_dim = points.shape[-1]
            voxels = np.zeros((num_voxels, max_points_per_voxel, feature_dim), dtype=points.dtype)
            voxel_coords = np.zeros((num_voxels, 3), dtype=np.int32)
            voxel_num_points = np.zeros((num_voxels,), dtype=np.int32)

            for idx, coord_key in enumerate(voxel_order):
                cur_points = np.stack(voxel_points[coord_key], axis=0)
                if cur_points.shape[0] > max_points_per_voxel:
                    if cur_points.shape[1] > 3:
                        importance = cur_points[:, 3]
                        order = np.argsort(-importance)
                    else:
                        center = (np.array(coord_key, dtype=np.float32) + 0.5) * voxel_size + point_cloud_min
                        distances = np.linalg.norm(cur_points[:, :3] - center, axis=1)
                        order = np.argsort(distances)
                    selected_idx = order[:max_points_per_voxel]
                    cur_points = cur_points[selected_idx]
                voxel_num = min(cur_points.shape[0], max_points_per_voxel)
                voxels[idx, :voxel_num] = cur_points[:voxel_num]
                voxel_coords[idx] = np.array(coord_key, dtype=np.int32)
                voxel_num_points[idx] = voxel_num

            return voxels, voxel_coords, voxel_num_points

        points = data_dict['points']
        voxels, coordinates, num_points = build_voxels(points)

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            for flipped_points in [points_yflip, points_xflip, points_xyflip]:
                cur_voxels, cur_coords, cur_num = build_voxels(flipped_points)
                if not data_dict['use_lead_xyz']:
                    cur_voxels = cur_voxels[..., 3:]
                voxels_list.append(cur_voxels)
                voxel_coords_list.append(cur_coords)
                voxel_num_points_list.append(cur_num)

            data_dict['voxels'] = voxels_list
            data_dict['voxel_coords'] = voxel_coords_list
            data_dict['voxel_num_points'] = voxel_num_points_list
        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict

    def lidsor_filter(self, data_dict=None, config=None):
        if data_dict is None:  # 함수 초기화 시 config를 캡처하도록 partial 반환
            return partial(self.lidsor_filter, config=config)

        if o3d is None:  # Open3D가 없으면 필터를 사용할 수 없으므로 즉시 알림
            raise ImportError("Open3D is required for the LIDSOR filter but is not installed.")

        points = data_dict.get('points', None)  # 필터 대상 포인트 클라우드를 조회
        if points is None or points.shape[0] == 0:  # 포인트가 없으면 그대로 반환
            return data_dict

        enabled_cfg = config.get('ENABLED', None)  # split별 사용 여부 설정값 읽기
        if enabled_cfg is not None:  # ENABLED가 지정돼 있으면 조건에 따라 조기 종료
            if isinstance(enabled_cfg, dict):  # train/test 별로 제어하는 경우
                if not enabled_cfg.get(self.mode, True):  # 현재 모드가 비활성화면 종료
                    return data_dict
            elif not enabled_cfg:  # 단일 불리언이 False면 종료
                return data_dict

        mean_k = int(config.get('MEAN_K', 50))  # KNN 이웃 수(평균 거리 계산에 사용)
        std_mul = float(config.get('STD_MUL', 0.15))  # 표준편차 배수 계수
        range_multiplier = float(config.get('RANGE_MULTIPLIER', 0.05))  # 거리 기반 스케일 팩터
        distance_threshold = float(config.get('DISTANCE_THRESHOLD', np.inf))  # 최대 거리 제한
        intensity_threshold = config.get('INTENSITY_THRESHOLD', None)  # 강도 임계값
        intensity_index = int(config.get('INTENSITY_INDEX', 3))  # 강도 컬럼 위치

        num_points = points.shape[0]  # 전체 포인트 수 계산
        k = min(mean_k, num_points)  # 실제 사용 이웃 수(포인트 수보다 크면 축소)
        if k <= 1:  # 유효한 이웃이 없으면 필터가 의미가 없으므로 그대로 반환
            return data_dict

        xyz = points[:, :3].astype(np.float64, copy=False)  # KDTree용 좌표 배열 생성
        cloud = o3d.geometry.PointCloud()  # Open3D 포인트클라우드 객체 준비
        cloud.points = o3d.utility.Vector3dVector(xyz)  # 좌표 데이터를 Open3D 형식으로 주입
        kdtree = o3d.geometry.KDTreeFlann(cloud)  # KNN 탐색용 KDTree 구성

        mean_distances = np.zeros(num_points, dtype=np.float64)  # 포인트별 평균 이웃 거리 버퍼
        for idx in range(num_points):  # 모든 포인트에 대해 반복하며 KNN 거리 계산
            _, _, dists = kdtree.search_knn_vector_3d(cloud.points[idx], k)  # 자기 포함 KNN 조회
            if len(dists) <= 1:  # 자기 자신만 반환되면 다음 포인트로 건너뜀
                continue
            neighbours = np.sqrt(np.asarray(dists[1:], dtype=np.float64))  # 제곱거리를 실제 거리로 변환
            if neighbours.size == 0:  # 이웃이 없으면 평균을 저장하지 않음
                continue
            mean_distances[idx] = neighbours.mean()  # 계산된 평균 이웃 거리를 기록

        valid_mask = mean_distances > 0  # 평균 거리가 유효하게 계산된 포인트만 선택
        if not np.any(valid_mask):  # 유효 포인트가 없으면 필터링을 생략
            return data_dict

        stats = mean_distances[valid_mask]  # 유효 포인트들에 대한 통계 배열 준비
        mean_val = stats.mean()  # 전체 평균 이웃 거리
        std_val = stats.std(ddof=1) if stats.size > 1 else 0.0  # 표준편차(샘플 분모 n-1)
        base_threshold = mean_val + std_mul * std_val  # 기본 임계값 계산

        ranges = np.linalg.norm(xyz, axis=1)  # 각 포인트의 원점 기준 거리
        dynamic_thresholds = base_threshold * range_multiplier * ranges  # 거리 비례 동적 임계값

        mean_condition = mean_distances > dynamic_thresholds  # 평균 거리가 임계값을 넘는지 검사

        if intensity_threshold is None or intensity_index >= points.shape[1]:  # 강도 조건 사용 여부 판별
            intensity_condition = np.ones(num_points, dtype=bool)  # 사용하지 않으면 항상 True
        else:
            intensity_vals = points[:, intensity_index]  # 강도 값 추출
            intensity_condition = intensity_vals < intensity_threshold  # 강도 조건 평가

        if not np.isfinite(distance_threshold):  # 거리 제한을 사용하지 않는 경우
            distance_condition = np.ones(num_points, dtype=bool)  # 모두 통과시킴
        else:
            distance_condition = ranges < distance_threshold  # 거리 조건 평가

        noise_mask = mean_condition & intensity_condition & distance_condition  # 세 조건을 모두 만족하는 포인트만 노이즈로 간주
        if not np.any(noise_mask):  # 노이즈가 한 개도 없으면 그대로 반환
            return data_dict

        keep_mask = ~noise_mask  # 유지할 포인트 마스크 계산
        data_dict['points'] = points[keep_mask]  # 노이즈 포인트를 제거한 배열로 교체
        return data_dict  # 필터링된 데이터 반환

    def voxel_mean_downsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.voxel_mean_downsample, config=config)

        points = data_dict.get('points', None)
        if points is None or len(points) == 0:
            return data_dict

        if not hasattr(self, 'voxel_mean_generator') or self.voxel_mean_generator is None:
            max_num_points = config.get('MAX_POINTS_PER_VOXEL', 5)
            max_num_voxels_cfg = config.get('MAX_NUMBER_OF_VOXELS', 200000)
            if isinstance(max_num_voxels_cfg, dict):
                max_num_voxels = max_num_voxels_cfg[self.mode]
            else:
                max_num_voxels = max_num_voxels_cfg

            self.voxel_mean_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=max_num_points,
                max_num_voxels=max_num_voxels
            )

        voxels, _, num_points = self.voxel_mean_generator.generate(points)
        valid_mask = num_points > 0
        if not np.any(valid_mask):
            data_dict['points'] = points[:0]
            return data_dict

        voxels = voxels[valid_mask]
        num_points = num_points[valid_mask]

        mean_points = voxels.sum(axis=1) / num_points[:, None]
        data_dict['points'] = mean_points.astype(points.dtype, copy=False)
        if config.get('SAVE_NUM_POINTS', False):
            data_dict['mean_points_per_voxel'] = num_points
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
