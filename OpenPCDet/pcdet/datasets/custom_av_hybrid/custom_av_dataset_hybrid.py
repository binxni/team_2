import copy
import pickle
import os
<<<<<<< HEAD
=======
from pathlib import Path
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38

import numpy as np
import torch 

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate



class CustomAvDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.custom_av_infos = []
<<<<<<< HEAD
=======
        # Allow selecting which point folder to read (e.g. points vs points_lisa)
        self.point_dir = self.dataset_cfg.get('POINT_DIR', 'points_lisa')
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    def include_data(self, mode):
<<<<<<< HEAD
        self.logger.info('Loading Custom AV dataset.')
=======
        self.logger.info('Loading Custom AV LISA dataset.')
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38
        custom_av_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_av_infos.extend(infos)

        self.custom_av_infos.extend(custom_av_infos)
<<<<<<< HEAD
        self.logger.info('Total samples for Custom AV dataset: %d' % (len(custom_av_infos)))
=======
        self.logger.info('Total samples for Custom AV LISA dataset: %d' % (len(custom_av_infos)))
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38

    def get_label(self, idx):
        label_file = self.root_path / 'labels' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
<<<<<<< HEAD
        lidar_file = self.root_path / 'points' / ('%s.npy' % idx)
        assert lidar_file.exists()
=======
        # Use the active point directory when fetching lidar frames
        lidar_file = self.root_path / self.point_dir / ('%s.npy' % idx)
        if not lidar_file.exists():
            raise FileNotFoundError(f'Point file not found: {lidar_file}')
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38
        point_features = np.load(lidar_file)
        return point_features

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
<<<<<<< HEAD
=======
        # Refresh point_dir if caller updated dataset_cfg between splits
        self.point_dir = self.dataset_cfg.get('POINT_DIR', self.point_dir)
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.sample_id_list = [sample_id for sample_id in self.sample_id_list if sample_id.strip()]


    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.custom_av_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_av_infos)

        info = copy.deepcopy(self.custom_av_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        # print("self.sample_id_list \n")
        # print(self.sample_id_list)


        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            # annos = common_utils.drop_info_with_name(annos, name='DontCare')
            
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_av_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval_martin as kitti_eval
            from ..kitti import kitti_utils
            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict
        def waymo_eval(eval_det_annos, eval_gt_annos):
            from ..waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_av_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))

            # 1) 포인트 로드
            points = self.get_lidar(sample_idx)

            # 2) 채널 수 추정 (ring이 5번째 컬럼이라고 가정)
            num_channels = None
            lidar_type = None
            if points.ndim == 2 and points.shape[1] >= 5:
                ring = points[:, 4].astype(np.int32, copy=False)
                # sanity check: 합리적 범위 내에서만 인정
                rmin, rmax = int(ring.min()), int(ring.max())
                if 0 <= rmin <= rmax <= 2048:
                    num_channels = rmax + 1
                    if num_channels == 64:
                        lidar_type = 'Pandar64'
                    elif num_channels == 128:
                        lidar_type = 'Pandar128'

            # 3) info 딕셔너리 구성
            info = {}
            pc_info = {
                'num_features': num_features,
                'lidar_idx'  : sample_idx,
                'num_channels': num_channels,       # None일 수 있음
                'lidar_type' : lidar_type           # None일 수 있음
            }
            info['point_cloud'] = pc_info

            # 4) 라벨 (옵션)
            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                num_pts_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes_lidar[:, :7])
                ).sum(dim=1).float().cpu().numpy()

                annotations['num_points_in_gt'] = num_pts_in_gt.astype(np.int64)
                annotations['difficulty'] = np.array([0] * gt_boxes_lidar.shape[0])
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

<<<<<<< HEAD
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('custom_av_dbinfos_%s.pkl' % split)
=======
        point_dir_str = str(self.point_dir)
        dir_tag = point_dir_str.strip().replace(os.sep, '_')
        suffix = '' if dir_tag in ('points_lisa', '') else f'_{dir_tag}'
        db_dir_name = 'gt_database' if split == 'train' else f'gt_database_{split}'
        database_save_path = Path(self.root_path) / f'{db_dir_name}{suffix}'
        db_info_save_path = Path(self.root_path) / f'custom_av_dbinfos_{split}{suffix}.pkl'
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
<<<<<<< HEAD
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
=======
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database*/xxxxx.bin
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


<<<<<<< HEAD
def create_custom_av_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
=======
def create_custom_av_infos(dataset_cfg, class_names, data_path, save_path, workers=4, point_dirs=None):  # pkl 만드는 함수
    data_path = Path(data_path)
    save_path = Path(save_path)
    # Build the list of point folders to process (single or multiple)
    raw_point_dirs = point_dirs or dataset_cfg.get('POINT_DIRS')
    if raw_point_dirs is None:
        raw_point_dirs = [dataset_cfg.get('POINT_DIR', 'points_lisa')]
    if isinstance(raw_point_dirs, str):
        raw_point_dirs = [raw_point_dirs]
    if not raw_point_dirs:
        raise ValueError('No point directories provided for info generation.')

    dataset_cfg.POINT_DIR = str(raw_point_dirs[0])

>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38
    dataset = CustomAvDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

<<<<<<< HEAD
    train_filename = save_path / ('custom_av_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_av_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_av_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_av_infos_train, f)
    print('custom_av info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    custom_av_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_av_infos_val, f)
    print('custom_av info val file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')
=======
    print('------------------------Start to generate data infos------------------------')

    for point_dir in raw_point_dirs:
        point_dir_str = str(point_dir)
        dir_tag = point_dir_str.strip().replace(os.sep, '_') or 'points'
        if len(raw_point_dirs) == 1 and save_path.name.endswith('_pkl'):
            output_dir = save_path
        else:
            output_dir = save_path / f'{dir_tag}_pkl'
        output_dir.mkdir(parents=True, exist_ok=True)

        train_filename = output_dir / ('custom_av_infos_%s.pkl' % train_split)
        val_filename = output_dir / ('custom_av_infos_%s.pkl' % val_split)

        # Iterate per point folder and emit independent info/db outputs
        print(f'Processing point directory "{point_dir_str}" -> {output_dir}')

        dataset.dataset_cfg.POINT_DIR = point_dir_str
        dataset.point_dir = point_dir_str

        dataset.set_split(train_split)
        custom_av_infos_train = dataset.get_infos(
            class_names, num_workers=workers, has_label=True, num_features=num_features
        )
        with open(train_filename, 'wb') as f:
            pickle.dump(custom_av_infos_train, f)
        print('custom_av info train file is saved to %s' % train_filename)

        dataset.set_split(val_split)
        custom_av_infos_val = dataset.get_infos(
            class_names, num_workers=workers, has_label=True, num_features=num_features
        )
        with open(val_filename, 'wb') as f:
            pickle.dump(custom_av_infos_val, f)
        print('custom_av info val file is saved to %s' % val_filename)

        print('------------------------Start create groundtruth database for data augmentation------------------------')
        dataset.set_split(train_split)
        dataset.create_groundtruth_database(train_filename, split=train_split)
        print('------------------------Data preparation done------------------------')
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_av_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_av_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
<<<<<<< HEAD
            data_path=ROOT_DIR / 'data' / 'custom_av_hybrid',
            save_path=ROOT_DIR / 'data' / 'custom_av_hybrid',
=======
            data_path=ROOT_DIR / 'data' / 'custom_av_64' ,
            save_path=ROOT_DIR / 'data' / 'custom_av_64' / 'points_hybrid',
>>>>>>> 078bf74d65cea9ec4e646ed65d42ceaf314c3d38
        )
