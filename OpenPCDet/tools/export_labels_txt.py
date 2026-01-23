import _init_path  # ensure 'pcdet' is importable without installation

import argparse
from pathlib import Path
import os
import glob
import numpy as np
import torch
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class NPYFolderDataset(DatasetTemplate):
    """Lightweight dataset that reads raw .npy point clouds from a folder.

    It mirrors tools/demo.py behavior but exposes frame_id (filename stem)
    so we can save per-frame prediction files.
    """

    def __init__(self, dataset_cfg, class_names, root_path: Path, logger=None, ext: str = ".npy"):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=root_path, logger=logger)
        self.root_path = Path(root_path)
        self.ext = ext
        if self.root_path.is_dir():
            self.sample_file_list = sorted(glob.glob(str(self.root_path / f"*{self.ext}")))
        else:
            self.sample_file_list = [str(self.root_path)]

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        path = self.sample_file_list[index]
        if self.ext == ".npy":
            points = np.load(path)
        elif self.ext == ".bin":
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        else:
            raise NotImplementedError(f"Unsupported extension: {self.ext}")

        frame_id = Path(path).stem
        input_dict = {
            "points": points,
            "frame_id": frame_id,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_args():
    ap = argparse.ArgumentParser(description="Run inference over a folder of .npy/.bin and export labels to txt")
    ap.add_argument("--cfg_file", type=str, required=True, help="Path to model config yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    ap.add_argument("--data_dir", type=str, required=True, help="Folder of point clouds (.npy or .bin)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for label txt files")
    ap.add_argument("--ext", type=str, default=".npy", help="Point file extension: .npy or .bin")
    ap.add_argument("--score_thresh", type=float, default=0.0, help="Minimum score to export")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    return ap.parse_args()


def main():
    args = parse_args()

    # OpenPCDet 루트로 작업 디렉토리 변경 (상대 경로 해결을 위해)
    import sys
    script_dir = Path(__file__).resolve().parent
    openpcdet_root = script_dir.parent
    original_cwd = Path.cwd()
    os.chdir(openpcdet_root)
    
    # 경로들을 절대 경로로 변환
    cfg_file = Path(args.cfg_file)
    if not cfg_file.is_absolute():
        cfg_file = original_cwd / cfg_file
    
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = original_cwd / data_dir
    
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = original_cwd / out_dir
    
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = original_cwd / ckpt_path
    
    # 체크포인트 경로를 절대 경로로 업데이트
    args.ckpt = str(ckpt_path)

    cfg_from_yaml_file(str(cfg_file), cfg)
    logger = common_utils.create_logger()

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = NPYFolderDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=data_dir,
        logger=logger,
        ext=args.ext,
    )
    logger.info(f"Total samples: {len(dataset)} from {data_dir}")

    # If running on CPU, ensure post-processing uses CPU NMS.
    if args.device == "cpu":
        try:
            cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG.NMS_TYPE = 'nms_cpu'
            logger.info("Using CPU NMS for post-processing")
        except Exception:
            logger.info("Could not set NMS_TYPE to nms_cpu; proceeding anyway")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=(args.device == "cpu"))
    model.to(args.device)
    model.eval()

    def load_data_to_device(batch_dict, device: str):
        """Device-agnostic alternative to load_data_to_gpu.

        Mirrors the behavior of pcdet.models.load_data_to_gpu but allows CPU.
        """
        import numpy as _np
        import torch as _torch
        try:
            import kornia as _kornia
        except Exception:
            _kornia = None

        for key, val in batch_dict.items():
            if key == 'camera_imgs':
                batch_dict[key] = val.to(device)
            elif not isinstance(val, _np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib', 'image_paths', 'ori_shape', 'img_process_infos']:
                continue
            elif key in ['images'] and _kornia is not None:
                batch_dict[key] = _kornia.image_to_tensor(val).float().to(device).contiguous()
            elif key in ['image_shape']:
                batch_dict[key] = _torch.from_numpy(val).int().to(device)
            else:
                batch_dict[key] = _torch.from_numpy(val).float().to(device)

    with torch.no_grad():
        for data_dict in tqdm(dataset, desc="Exporting labels"):
            batch = dataset.collate_batch([data_dict])
            if args.device == "cuda":
                load_data_to_gpu(batch)
            else:
                load_data_to_device(batch, args.device)
            pred_dicts, _ = model.forward(batch)
            pred = pred_dicts[0]

            # Move to CPU numpy
            boxes = pred["pred_boxes"].detach().cpu().numpy() if len(pred["pred_boxes"]) > 0 else np.zeros((0, 7), dtype=np.float32)
            scores = pred["pred_scores"].detach().cpu().numpy() if len(pred["pred_scores"]) > 0 else np.zeros((0,), dtype=np.float32)
            labels = pred["pred_labels"].detach().cpu().numpy() if len(pred["pred_labels"]) > 0 else np.zeros((0,), dtype=np.int64)

            # Filter by score
            if boxes.shape[0] > 0 and args.score_thresh > 0:
                keep = scores >= args.score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            # Sort by score desc for readability
            if boxes.shape[0] > 0:
                order = scores.argsort()[::-1]
                boxes = boxes[order]
                scores = scores[order]
                labels = labels[order]

            frame_id = batch["frame_id"][0] if isinstance(batch["frame_id"][0], str) else str(batch["frame_id"][0])
            out_path = out_dir / f"{frame_id}.txt"

            # Each line: x y z dx dy dz yaw score class_id
            with open(out_path, "w") as f:
                for i in range(boxes.shape[0]):
                    x, y, z, dx, dy, dz, yaw = boxes[i].tolist()
                    score = float(scores[i])
                    cls_id = int(labels[i])  # 1-based in OpenPCDet
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {dx:.6f} {dy:.6f} {dz:.6f} {yaw:.6f} {score:.6f} {cls_id}\n")

    logger.info(f"Done. Wrote labels to: {out_dir}")


if __name__ == "__main__":
    main()
