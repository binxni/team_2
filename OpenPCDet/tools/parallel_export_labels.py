"""
3개 GPU를 사용하여 병렬로 라벨 추론을 수행하는 스크립트
데이터를 3등분하여 각 GPU에서 동시에 처리
"""

import argparse
import subprocess
import os
from pathlib import Path
import glob
import math


def parse_args():
    ap = argparse.ArgumentParser(description="Run parallel inference on 3 GPUs")
    ap.add_argument("--cfg_file", type=str, required=True, help="Path to model config yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    ap.add_argument("--data_dir", type=str, required=True, help="Folder of point clouds")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for label txt files")
    ap.add_argument("--ext", type=str, default=".npy", help="Point file extension")
    ap.add_argument("--score_thresh", type=float, default=0.3, help="Minimum score to export")
    ap.add_argument("--gpus", type=str, default="0,1,2", help="GPU IDs to use (comma separated)")
    return ap.parse_args()


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU 리스트 파싱
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    
    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    
    # 모든 파일 리스트 가져오기
    all_files = sorted(glob.glob(str(data_dir / f"*{args.ext}")))
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")
    
    # 파일을 GPU 개수만큼 분할
    files_per_gpu = math.ceil(total_files / num_gpus)
    
    # 임시 디렉토리 생성 (각 GPU가 처리할 파일들을 저장)
    temp_dirs = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * files_per_gpu
        end_idx = min((i + 1) * files_per_gpu, total_files)
        gpu_files = all_files[start_idx:end_idx]
        
        temp_dir = data_dir.parent / f"temp_gpu_{gpu_id}"
        temp_dir.mkdir(exist_ok=True)
        temp_dirs.append(temp_dir)
        
        print(f"GPU {gpu_id}: Processing {len(gpu_files)} files (indices {start_idx} to {end_idx-1})")
        
        # 심볼릭 링크 생성 (복사하지 않고 링크만 생성하여 빠르게)
        for file_path in gpu_files:
            src = Path(file_path)
            dst = temp_dir / src.name
            if not dst.exists():
                os.symlink(src, dst)
    
    # OpenPCDet 루트 디렉토리 찾기
    script_dir = Path(__file__).resolve().parent
    openpcdet_root = script_dir.parent
    
    # 설정 파일 경로를 절대 경로로 변환
    cfg_file_path = Path(args.cfg_file)
    if not cfg_file_path.is_absolute():
        cfg_file_path = openpcdet_root / cfg_file_path
    
    # 체크포인트 경로를 절대 경로로 변환
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = openpcdet_root / ckpt_path
    
    # 각 GPU에서 병렬로 실행
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        temp_dir = temp_dirs[i]
        
        cmd = [
            "python", "tools/export_labels_txt.py",
            "--cfg_file", str(cfg_file_path),
            "--ckpt", str(ckpt_path),
            "--data_dir", str(temp_dir.resolve()),
            "--out_dir", str(out_dir.resolve()),
            "--ext", args.ext,
            "--score_thresh", str(args.score_thresh),
            "--device", "cuda"
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # PYTHONPATH에 OpenPCDet 루트 디렉토리 추가
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{openpcdet_root}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(openpcdet_root)
        
        print(f"\nStarting process on GPU {gpu_id}...")
        print(f"Command: CUDA_VISIBLE_DEVICES={gpu_id} PYTHONPATH={openpcdet_root} {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd, env=env, cwd=str(openpcdet_root))
        processes.append(process)
    
    # 모든 프로세스가 끝날 때까지 대기
    print("\n" + "="*80)
    print("All processes started. Waiting for completion...")
    print("="*80 + "\n")
    
    for i, process in enumerate(processes):
        gpu_id = gpu_ids[i]
        print(f"Waiting for GPU {gpu_id} process to finish...")
        process.wait()
        if process.returncode == 0:
            print(f"✓ GPU {gpu_id} completed successfully")
        else:
            print(f"✗ GPU {gpu_id} failed with return code {process.returncode}")
    
    # 임시 디렉토리 정리
    print("\nCleaning up temporary directories...")
    for temp_dir in temp_dirs:
        # 심볼릭 링크 삭제
        for link in temp_dir.glob("*"):
            link.unlink()
        temp_dir.rmdir()
    
    print("\n" + "="*80)
    print("All done! Labels saved to:", out_dir)
    print("="*80)


if __name__ == "__main__":
    main()
