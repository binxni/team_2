#!/usr/bin/env python3
"""
비 노이즈가 추가된 포인트클라우드 데이터셋 생성기

Usage:
    python generate_rain_noise_dataset.py [--config config_file.yaml] [--start 0] [--end 100]
"""

import numpy as np
import os
import argparse
import yaml
from datetime import datetime
import time
import tempfile
import shutil
from rain_noise_utils import RainNoiseSimulator

# Configuration
DATA_PATH = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_aligned64"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "train_origin.txt")
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
LABELS_FOLDER = os.path.join(DATA_PATH, "labels")
OUTPUT_DIR = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_noise/points"

# 기본 노이즈 설정 (dataset_change.yaml과 동일)
DEFAULT_NOISE_CONFIG = {
    'APPLY_PROBABILITY': 0.9,                    # 모든 프레임에 적용
    'RAIN_INTENSITY_RANGE': [0.1, 2.5],         # 비 강도 범위
    'BASE_NOISE_DENSITY': 0.01,                 # 기본 노이즈 밀도
    'NOISE_RADIUS': 3.0,                        # 원점 기준 노이즈 생성 반경 (미터)
    'Z_BIAS_RANGE': [0.0, 6.0],                 # 노이즈 Z 범위
    'RAIN_INTENSITY_VALUES': [0.02, 0.25],      # 빗방울 intensity 범위
    'MAX_ATTENUATION_DISTANCE': 70.0,           # 감쇠 적용 최대 거리
    'BASE_ATTENUATION_RATE': 0.1,              # 기본 감쇠율
    'MAX_ATTENUATION_PROB': 0.35,               # 최대 감쇠 확률
    'INTENSITY_REDUCTION_FACTOR': 0.25,         # Intensity 감소 계수
    'DROPOUT_RATIO_RANGE': [0.08, 0.35],        # 포인트 드롭아웃 범위
}

class RainNoiseDatasetGenerator:
    def __init__(self, config=None):
        """
        비 노이즈 데이터셋 생성기 초기화
        
        Args:
            config: 노이즈 설정 딕셔너리
        """
        self.config = config if config is not None else DEFAULT_NOISE_CONFIG
        self.rain_simulator = RainNoiseSimulator(config=self.config)
    
    def load_frame_ids(self):
        """프레임 ID 목록을 로드합니다."""
        with open(FRAME_LIST_FILE, 'r') as f:
            frame_ids = [line.strip() for line in f.readlines()]
        return frame_ids
    
    def load_pointcloud(self, file_path):
        """포인트클라우드를 로드합니다."""
        if not os.path.exists(file_path):
            return None
        return np.load(file_path)
    
    def apply_rain_noise_with_dropout(self, points, rain_intensity=None):
        """
        포인트클라우드에 비 노이즈와 드롭아웃을 적용합니다.
        data_processor.py의 로직과 동일하게 구현
        """
        if rain_intensity is None:
            rain_intensity = np.random.uniform(
                self.config['RAIN_INTENSITY_RANGE'][0],
                self.config['RAIN_INTENSITY_RANGE'][1]
            )
        
        # 입력 데이터를 복사하여 원본 보호
        points = np.copy(points).astype(np.float32)
        
        # 1. 허위 반사점 생성 (빗방울)
        noise_points = self._generate_rain_droplet_noise(points, rain_intensity)
        
        # 2. 거리별 포인트 감쇠 시뮬레이션
        attenuated_points = self._apply_distance_attenuation(points, rain_intensity)
        
        # 3. Intensity 감소 시뮬레이션
        intensity_reduced_points = self._apply_intensity_attenuation(attenuated_points, rain_intensity)
        
        # 4. 추가 포인트 드롭아웃 (weather_point_dropout)
        final_points = self._apply_weather_dropout(intensity_reduced_points)
        
        # 5. 최종 포인트 결합
        if len(noise_points) > 0:
            # 데이터 타입과 차원 맞추기
            noise_points = noise_points.astype(np.float32)
            final_points = final_points.astype(np.float32)
            
            # 차원 확인 및 조정
            if final_points.shape[1] != noise_points.shape[1]:
                min_cols = min(final_points.shape[1], noise_points.shape[1])
                final_points = final_points[:, :min_cols]
                noise_points = noise_points[:, :min_cols]
            
            result = np.concatenate([final_points, noise_points], axis=0)
        else:
            result = final_points.astype(np.float32)
        
        # 메모리 연속성 보장
        result = np.ascontiguousarray(result)
        
        return result, rain_intensity
    
    def _generate_rain_droplet_noise(self, points, rain_intensity):
        """빗방울로 인한 허위 반사점 생성 (data_processor.py와 동일)"""
        base_noise_density = self.config.get('BASE_NOISE_DENSITY', 0.002)
        noise_density = base_noise_density * rain_intensity
        num_noise_points = int(len(points) * noise_density)
        
        if num_noise_points == 0:
            return np.array([]).reshape(0, points.shape[1])
        
        noise_radius = self.config.get('NOISE_RADIUS', 2.0)
        
        # 원점 기준 반경 내에서 노이즈 포인트 생성
        generated_points = []
        attempts = 0
        max_attempts = num_noise_points * 10
        
        while len(generated_points) < num_noise_points and attempts < max_attempts:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, noise_radius)
            
            noise_x = radius * np.cos(angle)
            noise_y = radius * np.sin(angle)
            
            z_bias_range = self.config.get('Z_BIAS_RANGE', [0.5, 8.0])
            noise_z = np.random.uniform(z_bias_range[0], z_bias_range[1])
            
            rain_intensity_range = self.config.get('RAIN_INTENSITY_VALUES', [0.05, 0.3])
            noise_intensity = np.random.uniform(
                rain_intensity_range[0], 
                rain_intensity_range[1]
            )
            
            generated_points.append([noise_x, noise_y, noise_z, noise_intensity])
            attempts += 1
        
        if len(generated_points) == 0:
            return np.array([]).reshape(0, points.shape[1])
        
        return np.array(generated_points)
    
    def _apply_distance_attenuation(self, points, rain_intensity):
        """거리별 포인트 감쇠 시뮬레이션 (data_processor.py와 동일)"""
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        
        max_distance = self.config.get('MAX_ATTENUATION_DISTANCE', 70.0)
        base_attenuation = self.config.get('BASE_ATTENUATION_RATE', 0.05)
        
        attenuation_rate = base_attenuation * rain_intensity
        attenuation_probs = np.clip(
            attenuation_rate * (distances / max_distance), 
            0.0, 
            self.config.get('MAX_ATTENUATION_PROB', 0.3)
        )
        
        keep_mask = np.random.random(len(points)) > attenuation_probs
        return points[keep_mask]
    
    def _apply_intensity_attenuation(self, points, rain_intensity):
        """Intensity 감소 시뮬레이션 (data_processor.py와 동일)"""
        if points.shape[1] < 4:
            return points
        
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        intensity_reduction = self.config.get('INTENSITY_REDUCTION_FACTOR', 0.1) * rain_intensity
        
        reduction_factors = 1.0 - (intensity_reduction * distances / 50.0)
        reduction_factors = np.clip(reduction_factors, 0.3, 1.0)
        
        points[:, 3] *= reduction_factors
        return points
    
    def _apply_weather_dropout(self, points):
        """날씨로 인한 추가 포인트 손실 시뮬레이션"""
        dropout_ratio = np.random.uniform(*self.config['DROPOUT_RATIO_RANGE'])
        keep_indices = np.random.choice(
            len(points), 
            int(len(points) * (1 - dropout_ratio)), 
            replace=False
        )
        return points[keep_indices]
    
    def process_single_frame(self, frame_id, rain_intensity=None):
        """단일 프레임에 비 노이즈를 적용하고 저장합니다."""
        start_time = time.time()
        
        # 원본 포인트클라우드 로드
        input_path = os.path.join(POINTS_FOLDER, f"{frame_id}.npy")
        points = self.load_pointcloud(input_path)
        
        if points is None:
            return None, f"File not found: {input_path}"
        
        original_point_count = len(points)
        
        # 비 노이즈 적용
        noisy_points, actual_rain_intensity = self.apply_rain_noise_with_dropout(points, rain_intensity)
        
        # 배열 정리 및 데이터 타입 확인
        noisy_points = np.ascontiguousarray(noisy_points, dtype=np.float32)
        
        # 출력 파일 저장 (안전한 방식)
        output_path = os.path.join(OUTPUT_DIR, f"{frame_id}.npy")
        try:
            np.save(output_path, noisy_points)
        except OSError as e:
            # 메모리 문제 시 임시 파일로 저장 후 이동
            import tempfile
            import shutil
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npy', dir=os.path.dirname(output_path)) as tmp_file:
                np.save(tmp_file.name, noisy_points)
                shutil.move(tmp_file.name, output_path)
        
        return {
            'frame_id': frame_id,
            'original_points': original_point_count,
            'noise_points': len(noisy_points),
            'rain_intensity': actual_rain_intensity,
            'processing_time': time.time() - start_time
        }, None
    
    def generate_dataset(self, start_idx=None, end_idx=None, fixed_rain_intensity=None):
        """
        비 노이즈 데이터셋을 생성합니다.
        
        Args:
            start_idx: 시작 프레임 인덱스
            end_idx: 끝 프레임 인덱스  
            fixed_rain_intensity: 고정 비 강도 (None이면 랜덤)
        """
        # 출력 디렉토리 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 프레임 ID 로드
        frame_ids = self.load_frame_ids()
        total_frames = len(frame_ids)
        
        # 범위 설정
        start_idx = start_idx if start_idx is not None else 0
        end_idx = end_idx if end_idx is not None else total_frames
        
        start_idx = max(0, start_idx)
        end_idx = min(total_frames, end_idx)
        
        if start_idx >= end_idx:
            print("❌ Invalid frame range!")
            return
        
        target_frame_count = end_idx - start_idx
        
        print(f"\n🌧️ Starting Rain Noise Dataset Generation...")
        print(f"📊 Processing frames {start_idx} to {end_idx-1} ({target_frame_count:,} total frames)")
        print(f"📁 Input path: {POINTS_FOLDER}")
        print(f"📁 Output path: {OUTPUT_DIR}")
        print(f"🎯 Rain intensity: {'Fixed ' + str(fixed_rain_intensity) if fixed_rain_intensity else 'Random'}")
        print("-" * 80)
        
        # 처리 시작
        start_time = time.time()
        successful_frames = []
        failed_frames = []
        
        for idx, i in enumerate(range(start_idx, end_idx)):
            frame_id = frame_ids[i]
            
            # 진행률 출력
            progress = (idx + 1) / target_frame_count
            elapsed_time = time.time() - start_time
            
            if idx % max(1, target_frame_count // 50) == 0 or idx == target_frame_count - 1:
                eta = elapsed_time / (idx + 1) * (target_frame_count - idx - 1) if idx > 0 else 0
                print(f"\\r⏳ Progress: {progress*100:5.1f}% [{idx+1:,}/{target_frame_count:,}] "
                      f"| Frame: {frame_id} | Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s", end="", flush=True)
            
            # 프레임 처리
            result, error = self.process_single_frame(frame_id, fixed_rain_intensity)
            
            if error:
                failed_frames.append((frame_id, error))
            else:
                successful_frames.append(result)
        
        print()  # 새 줄
        total_time = time.time() - start_time
        
        # 간단한 요약만 출력
        print(f"\n✅ Dataset generation completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"🎯 Successful frames: {len(successful_frames):,}")
        print(f"❌ Failed frames: {len(failed_frames):,}")
        
        if failed_frames:
            print(f"\n❌ Failed frames:")
            for frame_id, error in failed_frames[:10]:  # 처음 10개만 출력
                print(f"   {frame_id}: {error}")
            if len(failed_frames) > 10:
                print(f"   ... and {len(failed_frames) - 10} more")
        
        return successful_frames, failed_frames

def load_config_from_yaml(config_file):
    """YAML 설정 파일에서 노이즈 설정을 로드합니다."""
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return None
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # DATA_PROCESSOR에서 rain_noise_simulation 설정 추출
    if 'DATA_CONFIG' in config and 'DATA_PROCESSOR' in config['DATA_CONFIG']:
        for processor in config['DATA_CONFIG']['DATA_PROCESSOR']:
            if processor.get('NAME') == 'rain_noise_simulation':
                return {key: value for key, value in processor.items() if key != 'NAME'}
    
    return None

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Generate rain noise dataset from original point cloud data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (optional)')
    parser.add_argument('--start', type=int, default=None,
                       help='Start frame index (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='End frame index (default: all frames)')
    parser.add_argument('--rain_intensity', type=float, default=None,
                       help='Fixed rain intensity (default: random)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 랜덤 시드 설정
    np.random.seed(args.seed)
    
    # 설정 로드
    if args.config:
        config = load_config_from_yaml(args.config)
        if config is None:
            print("Failed to load config, using default settings")
            config = DEFAULT_NOISE_CONFIG
    else:
        config = DEFAULT_NOISE_CONFIG
    
    # 데이터셋 생성기 초기화
    generator = RainNoiseDatasetGenerator(config)
    
    # 데이터셋 생성
    successful_frames, failed_frames = generator.generate_dataset(
        start_idx=args.start,
        end_idx=args.end,
        fixed_rain_intensity=args.rain_intensity
    )
    
    print(f"\\n🎉 Rain noise dataset generation completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()