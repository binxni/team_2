#!/usr/bin/env python3
"""
포인트 클라우드 데이터에서 거리별 복셀 내 포인트 개수 분석
"""

import numpy as np
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def create_voxels_optimized(points, inv_voxel_size, point_cloud_range_min, grid_size, 
                           max_points_per_voxel, max_number_of_voxels=150000):
    """
    포인트 클라우드를 복셀로 변환 (고도 최적화된 버전)
    """
    # 포인트 클라우드 범위 내의 포인트만 필터링 (벡터화)
    mask = (points[:, 0] >= point_cloud_range_min[0]) & (points[:, 0] <= point_cloud_range_min[0] + grid_size[0] * inv_voxel_size[0]) & \
           (points[:, 1] >= point_cloud_range_min[1]) & (points[:, 1] <= point_cloud_range_min[1] + grid_size[1] * inv_voxel_size[1]) & \
           (points[:, 2] >= point_cloud_range_min[2]) & (points[:, 2] <= point_cloud_range_min[2] + grid_size[2] * inv_voxel_size[2])
    points = points[mask]
    
    if len(points) == 0:
        return np.array([]), np.array([])
    
    # 포인트를 복셀 인덱스로 변환 (사전 계산된 값 사용)
    voxel_indices = ((points[:, :3] - point_cloud_range_min) * inv_voxel_size).astype(np.int32)
    
    # 복셀 인덱스를 단일 값으로 변환 (해시) - 사전 계산된 상수 사용
    gy_gz = grid_size[1] * grid_size[2]
    voxel_hash = voxel_indices[:, 0] * gy_gz + voxel_indices[:, 1] * grid_size[2] + voxel_indices[:, 2]
    
    # 각 복셀별 포인트 그룹화
    unique_hashes, counts = np.unique(voxel_hash, return_counts=True)
    
    # MAX_NUMBER_OF_VOXELS 제한 적용 (argpartition 사용)
    if len(unique_hashes) > max_number_of_voxels:
        # Top-k 선택을 argpartition으로 최적화
        topk_idx = np.argpartition(counts, -max_number_of_voxels)[-max_number_of_voxels:]
        topk_idx = topk_idx[np.argsort(counts[topk_idx])[::-1]]
        
        unique_hashes = unique_hashes[topk_idx]
        counts = counts[topk_idx]
    
    # 복셀 좌표 계산 (완전 벡터화 - 루프 제거)
    ix = unique_hashes // gy_gz
    rem = unique_hashes % gy_gz
    iy = rem // grid_size[2]
    iz = rem % grid_size[2]
    coordinates = np.stack([ix, iy, iz], axis=1).astype(np.int32)
    
    # 랜덤 샘플링 제거 - 단순 클리핑만
    num_points_per_voxel = np.minimum(counts, max_points_per_voxel)
    
    return coordinates, num_points_per_voxel

def calculate_distance_squared_from_origin(coordinates, voxel_size, point_cloud_range_min):
    """
    복셀 좌표에서 원점까지의 거리 제곱 계산 (sqrt 제거)
    """
    # 복셀 중심점 계산 (사전 계산된 값 사용)
    voxel_centers = (coordinates + 0.5) * voxel_size + point_cloud_range_min
    
    # 원점(0, 0, 0)으로부터의 거리 제곱 계산 (XY 평면에서)
    distances_squared = voxel_centers[:, 0]**2 + voxel_centers[:, 1]**2
    
    return distances_squared

class StreamingStats:
    """스트리밍 통계 클래스 - 메모리 효율적인 통계 계산"""
    def __init__(self, max_points_per_voxel=5):
        self.max_points_per_voxel = max_points_per_voxel
        self.reset()
    
    def reset(self):
        self.n_voxels = 0
        self.sum_counts = 0
        self.sumsq_counts = 0
        # 히스토그램: 1, 2, 3, 4, 5개 포인트를 가진 복셀 수
        self.hist_counts = np.zeros(self.max_points_per_voxel, dtype=np.int64)
        
    def update(self, counts):
        """새로운 복셀 포인트 수 배열로 통계 업데이트"""
        if len(counts) == 0:
            return
            
        self.n_voxels += len(counts)
        self.sum_counts += np.sum(counts)
        self.sumsq_counts += np.sum(counts.astype(np.float64) ** 2)
        
        # 히스토그램 업데이트 (벡터화)
        for i in range(1, self.max_points_per_voxel + 1):
            self.hist_counts[i-1] += np.sum(counts == i)
    
    def get_stats(self):
        """현재 통계 반환"""
        if self.n_voxels == 0:
            return {
                'n_voxels': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0,
                'max': 0,
                'median': 0.0,
                'histogram': self.hist_counts.copy()
            }
        
        mean = self.sum_counts / self.n_voxels
        variance = (self.sumsq_counts / self.n_voxels) - (mean ** 2)
        std = np.sqrt(max(0, variance))
        
        # 히스토그램에서 최소/최대값 계산
        min_val = 0
        max_val = 0
        for i in range(self.max_points_per_voxel):
            if self.hist_counts[i] > 0:
                if min_val == 0:
                    min_val = i + 1
                max_val = i + 1
        
        # 중간값 추정 (히스토그램 기반)
        cumsum = np.cumsum(self.hist_counts)
        median_pos = self.n_voxels // 2
        median_idx = np.searchsorted(cumsum, median_pos)
        median = median_idx + 1 if median_idx < self.max_points_per_voxel else self.max_points_per_voxel
        
        return {
            'n_voxels': self.n_voxels,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'median': float(median),
            'histogram': self.hist_counts.copy()
        }

def process_single_file(args):
    """단일 파일 처리 함수 (멀티프로세싱용)"""
    file_path, inv_voxel_size, point_cloud_range_min, grid_size, max_points_per_voxel, max_number_of_voxels, bins_squared, file_patterns = args
    
    # 파일 패턴 식별
    filename = os.path.basename(file_path)
    file_pattern = None
    for pattern in file_patterns:
        if filename.startswith(pattern):
            file_pattern = pattern
            break
    
    # 포인트 클라우드 로드 (메모리 맵 사용)
    points = np.load(file_path, mmap_mode='r')
    original_point_count = len(points)
    
    # 복셀 생성
    coordinates, num_points_per_voxel = create_voxels_optimized(
        points, inv_voxel_size, point_cloud_range_min, grid_size, 
        max_points_per_voxel, max_number_of_voxels
    )
    
    # 복셀화 후 남은 포인트 수 계산
    remaining_point_count = np.sum(num_points_per_voxel) if len(num_points_per_voxel) > 0 else 0
    
    result = {
        'file_pattern': file_pattern,
        'original_points': original_point_count,
        'remaining_points': remaining_point_count,
        'total_voxels': len(coordinates),
        'distance_stats': {}
    }
    
    if len(coordinates) == 0:
        return result
    
    # 거리 제곱 계산 (sqrt 제거)
    distances_squared = calculate_distance_squared_from_origin(
        coordinates, inv_voxel_size, point_cloud_range_min
    )
    
    # 거리별 분류 (제곱 거리로 비교)
    for i, (min_dist_sq, max_dist_sq, range_key) in enumerate(bins_squared):
        mask = (distances_squared >= min_dist_sq) & (distances_squared < max_dist_sq)
        if np.any(mask):
            points_in_range = num_points_per_voxel[mask]
            
            # 스트리밍 통계로 저장
            stats = StreamingStats(max_points_per_voxel)
            stats.update(points_in_range)
            result['distance_stats'][range_key] = stats.get_stats()
    
    return result

def analyze_voxel_points_by_distance(data_path, point_cloud_range, voxel_size, max_points_per_voxel, 
                                   distance_bins=None, max_number_of_voxels=150000):
    """
    거리별 및 파일 패턴별 복셀 내 포인트 개수 분석 (최적화된 버전)
    """
    if distance_bins is None:
        distance_bins = list(range(0, 81, 10))  # 0-10m, 10-20m, ..., 70-80m
    
    # 사전 계산 - 고정값들
    inv_voxel_size = 1.0 / voxel_size
    point_cloud_range_min = point_cloud_range[:3]
    grid_size = np.array([
        (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0],
        (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1],
        (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]
    ], dtype=np.int32)
    
    # 거리 구간을 제곱으로 미리 계산
    bins_squared = []
    for j in range(len(distance_bins) - 1):
        min_dist = distance_bins[j]
        max_dist = distance_bins[j + 1]
        range_key = f"{min_dist}-{max_dist}m"
        bins_squared.append((min_dist**2, max_dist**2, range_key))
    
    # 파일 패턴 정의
    file_patterns = ['000', '001', '002', '003', '100', '101']
    
    all_files = []
    pattern_files = {}
    
    print("파일 패턴별 검색 중...")
    for pattern in tqdm(file_patterns, desc="패턴별 파일 검색"):
        files = list(glob.iglob(os.path.join(data_path, f"{pattern}*.npy")))
        pattern_files[pattern] = files
        all_files.extend(files)
    
    all_files.sort()
    print(f"총 {len(all_files)}개의 파일을 찾았습니다.")
    
    # 패턴별 파일 개수 출력
    for pattern, files in pattern_files.items():
        print(f"  {pattern}*: {len(files)}개 파일")
    
    # 멀티프로세싱으로 파일 처리
    # num_processes = min(mp.cpu_count(), len(all_files))
    num_processes = min(8, len(all_files))
    print(f"멀티프로세싱 사용: {num_processes}개 프로세스")
    
    # 각 파일 처리를 위한 인자 준비
    process_args = [
        (file_path, inv_voxel_size, point_cloud_range_min, grid_size, 
         max_points_per_voxel, max_number_of_voxels, bins_squared, file_patterns)
        for file_path in all_files
    ]
    
    # 결과 수집을 위한 구조체 초기화
    distance_stats = {}
    pattern_distance_stats = {}  
    pattern_point_stats = {}
    
    for pattern in file_patterns:
        pattern_distance_stats[pattern] = {}
        pattern_point_stats[pattern] = {
            'original_points': 0,
            'remaining_points': 0,
            'total_voxels': 0,
            'files_processed': 0
        }
    
    # 거리 범위별 스트리밍 통계 초기화
    for _, _, range_key in bins_squared:
        distance_stats[range_key] = StreamingStats(max_points_per_voxel)
        for pattern in file_patterns:
            pattern_distance_stats[pattern][range_key] = StreamingStats(max_points_per_voxel)
    
    # 멀티프로세싱 실행
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(
            executor.map(process_single_file, process_args),
            total=len(all_files),
            desc="파일 처리 중"
        ))
    
    # 결과 집계
    for result in results:
        file_pattern = result['file_pattern']
        
        # 패턴별 포인트 통계 업데이트
        if file_pattern:
            pattern_point_stats[file_pattern]['original_points'] += result['original_points']
            pattern_point_stats[file_pattern]['remaining_points'] += result['remaining_points']
            pattern_point_stats[file_pattern]['total_voxels'] += result['total_voxels']
            pattern_point_stats[file_pattern]['files_processed'] += 1
        
        # 거리별 통계 집계
        for range_key, stats_dict in result['distance_stats'].items():
            if stats_dict['n_voxels'] > 0:
                # 전체 통계에 직접 합산 (스트리밍 통계 객체 사용)
                # 실제로는 각 파일의 통계를 재구성해서 합산해야 하지만,
                # 여기서는 단순화를 위해 히스토그램만 합산
                distance_stats[range_key].n_voxels += stats_dict['n_voxels']
                distance_stats[range_key].sum_counts += int(stats_dict['mean'] * stats_dict['n_voxels'])
                distance_stats[range_key].sumsq_counts += int((stats_dict['std']**2 + stats_dict['mean']**2) * stats_dict['n_voxels'])
                distance_stats[range_key].hist_counts += stats_dict['histogram']
                
                # 패턴별 통계도 동일하게 업데이트
                if file_pattern:
                    pattern_distance_stats[file_pattern][range_key].n_voxels += stats_dict['n_voxels']
                    pattern_distance_stats[file_pattern][range_key].sum_counts += int(stats_dict['mean'] * stats_dict['n_voxels'])
                    pattern_distance_stats[file_pattern][range_key].sumsq_counts += int((stats_dict['std']**2 + stats_dict['mean']**2) * stats_dict['n_voxels'])
                    pattern_distance_stats[file_pattern][range_key].hist_counts += stats_dict['histogram']
    
    return distance_stats, pattern_distance_stats, pattern_point_stats

def plot_results(distance_stats, save_path='/home/ailab/git/Team_2/Seokjae/OpenPCDet/dataset_analysis'):
    """
    결과 시각화 및 저장 (스트리밍 통계 버전)
    """
    # 저장 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 스트리밍 통계에서 값 추출
    ranges = list(distance_stats.keys())
    stats_data = [distance_stats[range_key].get_stats() for range_key in ranges]
    
    avg_points = [stat['mean'] for stat in stats_data]
    std_points = [stat['std'] for stat in stats_data]
    voxel_counts = [stat['n_voxels'] for stat in stats_data]
    
    # 1. 거리별 평균 포인트 수
    plt.subplot(2, 2, 1)
    plt.bar(ranges, avg_points, yerr=std_points, capsize=5, alpha=0.7)
    plt.title('거리별 복셀 내 평균 포인트 수')
    plt.xlabel('거리 범위')
    plt.ylabel('평균 포인트 수')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. 거리별 복셀 수
    plt.subplot(2, 2, 2)
    plt.bar(ranges, voxel_counts, alpha=0.7, color='orange')
    plt.title('거리별 복셀 수')
    plt.xlabel('거리 범위')
    plt.ylabel('복셀 수')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. 포인트 수 분포 히스토그램 (히스토그램 기반)
    plt.subplot(2, 2, 3)
    unique_ranges = ranges[:4]  # 처음 4개만 표시
    
    x_positions = np.arange(1, 6)  # 1, 2, 3, 4, 5 포인트
    width = 0.8 / len(unique_ranges)
    
    for i, range_name in enumerate(unique_ranges):
        stats = distance_stats[range_name].get_stats()
        hist = stats['histogram']
        # 정규화 (확률 밀도)
        if stats['n_voxels'] > 0:
            hist_normalized = hist / stats['n_voxels']
            plt.bar(x_positions + i * width, hist_normalized, width, 
                   alpha=0.7, label=range_name)
    
    plt.title('거리별 복셀 내 포인트 수 분포')
    plt.xlabel('복셀 내 포인트 수')
    plt.ylabel('확률 밀도')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x_positions + width * (len(unique_ranges) - 1) / 2, 
               ['1', '2', '3', '4', '5'])
    
    # 4. 통계 요약 테이블
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 테이블 데이터 준비
    table_data = []
    for i, range_name in enumerate(ranges):
        stats = stats_data[i]
        table_data.append([
            range_name,
            f"{stats['n_voxels']:,}",
            f"{stats['mean']:.2f}",
            f"{stats['std']:.2f}",
            f"{stats['median']:.1f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['거리범위', '복셀수', '평균', '표준편차', '중간값'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('거리별 통계 요약', pad=20)
    
    plt.tight_layout()
    
    # 그래프 저장
    plot_filename = os.path.join(save_path, 'voxel_analysis_plots.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"시각화 결과가 저장되었습니다: {plot_filename}")
    
    plt.show()
    
    # 개별 그래프들도 저장
    save_individual_plots(distance_stats, save_path)

def save_individual_plots(distance_stats, save_path):
    """
    개별 그래프들을 따로 저장 (스트리밍 통계 버전)
    """
    ranges = list(distance_stats.keys())
    stats_data = [distance_stats[range_key].get_stats() for range_key in ranges]
    avg_points = [stat['mean'] for stat in stats_data]
    std_points = [stat['std'] for stat in stats_data]
    voxel_counts = [stat['n_voxels'] for stat in stats_data]
    
    # 1. 거리별 평균 포인트 수 그래프
    plt.figure(figsize=(10, 6))
    plt.bar(ranges, avg_points, yerr=std_points, capsize=5, alpha=0.7)
    plt.title('거리별 복셀 내 평균 포인트 수', fontsize=14)
    plt.xlabel('거리 범위', fontsize=12)
    plt.ylabel('평균 포인트 수', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'avg_points_per_voxel.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 거리별 복셀 수 그래프
    plt.figure(figsize=(10, 6))
    plt.bar(ranges, voxel_counts, alpha=0.7, color='orange')
    plt.title('거리별 복셀 수', fontsize=14)
    plt.xlabel('거리 범위', fontsize=12)
    plt.ylabel('복셀 수', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'voxel_count_by_distance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 포인트 수 분포 히스토그램 (모든 거리 범위) - 스트리밍 통계 버전
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(ranges)))
    
    x_positions = np.arange(1, 6)  # 1, 2, 3, 4, 5 포인트
    width = 0.8 / len(ranges)
    
    for i, range_name in enumerate(ranges):
        stats = stats_data[i]
        if stats['n_voxels'] > 0:
            hist_normalized = stats['histogram'] / stats['n_voxels']
            plt.bar(x_positions + i * width, hist_normalized, width,
                   alpha=0.7, label=range_name, color=colors[i])
    
    plt.title('거리별 복셀 내 포인트 수 분포', fontsize=14)
    plt.xlabel('복셀 내 포인트 수', fontsize=12)
    plt.ylabel('확률 밀도', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(x_positions + width * (len(ranges) - 1) / 2, ['1', '2', '3', '4', '5'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'point_distribution_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 통계 요약 그래프 (박스플롯 대신)
    plt.figure(figsize=(12, 8))
    
    # 여러 통계치를 막대 그래프로 표시
    x = np.arange(len(ranges))
    width = 0.2
    
    plt.bar(x - width, avg_points, width, label='평균', alpha=0.8)
    plt.bar(x, [stat['median'] for stat in stats_data], width, label='중간값', alpha=0.8)
    plt.bar(x + width, [stat['max'] for stat in stats_data], width, label='최대값', alpha=0.8)
    
    plt.title('거리별 복셀 내 포인트 수 통계 비교', fontsize=14)
    plt.xlabel('거리 범위', fontsize=12)
    plt.ylabel('포인트 수', fontsize=12)
    plt.xticks(x, ranges, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'statistics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"개별 그래프들이 저장되었습니다:")
    print(f"  - {os.path.join(save_path, 'avg_points_per_voxel.png')}")
    print(f"  - {os.path.join(save_path, 'voxel_count_by_distance.png')}")
    print(f"  - {os.path.join(save_path, 'point_distribution_histogram.png')}")
    print(f"  - {os.path.join(save_path, 'statistics_comparison.png')}")

def plot_pattern_comparison(pattern_distance_stats, save_path):
    """
    파일 패턴별 비교 시각화
    """
    patterns = list(pattern_distance_stats.keys())
    if not patterns:
        return
    
    # 거리 범위 추출 (첫 번째 패턴에서)
    distance_ranges = list(pattern_distance_stats[patterns[0]].keys())
    
    # 1. 패턴별 평균 포인트 수 비교
    plt.figure(figsize=(15, 10))
    
    # 거리별로 패턴 비교
    x = np.arange(len(distance_ranges))
    width = 0.8 / len(patterns)
    
    plt.subplot(2, 2, 1)
    for i, pattern in enumerate(patterns):
        avg_points = []
        for range_key in distance_ranges:
            if range_key in pattern_distance_stats[pattern] and len(pattern_distance_stats[pattern][range_key]) > 0:
                avg_points.append(np.mean(pattern_distance_stats[pattern][range_key]))
            else:
                avg_points.append(0)
        
        plt.bar(x + i * width, avg_points, width, label=f"{pattern}*", alpha=0.8)
    
    plt.title('파일 패턴별 거리별 평균 포인트 수')
    plt.xlabel('거리 범위')
    plt.ylabel('평균 포인트 수')
    plt.xticks(x + width * (len(patterns) - 1) / 2, distance_ranges, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 패턴별 복셀 수 비교
    plt.subplot(2, 2, 2)
    for i, pattern in enumerate(patterns):
        voxel_counts = []
        for range_key in distance_ranges:
            if range_key in pattern_distance_stats[pattern]:
                voxel_counts.append(len(pattern_distance_stats[pattern][range_key]))
            else:
                voxel_counts.append(0)
        
        plt.bar(x + i * width, voxel_counts, width, label=f"{pattern}*", alpha=0.8)
    
    plt.title('파일 패턴별 거리별 복셀 수')
    plt.xlabel('거리 범위')
    plt.ylabel('복셀 수')
    plt.xticks(x + width * (len(patterns) - 1) / 2, distance_ranges, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 패턴별 전체 통계 (거리 무관)
    plt.subplot(2, 2, 3)
    pattern_total_avg = []
    pattern_total_std = []
    pattern_labels = []
    
    for pattern in patterns:
        all_points = []
        for range_key in distance_ranges:
            if range_key in pattern_distance_stats[pattern]:
                all_points.extend(pattern_distance_stats[pattern][range_key])
        
        if all_points:
            pattern_total_avg.append(np.mean(all_points))
            pattern_total_std.append(np.std(all_points))
            pattern_labels.append(f"{pattern}*")
    
    plt.bar(pattern_labels, pattern_total_avg, yerr=pattern_total_std, capsize=5, alpha=0.7)
    plt.title('파일 패턴별 전체 평균 포인트 수')
    plt.xlabel('파일 패턴')
    plt.ylabel('평균 포인트 수')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. 패턴별 포인트 수 분포 (히트맵)
    plt.subplot(2, 2, 4)
    
    # 히트맵 데이터 준비
    heatmap_data = []
    for pattern in patterns:
        pattern_row = []
        for range_key in distance_ranges:
            if range_key in pattern_distance_stats[pattern] and len(pattern_distance_stats[pattern][range_key]) > 0:
                pattern_row.append(np.mean(pattern_distance_stats[pattern][range_key]))
            else:
                pattern_row.append(0)
        heatmap_data.append(pattern_row)
    
    im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='평균 포인트 수')
    plt.title('파일 패턴별 거리별 평균 포인트 수 히트맵')
    plt.xlabel('거리 범위')
    plt.ylabel('파일 패턴')
    plt.xticks(range(len(distance_ranges)), distance_ranges, rotation=45)
    plt.yticks(range(len(patterns)), [f"{p}*" for p in patterns])
    
    # 히트맵에 값 표시
    for i in range(len(patterns)):
        for j in range(len(distance_ranges)):
            plt.text(j, i, f'{heatmap_data[i][j]:.2f}', ha='center', va='center')
    
    plt.tight_layout()
    
    # 저장
    plot_filename = os.path.join(save_path, 'pattern_comparison_plots.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"패턴 비교 시각화가 저장되었습니다: {plot_filename}")
    
    plt.show()

def print_pattern_point_statistics(pattern_point_stats):
    """
    패턴별 포인트 수 통계 출력 (복셀화 전후 비교)
    """
    print("\n" + "="*80)
    print("파일 패턴별 복셀화 전후 포인트 수 통계")
    print("="*80)
    
    for pattern, stats in pattern_point_stats.items():
        if stats['files_processed'] > 0:
            retention_rate = (stats['remaining_points'] / stats['original_points']) * 100 if stats['original_points'] > 0 else 0
            avg_voxels_per_file = stats['total_voxels'] / stats['files_processed']
            avg_original_per_file = stats['original_points'] / stats['files_processed']
            avg_remaining_per_file = stats['remaining_points'] / stats['files_processed']
            
            print(f"\n{pattern}* 패턴:")
            print("-" * 50)
            print(f"  처리된 파일 수: {stats['files_processed']:,}")
            print(f"  전체 원본 포인트 수: {stats['original_points']:,}")
            print(f"  전체 복셀화 후 포인트 수: {stats['remaining_points']:,}")
            print(f"  포인트 보존율: {retention_rate:.2f}%")
            print(f"  전체 생성된 복셀 수: {stats['total_voxels']:,}")
            print(f"  파일당 평균:")
            print(f"    원본 포인트 수: {avg_original_per_file:,.0f}")
            print(f"    복셀화 후 포인트 수: {avg_remaining_per_file:,.0f}")
            print(f"    생성된 복셀 수: {avg_voxels_per_file:,.0f}")

def print_pattern_statistics(pattern_distance_stats):
    """
    패턴별 통계 정보 출력 (스트리밍 통계 버전)
    """
    print("\n" + "="*80)
    print("파일 패턴별 복셀 내 포인트 수 통계")
    print("="*80)
    
    for pattern, distance_data in pattern_distance_stats.items():
        print(f"\n{pattern}* 패턴:")
        print("-" * 50)
        
        # 패턴별 전체 통계 계산
        total_voxels = 0
        total_sum = 0
        total_sumsq = 0
        combined_hist = np.zeros(5, dtype=np.int64)
        
        for range_key, streaming_stats in distance_data.items():
            stats = streaming_stats.get_stats()
            total_voxels += stats['n_voxels']
            total_sum += stats['mean'] * stats['n_voxels']
            total_sumsq += (stats['std']**2 + stats['mean']**2) * stats['n_voxels']
            combined_hist += stats['histogram']
        
        if total_voxels > 0:
            overall_mean = total_sum / total_voxels
            overall_variance = (total_sumsq / total_voxels) - (overall_mean ** 2)
            overall_std = np.sqrt(max(0, overall_variance))
            
            # 최소/최대값 계산
            min_val = 0
            max_val = 0
            for i in range(5):
                if combined_hist[i] > 0:
                    if min_val == 0:
                        min_val = i + 1
                    max_val = i + 1
            
            # 중간값 추정
            cumsum = np.cumsum(combined_hist)
            median_pos = total_voxels // 2
            median_idx = np.searchsorted(cumsum, median_pos)
            median = median_idx + 1 if median_idx < 5 else 5
            
            print(f"  전체 복셀 수: {total_voxels:,}")
            print(f"  전체 평균 포인트 수: {overall_mean:.2f}")
            print(f"  전체 표준편차: {overall_std:.2f}")
            print(f"  전체 최소값: {min_val}")
            print(f"  전체 최대값: {max_val}")
            print(f"  전체 중간값: {median:.2f}")
            
            print(f"  거리별 세부 통계:")
            for range_key, streaming_stats in distance_data.items():
                stats = streaming_stats.get_stats()
                if stats['n_voxels'] > 0:
                    print(f"    {range_key}:")
                    print(f"      복셀 수: {stats['n_voxels']:,}")
                    print(f"      평균: {stats['mean']:.2f}")
                    print(f"      표준편차: {stats['std']:.2f}")
                    
                    # 포인트 수 분포
                    distribution = ", ".join([f"{i+1}개:{count}개({count/stats['n_voxels']*100:.1f}%)" 
                                            for i, count in enumerate(stats['histogram']) if count > 0])
                    print(f"      분포: {distribution}")

def save_pattern_statistics_to_csv(pattern_distance_stats, save_path):
    """
    패턴별 통계를 CSV로 저장
    """
    import pandas as pd
    
    # 패턴별 요약 통계
    pattern_summary = []
    detailed_data = []
    
    for pattern, distance_data in pattern_distance_stats.items():
        # 전체 통계
        all_points = []
        for points in distance_data.values():
            all_points.extend(points)
        
        if all_points:
            pattern_summary.append({
                '파일_패턴': f"{pattern}*",
                '전체_복셀_수': len(all_points),
                '전체_평균_포인트_수': np.mean(all_points),
                '전체_표준편차': np.std(all_points),
                '전체_최소값': np.min(all_points),
                '전체_최대값': np.max(all_points),
                '전체_중간값': np.median(all_points)
            })
        
        # 거리별 세부 통계
        for range_key, points in distance_data.items():
            if len(points) > 0:
                detailed_data.append({
                    '파일_패턴': f"{pattern}*",
                    '거리_범위': range_key,
                    '복셀_수': len(points),
                    '평균_포인트_수': np.mean(points),
                    '표준편차': np.std(points),
                    '최소값': np.min(points),
                    '최대값': np.max(points),
                    '중간값': np.median(points)
                })
    
    # CSV 저장
    if pattern_summary:
        df_summary = pd.DataFrame(pattern_summary)
        summary_path = os.path.join(save_path, 'pattern_summary_statistics.csv')
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"패턴별 요약 통계가 저장되었습니다: {summary_path}")
    
    if detailed_data:
        df_detailed = pd.DataFrame(detailed_data)
        detailed_path = os.path.join(save_path, 'pattern_detailed_statistics.csv')
        df_detailed.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        print(f"패턴별 세부 통계가 저장되었습니다: {detailed_path}")

def print_statistics(distance_stats):
    """
    통계 정보 출력 (스트리밍 통계 버전)
    """
    print("\n" + "="*80)
    print("거리별 복셀 내 포인트 수 통계")
    print("="*80)
    
    for range_name, streaming_stats in distance_stats.items():
        stats = streaming_stats.get_stats()
        if stats['n_voxels'] > 0:
            print(f"\n{range_name}:")
            print(f"  복셀 수: {stats['n_voxels']:,}")
            print(f"  평균 포인트 수: {stats['mean']:.2f}")
            print(f"  표준편차: {stats['std']:.2f}")
            print(f"  최소값: {stats['min']}")
            print(f"  최대값: {stats['max']}")
            print(f"  중간값: {stats['median']:.2f}")
            
            # 포인트 수별 분포 (히스토그램 기반)
            print(f"  포인트 수 분포:")
            for i, count in enumerate(stats['histogram']):
                if count > 0:
                    percentage = (count / stats['n_voxels']) * 100
                    print(f"    {i+1}개: {count:,}개 ({percentage:.1f}%)")

def main():
    # 설정
    data_path = '/home/ailab/git/Team_4/Ai_challenge/OpenPCDet/data/custom_av/points'
    point_cloud_range = np.array([-70.0, -70.0, -4.0, 70.0, 70.0, 4.0])
    voxel_size = np.array([0.1, 0.1, 0.15])
    max_points_per_voxel = 5
    distance_bins = list(range(0, 81, 10))  # 0-10m, 10-20m, ..., 70-80m
    
    print("포인트 클라우드 데이터 분석 시작...")
    print(f"데이터 경로: {data_path}")
    print(f"포인트 클라우드 범위: {point_cloud_range}")
    print(f"복셀 크기: {voxel_size}")
    print(f"복셀당 최대 포인트 수: {max_points_per_voxel}")
    print(f"거리 구간: {distance_bins}")
    
    # 데이터 경로 확인
    if not os.path.exists(data_path):
        print(f"오류: 데이터 경로를 찾을 수 없습니다: {data_path}")
        return
    
    # 분석 실행 (MAX_NUMBER_OF_VOXELS 반영)
    max_number_of_voxels = 150000
    print(f"최대 복셀 수: {max_number_of_voxels:,}")
    
    distance_stats, pattern_distance_stats, pattern_point_stats = analyze_voxel_points_by_distance(
        data_path, point_cloud_range, voxel_size, max_points_per_voxel, distance_bins, max_number_of_voxels
    )
    
    if not distance_stats:
        print("분석할 데이터가 없습니다.")
        return
    
    # 저장 디렉토리 생성
    save_dir = '/home/ailab/git/Team_2/Seokjae/OpenPCDet/dataset_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    # 패턴별 포인트 통계 출력 (복셀화 전후 비교)
    print_pattern_point_statistics(pattern_point_stats)
    
    # 전체 결과 출력 및 시각화
    print_statistics(distance_stats)
    plot_results(distance_stats, save_dir)
    
    # 패턴별 결과 출력 및 시각화
    print_pattern_statistics(pattern_distance_stats)
    plot_pattern_comparison(pattern_distance_stats, save_dir)
    
    # 결과를 CSV로 저장
    import pandas as pd
    
    # 전체 요약 통계를 DataFrame으로 변환 (스트리밍 통계 버전)
    summary_data = []
    for range_name, streaming_stats in distance_stats.items():
        stats = streaming_stats.get_stats()
        if stats['n_voxels'] > 0:
            summary_data.append({
                '거리_범위': range_name,
                '복셀_수': stats['n_voxels'],
                '평균_포인트_수': stats['mean'],
                '표준편차': stats['std'],
                '최소값': stats['min'],
                '최대값': stats['max'],
                '중간값': stats['median']
            })
    
    df_summary = pd.DataFrame(summary_data)
    output_path = os.path.join(save_dir, 'voxel_analysis_summary.csv')
    df_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n전체 요약 결과가 저장되었습니다: {output_path}")
    
    # 패턴별 통계를 CSV로 저장
    save_pattern_statistics_to_csv(pattern_distance_stats, save_dir)

if __name__ == "__main__":
    main()