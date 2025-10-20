import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from datetime import datetime
import glob

# Configuration
DATA_PATH = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av"
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
OUTPUT_DIR = "/home/ailab/git/Team_2/Seokjae/OpenPCDet/dataset_analysis"

def load_npy_pointcloud(file_path):
    """포인트 클라우드를 로드합니다."""
    if not os.path.exists(file_path):
        return None
    try:
        points = np.load(file_path)
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_frame_prefix(frame_id):
    """프레임 ID에서 접두사를 추출합니다 (000, 001, 002, 003, 100)"""
    if frame_id.startswith('000'):
        return '000'
    elif frame_id.startswith('001'):
        return '001'
    elif frame_id.startswith('002'):
        return '002'
    elif frame_id.startswith('003'):
        return '003'
    elif frame_id.startswith('100'):
        return '100'
    else:
        return 'other'

def get_available_npy_files():
    """points 폴더에서 사용 가능한 .npy 파일들을 찾습니다."""
    npy_pattern = os.path.join(POINTS_FOLDER, "*.npy")
    npy_files = glob.glob(npy_pattern)
    
    frame_info = []
    for file_path in npy_files:
        frame_id = os.path.basename(file_path).replace('.npy', '')
        prefix = extract_frame_prefix(frame_id)
        
        # 관심 있는 접두사만 필터링
        if prefix in ['000', '001', '002', '003', '100']:
            frame_info.append({
                'frame_id': frame_id,
                'file_path': file_path,
                'prefix': prefix
            })
    
    # frame_id로 정렬
    frame_info.sort(key=lambda x: x['frame_id'])
    
    return frame_info

def analyze_intensity_by_prefix():
    """frame_id 접두사별로 intensity를 분석합니다."""
    frame_info = get_available_npy_files()
    
    if not frame_info:
        print("No .npy files found with specified prefixes (000, 001, 002, 003, 100)")
        return None
    
    # 접두사별로 데이터 그룹화
    intensity_data = defaultdict(list)
    detailed_results = []
    
    print(f"Found {len(frame_info)} files to analyze...")
    
    for i, info in enumerate(frame_info):
        frame_id = info['frame_id']
        file_path = info['file_path']
        prefix = info['prefix']
        
        if (i + 1) % 100 == 0:
            print(f"Processing... {i + 1}/{len(frame_info)}")
        
        # 포인트 클라우드 로드
        points = load_npy_pointcloud(file_path)
        
        if points is None:
            continue
        
        # intensity는 보통 4번째 컬럼 (x, y, z, intensity)
        if points.shape[1] >= 4:
            intensity_values = points[:, 3]  # 4번째 컬럼이 intensity
            
            # NaN이나 inf 값 제거
            valid_intensity = intensity_values[np.isfinite(intensity_values)]
            
            if len(valid_intensity) > 0:
                mean_intensity = np.mean(valid_intensity)
                std_intensity = np.std(valid_intensity)
                min_intensity = np.min(valid_intensity)
                max_intensity = np.max(valid_intensity)
                
                intensity_data[prefix].append(mean_intensity)
                
                detailed_results.append({
                    'frame_id': frame_id,
                    'prefix': prefix,
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'min_intensity': min_intensity,
                    'max_intensity': max_intensity,
                    'num_points': len(valid_intensity)
                })
            else:
                print(f"Warning: No valid intensity values in {frame_id}")
        else:
            print(f"Warning: {frame_id} has only {points.shape[1]} columns, expected at least 4")
    
    return intensity_data, detailed_results

def create_visualizations(intensity_data, detailed_results):
    """다양한 시각화를 생성합니다."""
    if not intensity_data or not detailed_results:
        print("No data to visualize")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # DataFrame 생성
    df = pd.DataFrame(detailed_results)
    
    # 색상 팔레트 설정
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    prefix_colors = {prefix: colors[i] for i, prefix in enumerate(['000', '001', '002', '003', '100'])}
    
    # 1. 접두사별 평균 intensity 박스플롯
    plt.figure(figsize=(12, 8))
    
    # 데이터 준비
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    for prefix in ['000', '001', '002', '003', '100']:
        if prefix in intensity_data and len(intensity_data[prefix]) > 0:
            plot_data.append(intensity_data[prefix])
            plot_labels.append(f'{prefix} (n={len(intensity_data[prefix])})')
            plot_colors.append(prefix_colors[prefix])
    
    if plot_data:
        bp = plt.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        
        # 박스 색상 설정
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    plt.title('Average Intensity Distribution by Frame ID Prefix', fontsize=16, fontweight='bold')
    plt.xlabel('Frame ID Prefix', fontsize=12)
    plt.ylabel('Average Intensity', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'intensity_boxplot_by_prefix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 접두사별 평균값 바 차트
    plt.figure(figsize=(10, 6))
    
    prefix_means = {}
    prefix_stds = {}
    
    for prefix in ['000', '001', '002', '003', '100']:
        if prefix in intensity_data and len(intensity_data[prefix]) > 0:
            prefix_means[prefix] = np.mean(intensity_data[prefix])
            prefix_stds[prefix] = np.std(intensity_data[prefix])
    
    if prefix_means:
        prefixes = list(prefix_means.keys())
        means = list(prefix_means.values())
        stds = list(prefix_stds.values())
        colors_list = [prefix_colors[p] for p in prefixes]
        
        bars = plt.bar(prefixes, means, yerr=stds, capsize=5, color=colors_list, alpha=0.8, edgecolor='black')
        
        # 값 표시
        for bar, mean_val in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.1, 
                    f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Average Intensity by Frame ID Prefix', fontsize=16, fontweight='bold')
    plt.xlabel('Frame ID Prefix', fontsize=12)
    plt.ylabel('Mean Intensity', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'intensity_bar_chart_by_prefix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 시간 순서에 따른 intensity 변화 (각 접두사별로)
    plt.figure(figsize=(15, 10))
    
    for i, prefix in enumerate(['000', '001', '002', '003', '100']):
        if prefix in intensity_data and len(intensity_data[prefix]) > 0:
            prefix_df = df[df['prefix'] == prefix].sort_values('frame_id')
            
            plt.subplot(3, 2, i+1)
            plt.plot(range(len(prefix_df)), prefix_df['mean_intensity'], 
                    color=prefix_colors[prefix], linewidth=2, marker='o', markersize=3)
            plt.fill_between(range(len(prefix_df)), 
                           prefix_df['mean_intensity'] - prefix_df['std_intensity'],
                           prefix_df['mean_intensity'] + prefix_df['std_intensity'],
                           alpha=0.3, color=prefix_colors[prefix])
            
            plt.title(f'Intensity Variation - Prefix {prefix}', fontweight='bold')
            plt.xlabel('Frame Index')
            plt.ylabel('Intensity')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'intensity_time_series_by_prefix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 히트맵 (접두사별 통계 요약)
    plt.figure(figsize=(10, 6))
    
    summary_data = []
    for prefix in ['000', '001', '002', '003', '100']:
        if prefix in intensity_data and len(intensity_data[prefix]) > 0:
            data = intensity_data[prefix]
            summary_data.append([
                np.mean(data),      # 평균
                np.std(data),       # 표준편차
                np.min(data),       # 최소값
                np.max(data),       # 최대값
                len(data)           # 샘플 수
            ])
        else:
            summary_data.append([0, 0, 0, 0, 0])
    
    summary_df = pd.DataFrame(summary_data, 
                             index=['000', '001', '002', '003', '100'],
                             columns=['Mean', 'Std', 'Min', 'Max', 'Count'])
    
    # Count 컬럼을 제외하고 정규화
    summary_normalized = summary_df.copy()
    for col in ['Mean', 'Std', 'Min', 'Max']:
        if summary_df[col].max() > 0:
            summary_normalized[col] = summary_df[col] / summary_df[col].max()
    
    sns.heatmap(summary_normalized, annot=summary_df, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized Value'})
    plt.title('Intensity Statistics Summary by Prefix', fontsize=16, fontweight='bold')
    plt.ylabel('Frame ID Prefix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'intensity_heatmap_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()

def save_detailed_report(intensity_data, detailed_results):
    """상세한 분석 보고서를 저장합니다."""
    if not detailed_results:
        return
    
    report_path = os.path.join(OUTPUT_DIR, 'intensity_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Point Cloud Intensity Analysis Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Path: {POINTS_FOLDER}\n")
        f.write(f"Total Files Analyzed: {len(detailed_results)}\n\n")
        
        # 접두사별 요약
        f.write("Summary by Frame ID Prefix:\n")
        f.write("-" * 30 + "\n")
        
        for prefix in ['000', '001', '002', '003', '100']:
            if prefix in intensity_data and len(intensity_data[prefix]) > 0:
                data = intensity_data[prefix]
                f.write(f"\nPrefix {prefix}:\n")
                f.write(f"  Files: {len(data)}\n")
                f.write(f"  Mean Intensity: {np.mean(data):.4f}\n")
                f.write(f"  Std Intensity: {np.std(data):.4f}\n")
                f.write(f"  Min Intensity: {np.min(data):.4f}\n")
                f.write(f"  Max Intensity: {np.max(data):.4f}\n")
            else:
                f.write(f"\nPrefix {prefix}: No files found\n")
        
        # 상세 결과
        f.write("\n\nDetailed Results:\n")
        f.write("-" * 20 + "\n")
        f.write("Frame_ID\tPrefix\tMean_Int\tStd_Int\tMin_Int\tMax_Int\tNum_Points\n")
        
        for result in detailed_results:
            f.write(f"{result['frame_id']}\t{result['prefix']}\t"
                   f"{result['mean_intensity']:.4f}\t{result['std_intensity']:.4f}\t"
                   f"{result['min_intensity']:.4f}\t{result['max_intensity']:.4f}\t"
                   f"{result['num_points']}\n")
    
    print(f"Detailed report saved to: {report_path}")

def main():
    """메인 함수"""
    print("Starting intensity analysis...")
    print(f"Analyzing files in: {POINTS_FOLDER}")
    
    # 파일 존재 확인
    if not os.path.exists(POINTS_FOLDER):
        print(f"Error: Points folder not found: {POINTS_FOLDER}")
        return
    
    # intensity 분석
    intensity_data, detailed_results = analyze_intensity_by_prefix()
    
    if not intensity_data:
        print("No valid intensity data found.")
        return
    
    print(f"\nAnalysis completed!")
    print(f"Total files processed: {len(detailed_results)}")
    
    # 간단한 요약 출력
    for prefix in ['000', '001', '002', '003', '100']:
        if prefix in intensity_data and len(intensity_data[prefix]) > 0:
            mean_int = np.mean(intensity_data[prefix])
            print(f"Prefix {prefix}: {len(intensity_data[prefix])} files, Mean intensity: {mean_int:.4f}")
    
    # 시각화 생성
    print("\nGenerating visualizations...")
    create_visualizations(intensity_data, detailed_results)
    
    # 상세 보고서 저장
    print("\nSaving detailed report...")
    save_detailed_report(intensity_data, detailed_results)
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()