import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# 경로 설정
data_root = '/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av'
points_dir = os.path.join(data_root, 'points')
test_txt = os.path.join(data_root, 'ImageSets', 'test.txt')

# test.txt 읽기
with open(test_txt, 'r') as f:
    test_ids = [line.strip() for line in f.readlines()]

print(f"총 테스트 샘플 수: {len(test_ids)}")

# Noise와 Clean 구간 정의
noise_ranges = [
    (10006432, 10009318),
    (10100400, 10104398)
]

def is_noise(file_id):
    """파일 ID가 noise 구간에 속하는지 확인"""
    file_num = int(file_id)
    for start, end in noise_ranges:
        if start <= file_num <= end:
            return True
    return False

def count_points_in_radius(points, radius=2.0):
    """
    반경 radius 이내의 포인트 개수를 계산
    원점 (0, 0, 0)으로부터의 거리 기준
    """
    # x, y, z 좌표만 사용
    coords = points[:, :3]
    
    # 원점으로부터의 거리 계산
    distances = np.sqrt(np.sum(coords**2, axis=1))
    
    # 반경 이내의 포인트 개수
    count = np.sum(distances <= radius)
    
    return count

def analyze_intensity_in_radius(points, radius=2.0):
    """
    반경 radius 이내의 포인트들의 intensity 분석
    """
    # x, y, z 좌표
    coords = points[:, :3]
    
    # 원점으로부터의 거리 계산
    distances = np.sqrt(np.sum(coords**2, axis=1))
    
    # 반경 이내의 포인트 마스크
    mask = distances <= radius
    
    # intensity는 4번째 열 (index 3)
    if points.shape[1] >= 4:
        intensities = points[mask, 3]
        
        if len(intensities) > 0:
            return {
                'mean': np.mean(intensities),
                'median': np.median(intensities),
                'std': np.std(intensities),
                'min': np.min(intensities),
                'max': np.max(intensities),
                'intensities': intensities
            }
    
    return {
        'mean': 0,
        'median': 0,
        'std': 0,
        'min': 0,
        'max': 0,
        'intensities': np.array([])
    }

# 분석 시작
noise_counts = []
clean_counts = []
noise_intensity_stats = []
clean_intensity_stats = []
noise_all_intensities = []
clean_all_intensities = []

noise_files = []
clean_files = []

print("\n포인트 클라우드 및 Intensity 분석 중...")
for file_id in tqdm(test_ids):
    file_path = os.path.join(points_dir, f"{file_id}.npy")
    
    if not os.path.exists(file_path):
        continue
    
    # 포인트 클라우드 로드
    points = np.load(file_path)
    
    # 반경 2m 내 포인트 개수 계산
    count = count_points_in_radius(points, radius=2.0)
    
    # Intensity 분석
    intensity_stat = analyze_intensity_in_radius(points, radius=2.0)
    
    # Noise vs Clean 분류
    if is_noise(file_id):
        noise_counts.append(count)
        noise_files.append(file_id)
        noise_intensity_stats.append(intensity_stat)
        if len(intensity_stat['intensities']) > 0:
            noise_all_intensities.extend(intensity_stat['intensities'])
    else:
        clean_counts.append(count)
        clean_files.append(file_id)
        clean_intensity_stats.append(intensity_stat)
        if len(intensity_stat['intensities']) > 0:
            clean_all_intensities.extend(intensity_stat['intensities'])

# 통계 계산
noise_counts = np.array(noise_counts)
clean_counts = np.array(clean_counts)
noise_all_intensities = np.array(noise_all_intensities)
clean_all_intensities = np.array(clean_all_intensities)

# Intensity 평균 통계
noise_intensity_means = [stat['mean'] for stat in noise_intensity_stats if stat['mean'] > 0]
clean_intensity_means = [stat['mean'] for stat in clean_intensity_stats if stat['mean'] > 0]

print("\n" + "="*80)
print("분석 결과")
print("="*80)

print(f"\n{'[Noise 데이터]':^80}")
print(f"  - 샘플 수: {len(noise_counts)}")
print(f"  - 구간: 10006432~10009318, 10100400~10104398")
print(f"\n  [포인트 수 (반경 2m)]")
print(f"    • 평균: {noise_counts.mean():.2f}")
print(f"    • 중앙값: {np.median(noise_counts):.2f}")
print(f"    • 표준편차: {noise_counts.std():.2f}")
print(f"    • 범위: {noise_counts.min()} ~ {noise_counts.max()}")
print(f"\n  [Intensity (반경 2m 내 포인트)]")
if len(noise_all_intensities) > 0:
    print(f"    • 전체 포인트 수: {len(noise_all_intensities):,}")
    print(f"    • 평균: {noise_all_intensities.mean():.4f}")
    print(f"    • 중앙값: {np.median(noise_all_intensities):.4f}")
    print(f"    • 표준편차: {noise_all_intensities.std():.4f}")
    print(f"    • 범위: {noise_all_intensities.min():.4f} ~ {noise_all_intensities.max():.4f}")

print(f"\n{'[Clean 데이터]':^80}")
print(f"  - 샘플 수: {len(clean_counts)}")
print(f"\n  [포인트 수 (반경 2m)]")
print(f"    • 평균: {clean_counts.mean():.2f}")
print(f"    • 중앙값: {np.median(clean_counts):.2f}")
print(f"    • 표준편차: {clean_counts.std():.2f}")
print(f"    • 범위: {clean_counts.min()} ~ {clean_counts.max()}")
print(f"\n  [Intensity (반경 2m 내 포인트)]")
if len(clean_all_intensities) > 0:
    print(f"    • 전체 포인트 수: {len(clean_all_intensities):,}")
    print(f"    • 평균: {clean_all_intensities.mean():.4f}")
    print(f"    • 중앙값: {np.median(clean_all_intensities):.4f}")
    print(f"    • 표준편차: {clean_all_intensities.std():.4f}")
    print(f"    • 범위: {clean_all_intensities.min():.4f} ~ {clean_all_intensities.max():.4f}")

print(f"\n{'[비교 분석]':^80}")
diff_mean = noise_counts.mean() - clean_counts.mean()
diff_percent = (diff_mean / clean_counts.mean()) * 100 if clean_counts.mean() > 0 else 0
print(f"  [포인트 수]")
print(f"    • 평균 차이: {diff_mean:.2f} 포인트")
print(f"    • 비율: Noise가 Clean 대비 {diff_percent:+.2f}% {'많음' if diff_percent > 0 else '적음'}")

if len(noise_all_intensities) > 0 and len(clean_all_intensities) > 0:
    intensity_diff = noise_all_intensities.mean() - clean_all_intensities.mean()
    intensity_percent = (intensity_diff / clean_all_intensities.mean()) * 100 if clean_all_intensities.mean() > 0 else 0
    print(f"\n  [Intensity]")
    print(f"    • 평균 차이: {intensity_diff:+.4f}")
    print(f"    • 비율: Noise가 Clean 대비 {intensity_percent:+.2f}% {'높음' if intensity_percent > 0 else '낮음'}")

print("\n" + "="*80)

# 시각화 1: 포인트 수 분석
fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle('Point Count Analysis (2m Radius)', fontsize=16, fontweight='bold', y=0.995)

# 1-1. 히스토그램 비교
ax1 = axes[0, 0]
ax1.hist(clean_counts, bins=50, alpha=0.6, label='Clean', color='#3498db', edgecolor='black', linewidth=0.5)
ax1.hist(noise_counts, bins=50, alpha=0.6, label='Noise', color='#e74c3c', edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Number of Points', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Distribution Histogram', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# 통계값 표시
ax1.axvline(clean_counts.mean(), color='#3498db', linestyle='--', linewidth=2, alpha=0.8, label=f'Clean Mean: {clean_counts.mean():.1f}')
ax1.axvline(noise_counts.mean(), color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8, label=f'Noise Mean: {noise_counts.mean():.1f}')

# 1-2. 박스플롯
ax2 = axes[0, 1]
box_data = [clean_counts, noise_counts]
bp = ax2.boxplot(box_data, tick_labels=['Clean', 'Noise'], patch_artist=True, 
                 widths=0.6, showmeans=True, meanline=True)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#e74c3c')
bp['boxes'][1].set_alpha(0.7)
ax2.set_ylabel('Number of Points', fontsize=11, fontweight='bold')
ax2.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# 1-3. 바이올린 플롯
ax3 = axes[0, 2]
parts = ax3.violinplot([clean_counts, noise_counts], positions=[1, 2], showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#3498db', '#e74c3c'][i])
    pc.set_alpha(0.6)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Clean', 'Noise'])
ax3.set_ylabel('Number of Points', fontsize=11, fontweight='bold')
ax3.set_title('Violin Plot', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

# 1-4. 누적 분포 함수 (CDF)
ax4 = axes[1, 0]
clean_sorted = np.sort(clean_counts)
clean_cdf = np.arange(1, len(clean_sorted) + 1) / len(clean_sorted)
noise_sorted = np.sort(noise_counts)
noise_cdf = np.arange(1, len(noise_sorted) + 1) / len(noise_sorted)

ax4.plot(clean_sorted, clean_cdf, label='Clean', color='#3498db', linewidth=2.5, alpha=0.8)
ax4.plot(noise_sorted, noise_cdf, label='Noise', color='#e74c3c', linewidth=2.5, alpha=0.8)
ax4.set_xlabel('Number of Points', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
ax4.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10, framealpha=0.9)
ax4.grid(True, alpha=0.3, linestyle='--')

# 1-5. 통계 요약 표
ax5 = axes[1, 1]
ax5.axis('off')

stats_data = [
    ['Metric', 'Clean', 'Noise', 'Diff (N-C)'],
    ['Samples', f'{len(clean_counts)}', f'{len(noise_counts)}', '-'],
    ['Mean', f'{clean_counts.mean():.2f}', f'{noise_counts.mean():.2f}', 
     f'{noise_counts.mean() - clean_counts.mean():+.2f}'],
    ['Median', f'{np.median(clean_counts):.2f}', f'{np.median(noise_counts):.2f}',
     f'{np.median(noise_counts) - np.median(clean_counts):+.2f}'],
    ['Std Dev', f'{clean_counts.std():.2f}', f'{noise_counts.std():.2f}',
     f'{noise_counts.std() - clean_counts.std():+.2f}'],
    ['Min', f'{clean_counts.min()}', f'{noise_counts.min()}',
     f'{int(noise_counts.min() - clean_counts.min()):+d}'],
    ['Max', f'{clean_counts.max()}', f'{noise_counts.max()}',
     f'{int(noise_counts.max() - clean_counts.max()):+d}'],
    ['25%', f'{np.percentile(clean_counts, 25):.2f}', f'{np.percentile(noise_counts, 25):.2f}',
     f'{np.percentile(noise_counts, 25) - np.percentile(clean_counts, 25):+.2f}'],
    ['75%', f'{np.percentile(clean_counts, 75):.2f}', f'{np.percentile(noise_counts, 75):.2f}',
     f'{np.percentile(noise_counts, 75) - np.percentile(clean_counts, 75):+.2f}'],
]

table = ax5.table(cellText=stats_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# 헤더 스타일링
for i in range(4):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 데이터 행 스타일링
for i in range(1, len(stats_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax5.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)

# 1-6. 비교 막대 그래프
ax6 = axes[1, 2]
metrics = ['Mean', 'Median', 'Std Dev']
clean_vals = [clean_counts.mean(), np.median(clean_counts), clean_counts.std()]
noise_vals = [noise_counts.mean(), np.median(noise_counts), noise_counts.std()]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax6.bar(x - width/2, clean_vals, width, label='Clean', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax6.bar(x + width/2, noise_vals, width, label='Noise', color='#e74c3c', alpha=0.8, edgecolor='black')

ax6.set_xlabel('Metrics', fontsize=11, fontweight='bold')
ax6.set_ylabel('Value', fontsize=11, fontweight='bold')
ax6.set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics)
ax6.legend(fontsize=10, framealpha=0.9)
ax6.grid(True, alpha=0.3, axis='y', linestyle='--')

# 막대 위에 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ailab/git/Team_2/Subin/OpenPCDet/point_density_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n[저장] 포인트 수 시각화: point_density_comparison.png")

# 시각화 2: Intensity 분석
if len(noise_all_intensities) > 0 and len(clean_all_intensities) > 0:
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
    fig2.suptitle('Intensity Analysis (Points within 2m Radius)', fontsize=16, fontweight='bold', y=0.995)
    
    # 2-1. Intensity 히스토그램
    ax21 = axes2[0, 0]
    # 샘플링 (너무 많은 데이터는 샘플링)
    clean_intensity_sample = clean_all_intensities if len(clean_all_intensities) < 100000 else np.random.choice(clean_all_intensities, 100000, replace=False)
    noise_intensity_sample = noise_all_intensities if len(noise_all_intensities) < 100000 else np.random.choice(noise_all_intensities, 100000, replace=False)
    
    ax21.hist(clean_intensity_sample, bins=100, alpha=0.6, label='Clean', color='#3498db', edgecolor='black', linewidth=0.5, density=True)
    ax21.hist(noise_intensity_sample, bins=100, alpha=0.6, label='Noise', color='#e74c3c', edgecolor='black', linewidth=0.5, density=True)
    ax21.set_xlabel('Intensity', fontsize=11, fontweight='bold')
    ax21.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax21.set_title('Intensity Distribution (Normalized)', fontsize=12, fontweight='bold')
    ax21.legend(fontsize=10, framealpha=0.9)
    ax21.grid(True, alpha=0.3, linestyle='--')
    
    # 평균선 표시
    ax21.axvline(clean_all_intensities.mean(), color='#3498db', linestyle='--', linewidth=2, alpha=0.8)
    ax21.axvline(noise_all_intensities.mean(), color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
    
    # 2-2. Intensity 박스플롯
    ax22 = axes2[0, 1]
    bp2 = ax22.boxplot([clean_intensity_sample, noise_intensity_sample], 
                        tick_labels=['Clean', 'Noise'], patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
    bp2['boxes'][0].set_facecolor('#3498db')
    bp2['boxes'][0].set_alpha(0.7)
    bp2['boxes'][1].set_facecolor('#e74c3c')
    bp2['boxes'][1].set_alpha(0.7)
    ax22.set_ylabel('Intensity', fontsize=11, fontweight='bold')
    ax22.set_title('Intensity Box Plot', fontsize=12, fontweight='bold')
    ax22.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 2-3. Intensity CDF
    ax23 = axes2[0, 2]
    clean_int_sorted = np.sort(clean_intensity_sample)
    clean_int_cdf = np.arange(1, len(clean_int_sorted) + 1) / len(clean_int_sorted)
    noise_int_sorted = np.sort(noise_intensity_sample)
    noise_int_cdf = np.arange(1, len(noise_int_sorted) + 1) / len(noise_int_sorted)
    
    ax23.plot(clean_int_sorted, clean_int_cdf, label='Clean', color='#3498db', linewidth=2.5, alpha=0.8)
    ax23.plot(noise_int_sorted, noise_int_cdf, label='Noise', color='#e74c3c', linewidth=2.5, alpha=0.8)
    ax23.set_xlabel('Intensity', fontsize=11, fontweight='bold')
    ax23.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax23.set_title('Intensity CDF', fontsize=12, fontweight='bold')
    ax23.legend(fontsize=10, framealpha=0.9)
    ax23.grid(True, alpha=0.3, linestyle='--')
    
    # 2-4. Intensity 통계 표
    ax24 = axes2[1, 0]
    ax24.axis('off')
    
    intensity_stats_data = [
        ['Metric', 'Clean', 'Noise', 'Diff (N-C)'],
        ['Total Points', f'{len(clean_all_intensities):,}', f'{len(noise_all_intensities):,}', '-'],
        ['Mean', f'{clean_all_intensities.mean():.4f}', f'{noise_all_intensities.mean():.4f}',
         f'{noise_all_intensities.mean() - clean_all_intensities.mean():+.4f}'],
        ['Median', f'{np.median(clean_all_intensities):.4f}', f'{np.median(noise_all_intensities):.4f}',
         f'{np.median(noise_all_intensities) - np.median(clean_all_intensities):+.4f}'],
        ['Std Dev', f'{clean_all_intensities.std():.4f}', f'{noise_all_intensities.std():.4f}',
         f'{noise_all_intensities.std() - clean_all_intensities.std():+.4f}'],
        ['Min', f'{clean_all_intensities.min():.4f}', f'{noise_all_intensities.min():.4f}',
         f'{noise_all_intensities.min() - clean_all_intensities.min():+.4f}'],
        ['Max', f'{clean_all_intensities.max():.4f}', f'{noise_all_intensities.max():.4f}',
         f'{noise_all_intensities.max() - clean_all_intensities.max():+.4f}'],
        ['25%', f'{np.percentile(clean_all_intensities, 25):.4f}', 
         f'{np.percentile(noise_all_intensities, 25):.4f}',
         f'{np.percentile(noise_all_intensities, 25) - np.percentile(clean_all_intensities, 25):+.4f}'],
        ['75%', f'{np.percentile(clean_all_intensities, 75):.4f}', 
         f'{np.percentile(noise_all_intensities, 75):.4f}',
         f'{np.percentile(noise_all_intensities, 75) - np.percentile(clean_all_intensities, 75):+.4f}'],
    ]
    
    table2 = ax24.table(cellText=intensity_stats_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2)
    
    for i in range(4):
        table2[(0, i)].set_facecolor('#2c3e50')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(intensity_stats_data)):
        for j in range(4):
            if i % 2 == 0:
                table2[(i, j)].set_facecolor('#ecf0f1')
    
    ax24.set_title('Intensity Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # 2-5. Intensity 비교 막대 그래프
    ax25 = axes2[1, 1]
    metrics2 = ['Mean', 'Median', 'Std Dev']
    clean_int_vals = [clean_all_intensities.mean(), np.median(clean_all_intensities), clean_all_intensities.std()]
    noise_int_vals = [noise_all_intensities.mean(), np.median(noise_all_intensities), noise_all_intensities.std()]
    
    x2 = np.arange(len(metrics2))
    bars3 = ax25.bar(x2 - width/2, clean_int_vals, width, label='Clean', color='#3498db', alpha=0.8, edgecolor='black')
    bars4 = ax25.bar(x2 + width/2, noise_int_vals, width, label='Noise', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax25.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax25.set_ylabel('Intensity Value', fontsize=11, fontweight='bold')
    ax25.set_title('Intensity Metrics Comparison', fontsize=12, fontweight='bold')
    ax25.set_xticks(x2)
    ax25.set_xticklabels(metrics2)
    ax25.legend(fontsize=10, framealpha=0.9)
    ax25.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax25.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2-6. 2D 히스토그램 (Point Count vs Intensity Mean)
    ax26 = axes2[1, 2]
    
    # 각 파일별 평균 intensity 계산
    noise_file_intensity_means = [stat['mean'] for stat in noise_intensity_stats]
    clean_file_intensity_means = [stat['mean'] for stat in clean_intensity_stats]
    
    ax26.scatter(clean_counts, clean_file_intensity_means, alpha=0.4, s=30, color='#3498db', label='Clean', edgecolors='black', linewidth=0.3)
    ax26.scatter(noise_counts, noise_file_intensity_means, alpha=0.4, s=30, color='#e74c3c', label='Noise', edgecolors='black', linewidth=0.3)
    ax26.set_xlabel('Point Count (2m radius)', fontsize=11, fontweight='bold')
    ax26.set_ylabel('Mean Intensity', fontsize=11, fontweight='bold')
    ax26.set_title('Point Count vs Mean Intensity', fontsize=12, fontweight='bold')
    ax26.legend(fontsize=10, framealpha=0.9)
    ax26.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('/home/ailab/git/Team_2/Subin/OpenPCDet/intensity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[저장] Intensity 시각화: intensity_comparison.png")

# 상세 결과를 CSV로 저장
output_file = '/home/ailab/git/Team_2/Subin/OpenPCDet/point_density_analysis.csv'
with open(output_file, 'w') as f:
    f.write("file_id,type,point_count_2m_radius,mean_intensity,median_intensity,std_intensity,min_intensity,max_intensity\n")
    for file_id, count, intensity_stat in zip(clean_files, clean_counts, clean_intensity_stats):
        f.write(f"{file_id},clean,{count},{intensity_stat['mean']:.6f},{intensity_stat['median']:.6f},"
                f"{intensity_stat['std']:.6f},{intensity_stat['min']:.6f},{intensity_stat['max']:.6f}\n")
    for file_id, count, intensity_stat in zip(noise_files, noise_counts, noise_intensity_stats):
        f.write(f"{file_id},noise,{count},{intensity_stat['mean']:.6f},{intensity_stat['median']:.6f},"
                f"{intensity_stat['std']:.6f},{intensity_stat['min']:.6f},{intensity_stat['max']:.6f}\n")

print(f"[저장] 상세 CSV 결과: point_density_analysis.csv\n")
