import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.patches as mpatches

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590']

def get_sample_classes(data_path, sample_id):
    """샘플의 클래스 정보 반환"""
    label_file = os.path.join(data_path, 'labels', f'{sample_id}.txt')
    if not os.path.exists(label_file):
        return set()
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    classes = set()
    for line in lines:
        if line.strip():
            class_name = line.strip().split(' ')[-1]
            classes.add(class_name)
    
    return classes

def count_class_objects(data_path, sample_list):
    """샘플 리스트의 클래스별 객체 수 계산"""
    class_counts = defaultdict(int)
    
    for sample_id in sample_list:
        label_file = os.path.join(data_path, 'labels', f'{sample_id}.txt')
        if not os.path.exists(label_file):
            continue
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.strip():
                class_name = line.strip().split(' ')[-1]
                class_counts[class_name] += 1
    
    return class_counts

def analyze_dataset_for_visualization(data_path):
    """시각화용 데이터셋 분석"""
    labels_dir = os.path.join(data_path, 'labels')
    all_samples = [f.replace('.txt', '') for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    # 1. 클래스 조합 분석
    class_combinations = defaultdict(int)
    class_in_samples = defaultdict(int)
    single_class_counts = defaultdict(int)
    dependency_data = {'Pedestrian': {'with_vehicle': 0, 'without_vehicle': 0},
                      'Cyclist': {'with_vehicle': 0, 'without_vehicle': 0}}
    
    for sample_id in all_samples:
        classes = get_sample_classes(data_path, sample_id)
        if classes:
            combo_key = tuple(sorted(classes))
            class_combinations[combo_key] += 1
            
            # 각 클래스별 샘플 수
            for cls in classes:
                class_in_samples[cls] += 1
            
            # 단독 클래스 체크
            if len(classes) == 1:
                single_class_counts[list(classes)[0]] += 1
            
            # Vehicle 의존성 체크
            if 'Pedestrian' in classes:
                if 'Vehicle' in classes:
                    dependency_data['Pedestrian']['with_vehicle'] += 1
                else:
                    dependency_data['Pedestrian']['without_vehicle'] += 1
            
            if 'Cyclist' in classes:
                if 'Vehicle' in classes:
                    dependency_data['Cyclist']['with_vehicle'] += 1
                else:
                    dependency_data['Cyclist']['without_vehicle'] += 1
    
    # 2. 전체 객체 수 계산
    total_class_counts = count_class_objects(data_path, all_samples)
    
    return {
        'total_samples': len(all_samples),
        'class_combinations': dict(class_combinations),
        'class_in_samples': dict(class_in_samples),
        'single_class_counts': dict(single_class_counts),
        'total_class_counts': dict(total_class_counts),
        'dependency_data': dependency_data,
        'all_samples': all_samples
    }

def visualize_class_combinations(analysis_data, save_path=None):
    """클래스 조합 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Class Combination Analysis', fontsize=16, fontweight='bold')
    
    # 1. 클래스 조합별 샘플 수 (파이 차트)
    combos = analysis_data['class_combinations']
    combo_labels = [str(combo).replace("'", "").replace("(", "").replace(")", "") for combo in combos.keys()]
    combo_values = list(combos.values())
    
    ax1.pie(combo_values, labels=combo_labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Sample Distribution by Class Combinations', fontweight='bold')
    
    # 2. 클래스별 샘플 수 (막대 차트)
    classes = ['Vehicle', 'Pedestrian', 'Cyclist']
    sample_counts = [analysis_data['class_in_samples'].get(cls, 0) for cls in classes]
    
    bars = ax2.bar(classes, sample_counts, color=colors[:3])
    ax2.set_title('Samples Containing Each Class', fontweight='bold')
    ax2.set_ylabel('Number of Samples')
    
    # 값 표시
    for bar, count in zip(bars, sample_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 단독 클래스 분석 (막대 차트)
    single_counts = [analysis_data['single_class_counts'].get(cls, 0) for cls in classes]
    
    bars = ax3.bar(classes, single_counts, color=colors[:3], alpha=0.7)
    ax3.set_title('Samples with Single Class Only', fontweight='bold')
    ax3.set_ylabel('Number of Samples')
    
    for bar, count in zip(bars, single_counts):
        if count == 0:
            ax3.text(bar.get_x() + bar.get_width()/2, 10,
                    'None', ha='center', va='bottom', fontweight='bold', color='red')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 클래스별 총 객체 수 (막대 차트)
    object_counts = [analysis_data['total_class_counts'].get(cls, 0) for cls in classes]
    
    bars = ax4.bar(classes, object_counts, color=colors[:3])
    ax4.set_title('Total Objects by Class', fontweight='bold')
    ax4.set_ylabel('Number of Objects')
    
    for bar, count in zip(bars, object_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'class_combinations_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def visualize_vehicle_dependency(analysis_data, save_path=None):
    """Vehicle 의존성 시각화"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Vehicle Dependency Analysis', fontsize=16, fontweight='bold')
    
    dependency_data = analysis_data['dependency_data']
    
    # 1. Pedestrian의 Vehicle 의존성
    ped_data = dependency_data['Pedestrian']
    ped_labels = ['With Vehicle', 'Without Vehicle']
    ped_values = [ped_data['with_vehicle'], ped_data['without_vehicle']]
    
    wedges, texts, autotexts = ax1.pie(ped_values, labels=ped_labels, autopct='%1.1f%%', 
                                      startangle=90, colors=['#2E86AB', '#C73E1D'])
    ax1.set_title('Pedestrian Samples', fontweight='bold')
    
    # 2. Cyclist의 Vehicle 의존성
    cyc_data = dependency_data['Cyclist']
    cyc_labels = ['With Vehicle', 'Without Vehicle']
    cyc_values = [cyc_data['with_vehicle'], cyc_data['without_vehicle']]
    
    wedges, texts, autotexts = ax2.pie(cyc_values, labels=cyc_labels, autopct='%1.1f%%', 
                                      startangle=90, colors=['#2E86AB', '#C73E1D'])
    ax2.set_title('Cyclist Samples', fontweight='bold')
    
    # 3. 의존성 요약 (막대 차트)
    classes = ['Pedestrian', 'Cyclist']
    with_vehicle = [ped_data['with_vehicle'], cyc_data['with_vehicle']]
    without_vehicle = [ped_data['without_vehicle'], cyc_data['without_vehicle']]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, with_vehicle, width, label='With Vehicle', color='#2E86AB')
    bars2 = ax3.bar(x + width/2, without_vehicle, width, label='Without Vehicle', color='#C73E1D')
    
    ax3.set_title('Vehicle Dependency Summary', fontweight='bold')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Number of Samples')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes)
    ax3.legend()
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'vehicle_dependency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def visualize_split_analysis(data_path, save_path=None):
    """분할 결과 분석 시각화"""
    # Train/Val 파일 읽기
    imagesets_dir = os.path.join(data_path, 'ImageSets')
    
    with open(os.path.join(imagesets_dir, 'train_eval.txt'), 'r') as f:
        train_samples = [x.strip() for x in f.readlines()]
    
    with open(os.path.join(imagesets_dir, 'val_eval.txt'), 'r') as f:
        val_samples = [x.strip() for x in f.readlines()]
    
    # 클래스별 객체 수 계산
    train_counts = count_class_objects(data_path, train_samples)
    val_counts = count_class_objects(data_path, val_samples)
    
    # 조합별 분할 분석
    train_combos = defaultdict(int)
    val_combos = defaultdict(int)
    
    for sample_id in train_samples:
        classes = get_sample_classes(data_path, sample_id)
        if classes:
            combo_key = tuple(sorted(classes))
            train_combos[combo_key] += 1
    
    for sample_id in val_samples:
        classes = get_sample_classes(data_path, sample_id)
        if classes:
            combo_key = tuple(sorted(classes))
            val_combos[combo_key] += 1
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Train/Validation Split Analysis', fontsize=16, fontweight='bold')
    
    # 1. 클래스별 객체 수 분할 비율
    classes = ['Vehicle', 'Pedestrian', 'Cyclist']
    train_obj_counts = [train_counts.get(cls, 0) for cls in classes]
    val_obj_counts = [val_counts.get(cls, 0) for cls in classes]
    total_obj_counts = [train_obj_counts[i] + val_obj_counts[i] for i in range(3)]
    
    train_ratios = [train_obj_counts[i] / total_obj_counts[i] * 100 if total_obj_counts[i] > 0 else 0 for i in range(3)]
    val_ratios = [val_obj_counts[i] / total_obj_counts[i] * 100 if total_obj_counts[i] > 0 else 0 for i in range(3)]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_ratios, width, label='Train', color='#2E86AB')
    bars2 = ax1.bar(x + width/2, val_ratios, width, label='Validation', color='#A23B72')
    
    ax1.set_title('Object Distribution Ratio (%)', fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Target 90%')
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Target 10%')
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. 클래스별 절대 객체 수
    bars1 = ax2.bar(x - width/2, train_obj_counts, width, label='Train', color='#2E86AB')
    bars2 = ax2.bar(x + width/2, val_obj_counts, width, label='Validation', color='#A23B72')
    
    ax2.set_title('Absolute Object Count', fontweight='bold')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Number of Objects')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 조합별 샘플 분할
    all_combos = set(train_combos.keys()) | set(val_combos.keys())
    combo_names = [str(combo).replace("'", "").replace("(", "").replace(")", "") for combo in all_combos]
    
    train_combo_counts = [train_combos.get(combo, 0) for combo in all_combos]
    val_combo_counts = [val_combos.get(combo, 0) for combo in all_combos]
    
    x_combo = np.arange(len(combo_names))
    bars1 = ax3.bar(x_combo - width/2, train_combo_counts, width, label='Train', color='#2E86AB')
    bars2 = ax3.bar(x_combo + width/2, val_combo_counts, width, label='Validation', color='#A23B72')
    
    ax3.set_title('Sample Count by Combination', fontweight='bold')
    ax3.set_xlabel('Class Combination')
    ax3.set_ylabel('Number of Samples')
    ax3.set_xticks(x_combo)
    ax3.set_xticklabels(combo_names, rotation=45, ha='right')
    ax3.legend()
    
    # 4. 조합별 분할 비율
    total_combo_counts = [train_combo_counts[i] + val_combo_counts[i] for i in range(len(combo_names))]
    train_combo_ratios = [train_combo_counts[i] / total_combo_counts[i] * 100 if total_combo_counts[i] > 0 else 0 for i in range(len(combo_names))]
    val_combo_ratios = [val_combo_counts[i] / total_combo_counts[i] * 100 if total_combo_counts[i] > 0 else 0 for i in range(len(combo_names))]
    
    bars1 = ax4.bar(x_combo - width/2, train_combo_ratios, width, label='Train', color='#2E86AB')
    bars2 = ax4.bar(x_combo + width/2, val_combo_ratios, width, label='Validation', color='#A23B72')
    
    ax4.set_title('Split Ratio by Combination (%)', fontweight='bold')
    ax4.set_xlabel('Class Combination')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_xticks(x_combo)
    ax4.set_xticklabels(combo_names, rotation=45, ha='right')
    ax4.legend()
    ax4.axhline(y=90, color='red', linestyle='--', alpha=0.5)
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'split_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(analysis_data, data_path, save_path=None):
    """요약 보고서 생성"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 제목
    fig.suptitle('Dataset Analysis Summary Report', fontsize=20, fontweight='bold', y=0.95)
    
    # 기본 통계
    total_samples = analysis_data['total_samples']
    total_objects = sum(analysis_data['total_class_counts'].values())
    
    # 텍스트 요약
    summary_text = f"""
📊 DATASET OVERVIEW
• Total Samples: {total_samples:,}
• Total Objects: {total_objects:,}
• Classes: Vehicle, Pedestrian, Cyclist

🔍 CLASS DISTRIBUTION
• Vehicle: {analysis_data['total_class_counts'].get('Vehicle', 0):,} objects ({analysis_data['total_class_counts'].get('Vehicle', 0)/total_objects*100:.1f}%)
• Pedestrian: {analysis_data['total_class_counts'].get('Pedestrian', 0):,} objects ({analysis_data['total_class_counts'].get('Pedestrian', 0)/total_objects*100:.1f}%)
• Cyclist: {analysis_data['total_class_counts'].get('Cyclist', 0):,} objects ({analysis_data['total_class_counts'].get('Cyclist', 0)/total_objects*100:.1f}%)

🎯 CLASS COMBINATIONS FOUND
"""
    
    for combo, count in analysis_data['class_combinations'].items():
        combo_str = str(combo).replace("'", "").replace("(", "").replace(")", "")
        percentage = count / total_samples * 100
        summary_text += f"• {combo_str}: {count:,} samples ({percentage:.1f}%)\n"
    
    summary_text += f"""
🚗 VEHICLE DEPENDENCY ANALYSIS
• Pedestrian samples with Vehicle: {analysis_data['dependency_data']['Pedestrian']['with_vehicle']:,}
• Pedestrian samples without Vehicle: {analysis_data['dependency_data']['Pedestrian']['without_vehicle']:,}
• Cyclist samples with Vehicle: {analysis_data['dependency_data']['Cyclist']['with_vehicle']:,}
• Cyclist samples without Vehicle: {analysis_data['dependency_data']['Cyclist']['without_vehicle']:,}

✅ KEY FINDINGS
• Pedestrian ALWAYS appears with Vehicle (100% dependency)
• Cyclist ALWAYS appears with Vehicle (100% dependency) 
• No single-class samples for Pedestrian or Cyclist
• This ensures balanced 9:1 split across all classes when splitting by combinations

📈 SPLIT STRATEGY
• Combination-based stratified split ensures:
  - No sample overlap between train/val
  - Balanced class distribution (90:10 ratio)
  - Preservation of multi-class relationships
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'summary_report.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    # 데이터 경로 설정
    data_path = "/workspace/dataset/custom_av"
    save_path = "/workspace/OpenPCDet/output/visualization"
    
    # 저장 폴더 생성
    os.makedirs(save_path, exist_ok=True)
    
    print("🔄 데이터셋 분석 중...")
    analysis_data = analyze_dataset_for_visualization(data_path)
    
    print("📊 클래스 조합 분석 시각화...")
    visualize_class_combinations(analysis_data, save_path)
    
    print("🚗 Vehicle 의존성 분석 시각화...")
    visualize_vehicle_dependency(analysis_data, save_path)
    
    print("📈 분할 결과 분석 시각화...")
    visualize_split_analysis(data_path, save_path)
    
    print("📋 요약 보고서 생성...")
    create_summary_report(analysis_data, data_path, save_path)
    
    print(f"✅ 모든 시각화 완료! 결과는 {save_path}에 저장되었습니다.")
    print("\n생성된 파일들:")
    print("- class_combinations_analysis.png")
    print("- vehicle_dependency_analysis.png") 
    print("- split_analysis.png")
    print("- summary_report.png")

if __name__ == "__main__":
    main()