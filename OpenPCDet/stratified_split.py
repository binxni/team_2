import os
import numpy as np
from collections import defaultdict
import random

def analyze_dataset_distribution(data_path, sample_list):
    """데이터셋의 클래스별 분포 분석"""
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

def stratified_split_by_noise(data_path, train_ratio=0.9, random_seed=42):
    """노이즈 유무별로 분리하여 각각 클래스 조합 기반 분할"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 1. 전체 sample_id 리스트 읽기
    imagesets_dir = os.path.join(data_path, 'ImageSets')
    train_file = os.path.join(imagesets_dir, 'train_origin.txt')
    
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            all_samples = [x.strip() for x in f.readlines() if x.strip()]
    else:
        # train.txt가 없으면 labels 폴더에서 모든 파일 읽기
        labels_dir = os.path.join(data_path, 'labels')
        all_samples = [f.replace('.txt', '') for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    print(f"전체 샘플 수: {len(all_samples)}")
    
    # 2. 노이즈 유무별로 데이터 분리
    no_noise_samples = []
    noise_samples = []
    
    for sample_id in all_samples:
        sample_num = int(sample_id)
        if 0 <= sample_num <= 14810:
            no_noise_samples.append(sample_id)
        elif 14811 <= sample_num <= 29601:
            noise_samples.append(sample_id)
    
    print(f"\n데이터 분리 결과:")
    print(f"- 노이즈 없는 데이터: {len(no_noise_samples)}개 (00000000~00014810)")
    print(f"- 노이즈 있는 데이터: {len(noise_samples)}개 (00014811~00029601)")
    
    # 3. 각 그룹별로 클래스별 분포 분석
    no_noise_class_counts = analyze_dataset_distribution(data_path, no_noise_samples)
    noise_class_counts = analyze_dataset_distribution(data_path, noise_samples)
    
    print("\n노이즈 없는 데이터 클래스별 객체 수:")
    for class_name in ['Vehicle', 'Pedestrian', 'Cyclist']:
        count = no_noise_class_counts.get(class_name, 0)
        print(f"  {class_name}: {count}")
    
    print("\n노이즈 있는 데이터 클래스별 객체 수:")
    for class_name in ['Vehicle', 'Pedestrian', 'Cyclist']:
        count = noise_class_counts.get(class_name, 0)
        print(f"  {class_name}: {count}")
    
    # 4. 노이즈 없는 데이터의 클래스 조합별 분할
    print("\n=== 노이즈 없는 데이터 처리 ===")
    no_noise_class_combinations = defaultdict(list)
    
    for sample_id in no_noise_samples:
        classes = get_sample_classes(data_path, sample_id)
        if classes:  # 빈 라벨이 아닌 경우만
            combo_key = tuple(sorted(classes))
            no_noise_class_combinations[combo_key].append(sample_id)
    
    print("노이즈 없는 데이터 클래스 조합별 샘플 분포:")
    for combo, samples in sorted(no_noise_class_combinations.items()):
        print(f"  {combo}: {len(samples)} 샘플")
    
    no_noise_train = []
    no_noise_val = []
    
    print("\n노이즈 없는 데이터 조합별 분할:")
    for combo, samples in no_noise_class_combinations.items():
        random.shuffle(samples)
        train_size = int(len(samples) * train_ratio)
        
        combo_train = samples[:train_size]
        combo_val = samples[train_size:]
        
        no_noise_train.extend(combo_train)
        no_noise_val.extend(combo_val)
        
        print(f"  {combo}: Train {len(combo_train)}, Val {len(combo_val)}")
    
    # 5. 노이즈 있는 데이터의 클래스 조합별 분할
    print("\n=== 노이즈 있는 데이터 처리 ===")
    noise_class_combinations = defaultdict(list)
    
    for sample_id in noise_samples:
        classes = get_sample_classes(data_path, sample_id)
        if classes:  # 빈 라벨이 아닌 경우만
            combo_key = tuple(sorted(classes))
            noise_class_combinations[combo_key].append(sample_id)
    
    print("노이즈 있는 데이터 클래스 조합별 샘플 분포:")
    for combo, samples in sorted(noise_class_combinations.items()):
        print(f"  {combo}: {len(samples)} 샘플")
    
    noise_train = []
    noise_val = []
    
    print("\n노이즈 있는 데이터 조합별 분할:")
    for combo, samples in noise_class_combinations.items():
        random.shuffle(samples)
        train_size = int(len(samples) * train_ratio)
        
        combo_train = samples[:train_size]
        combo_val = samples[train_size:]
        
        noise_train.extend(combo_train)
        noise_val.extend(combo_val)
        
        print(f"  {combo}: Train {len(combo_train)}, Val {len(combo_val)}")
    
    # 6. 최종 train/val 결합
    final_train = no_noise_train + noise_train
    final_val = no_noise_val + noise_val
    
    # 정렬
    train_list = sorted(final_train)
    val_list = sorted(final_val)
    
    # 중복 검증
    train_set = set(train_list)
    val_set = set(val_list)
    overlap = train_set & val_set
    
    print(f"\n=== 중복 검증 ===")
    if overlap:
        print(f"❌ 중복된 샘플 발견: {len(overlap)}개")
        print(f"중복 샘플 예시: {list(overlap)[:5]}")
        return None, None
    else:
        print("✅ 중복 없음")
    
    print(f"\n=== 최종 분할 결과 ===")
    print(f"노이즈 없는 데이터:")
    print(f"  - Train: {len(no_noise_train)}")
    print(f"  - Val: {len(no_noise_val)}")
    print(f"노이즈 있는 데이터:")
    print(f"  - Train: {len(noise_train)}")
    print(f"  - Val: {len(noise_val)}")
    print(f"전체:")
    print(f"  - Train: {len(train_list)}")
    print(f"  - Val: {len(val_list)}")
    print(f"  - 총합: {len(train_list) + len(val_list)} (전체: {len(all_samples)})")
    
    # 7. 분할 후 클래스별 분포 검증
    print("\n=== 분할 결과 검증 ===")
    train_class_counts = analyze_dataset_distribution(data_path, train_list)
    val_class_counts = analyze_dataset_distribution(data_path, val_list)
    total_class_counts = analyze_dataset_distribution(data_path, all_samples)
    
    print("Train 클래스별 객체 수:")
    for class_name in ['Vehicle', 'Pedestrian', 'Cyclist']:
        train_count = train_class_counts.get(class_name, 0)
        total_count = total_class_counts.get(class_name, 0)
        if total_count > 0:
            ratio = train_count / total_count * 100
            print(f"  {class_name}: {train_count}/{total_count} ({ratio:.1f}%)")
    
    print("Val 클래스별 객체 수:")
    for class_name in ['Vehicle', 'Pedestrian', 'Cyclist']:
        val_count = val_class_counts.get(class_name, 0)
        total_count = total_class_counts.get(class_name, 0)
        if total_count > 0:
            ratio = val_count / total_count * 100
            print(f"  {class_name}: {val_count}/{total_count} ({ratio:.1f}%)")
    
    return train_list, val_list

def save_split_files(data_path, train_list, val_list):
    """분할된 리스트를 파일로 저장"""
    if train_list is None or val_list is None:
        print("분할에 실패했습니다. 파일을 저장하지 않습니다.")
        return
        
    imagesets_dir = os.path.join(data_path, 'ImageSets')
    os.makedirs(imagesets_dir, exist_ok=True)
    
    # train.txt 저장
    train_file = os.path.join(imagesets_dir, 'train.txt')
    with open(train_file, 'w') as f:
        for sample_id in train_list:
            f.write(f"{sample_id}\n")
    print(f"Train 파일 저장: {train_file}")
    
    # val.txt 저장  
    val_file = os.path.join(imagesets_dir, 'val.txt')
    with open(val_file, 'w') as f:
        for sample_id in val_list:
            f.write(f"{sample_id}\n")
    print(f"Val 파일 저장: {val_file}")

if __name__ == "__main__":
    # 데이터 경로 설정
    data_path = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_hybrid"
    
    # 클래스 조합 기반 분할 수행 (노이즈 유무별 분리)
    train_list, val_list = stratified_split_by_noise(
        data_path=data_path,
        train_ratio=0.9,
        random_seed=42
    )
    
    # 파일로 저장
    save_split_files(data_path, train_list, val_list)
    
    print("\n노이즈 유무별 클래스 조합 기반 분할이 완료되었습니다!")