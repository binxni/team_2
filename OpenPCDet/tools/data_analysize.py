import numpy as np
import json
import pickle
from pathlib import Path
from collections import Counter
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

# ========== 데이터 로딩 함수 ==========
def load_data(data_path):
    """
    OpenPCDet 형식의 데이터 로딩
    data_path: .pkl 파일 경로 또는 디렉토리
    """
    data_path = Path(data_path)
    
    if data_path.is_file() and data_path.suffix == '.pkl':
        # 단일 pkl 파일
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data if isinstance(data, list) else [data]
    
    elif data_path.is_dir():
        # 디렉토리에서 모든 pkl 파일 로드
        data_list = []
        for pkl_file in sorted(data_path.glob('*.pkl')):
            with open(pkl_file, 'rb') as f:
                data_list.append(pickle.load(f))
        return data_list
    
    else:
        raise ValueError(f"Invalid path: {data_path}")

# ========== 유틸리티 함수 ==========
def point_in_box(points, box):
    """
    points: (N, 3) - x, y, z
    box: (7,) - x, y, z, l, w, h, heading
    """
    cx, cy, cz, l, w, h, heading = box
    
    # 박스 중심으로 이동
    points_local = points - np.array([cx, cy, cz])
    
    # Rotation matrix
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rot_matrix = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    
    # 회전 적용
    points_rotated = points_local @ rot_matrix.T
    
    # 박스 내부 체크
    mask = (np.abs(points_rotated[:, 0]) <= l/2) & \
           (np.abs(points_rotated[:, 1]) <= w/2) & \
           (np.abs(points_rotated[:, 2]) <= h/2)
    
    return mask

# ========== 분석 함수들 ==========
def analyze_point_density(clear_data, adverse_data):
    metrics = {}
    
    clear_pts = [len(frame['points']) for frame in clear_data]
    adverse_pts = [len(frame['points']) for frame in adverse_data]
    
    metrics['avg_points_clear'] = float(np.mean(clear_pts))
    metrics['avg_points_adverse'] = float(np.mean(adverse_pts))
    metrics['point_reduction_rate'] = float(1 - np.mean(adverse_pts)/np.mean(clear_pts))
    metrics['std_clear'] = float(np.std(clear_pts))
    metrics['std_adverse'] = float(np.std(adverse_pts))
    
    return metrics

def distance_wise_density(points):
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
    bins = [(0, 30), (30, 50), (50, 70), (70, 100)]
    density_by_range = {}
    
    for min_d, max_d in bins:
        mask = (distances >= min_d) & (distances < max_d)
        density_by_range[f'{min_d}-{max_d}m'] = int(np.sum(mask))
    
    return density_by_range

def analyze_intensity(clear_data, adverse_data):
    clear_intensity = np.concatenate([f['points'][:, 3] for f in clear_data if f['points'].shape[1] > 3])
    adverse_intensity = np.concatenate([f['points'][:, 3] for f in adverse_data if f['points'].shape[1] > 3])
    
    metrics = {
        'clear_mean': float(np.mean(clear_intensity)),
        'clear_std': float(np.std(clear_intensity)),
        'adverse_mean': float(np.mean(adverse_intensity)),
        'adverse_std': float(np.std(adverse_intensity)),
        'mean_shift': float(np.abs(np.mean(clear_intensity) - np.mean(adverse_intensity))),
        'std_ratio': float(np.std(adverse_intensity) / np.std(clear_intensity))
    }
    
    # KL Divergence
    hist_clear, _ = np.histogram(clear_intensity, bins=50, density=True)
    hist_adverse, _ = np.histogram(adverse_intensity, bins=50, density=True)
    metrics['kl_divergence'] = float(entropy(hist_clear + 1e-10, hist_adverse + 1e-10))
    
    return metrics

def points_per_object(data):
    stats = {'Car': [], 'Pedestrian': [], 'Cyclist': []}
    
    for frame in data:
        points = frame['points']
        boxes = frame.get('gt_boxes', [])
        labels = frame.get('gt_names', [])
        
        if len(boxes) == 0:
            continue
            
        for box, label in zip(boxes, labels):
            if label not in stats:
                stats[label] = []
            mask = point_in_box(points[:, :3], box)
            pts_count = np.sum(mask)
            stats[label].append(pts_count)
    
    result = {}
    for k, v in stats.items():
        if len(v) > 0:
            result[k] = {
                'mean': float(np.mean(v)), 
                'std': float(np.std(v)), 
                'min': int(np.min(v)),
                'max': int(np.max(v)),
                'count': len(v)
            }
    return result

def estimate_noise_ratio(points, method='statistical'):
    if method == 'statistical':
        nbrs = NearestNeighbors(n_neighbors=min(10, len(points))).fit(points[:, :3])
        distances, _ = nbrs.kneighbors(points[:, :3])
        mean_dist = np.mean(distances, axis=1)
        
        threshold = np.mean(mean_dist) + 2 * np.std(mean_dist)
        noise_ratio = np.sum(mean_dist > threshold) / len(points)
        
    elif method == 'ground':
        z_threshold = np.percentile(points[:, 2], 95) + 5.0
        noise_ratio = np.sum(points[:, 2] > z_threshold) / len(points)
    
    return float(noise_ratio)

def spatial_uniformity(points, voxel_size=0.5):
    voxel_coords = np.floor(points[:, :3] / voxel_size).astype(int)
    unique_voxels = len(np.unique(voxel_coords, axis=0))
    total_points = len(points)
    
    occupancy_rate = unique_voxels / (total_points / 10)
    
    voxel_ids = [tuple(v) for v in voxel_coords]
    voxel_counts = Counter(voxel_ids)
    probs = np.array(list(voxel_counts.values())) / total_points
    spatial_entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return {
        'occupancy_rate': float(occupancy_rate),
        'spatial_entropy': float(spatial_entropy)
    }

def class_imbalance_metrics(clear_data, adverse_data):
    clear_labels = [name for f in clear_data for name in f.get('gt_names', [])]
    adverse_labels = [name for f in adverse_data for name in f.get('gt_names', [])]
    
    clear_counts = Counter(clear_labels)
    adverse_counts = Counter(adverse_labels)
    
    imbalance = {}
    for cls in clear_counts.keys():
        adverse_count = adverse_counts.get(cls, 0)
        ratio = adverse_count / clear_counts[cls] if clear_counts[cls] > 0 else 0
        imbalance[cls] = {
            'clear_count': clear_counts[cls],
            'adverse_count': adverse_count,
            'retention_rate': float(ratio)
        }
    
    return imbalance

def annotation_quality_metrics(data):
    from scipy import stats
    
    metrics = {}
    
    # 박스가 있는 프레임만 필터링
    valid_data = [f for f in data if len(f.get('gt_boxes', [])) > 0]
    
    if len(valid_data) == 0:
        return {'error': 'No annotations found'}
    
    boxes = np.concatenate([f['gt_boxes'] for f in valid_data])
    volumes = boxes[:, 3] * boxes[:, 4] * boxes[:, 5]
    
    # 박스 크기 이상치
    z_scores = np.abs(stats.zscore(volumes))
    metrics['abnormal_boxes_ratio'] = float(np.sum(z_scores > 3) / len(boxes))
    
    # 포인트 없는 박스
    empty_boxes = 0
    total_boxes = 0
    for frame in valid_data:
        for box in frame['gt_boxes']:
            pts_in_box = np.sum(point_in_box(frame['points'][:, :3], box))
            if pts_in_box < 5:
                empty_boxes += 1
            total_boxes += 1
    
    metrics['empty_boxes_ratio'] = float(empty_boxes / total_boxes) if total_boxes > 0 else 0
    metrics['total_boxes'] = total_boxes
    
    return metrics

def memory_footprint_analysis(data, batch_size=4):
    point_counts = [len(f['points']) for f in data]
    
    max_points = np.max(point_counts)
    avg_points = np.mean(point_counts)
    
    # 4 bytes per float, 4 features (x,y,z,intensity)
    max_memory_mb = (max_points * 4 * 4 * batch_size) / (1024**2)
    avg_memory_mb = (avg_points * 4 * 4 * batch_size) / (1024**2)
    
    cv = np.std(point_counts) / np.mean(point_counts)
    
    return {
        'max_memory_mb': float(max_memory_mb),
        'avg_memory_mb': float(avg_memory_mb),
        'coefficient_variation': float(cv),
        'oom_risk': 'High' if cv > 0.3 else 'Low'
    }

# ========== 메인 분석 함수 ==========
def comprehensive_preprocessing_analysis(clear_path, adverse_path):
    print("Loading data...")
    clear_data = load_data(clear_path)
    adverse_data = load_data(adverse_path)
    
    print(f"Loaded {len(clear_data)} clear frames and {len(adverse_data)} adverse frames")
    
    report = {}
    
    print("1. Analyzing point density...")
    report['density'] = analyze_point_density(clear_data, adverse_data)
    
    print("2. Analyzing intensity distribution...")
    report['intensity'] = analyze_intensity(clear_data, adverse_data)
    
    print("3. Analyzing points per object...")
    report['points_per_object'] = {
        'clear': points_per_object(clear_data),
        'adverse': points_per_object(adverse_data)
    }
    
    print("4. Estimating noise ratio...")
    report['noise'] = {
        'clear': float(np.mean([estimate_noise_ratio(f['points']) for f in clear_data[:100]])),  # 샘플링
        'adverse': float(np.mean([estimate_noise_ratio(f['points']) for f in adverse_data[:100]]))
    }
    
    print("5. Analyzing class imbalance...")
    report['class_imbalance'] = class_imbalance_metrics(clear_data, adverse_data)
    
    print("6. Analyzing memory footprint...")
    report['memory'] = {
        'clear': memory_footprint_analysis(clear_data),
        'adverse': memory_footprint_analysis(adverse_data)
    }
    
    print("7. Checking annotation quality...")
    report['annotation_quality'] = {
        'clear': annotation_quality_metrics(clear_data),
        'adverse': annotation_quality_metrics(adverse_data)
    }
    
    # 경고 생성
    warnings = []
    if report['density']['point_reduction_rate'] > 0.3:
        warnings.append("⚠️ 30% 이상 포인트 감소 - 샘플링 비율 조정 필요")
    
    if report['noise']['adverse'] > 0.1:
        warnings.append("⚠️ 10% 이상 노이즈 - 필터링 강화 필요")
    
    # Pedestrian 체크 (있는 경우만)
    if 'Pedestrian' in report['class_imbalance']:
        if report['class_imbalance']['Pedestrian']['retention_rate'] < 0.7:
            warnings.append("⚠️ Pedestrian 30% 이상 감소 - 클래스 가중치 조정")
    
    if report['memory']['adverse']['coefficient_variation'] > 0.35:
        warnings.append("⚠️ 높은 메모리 변동성 - 동적 배치 사이즈 고려")
    
    if report['intensity']['kl_divergence'] > 0.3:
        warnings.append("⚠️ 큰 Intensity 분포 차이 - 정규화 전략 재검토")
    
    report['warnings'] = warnings
    
    return report

# ========== 실행 예시 ==========
if __name__ == "__main__":
    # 사용 방법:
    # 1. pkl 파일이 있는 디렉토리 경로
    # 2. 단일 pkl 파일 경로
    
    clear_path = '../data/custom_av_64/points'  # 또는 'clear_data.pkl'
    adverse_path = '../data/custom_av_64/points_lisa'  # 또는 'adverse_data.pkl'
    
    report = comprehensive_preprocessing_analysis(clear_path, adverse_path)
    
    # 결과 저장
    with open('preprocessing_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*50)
    print("Analysis Report")
    print("="*50)
    print(json.dumps(report, indent=2))