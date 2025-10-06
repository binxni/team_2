import numpy as np

class RainNoiseSimulator:
    """비 노이즈 시뮬레이션을 위한 독립적인 클래스"""
    
    def __init__(self, config=None, noise_radius=2.0):
        """
        비 노이즈 시뮬레이션 설정 초기화
        
        Args:
            config: 비 노이즈 설정 딕셔너리
            noise_radius: 원점 기준 노이즈 생성 반경 (미터)
        """
        # 기본 설정값
        default_config = {
            'RAIN_INTENSITY_RANGE': [0.3, 1.2],
            'BASE_NOISE_DENSITY': 0.005,
            'Z_BIAS_RANGE': [0.0, 6.0],
            'RAIN_INTENSITY_VALUES': [0.02, 0.25],
            'MAX_ATTENUATION_DISTANCE': 50.0,
            'BASE_ATTENUATION_RATE': 0.08,
            'MAX_ATTENUATION_PROB': 0.25,
            'INTENSITY_REDUCTION_FACTOR': 0.15,
            'DROPOUT_RATIO_RANGE': [0.03, 0.12]
        }
        
        self.config = config if config is not None else default_config
        self.noise_radius = noise_radius  # 노이즈 생성 반경
    
    def set_noise_radius(self, radius):
        """노이즈 생성 반경 설정"""
        self.noise_radius = max(0.1, radius)  # 최소 0.1m
    
    def get_noise_radius(self):
        """현재 노이즈 생성 반경 반환"""
        return self.noise_radius
        
    def simulate_rain_noise(self, points, rain_intensity=None):
        """
        포인트 클라우드에 비 노이즈를 추가
        
        Args:
            points: 원본 포인트 클라우드 (N, 4) [x, y, z, intensity]
            rain_intensity: 비 강도 (None이면 랜덤 생성)
        
        Returns:
            noisy_points: 비 노이즈가 추가된 포인트 클라우드
        """
        if rain_intensity is None:
            rain_intensity = np.random.uniform(
                self.config['RAIN_INTENSITY_RANGE'][0],
                self.config['RAIN_INTENSITY_RANGE'][1]
            )
        
        # 1. 허위 반사점 생성 (빗방울)
        noise_points = self._generate_rain_droplet_noise(points, rain_intensity)
        
        # 2. 거리별 포인트 감쇠 시뮬레이션
        attenuated_points = self._apply_distance_attenuation(points, rain_intensity)
        
        # 3. Intensity 감소 시뮬레이션
        final_points = self._apply_intensity_attenuation(attenuated_points, rain_intensity)
        
        # 4. 최종 포인트 결합
        if len(noise_points) > 0:
            result = np.concatenate([final_points, noise_points], axis=0)
        else:
            result = final_points
            
        return result, rain_intensity
    
    def _generate_rain_droplet_noise(self, points, rain_intensity):
        """빗방울로 인한 허위 반사점 생성 (원점 기준 반경 내에서만)"""
        # 노이즈 포인트 개수 계산
        noise_density = self.config['BASE_NOISE_DENSITY'] * rain_intensity
        num_noise_points = int(len(points) * noise_density)
        
        if num_noise_points == 0:
            return np.array([]).reshape(0, points.shape[1])
        
        # 원점 기준 반경 내에서만 노이즈 포인트 생성
        generated_points = []
        attempts = 0
        max_attempts = num_noise_points * 10  # 무한 루프 방지
        
        while len(generated_points) < num_noise_points and attempts < max_attempts:
            # 원점 기준 반경 내에서 랜덤 포인트 생성
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, self.noise_radius)
            
            noise_x = radius * np.cos(angle)
            noise_y = radius * np.sin(angle)
            
            # Z축은 주로 지상 위쪽에 집중 (비는 위에서 아래로)
            z_bias_range = self.config['Z_BIAS_RANGE']
            noise_z = np.random.uniform(z_bias_range[0], z_bias_range[1])
            
            # 빗방울의 낮은 intensity 시뮬레이션
            rain_intensity_range = self.config['RAIN_INTENSITY_VALUES']
            noise_intensity = np.random.uniform(
                rain_intensity_range[0], 
                rain_intensity_range[1]
            )
            
            generated_points.append([noise_x, noise_y, noise_z, noise_intensity])
            attempts += 1
        
        if len(generated_points) == 0:
            return np.array([]).reshape(0, points.shape[1])
        
        noise_points = np.array(generated_points)
        
        return noise_points
    
    def _apply_distance_attenuation(self, points, rain_intensity):
        """거리별 포인트 감쇠 시뮬레이션"""
        # 원점으로부터의 거리 계산
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        
        # 거리별 감쇠 확률 계산
        max_distance = self.config['MAX_ATTENUATION_DISTANCE']
        base_attenuation = self.config['BASE_ATTENUATION_RATE']
        
        # 비 강도에 따른 감쇠율 조정
        attenuation_rate = base_attenuation * rain_intensity
        
        # 거리에 비례한 감쇠 확률 (멀수록 더 많이 제거)
        attenuation_probs = np.clip(
            attenuation_rate * (distances / max_distance), 
            0.0, 
            self.config['MAX_ATTENUATION_PROB']
        )
        
        # 랜덤 샘플링으로 포인트 제거
        keep_mask = np.random.random(len(points)) > attenuation_probs
        
        return points[keep_mask]
    
    def _apply_intensity_attenuation(self, points, rain_intensity):
        """Intensity 감소 시뮬레이션"""
        if points.shape[1] < 4:  # intensity 채널이 없으면 스킵
            return points
        
        # 거리별 intensity 감소
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # 비 강도에 따른 intensity 감소율
        intensity_reduction = self.config['INTENSITY_REDUCTION_FACTOR'] * rain_intensity
        
        # 거리별 차등 적용
        reduction_factors = 1.0 - (intensity_reduction * distances / 50.0)
        reduction_factors = np.clip(reduction_factors, 0.3, 1.0)  # 최소 30%는 유지
        
        points[:, 3] *= reduction_factors
        
        return points
    
    def apply_weather_dropout(self, points, dropout_ratio=None):
        """날씨로 인한 포인트 손실 시뮬레이션"""
        if dropout_ratio is None:
            dropout_ratio = np.random.uniform(*self.config['DROPOUT_RATIO_RANGE'])
        
        # 균등 드롭아웃
        keep_indices = np.random.choice(
            len(points), 
            int(len(points) * (1 - dropout_ratio)), 
            replace=False
        )
        
        return points[keep_indices], dropout_ratio