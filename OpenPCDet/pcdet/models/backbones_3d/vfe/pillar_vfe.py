import torch  # PyTorch 텐서 연산을 위한 모듈 임포트
import torch.nn as nn  # 신경망 모듈(Linear, BatchNorm 등)을 사용하기 위한 임포트
import torch.nn.functional as F  # 활성화 함수 등 함수형 API 임포트

from .vfe_template import VFETemplate  # VFE 공통 템플릿 클래스 임포트


class PFNLayer(nn.Module):  # Pillar Feature Net의 기본 구성 블록 정의
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer  # 마지막 VFE 블록인지 여부 저장
        self.use_norm = use_norm  # 배치 정규화 사용 여부 저장
        if not self.last_vfe:
            out_channels = out_channels // 2  # 마지막 블록이 아니면 출력 채널을 절반으로 줄여 concat 대비

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)  # BN 전 bias를 제거한 Linear 층
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)  # 안정화를 위한 BatchNorm1d
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)  # 정규화를 쓰지 않으면 bias 포함 Linear 층

        self.part = 50000  # 너무 큰 배치를 나누어 처리하기 위한 임계값

    def forward(self, inputs):  # PFNLayer의 순전파 정의
        if inputs.shape[0] > self.part:  # 배치가 매우 클 때 안정성을 위해 분할 처리
            num_parts = inputs.shape[0] // self.part  # 분할 개수 계산
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]  # 각 분할에 Linear 적용
            x = torch.cat(part_linear_out, dim=0)  # 다시 하나의 텐서로 결합
        else:
            x = self.linear(inputs)  # 일반 경우에는 그대로 Linear 통과
        torch.backends.cudnn.enabled = False  # BN 시 CUDNN 사용을 잠시 비활성화 (안정성 보정)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x  # 필요 시 BatchNorm 적용 (채널 축 정렬)
        torch.backends.cudnn.enabled = True  # CUDNN 다시 활성화
        x = F.relu(x)  # 비선형 활성화 적용
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # pillar 내 포인트 차원에서 MaxPooling 수행

        if self.last_vfe:
            return x_max  # 마지막 블록이면 MaxPooling 결과만 반환 (pillar 당 하나의 벡터)
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)  # MaxPooling 결과를 포인트 수만큼 반복
            x_concatenated = torch.cat([x, x_repeat], dim=2)  # 반복된 결과와 원본을 concat해 skip-connection 역할 수행
            return x_concatenated  # 다음 PFNLayer에 전달


class PillarVFE(VFETemplate):  # Pillar 기반 Voxel Feature Encoder 정의
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM  # BN 사용 여부 설정 불러오기
        self.with_distance = self.model_cfg.WITH_DISTANCE  # 거리 값 사용 여부 설정 불러오기
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ  # 절대 좌표 사용 여부 설정 불러오기
        num_point_features += 6 if self.use_absolute_xyz else 3  # 절대 좌표를 쓰면 xyz와 중심 좌표 차이가 추가됨
        if self.with_distance:
            num_point_features += 1  # 거리 특징을 사용할 경우 차원을 추가

        self.num_filters = self.model_cfg.NUM_FILTERS  # PFNLayer 출력 채널 배열
        assert len(self.num_filters) > 0  # 최소 한 층 이상이어야 함을 보장
        num_filters = [num_point_features] + list(self.num_filters)  # 입력 차원을 앞에 붙여 층별 채널 리스트 구성

        pfn_layers = []  # PFNLayer 리스트 초기화
        for i in range(len(num_filters) - 1):  # 인접한 채널 쌍으로 순회하며 레이어 생성
            in_filters = num_filters[i]  # 입력 채널 수
            out_filters = num_filters[i + 1]  # 출력 채널 수
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )  # 마지막 인덱스에서는 last_layer 플래그를 True로 설정
        self.pfn_layers = nn.ModuleList(pfn_layers)  # ModuleList로 레이어 등록

        self.voxel_x = voxel_size[0]  # x축 복셀 크기 저장
        self.voxel_y = voxel_size[1]  # y축 복셀 크기 저장
        self.voxel_z = voxel_size[2]  # z축 복셀 크기 저장
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]  # pillar 중심 오프셋 계산 (x)
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]  # pillar 중심 오프셋 계산 (y)
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]  # pillar 중심 오프셋 계산 (z)

    def get_output_feature_dim(self):  # 출력 피처 차원을 반환하는 헬퍼 함수
        return self.num_filters[-1]  # 마지막 PFNLayer의 출력 채널 수 반환

    def get_paddings_indicator(self, actual_num, max_num, axis=0):  # padding 여부를 나타내는 마스크 생성
        actual_num = torch.unsqueeze(actual_num, axis + 1)  # 포인트 수 텐서 차원 확장
        max_num_shape = [1] * len(actual_num.shape)  # arange reshape를 위한 기본 형태 생성
        max_num_shape[axis + 1] = -1  # 비교 축에 -1을 설정해 arange가 맞는 차원으로 생성되도록 함
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)  # [0, ..., max_num-1]
        paddings_indicator = actual_num.int() > max_num  # 실제 포인트 수보다 인덱스가 크면 padding 영역으로 간주
        return paddings_indicator  # True/False 마스크 반환

    def forward(self, batch_dict, **kwargs):  # PillarVFE의 순전파 정의
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']  # 입력 포인트 및 메타 정보 추출
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)  # pillar 내 포인트 평균 좌표 계산
        f_cluster = voxel_features[:, :, :3] - points_mean  # 각 포인트의 평균에서의 차이(클러스터 피처)

        f_center = torch.zeros_like(voxel_features[:, :, :3])  # pillar 중심과의 거리 피처 초기화
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)  # x축 위치 보정
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)  # y축 위치 보정
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)  # z축 위치 보정

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]  # 원본 좌표와 보조 피처들을 모두 사용
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]  # intensity 등 후행 피처만 사용하고 좌표는 제외

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)  # 원점에서의 거리 피처 계산
            features.append(points_dist)  # 거리 피처를 피처 목록에 추가
        features = torch.cat(features, dim=-1)  # 여러 피처를 마지막 차원으로 concat

        voxel_count = features.shape[1]  # pillar 당 최대 포인트 수를 확인
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)  # padding 위치를 나타내는 마스크 생성
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)  # 피처 차원에 맞게 확장하고 dtype을 맞춤
        features *= mask  # padding 포인트에 해당하는 피처를 0으로 제거
        for pfn in self.pfn_layers:  # 정의한 PFNLayer 스택을 순서대로 적용
            features = pfn(features)  # PFNLayer 통과 후 특징 갱신
        features = features.squeeze()  # 불필요한 차원 제거로 (num_pillars, feature_dim) 형태 마련
        batch_dict['pillar_features'] = features  # pillar 특징을 batch_dict에 저장해 downstream 모듈로 전달
        return batch_dict  # 업데이트된 batch_dict 반환
