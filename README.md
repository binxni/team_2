# OpenPCDet 폴더 구조 안내

이 문서는 현재 저장소에 포함된 OpenPCDet 소스의 주요 디렉토리와 파일이 어떤 역할을 하는지 한글로 정리한 참고용 가이드입니다. 모델 학습이나 커스터마이징을 진행할 때 필요한 위치와 기능을 빠르게 파악할 수 있도록 구성되어 있습니다.

## 최상위 디렉토리
- `build/`: CUDA 확장 모듈을 빌드한 결과물이 저장되는 임시 디렉토리로, `lib.*`에는 파이썬에서 불러오는 `.so` 바이너리, `temp.*`에는 빌드 중간 산출물이 들어 있습니다.
- `data/`: 사용자 정의 데이터셋이 위치하는 폴더입니다. 기본 예시로 `custom_av/`가 포함되어 있으며 포인트 클라우드, 어노테이션 등을 배치합니다.
- `output/`: 학습·평가 결과(체크포인트, 로그, 예측 결과 등)가 저장되는 루트 폴더입니다. 실험별로 하위 폴더를 만들어 관리합니다.
- `pcdet/`: OpenPCDet의 핵심 파이썬 패키지입니다. 데이터셋 정의, 모델, 연산 커널, 유틸리티 코드가 모두 포함됩니다.
- `pcdet.egg-info/`: `setup.py`를 통해 패키지를 설치할 때 생성되는 메타데이터로, 의존성(`requires.txt`), 패키지 정보(`PKG-INFO`) 등이 기록됩니다.
- `requirements.txt`: 환경 구성에 필요한 파이썬 의존성 목록입니다.
- `setup.py`: OpenPCDet 패키지를 설치하고 CUDA 확장 모듈을 빌드하기 위한 스크립트입니다.
- `tools/`: 학습·평가·데모 실행 스크립트와 설정 파일이 들어 있는 실행 도구 모음입니다.
- `viewer.py`: 포인트 클라우드 뷰어를 구동하는 엔트리 스크립트로, `viewer_env/`에서 제공하는 가상환경을 사용합니다.
- `viewer_env/`: 뷰어 실행에 필요한 가상환경과 바이너리가 포함된 폴더입니다(`bin/`, `lib/`, `include/` 등 표준 venv 구조).

## `pcdet/` 패키지 상세
- `config.py`: YAML 설정을 파싱해 학습 파이프라인에 전달하는 설정 로더입니다.
- `version.py`: 패키지 버전 문자열을 정의합니다.
- `__init__.py`: 패키지 초기화를 담당하며, 주요 컴포넌트를 외부에 노출합니다.

### `pcdet/datasets/`
- `dataset.py`: 공통 데이터셋 베이스 클래스와 샘플 로딩 로직이 정의되어 있습니다.
- `augmentor/`: `data_augmentor.py`, `database_sampler.py` 등 포인트 클라우드 데이터 증강을 담당하는 모듈이 모여 있습니다. `augmentor_utils.py`는 보조 함수들을 제공합니다.
- `processor/`: 입력 포인트를 필터링·변환하는 `data_processor.py`, 특징 인코딩을 담당하는 `point_feature_encoder.py`, 통계 분석 도구(`analyze_label_distance.py`) 등이 포함됩니다.
- 데이터셋별 폴더: `kitti/`, `nuscenes/`, `lyft/`, `waymo/`, `argo2/`, `once/`, `pandaset/`, `custom/`, `custom_av/`, `custom_infra/` 등이 있으며, 각 폴더의 `*_dataset.py`에서 데이터셋 로더 클래스를 정의합니다. 예를 들어 `kitti/kitti_dataset.py`는 KITTI 포맷 로더, `waymo/waymo_dataset.py`는 Waymo Open Dataset 로더입니다. 폴더 내부의 `*_utils.py` 또는 `*_eval/` 디렉토리는 데이터 파싱과 벤치마크 평가 스크립트를 제공합니다.

### `pcdet/models/`
- `detectors/`: 학습 파이프라인의 최상위 감지기 클래스를 정의합니다. `pointpillar.py`, `pv_rcnn.py`, `centerpoint.py`, `voxel_rcnn.py`, `transfusion.py`, `bevfusion.py` 등 각 파일은 논문별 네트워크를 구현합니다. `detector3d_template.py`는 모든 감지기가 상속하는 공통 템플릿입니다.
- `backbones_3d/`: 3D 피쳐 백본을 구현한 모듈입니다. `spconv_backbone*.py`, `pointnet2_backbone.py`, `spconv_unet.py`는 대표적인 예시이며, `pfe/`는 Pillar/Voxel Feature Encoder, `focal_sparse_conv/`와 `dsvt.py`는 스파스 컨볼루션 기반 백본을 제공합니다.
- `backbones_2d/`: BEV(Bird’s Eye View) 피쳐 맵 생성을 위한 2D 백본입니다. `base_bev_backbone.py`는 공통 모듈, `map_to_bev/`와 `fuser/`는 LiDAR·이미지 피쳐 융합을 담당합니다.
- `backbones_image/`: 이미지 기반 백본(`swin.py` 등)과 Neck(`img_neck/`) 구성 요소를 포함합니다.
- `dense_heads/`: 포인트·보셀 기반 검출 헤드를 구현합니다. `anchor_head_single.py`, `center_head.py`, `point_head_*` 등은 출력 박스를 예측하는 로직을, `target_assigner/`는 학습용 타깃 생성 코드를 제공합니다.
- `roi_heads/`: RoI 기반 후처리 모듈입니다. `pointrcnn_head.py`, `pvrcnn_head.py`, `second_head.py`, `voxelrcnn_head.py` 등이 있고, `roi_head_template.py`가 공통 베이스 역할을 합니다.
- `model_utils/`: NMS, CenterNet 보조 함수(`centernet_utils.py`), Transformer 기반 모델 유틸(`transfusion_utils.py`), DSVT 보조 코드(`dsvt_utils.py`) 등 모델 전반에서 재사용되는 도우미 함수가 정의되어 있습니다.
- `view_transforms/`: 이미지 깊이맵을 BEV로 투영하는 `depth_lss.py` 등 멀티 센서 좌표 변환 모듈을 담고 있습니다.
- `__init__.py`: 모델 하위 모듈을 한 번에 임포트할 수 있도록 경로를 정리합니다.

### `pcdet/ops/`
- GPU 가속 커스텀 연산을 모아 둔 디렉토리입니다. 각 하위 폴더는 C++/CUDA 소스(`src/`)와 파이썬 래퍼(`*_op.py`, `*_utils.py`)로 구성됩니다.
  - `bev_pool/`: BEV 공간 풀링 연산(`bev_pool.py`)과 대응 CUDA 확장(`bev_pool_ext*.so`).
  - `ingroup_inds/`: 그룹 인덱스 연산을 위한 CUDA 커널과 래퍼.
  - `iou3d_nms/`: 3D IoU 계산 및 NMS 수행을 위한 연산.
  - `pointnet2/`: PointNet++ 샘플링·그룹화 커널(`pointnet2_batch/`, `pointnet2_stack/`).
  - `roiaware_pool3d/`, `roipoint_pool3d/`: RoI 기반 3D 풀링 연산.

### `pcdet/utils/`
- 박스 인코딩(`box_coder_utils.py`), 박스 기하(`box_utils.py`), KITTI 보정 행렬 처리(`calibration_kitti.py`), 손실 함수(`loss_utils.py`), 일반 유틸리티(`common_utils.py`, `transform_utils.py`), 분산 환경 보조 함수(`commu_utils.py`) 등을 제공합니다. 데이터셋별 객체 래퍼(`object3d_kitti.py`, `object3d_custom.py`)도 포함됩니다.

## `tools/` 디렉토리
- `cfgs/`: 학습·평가를 위한 YAML 설정 모음입니다. 데이터셋별(`kitti_models/`, `nuscenes_models/`, `waymo_models/`, `argo2_models/` 등)로 구성되어 있으며, 커스텀 실험은 `custom_models/`, `custom_av/`에 저장합니다.
- `train.py`, `test.py`, `val.py`: 각각 학습, 테스트, 검증을 수행하는 메인 스크립트입니다. 공통 초기화 로직은 `_init_path.py`에서 경로를 설정합니다.
- `train_utils/`: 학습 루프와 옵티마이저 설정을 관리하는 `train_utils.py`, `optimization/` 하위 모듈로 구성됩니다.
- `eval_utils/`: 평가 지표 계산과 후처리를 돕는 함수(`eval_utils.py`).
- `process_tools/`: 데이터셋 전처리 스크립트(`create_integrated_database.py`)가 위치합니다.
- `scripts/`: 분산 학습/검증 실행을 위한 쉘 스크립트(`dist_train.sh`, `torch_train.sh` 등).
- `visual_utils/`: Open3D 기반 시각화(`open3d_vis_utils.py`), 예측 결과 확인(`visualize_utils.py`) 도구 모음입니다.
- `util/`: 박스 연산(`box_np_ops.py`), 기하 함수(`geometry.py`), 평가 결과 시각화(`visualize_eval.py`) 등 보조 유틸리티입니다.
- `demo.py`: 사전학습 모델과 샘플 데이터를 이용해 빠르게 추론 결과를 확인하는 데모 스크립트입니다.
- `setup_viewer_env.sh`: 뷰어용 가상환경을 구축할 때 사용하는 셸 스크립트입니다.

## 기타 디렉토리 및 파일
- `data/custom_av/`: 사용자 정의 AV(자율주행) 데이터셋 예시. 실제 학습 시 여기에 포인트 클라우드(`.bin`), 라벨(`.json`/`.txt`) 등을 배치합니다.
- `output/custom_av/`: `tools/train.py` 실행 시 생성되는 체크포인트, TensorBoard 로그, 평가 결과 등이 저장됩니다.
- `viewer_env/bin/`, `viewer_env/lib/`: 뷰어 실행을 위한 파이썬 인터프리터와 의존성 패키지가 위치합니다.
- `viewer.py`: `viewer_env` 환경을 활성화한 뒤 LiDAR 씬을 시각화하는 실행 포인트입니다.

## 활용 팁
- 새로운 실험을 구성할 때는 `tools/cfgs/`에서 적절한 YAML을 복사해 수정한 뒤 `tools/train.py`에 전달합니다.
- 추가 데이터셋을 지원하려면 `pcdet/datasets/` 아래에 새 폴더를 만들고 `dataset.py`에서 베이스 클래스를 상속해 구현합니다.
- CUDA 연산을 수정한 경우 `python setup.py develop`으로 다시 빌드해야 `pcdet/ops/`의 변경 사항이 반영됩니다.

필요에 따라 위 구조를 참고해 원하는 기능을 빠르게 찾아보세요.
