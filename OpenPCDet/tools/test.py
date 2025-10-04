import _init_path  # 패키지 임포트를 위해 경로를 설정
import argparse  # 명령행 인자를 파싱하기 위한 모듈
import glob  # 패턴에 맞는 파일 목록을 찾기 위한 모듈
from pathlib import Path  # 경로 처리를 객체 지향적으로 수행하기 위한 Path 클래스
import numpy as np  # 수치 연산과 배열 처리를 위한 NumPy
import torch  # PyTorch 라이브러리
import os  # 운영체제 관련 기능을 사용하기 위한 모듈
from tqdm import tqdm  # 진행 상황을 시각화하기 위한 tqdm

from pcdet.config import cfg, cfg_from_yaml_file  # 설정 객체와 YAML 로더를 임포트
from pcdet.datasets import DatasetTemplate  # 데이터셋 템플릿 기본 클래스를 임포트
from pcdet.models import build_network, load_data_to_gpu  # 모델 생성 및 GPU 로딩 유틸
from pcdet.utils import common_utils  # 공용 유틸리티 함수 모음
import pickle  # 결과를 피클 형식으로 저장하기 위한 모듈
from thop import profile  # FLOPs 계산을 위한 thop 모듈
import time  # 실행 시간 측정을 위한 모듈

os.environ["NCCL_P2P_DISABLE"] = "1"  # 단일 GPU 환경에서 NCCL 통신 문제를 피하기 위한 설정


def get_filename_without_extension(file_path):  # 확장자를 제외한 파일명을 얻는 헬퍼 함수 정의
    """파일 전체 경로에서 확장자를 제외한 파일명을 추출합니다."""
    filename_with_extension = os.path.basename(file_path)  # 경로에서 파일명과 확장자를 분리
    filename_without_extension = os.path.splitext(filename_with_extension)[0]  # 확장자를 제거한 파일명 추출
    return filename_without_extension  # 처리된 파일명을 반환

class DemoDataset(DatasetTemplate):  # 테스트 데이터를 로드하기 위한 커스텀 데이터셋 클래스 정의
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):  # 초기화 함수 정의
        """
        Args:
            dataset_cfg (dict): 데이터셋 설정.
            class_names (list): 클래스 이름들.
            training (bool): 학습 여부.
            root_path (Path or str): 데이터셋 루트 경로.
            logger (Logger): 로거.
            ext (str): 파일 확장자.
        """
        super().__init__(  # 부모 클래스 초기화 메서드를 호출해 기본 구성 적용
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )  # 매개변수로 데이터 구성 정보를 전달
        self.root_path = root_path  # 데이터 루트 경로 저장
        self.ext = ext  # 포인트 클라우드 파일 확장자 저장

        test_frame_ids_filename = os.path.join(root_path, "ImageSets/test.txt")  # 테스트 프레임 목록 파일 경로 지정
        with open(test_frame_ids_filename, 'r') as f:  # 텍스트 파일을 열어
            test_frame_ids = [line.strip() for line in f.readlines()]  # 라인별 프레임 ID를 공백 제거 후 목록으로 저장

        self.sample_file_list = sorted(  # 정렬된 샘플 파일 경로 목록 생성
            [os.path.join(root_path, f"points/{frame_id}{ext}") for frame_id in test_frame_ids]
        )  # 각 프레임 ID에 확장자를 붙여 포인트 파일 경로 생성

    def __len__(self):  # 데이터셋 크기를 반환하는 매직 메서드 정의
        return len(self.sample_file_list)  # 샘플 파일 개수를 반환

    def __getitem__(self, index):  # 인덱스로 샘플을 조회하는 매직 메서드 정의
        if self.ext == '.bin':  # 확장자가 .bin인 경우
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)  # 바이너리 포인트 클라우드 로드
        elif self.ext == '.npy':  # 확장자가 .npy인 경우
            points = np.load(self.sample_file_list[index])  # NumPy 포맷 포인트 클라우드 로드
        else:  # 지원하지 않는 확장자인 경우
            raise NotImplementedError(f"File extension {self.ext} is not supported")  # 예외 발생

        frame_id = get_filename_without_extension(self.sample_file_list[index])  # 프레임 식별자 추출
        input_dict = {'points': points, 'frame_id': frame_id}  # 전처리를 위한 입력 딕셔너리 구성

        data_dict = self.prepare_data(data_dict=input_dict)  # 상위 클래스 전처리 파이프라인 적용
        return data_dict  # 전처리된 데이터를 반환

def parse_config():  # 설정 파일과 경로를 불러오는 함수 정의
    parser = argparse.ArgumentParser(description='Argument parser for demo')  # 인자 파서를 생성해 설명을 설정
    parser.add_argument('--cfg_file', type=str, required=False, help='Specify the config for demo')  # 설정 파일 경로 인자 등록
    parser.add_argument('--data_path', type=str, required=False, help='Specify the custom_av directory')  # 데이터 경로 인자 등록
    parser.add_argument('--ckpt', type=str, required=False, help='Specify the pretrained model')  # 체크포인트 경로 인자 등록
    parser.add_argument('--ext', type=str, default='.npy', help='Specify the extension of your point cloud data file')  # 포인트 파일 확장자 인자 등록

    args = parser.parse_args()  # 명령행에서 인자를 파싱
    args.cfg_file = "cfgs/custom_av/centerpoint_pillar_1x_long_epoch.yaml"  # 기본 설정 파일 경로를 강제로 지정
    args.ckpt = "../output/custom_av/centerpoint_pillar_1x_long_epoch/default/ckpt/checkpoint_epoch_80.pth"  # 기본 체크포인트 경로 지정
    args.data_path = "../data/custom_av"  # 기본 데이터 루트 경로 지정

    cfg_from_yaml_file(args.cfg_file, cfg)  # YAML 설정을 로드하여 전역 cfg에 반영
    return args, cfg  # 파싱된 인자와 설정 객체를 반환

def main():  # 스크립트의 진입점 함수 정의
    args, cfg = parse_config()  # 설정과 인자를 불러오기
    logger = common_utils.create_logger()  # 공용 로거 생성

    demo_dataset = DemoDataset(  # 추론용 데이터셋 인스턴스를 생성
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )  # 평가용 데이터셋 객체 생성
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')  # 전체 샘플 수를 로그로 출력

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)  # 설정에 맞게 모델 생성
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)  # 체크포인트 파라미터를 모델에 로드
    model.cuda()  # 모델을 GPU 메모리로 이동
    model.eval()  # 모델을 평가 모드로 전환

    dummy_data = demo_dataset.collate_batch([demo_dataset[0]])  # 첫 번째 샘플을 배치 형태로 묶어 더미 입력 생성
    load_data_to_gpu(dummy_data)  # 더미 입력을 GPU로 이동
    try:  # FLOPs 계산 작업을 시도
        macs, params = profile(model, inputs=(dummy_data,), verbose=False)  # 모델의 MAC과 파라미터 수를 계산
        flops = macs * 2 / 1e9  # MAC을 FLOPs로 환산해 GFLOPs 단위로 변환
        with open("flops.txt", "w") as f:  # FLOPs 결과를 저장할 파일을 열고
            f.write(f"{flops:.3f}\n")  # 소수점 셋째 자리까지 기록
        logger.info(f"FLOPs: {flops:.3f} GFLOPs (saved to flops.txt)")  # 로그에 FLOPs 정보를 출력
    except Exception as e:  # 예외가 발생한 경우
        logger.error(f"FLOPs 측정 실패: {e}")  # 오류 메시지를 로그로 남김

    for _ in range(20):  # GPU 워밍업을 20회 수행
        _ = model(dummy_data)  # 더미 데이터를 사용해 GPU 커널을 워밍업

    save_filename = "result.pkl"  # 결과를 저장할 파일명 지정

    det_annos = list()  # 탐지 결과를 누적할 리스트 초기화
    with torch.no_grad():  # 평가 시 그래디언트 계산 비활성화
        total_time = 0.0  # 총 추론 시간을 저장할 변수 초기화
        num_samples = 0  # 처리한 샘플 수를 저장할 변수 초기화

        for idx, data_dict in enumerate(tqdm(demo_dataset, desc="Processing dataset")):  # tqdm으로 진행 상태를 보면서 데이터 순회
            data_dict = demo_dataset.collate_batch([data_dict])  # 단일 샘플을 배치 형식으로 변환
            load_data_to_gpu(data_dict)  # 변환된 배치를 GPU로 이동
            torch.cuda.synchronize()  # 시작 시점 동기화
            start = time.time()  # 추론 시작 시간 기록
            pred_dicts, _ = model.forward(data_dict)  # 모델 추론 수행
            torch.cuda.synchronize()  # 종료 시점 동기화
            end = time.time()  # 추론 종료 시간 기록
            elapsed = (end - start) * 1000  # ms 단위 경과 시간 계산
            total_time += elapsed  # 총 시간에 누적
            num_samples += 1  # 처리한 샘플 수 증가
            pred_dict = pred_dicts[0]  # 배치에서 첫 번째 결과를 선택
            frame_id = data_dict['frame_id'][0]  # 해당 샘플의 프레임 ID 추출
            pred_dict = {k: v.cpu().numpy() for k, v in pred_dict.items()}  # 예측 텐서를 CPU NumPy 배열로 변환

            num_obj = len(pred_dict['pred_labels'])  # 감지된 객체 수를 계산

            if num_obj == 0:  # 감지된 객체가 없을 경우
                print("At least one object should be detected.")  # 사용자에게 메시지 출력
                assert num_obj != 0, "No objects detected. The program requires at least one object to be detected."  # 실행을 중단하는 단언문

            class_names = list()  # 객체 클래스 이름을 저장할 리스트 초기화
            for obj_idx in range(num_obj):  # 감지된 각 객체에 대해 반복
                class_id = pred_dict['pred_labels'][obj_idx] - 1  # 예측 라벨을 0 기반 인덱스로 변환
                class_name = cfg.CLASS_NAMES[class_id]  # 클래스 이름을 설정에서 조회
                class_names.append(class_name)  # 이름을 리스트에 추가

            det_anno = {  # OpenPCDet 평가 포맷에 맞춰 프레임별 결과 구성
                'name' : np.array(class_names, dtype='<U10'),  # 감지된 객체들의 클래스 이름 배열
                'score' : pred_dict['pred_scores'],  # 각 객체의 신뢰도 점수 배열
                'boxes_lidar': pred_dict['pred_boxes'],  # LiDAR 좌표계 기준 예측 박스 배열
                'pred_labels': pred_dict['pred_labels'],  # 클래스 ID 배열
                'frame_id': frame_id,  # 현재 프레임 식별자
            }

            det_annos.append(det_anno)  # 결과 리스트에 현재 프레임 결과 추가

        if num_samples > 0:  # 최소 한 개 이상의 샘플을 처리했을 경우
            avg_time = total_time / num_samples  # 평균 추론 시간을 계산
            logger.info(f"Average inference time per sample: {avg_time:.3f} ms")  # 평균 시간을 로그로 출력

    with open(save_filename, 'wb') as f:  # 결과 파일을 바이너리 쓰기 모드로 열고
        pickle.dump(det_annos, f)  # 탐지 결과 리스트를 피클로 저장

if __name__ == '__main__':  # 스크립트가 직접 실행될 때만 main 실행
    main()  # 메인 함수를 호출해 평가를 수행
