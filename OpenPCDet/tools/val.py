"""
val.py - 3D 객체 검출 모델의 평가를 위한 스크립트

이 스크립트는 학습된 모델의 성능을 평가하는 기능을 제공합니다.
주요 기능:
- 단일 체크포인트 평가
- 연속적인 체크포인트 평가
- 분산 평가 지원
- TensorBoard 로깅
"""

import _init_path  # 패키지 임포트를 위한 경로 설정 모듈 로드
import argparse  # 명령행 인자 처리를 위한 모듈 임포트
import datetime  # 날짜와 시간을 문자열로 처리하기 위한 모듈 임포트
import glob  # 파일 패턴 매칭을 위한 glob 모듈 임포트
import os  # 운영체제 관련 기능을 사용하기 위한 모듈 임포트
import re  # 정규 표현식을 사용하기 위한 모듈 임포트
import time  # 시간 지연 및 측정을 위한 모듈 임포트
from pathlib import Path  # 경로 처리를 위한 Path 클래스 임포트

import numpy as np  # 수치 연산을 위한 NumPy 임포트
import torch  # PyTorch 텐서 연산 및 모델 관리를 위한 모듈 임포트
from tensorboardX import SummaryWriter  # TensorBoard 로깅을 위한 요약 작성기 임포트

from eval_utils import eval_utils  # 평가 유틸리티 함수 모음 임포트
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file  # 설정 관련 함수와 객체 임포트
from pcdet.datasets import build_dataloader  # 데이터로더를 생성하는 함수 임포트
from pcdet.models import build_network  # 모델 생성을 위한 함수 임포트
from pcdet.utils import common_utils  # 공용 유틸 함수 모음 임포트

os.environ["NCCL_P2P_DISABLE"] = "1"  # 단일 GPU 환경에서 NCCL P2P 문제를 방지하기 위한 설정

def parse_config():  # 명령행 인자를 파싱하고 설정을 불러오는 함수 정의
    """
    명령행 인자를 파싱하고 설정을 로드하는 함수
    Returns:
        args: 파싱된 명령행 인자
        cfg: 설정 객체
    """
    parser = argparse.ArgumentParser(description='arg parser')  # 인자 파서를 생성하고 설명을 설정
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')  # 설정 파일 경로 인자 추가

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')  # 배치 크기 인자 추가
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')  # 데이터 로더 워커 수 인자 추가
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')  # 실험 태그 인자 추가
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')  # 시작 체크포인트 경로 인자 추가
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')  # 사전 학습 모델 경로 인자 추가
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')  # 분산 실행 방식 선택 인자 추가
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')  # 분산 학습용 포트 인자 추가
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')  # 로컬 랭크 인자 추가
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')  # 추가 설정을 명령행에서 덮어쓰는 인자 추가

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')  # 체크포인트 대기 시간 한계 인자 추가
    parser.add_argument('--start_epoch', type=int, default=0, help='')  # 평가 시작 epoch 인자 추가
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')  # 평가 결과 태그 인자 추가
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')  # 모든 체크포인트 평가 여부 인자 추가
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')  # 체크포인트 디렉터리 인자 추가
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')  # 결과 파일 저장 여부 인자 추가
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')  # 추론 시간 측정 여부 인자 추가

    args = parser.parse_args()  # 명령행 인자를 실제로 파싱

    # args.cfg_file = "cfgs/custom_av/centerpoint_pillar_1x_long_epoch.yaml"
    # args.batch_size = 1 
    # args.workers = 1 
    # args.ckpt = "../output/custom_av/centerpoint_pillar_1x_long_epoch/default/ckpt/checkpoint_epoch_10.pth"

    cfg_from_yaml_file(args.cfg_file, cfg)  # YAML 설정 파일을 읽어 전역 cfg에 반영
    cfg.TAG = Path(args.cfg_file).stem  # 설정 파일 이름을 TAG로 설정
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # cfgs/와 확장자를 제외한 경로를 실험 그룹으로 설정

    np.random.seed(1024)  # 재현성을 위한 난수 시드 고정

    if args.set_cfgs is not None:  # 추가 설정 덮어쓰기가 요청된 경우
        cfg_from_list(args.set_cfgs, cfg)  # 리스트 형태의 설정 덮어쓰기를 적용

    return args, cfg  # 파싱된 인자와 설정 객체 반환


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):  # 단일 체크포인트 평가 함수 정의
    """
    단일 체크포인트에 대한 평가를 수행하는 함수
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        args: 평가 설정이 포함된 인자
        eval_output_dir: 평가 결과를 저장할 디렉토리
        logger: 로깅을 위한 logger 객체
        epoch_id: 현재 평가 중인 epoch 번호
        dist_test: 분산 테스트 여부 (기본값: False)
    """
    # 체크포인트에서 모델 파라미터 로드
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)  # 지정된 체크포인트 파라미터를 모델에 로드
    model.cuda()  # 모델 가중치를 GPU 메모리로 이동
    
    # eval_utils를 사용하여 한 epoch에 대한 평가 수행
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )  # 한 에폭 기준으로 평가를 수행하고 결과를 저장


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):  # 아직 평가되지 않은 체크포인트를 찾는 함수 정의
    """
    아직 평가되지 않은 체크포인트를 찾는 함수
    Args:
        ckpt_dir: 체크포인트 파일들이 저장된 디렉토리 경로
        ckpt_record_file: 이미 평가된 체크포인트 목록이 기록된 파일 경로
        args: 평가 시작 epoch 등의 설정이 포함된 인자
    Returns:
        epoch_id: 발견된 체크포인트의 epoch 번호 (없으면 -1)
        cur_ckpt: 발견된 체크포인트 파일 경로 (없으면 None)
    """
    # 모든 체크포인트 파일을 찾고 수정 시간 순으로 정렬
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))  # 체크포인트 패턴에 맞는 파일 목록을 획득
    ckpt_list.sort(key=os.path.getmtime)  # 수정 시간 기준으로 오래된 순서대로 정렬
    
    # 이미 평가된 체크포인트의 epoch 번호 목록 로드
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]  # 이미 평가된 epoch 번호 목록 읽기
    
    # 각 체크포인트 파일에 대해 검사
    for cur_ckpt in ckpt_list:  # 정렬된 체크포인트들을 순회하면서
        # 파일명에서 epoch 번호 추출 (checkpoint_epoch_*.pth 형식)
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)  # 파일명에서 epoch 번호를 추출
        if num_list.__len__() == 0:  # epoch 번호가 없으면
            continue  # 다음 후보로 넘어감
            
        epoch_id = num_list[-1]  # 추출한 epoch 번호 중 마지막 값을 사용
        if 'optim' in epoch_id:  # optimizer 관련 체크포인트라면
            continue  # 평가 대상에서 제외
            
        # 아직 평가되지 않았고, 시작 epoch 이상인 체크포인트를 찾으면 반환
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:  # 아직 평가되지 않았고 시작 epoch 이상이면
            return epoch_id, cur_ckpt  # 대상 체크포인트를 반환
            
    # 평가할 체크포인트를 찾지 못한 경우
    return -1, None  # 적합한 체크포인트가 없다면 음수와 None을 반환


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):  # 반복적으로 새로운 체크포인트를 평가하는 함수 정의
    """
    주기적으로 새로운 체크포인트를 확인하고 평가를 수행하는 함수
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        args: 평가 설정이 포함된 인자
        eval_output_dir: 평가 결과를 저장할 디렉토리
        logger: 로깅을 위한 logger 객체
        ckpt_dir: 체크포인트가 저장된 디렉토리
        dist_test: 분산 테스트 여부 (기본값: False)
    """
    # 평가된 체크포인트 기록 파일 생성
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])  # 평가 기록 파일 경로 설정
    with open(ckpt_record_file, 'a'):  # 파일이 없으면 생성하기 위해 append 모드로 열기
        pass  # 내용은 작성하지 않고 파일만 보장

    # tensorboard log
    if cfg.LOCAL_RANK == 0:  # 마스터 프로세스에서만 TensorBoard 로깅 설정
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))  # 로깅 디렉터리 지정 후 작성기 생성
    total_time = 0  # 체크포인트 대기 시간을 누적할 변수 초기화
    first_eval = True  # 최초 평가 여부 플래그 초기화

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)  # 평가되지 않은 최신 체크포인트 탐색
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:  # 유효한 체크포인트가 없거나 시작 epoch 미만이면
            wait_second = 30  # 다음 확인까지 대기할 시간을 설정
            if cfg.LOCAL_RANK == 0:  # 마스터 프로세스에서만 진행 상황 출력
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)  # 대기 메시지를 출력
            time.sleep(wait_second)  # 일정 시간 동안 대기
            total_time += 30  # 누적 대기 시간을 갱신
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):  # 최대 대기 시간을 초과하면 루프 종료
                break  # 무한 대기를 방지하기 위해 종료
            continue  # 다음 반복으로 이동

        total_time = 0  # 체크포인트를 찾았으므로 대기 시간 초기화
        first_eval = False  # 첫 평가가 완료되었음을 표시

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)  # 최신 체크포인트 파라미터 로드
        model.cuda()  # 모델을 GPU로 이동

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']  # 현재 epoch 결과를 저장할 디렉터리 설정
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir
        )  # 평가를 수행하고 TensorBoard 로그 값을 수집

        if cfg.LOCAL_RANK == 0:  # 마스터 프로세스에서만 TensorBoard 로깅 실행
            for key, val in tb_dict.items():  # 수집된 지표를 순회하며
                tb_log.add_scalar(key, val, cur_epoch_id)  # 스칼라 값을 TensorBoard에 기록

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:  # 평가가 완료된 epoch를 파일에 기록
            print('%s' % cur_epoch_id, file=f)  # epoch 번호를 한 줄 추가
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)  # 로그에 평가 완료 메시지 남기기


def main():  # 평가 전체 프로세스를 수행하는 메인 함수 정의
    """
    평가 프로세스의 메인 함수
    - 설정을 파싱하고 초기화
    - 분산 테스트 설정
    - 데이터로더 생성
    - 모델 생성 및 평가 수행
    """
    # 설정 파싱
    args, cfg = parse_config()  # 명령행 인자와 설정을 파싱
  
    # 추론 시간 측정 모드
    if args.infer_time:  # 추론 시간 측정 모드가 요청된 경우
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # CUDA 호출을 동기화해 정확한 시간 측정을 보장

    if args.launcher == 'none':  # 분산 실행이 아닐 경우
        dist_test = False  # 분산 테스트 플래그를 비활성화
        total_gpus = 1  # 사용 GPU 수를 1로 설정
    else:  # 분산 실행 모드일 경우
        if args.local_rank is None:  # 로컬 랭크가 지정되지 않았다면
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))  # 환경 변수에서 랭크를 추출

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )  # 런처에 맞는 초기화 함수를 호출해 분산 환경을 설정하고 GPU 수를 얻음
        dist_test = True  # 분산 테스트 플래그를 활성화

    if args.batch_size is None:  # 배치 크기가 명시되지 않은 경우
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU  # 설정 파일의 기본 배치 크기를 사용
    else:  # 명령행에서 배치 크기를 지정한 경우
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'  # GPU 수와 나누어떨어지는지 검증
        args.batch_size = args.batch_size // total_gpus  # GPU당 배치 크기로 변환

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag  # 출력 디렉터리 경로 구성
    output_dir.mkdir(parents=True, exist_ok=True)  # 디렉터리가 없으면 생성

    eval_output_dir = output_dir / 'eval'  # 평가 결과를 저장할 하위 디렉터리 설정

    if not args.eval_all:  # 단일 체크포인트만 평가할 경우
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []  # 체크포인트 경로에서 숫자 추출
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'  # 마지막 숫자를 epoch ID로 사용
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']  # 해당 epoch 디렉터리를 생성
    else:  # 모든 체크포인트를 반복 평가할 경우
        eval_output_dir = eval_output_dir / 'eval_all_default'  # 공통 결과 디렉터리를 사용

    if args.eval_tag is not None:  # 추가 태그가 지정된 경우
        eval_output_dir = eval_output_dir / args.eval_tag  # 태그 하위 디렉터리를 사용

    eval_output_dir.mkdir(parents=True, exist_ok=True)  # 평가 디렉터리가 없다면 생성
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))  # 평가 로그 파일 경로 설정
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)  # 랭크별 로거 생성

    # log to file
    logger.info('**********************Start logging**********************')  # 로깅 시작 메시지 출력
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'  # 사용 중인 GPU 목록 확인
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)  # GPU 목록을 로그로 출력

    if dist_test:  # 분산 테스트인 경우
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))  # 전체 배치 크기를 로깅
    for key, val in vars(args).items():  # 인자 딕셔너리를 순회하면서
        logger.info('{:16} {}'.format(key, val))  # 설정된 인자 값을 한 줄씩 출력
    log_config_to_file(cfg, logger=logger)  # 설정 내용을 로그 파일에 기록

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'  # 체크포인트 디렉터리 경로 결정

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )  # 평가용 데이터셋과 데이터로더, 샘플러를 생성

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)  # 설정에 맞춰 평가 모델 생성
    with torch.no_grad():  # 평가 동안 그래디언트 계산을 비활성화
        if args.eval_all:  # 모든 체크포인트를 순회 평가할 경우
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)  # 반복 평가 루틴 실행
        else:  # 단일 체크포인트만 평가할 경우
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)  # 단일 평가 실행


if __name__ == '__main__':  # 스크립트가 직접 실행되는 경우에만
    main()  # 메인 함수를 호출해 평가를 시작
