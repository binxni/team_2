# feature_probe.py
import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

# ========== 유틸 ==========
@torch.no_grad()
def compute_stats(feats_list):
    """
    feats_list: [Tensor(B,C,H,W) ...] 또는 [Tensor(B,C,D,H,W)]
    모든 배치/공간을 펼쳐 (N, C)로 보고 채널 통계 계산
    """
    # concat over batches/layers -> (total_pixels, C)
    vecs = []
    for F in feats_list:
        # 3D or 2D 모두 지원: (B,C,*) 형태 가정
        B, C = F.shape[:2]
        v = F.permute(0,2,3,1) if F.dim()==4 else F.permute(0,2,3,4,1)
        v = v.reshape(-1, C)  # (N, C)
        vecs.append(v)
    V = torch.cat(vecs, dim=0)  # (N, C)

    mean = V.mean(dim=0)                 # (C,)
    var  = V.var(dim=0, unbiased=False)  # (C,)
    cov  = (V - mean).T @ (V - mean) / V.shape[0]  # (C,C) covariance (메모리 클 수 있음)
    # Gram은 단위정규화 후 내적합
    Vn = torch.nn.functional.normalize(V, dim=1)
    gram_trace = torch.trace((Vn.T @ Vn) / Vn.shape[0])  # 정규화된 자기유사도 총합의 크기 척도

    return {
        'mean': mean.cpu().numpy(),
        'var': var.cpu().numpy(),
        'cov_sampled_trace': float(torch.trace(cov)), # 전체 cov 저장은 무거우니 trace만
        'gram_trace': float(gram_trace),
        'count': int(V.shape[0])
    }

def cosine_between_means(stats_a, stats_b):
    a = torch.tensor(stats_a['mean'])
    b = torch.tensor(stats_b['mean'])
    a = a / (a.norm() + 1e-12)
    b = b / (b.norm() + 1e-12)
    return float((a*b).sum())

def frechet_distance(mu1, var1, mu2, var2, eps=1e-6):
    """
    FID 유사 통계 (채널 독립 가정: 공분산 대신 분산(diagonal)만 사용)
    """
    mu1 = torch.tensor(mu1); mu2 = torch.tensor(mu2)
    s1 = torch.tensor(var1); s2 = torch.tensor(var2)
    diff = (mu1 - mu2).pow(2).sum()
    # diagonal cov만 가정: trace(S1 + S2 - 2*sqrt(S1^0.5 S2 S1^0.5)) ~ trace(S1 + S2 - 2*sqrt(S1*S2))
    # diag일 때 sqrt는 원소별 sqrt
    trace_term = (s1 + s2 - 2*torch.sqrt((s1+eps)*(s2+eps))).sum()
    return float(diff + trace_term)

def mmd_rbf(X, Y, sigma=1.0):
    """
    간단한 RBF-kernel MMD (채널 평균 벡터만으로 근사도 비교)
    X,Y: (C,) 벡터 사용 (채널 통계 기반 근사)
    """
    X = torch.tensor(X).view(1,-1)
    Y = torch.tensor(Y).view(1,-1)
    def k(a, b): 
        return torch.exp(-((a-b).pow(2).sum(dim=1))/(2*sigma**2))
    return float(k(X,X) + k(Y,Y) - 2*k(X,Y))

# ========== 훅 등록 ==========
def register_feature_hooks(model, layer_names):
    """
    layer_names: 예) ['backbone_3d', 'backbone_2d.blocks.1'] 등
    반환: (handles, buffers_dict) - buffers_dict[name]에 feature 축적
    """
    buffers = defaultdict(list)
    handles = []

    def _make_hook(name):
        def hook(module, inp, out):
            # out: Tensor 또는 tuple
            if isinstance(out, (list, tuple)):
                out = out[0]
            buffers[name].append(out.detach())
        return hook

    # 문자열 경로로 모듈 찾아 훅 등록
    for name in layer_names:
        # 안전하게 모듈 탐색
        m = model
        for part in name.split('.'):
            if part.isdigit():
                m = m[int(part)]
            else:
                m = getattr(m, part)
        handles.append(m.register_forward_hook(_make_hook(name)))

    return handles, buffers

# ========== 데이터로더 ==========
def build_eval_loader(cfg_file, infos_path, batch_size=2, workers=4):
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()
    # training=False, dist=False 로 평가용 데이터로더 구성
    _, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=False
    )
    # infos 파일 경로 지정 필요 시, cfg.DATA_CONFIG._BASE_CONFIG_ 방식/CustomDataset에 맞게 수정
    # (OpenPCDet의 커스텀 데이터셋이면 DATA_CONFIG에 INFO_PATH가 이미 설정되어 있어야 함)
    return dataloader, cfg

def run_probe(model, dataloader, hook_targets, max_batches=10, device='cuda'):
    model.eval().to(device)
    handles, buffers = register_feature_hooks(model, hook_targets)

    with torch.no_grad():
        for i, batch_dict in enumerate(dataloader):
            for k,v in batch_dict.items():
                if isinstance(v, torch.Tensor):
                    batch_dict[k] = v.to(device)
            _ = model(batch_dict)  # forward; 훅이 buffers에 모음
            if i+1 >= max_batches:
                break

    for h in handles:
        h.remove()
    return {k: compute_stats(v) for k,v in buffers.items()}

# ========== 메인 ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--clear_infos', type=str, required=False, help='clear split에 해당하는 infos가 cfg에 설정되어 있어야 함')
    parser.add_argument('--adverse_infos', type=str, required=False)
    parser.add_argument('--batches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--layers', type=str, default='backbone_3d,backbone_2d')
    args = parser.parse_args()

    # 로거/랜덤시드
    logger = common_utils.create_logger()
    np.random.seed(0); torch.manual_seed(0)

    # 모델 빌드 & ckpt 로드
    cfg_from_yaml_file(args.cfg_file, cfg)
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)

    # Hook 타겟 파싱
    hook_targets = [x.strip() for x in args.layers.split(',') if x.strip()]

    # CLEAR
    clear_loader, _ = build_eval_loader(args.cfg_file, args.clear_infos, batch_size=args.batch_size, workers=args.workers)
    stats_clear = run_probe(model, clear_loader, hook_targets, max_batches=args.batches)

    # ADVERSE
    adverse_loader, _ = build_eval_loader(args.cfg_file, args.adverse_infos, batch_size=args.batch_size, workers=args.workers)
    stats_adverse = run_probe(model, adverse_loader, hook_targets, max_batches=args.batches)

    # 레이어별 비교 리포트
    report = {}
    for name in hook_targets:
        sc = stats_clear[name]; sa = stats_adverse[name]
        layer_report = {
            'cosine_between_channel_means': cosine_between_means(sc, sa),
            'fid_like': frechet_distance(sc['mean'], sc['var'], sa['mean'], sa['var']),
            'cov_trace_clear': sc['cov_sampled_trace'],
            'cov_trace_adverse': sa['cov_sampled_trace'],
            'gram_trace_clear': sc['gram_trace'],
            'gram_trace_adverse': sa['gram_trace'],
            'count_clear': sc['count'],
            'count_adverse': sa['count'],
            'mmd_rbf_means': mmd_rbf(sc['mean'], sa['mean'], sigma=1.0)
        }
        report[name] = layer_report

    import json
    print(json.dumps(report, indent=2))
