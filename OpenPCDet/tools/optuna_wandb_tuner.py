#!/usr/bin/env python3
"""Optuna + Weights & Biases hyper-parameter tuner for OpenPCDet.

This helper script samples a handful of important hyper-parameters from
`tools/cfgs/custom_av/centerpoint_pillar_val.yaml`, launches the standard
training / validation binaries, and streams the key metrics into WandB so that
trials can be compared visually.

Usage example:
    python tools/optuna_wandb_tuner.py \
        --cfg_file tools/cfgs/custom_av/centerpoint_pillar_val.yaml \
        --num_trials 10 \
        --wandb_project centerpoint-optuna \
        --gpus 0

The script expects Optuna and WandB to be installed:
    pip install optuna wandb optuna-integration

Note: heavy training / validation workloads are still executed by
`tools/train.py` and `tools/val.py`, so make sure the dataset path, GPU setup
and environment variables match your local configuration before launching a
sweep.
"""

from __future__ import annotations

import argparse  # 명령행 인자 처리를 위한 모듈
import os  # 환경변수 제어와 경로 결합을 위해 사용
import re  # 로그 문자열에서 수치를 추출하기 위한 정규표현식
import subprocess  # 학습/평가 스크립트를 외부 프로세스로 실행
import sys  # 파이썬 인터프리터 경로 사용을 위해
from pathlib import Path  # 출력 디렉터리 제어에 활용
from typing import Dict, Optional

import optuna  # 하이퍼파라미터 탐색 라이브러리
import wandb  # 실험 로깅 및 시각화를 위한 라이브러리

from pcdet.config import cfg, cfg_from_yaml_file  # OpenPCDet 설정 로더

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _parse_last_loss(log_text: str) -> Optional[float]:
    """Parse the final reported training loss from OpenPCDet stdout."""
    matches = list(re.finditer(r"Loss:\s*([0-9eE+\-.]+)", log_text))
    if not matches:
        return None
    try:
        return float(matches[-1].group(1))
    except ValueError:
        return None


def _parse_val_metric(log_text: str) -> Optional[float]:
    """Try to extract an overall mAP (or similar) score from val.py output."""
    patterns = [
        r"mAP:\s*([0-9.]+)",
        r"overall\s+AP[:=]\s*([0-9.]+)",
        r"ALL\s+AP[:=]\s*([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, log_text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


# -----------------------------------------------------------------------------
# Objective definition
# -----------------------------------------------------------------------------


def build_objective(args: argparse.Namespace, output_root: Path):
    """Create the Optuna objective callable."""

    def objective(trial: optuna.trial.Trial) -> float:
        # Optuna가 제안한 값으로 YAML 항목을 덮어쓸 하이퍼파라미터를 샘플링
        lr = trial.suggest_float("optim.lr", 5e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("optim.weight_decay", 1e-4, 5e-3, log=True)
        pct_start = trial.suggest_float("optim.pct_start", 0.1, 0.6)
        div_factor = trial.suggest_float("optim.div_factor", 5.0, 20.0)
        score_thresh = trial.suggest_float("model.score_thresh", 0.05, 0.20)

        run_tag = f"{args.run_prefix}trial_{trial.number:03d}"  # extra_tag와 WandB run 이름으로 활용
        run_dir = output_root / run_tag  # trial별 출력 디렉터리 경로

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_tag,
            config={
                "OPTIMIZATION.LR": lr,
                "OPTIMIZATION.WEIGHT_DECAY": weight_decay,
                "OPTIMIZATION.PCT_START": pct_start,
                "OPTIMIZATION.DIV_FACTOR": div_factor,
                "MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH": score_thresh,
            },
            mode="online" if not args.wandb_offline else "offline",
            reinit=True,
        )

        train_args = [
            f"--cfg_file={args.cfg_file}",
            f"--extra_tag={run_tag}",
            "--set",
            "OPTIMIZATION.LR",
            f"{lr}",
            "OPTIMIZATION.WEIGHT_DECAY",
            f"{weight_decay}",
            "OPTIMIZATION.PCT_START",
            f"{pct_start}",
            "OPTIMIZATION.DIV_FACTOR",
            f"{div_factor}",
            "MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH",
            f"{score_thresh:.4f}",
        ]

        gpu_ids: list[str] = []
        if args.gpus:
            gpu_ids = [gid.strip() for gid in args.gpus.split(',') if gid.strip()]
        num_gpus = len(gpu_ids)

        if num_gpus > 1:
            train_cmd = [
                "bash",
                "scripts/torch_train.sh",
                str(num_gpus),
                *train_args,
            ]
        else:
            train_cmd = [
                sys.executable,
                "tools/train.py",
                "--launcher",
                "none",
                *train_args,
            ]

        if args.train_extra:
            train_cmd.extend(args.train_extra)

        env = os.environ.copy()  # 자식 프로세스 환경변수 사본 생성
        if args.gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.gpus
        if args.wandb_api_key:
            env.setdefault("WANDB_API_KEY", args.wandb_api_key)

        train_process = subprocess.run(
            train_cmd,
            env=env,
            text=True,
            capture_output=True,
            cwd=args.repo_root,
        )

        train_loss = _parse_last_loss(train_process.stdout)  # 로깅된 최종 손실을 파싱
        if train_loss is not None:
            wandb.log({"train_loss_last": train_loss}, commit=False)

        if train_process.returncode != 0:
            wandb.log({"train_failed": True})
            trial.set_user_attr("train_stdout", train_process.stdout)
            trial.set_user_attr("train_stderr", train_process.stderr)
            wandb.finish()
            raise optuna.TrialPruned("Training command failed")

        # Locate the most recent checkpoint.
        ckpt_dir = run_dir / "ckpt"
        ckpt_candidates = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))  # 최신 체크포인트 탐색
        if not ckpt_candidates:
            wandb.finish()
            raise optuna.TrialPruned("No checkpoint was produced by training")
        latest_ckpt = ckpt_candidates[-1]

        val_cmd = [
            sys.executable,
            "tools/val.py",
            f"--cfg_file={args.cfg_file}",
            f"--ckpt={latest_ckpt}",
            "--eval_tag",
            run_tag,
            "--launcher",
            "none",
        ]
        if args.val_extra:
            val_cmd.extend(args.val_extra)

        val_process = subprocess.run(
            val_cmd,
            env=env,
            text=True,
            capture_output=True,
            cwd=args.repo_root,
        )
        if val_process.returncode != 0:
            wandb.log({"val_failed": True})
            trial.set_user_attr("val_stdout", val_process.stdout)
            trial.set_user_attr("val_stderr", val_process.stderr)
            wandb.finish()
            raise optuna.TrialPruned("Validation command failed")
        val_metric = _parse_val_metric(val_process.stdout)
        if val_metric is None:
            wandb.log({"val_metric_parse_failed": True})
            wandb.finish()
            raise optuna.TrialPruned("Unable to parse validation metric")

        wandb.log({"val_metric": val_metric})  # WandB에 최종 평가 값 기록

        wandb.finish()
        # Optuna maximizes by default when direction='maximize'.
        return val_metric

    return objective


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna + WandB tuner for OpenPCDet")
    parser.add_argument("--cfg_file", type=str,
                        default="tools/cfgs/custom_av/centerpoint_pillar_val.yaml",
                        help="Base YAML configuration to tune")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///optuna.db)")
    parser.add_argument("--study_name", type=str, default="centerpoint_tuning",
                        help="Optuna study name")
    parser.add_argument("--gpus", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES override for spawned processes")
    parser.add_argument("--train_extra", nargs='*', default=[],
                        help="Additional args appended to the train.py command")
    parser.add_argument("--val_extra", nargs='*', default=[],
                        help="Additional args appended to the val.py command")
    parser.add_argument("--wandb_project", type=str, required=True,
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity / team name")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="Optional WANDB_API_KEY override")
    parser.add_argument("--wandb_offline", action="store_true",
                        help="Run WandB in offline mode")
    parser.add_argument("--run_prefix", type=str, default="optuna_",
                        help="Prefix for extra_tag directories")
    parser.add_argument("--direction", choices=["minimize", "maximize"], default="maximize",
                        help="Optimization direction for the evaluation metric")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna")
    parser.add_argument("--repo_root", type=str, default=".",
                        help="Repository root where train/val scripts are executed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the base config once so that we can locate the output directory.
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1]) if '/' in args.cfg_file else ''
    output_root = Path(cfg.ROOT_DIR) / "output" / cfg.EXP_GROUP_PATH / cfg.TAG
    output_root.mkdir(parents=True, exist_ok=True)

    # Prepare Optuna study.
    sampler = optuna.samplers.TPESampler(seed=args.seed)  # 탐색용 TPE 샘플러 생성
    study = optuna.create_study(
        direction=args.direction,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.storage is not None,
        sampler=sampler,
    )

    objective = build_objective(args, output_root)

    try:
        study.optimize(objective, n_trials=args.num_trials)
    except KeyboardInterrupt:
        print("Interrupted by user. Current best value:", study.best_value, file=sys.stderr)

    try:
        best_trial = study.best_trial
    except ValueError:
        print("No trials completed successfully; nothing to report.")
        return

    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
