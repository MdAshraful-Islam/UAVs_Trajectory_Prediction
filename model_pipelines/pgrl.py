#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data-driven-backbone physics-informed hybrid learning for multi-step UAV
trajectory forecasting.

For each available physics-model output, this script aligns that physics
trajectory with the latest LSTM v2 full-flight prediction artifact, builds
multi-step anchors, and trains a separate PIML fusion network where the
data-driven LSTM horizon is the explicit backbone and physics acts as a
guidance signal:

    backbone(LSTM horizon) + correction(LSTM horizon, physics horizon) -> H future positions

The objective is:

    L = Ldata + lambda_phy Lphy + lambda_spd Lspd + lambda_acc Lacc
        + lambda_acc_bound Lacc_bound + lambda_jrk Ljrk + lambda_vz Lvz

Outputs are written to PIML_2_output/vN with predictions, checkpoints,
metrics, validation plots, error plots, and trajectory overlays for each
physics model.
"""

import argparse
import json
import math
import pickle
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


@dataclass
class Config:
    lstm_output_root: str = "output_lstm"
    physics_output_root: str = "Physics_model_outputs"
    physics_output_version: Optional[int] = 20
    mlp_output_root: str = "MLP_output"
    validation_data_dir: str = "data/sample_dataset"
    output_root: str = "PIML_2_output"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    epochs: int = 450
    phase_balanced_sampling: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 55
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    horizon_steps: int = 250
    horizon_discount: float = 0.98
    max_time_gap_seconds: float = 0.08
    min_anchors: int = 25
    dt_seconds: float = 0.12
    lambda_spd: float = 0.03
    lambda_acc: float = 0.002
    lambda_acc_bound: float = 0.002
    lambda_jrk: float = 0.0005
    lambda_vz: float = 0.04
    lambda_phy: float = 0.015
    calibrate_phase_residual: bool = True
    calibrate_external_residual: bool = True
    residual_alpha_grid_steps: int = 101
    vmax_takeoff: float = 8.0
    vmax_cruise: float = 14.0
    vmax_landing: float = 5.0
    landing_descent_rate_max: float = 4.0
    takeoff_vz_min: float = 0.5
    takeoff_vz_max: float = 2.4
    landing_vz_min: float = -2.3
    landing_vz_max: float = -0.35
    accel_limit_xy: float = 1.8
    accel_limit_z: float = 1.4
    max_integration_accel: float = 2.5


CFG = Config()
PHASE_TO_ID = {"takeoff": 0, "cruise": 1, "landing": 2}
ID_TO_PHASE = {v: k for k, v in PHASE_TO_ID.items()}


class PIMLDataBackboneNet(nn.Module):
    def __init__(self, horizon_steps: int):
        super().__init__()
        self.horizon_steps = horizon_steps
        flat_lstm = horizon_steps * 3
        flat_joint = horizon_steps * 12
        self.backbone = nn.Sequential(
            nn.Linear(flat_lstm, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, horizon_steps * 3),
        )
        self.correction = nn.Sequential(
            nn.Linear(flat_joint, 192),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, horizon_steps * 3),
        )
        self.physics_gate = nn.Sequential(
            nn.Linear(flat_joint, 96),
            nn.ReLU(),
            nn.Linear(96, horizon_steps * 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        blocks = x.view(-1, self.horizon_steps, 6)
        physics = blocks[:, :, 0:3]
        lstm = blocks[:, :, 3:6]

        backbone = self.backbone(lstm.reshape(len(blocks), -1)).view(-1, self.horizon_steps, 3)
        joint = torch.cat([physics, lstm, physics - lstm, physics + lstm], dim=2).reshape(len(blocks), -1)
        correction = self.correction(joint).view(-1, self.horizon_steps, 3)
        gate = self.physics_gate(joint).view(-1, self.horizon_steps, 3)
        return backbone + gate * correction


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def version_number(path: Path) -> int:
    match = re.fullmatch(r"v(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def latest_version_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and version_number(p) >= 0]
    if not candidates:
        return root
    return max(candidates, key=version_number)


def parse_physics_version(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "latest", "none", "auto"}:
        return None
    if text.startswith("v"):
        text = text[1:]
    try:
        version = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Physics version must be an integer such as 20, v20, or 'latest'; got {value!r}"
        ) from exc
    if version < 0:
        raise argparse.ArgumentTypeError("Physics version must be non-negative")
    return version


def resolve_physics_output_dir(root: Path, version: Optional[int]) -> Optional[Path]:
    if not root.exists():
        return None
    if version is None:
        return latest_version_dir(root)
    if version_number(root) == version:
        return root
    version_dir = root / f"v{version}"
    if version_dir.exists():
        return version_dir
    return None


def get_next_version_dir(output_root: Path) -> Path:
    ensure_dir(output_root)
    versions = [version_number(p) for p in output_root.iterdir() if p.is_dir() and version_number(p) >= 0]
    out_dir = output_root / f"v{1 if not versions else max(versions) + 1}"
    ensure_dir(out_dir)
    return out_dir


def normalize_phase(value: object) -> str:
    text = str(value).strip().lower()
    if "take" in text or "ascend" in text:
        return "takeoff"
    if "land" in text or "descend" in text:
        return "landing"
    if "cruise" in text or "hover" in text or "turn" in text:
        return "cruise"
    return text if text else "unknown"


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lookup:
            return lookup[key]
    return None


def load_latest_lstm_predictions(root: Path) -> Tuple[pd.DataFrame, Path]:
    version_dir = latest_version_dir(root)
    if version_dir is None:
        raise FileNotFoundError(f"No LSTM v2 output directory found at {root}")
    artifact = version_dir / "artifacts" / "all_test_nextstep_predictions.csv"
    if not artifact.exists():
        artifact = version_dir / "artifacts" / "full_flight_nextstep_predictions.csv"
    if not artifact.exists():
        matches = sorted(root.glob("**/all_test_nextstep_predictions.csv"))
        if not matches:
            matches = sorted(root.glob("**/full_flight_nextstep_predictions.csv"))
        if not matches:
            raise FileNotFoundError("Could not find all_test_nextstep_predictions.csv or full_flight_nextstep_predictions.csv")
        artifact = matches[-1]

    df = pd.read_csv(artifact)
    required = {"phase", "t_next", "true_x", "true_y", "true_z", "pred_x", "pred_y", "pred_z"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {missing}")
    time_source = "phase_time_rel" if "phase_time_rel" in df.columns else "t_next"
    out = pd.DataFrame(
        {
            "phase": df["phase"].map(normalize_phase),
            "t_next": numeric_series(df, "t_next"),
            "phase_time_rel": numeric_series(df, time_source),
            "true_x": numeric_series(df, "true_x"),
            "true_y": numeric_series(df, "true_y"),
            "true_z": numeric_series(df, "true_z"),
            "lstm_x": numeric_series(df, "pred_x"),
            "lstm_y": numeric_series(df, "pred_y"),
            "lstm_z": numeric_series(df, "pred_z"),
        }
    )
    if "file" in df.columns:
        out["file"] = df["file"].astype(str)
    return out.replace([np.inf, -np.inf], np.nan).dropna(), artifact


def infer_lstm_timing(root: Path) -> Tuple[float, int]:
    version_dir = latest_version_dir(root)
    if version_dir is None:
        return CFG.dt_seconds, CFG.horizon_steps
    run_info = version_dir / "run_info.json"
    if run_info.exists():
        try:
            payload = json.loads(run_info.read_text(encoding="utf-8"))
            dt = float(payload.get("dt", CFG.dt_seconds))
            horizon_steps = int(payload.get("horizon_steps", CFG.horizon_steps))
            return dt, horizon_steps
        except Exception:
            pass
    return CFG.dt_seconds, CFG.horizon_steps


def load_latest_lstm_horizon_artifacts(
    root: Path,
    horizon_steps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    version_dir = latest_version_dir(root)
    if version_dir is None:
        raise FileNotFoundError(f"No LSTM v2 output directory found at {root}")

    artifacts_dir = version_dir / "artifacts"
    ytrue_path = artifacts_dir / "Ytrue_test.npy"
    ypred_path = artifacts_dir / "Ypred_test.npy"
    pte_path = artifacts_dir / "Pte_test.npy"
    fte_path = artifacts_dir / "Fte_test.npy"
    tte_path = artifacts_dir / "Tte_test.npy"
    needed = [ytrue_path, ypred_path]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing LSTM horizon artifacts: {missing}")

    y_true = np.load(ytrue_path)
    y_pred = np.load(ypred_path)
    if pte_path.exists() and fte_path.exists() and tte_path.exists():
        phase_lbl = np.load(pte_path, allow_pickle=True).astype(str)
        file_lbl = np.load(fte_path, allow_pickle=True).astype(str)
        t_start = np.load(tte_path, allow_pickle=True).astype(float)
    else:
        phase_lbl, file_lbl, t_start = reconstruct_lstm_test_metadata(version_dir, len(y_true))

    if horizon_steps is None or horizon_steps > y_true.shape[1]:
        horizon_steps = int(y_true.shape[1])

    return (
        y_true[:, :horizon_steps, :],
        y_pred[:, :horizon_steps, :],
        phase_lbl,
        file_lbl,
        t_start,
        version_dir,
    )


def reconstruct_lstm_test_metadata(version_dir: Path, expected_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_info_path = version_dir / "run_info.json"
    if not run_info_path.exists():
        raise FileNotFoundError(
            f"Missing Pte/Fte/Tte artifacts and no run_info.json found to reconstruct metadata in {version_dir}"
        )

    import lstm_unified_nohyp as lstm_source

    payload = json.loads(run_info_path.read_text(encoding="utf-8"))
    saved_cfg = payload.get("config", {})
    for key, value in saved_cfg.items():
        if hasattr(lstm_source.CFG, key):
            setattr(lstm_source.CFG, key, value)

    data_dir_text = payload.get("data_dir")
    data_dir = Path(data_dir_text) if data_dir_text else Path(lstm_source.CFG.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent / data_dir
    if not data_dir.is_dir():
        raise RuntimeError(f"Cannot reconstruct LSTM metadata; data directory not found: {data_dir}")

    all_dfs = lstm_source.load_all_csvs(str(data_dir))
    big_df, feature_name_map = lstm_source.build_unified_dataframe(all_dfs, lstm_source.CFG)
    x, y, phase_lbl, file_lbl, tnext, _dt, _feature_cols = lstm_source.create_sequences_all(
        big_df, feature_name_map, lstm_source.CFG
    )
    splits = lstm_source.split_by_file(x, y, phase_lbl, file_lbl, tnext, lstm_source.CFG)
    _xte, _yte, pte, fte, tte = splits["test"]
    if len(pte) != expected_n:
        raise ValueError(
            f"Reconstructed {len(pte)} test labels, but LSTM horizon artifact has {expected_n} anchors."
        )

    artifacts_dir = version_dir / "artifacts"
    np.save(artifacts_dir / "Pte_test.npy", pte)
    np.save(artifacts_dir / "Fte_test.npy", fte)
    np.save(artifacts_dir / "Tte_test.npy", tte)
    print(f"[INFO] Reconstructed and saved missing LSTM test metadata in {artifacts_dir}")
    return pte.astype(str), fte.astype(str), tte.astype(float)


def standardize_physics_df(df: pd.DataFrame, model_name: str, phase: str) -> pd.DataFrame:
    time_col = find_col(df, ("time", "Time", "t", "time_rel"))
    x_col = find_col(df, ("x", "pred_x", "mean_x"))
    y_col = find_col(df, ("y", "pred_y", "mean_y"))
    z_col = find_col(df, ("z", "pred_z", "mean_z"))
    if time_col is None or x_col is None or y_col is None or z_col is None:
        raise ValueError(f"{model_name}/{phase} has no usable time/x/y/z columns")

    phase_col = find_col(df, ("phase", "Phase", "flight_phase"))
    phase_values = df[phase_col].map(normalize_phase) if phase_col else phase
    out = pd.DataFrame(
        {
            "physics_model": model_name,
            "phase": phase_values,
            "t_physics": numeric_series(df, time_col),
            "phy_x": numeric_series(df, x_col),
            "phy_y": numeric_series(df, y_col),
            "phy_z": numeric_series(df, z_col),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def load_phase_file(path: Path, model_name: str, phase: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return standardize_physics_df(pd.read_csv(path), model_name, phase)
    except Exception as exc:
        print(f"[WARN] Skipping {path}: {exc}")
        return None


def collect_v1() -> Optional[pd.DataFrame]:
    root = Path("physics_model_v1_rigidbody_outputs")
    base = latest_version_dir(root)
    if base is None:
        return None
    frames = []
    for phase in ("takeoff", "cruise", "landing"):
        frame = load_phase_file(base / f"{phase}_predicted_trajectory.csv", "physics_v1_rigidbody", phase)
        if frame is not None:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def collect_v2() -> Optional[pd.DataFrame]:
    root = Path("physics_model_v2_4rotor_outputs")
    base = latest_version_dir(root)
    if base is None:
        return None
    frames = []
    for phase in ("takeoff", "cruise", "landing"):
        frame = load_phase_file(base / f"{phase}_predicted_trajectory_v2.csv", "physics_v2_4rotor", phase)
        if frame is not None:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def collect_v3() -> Optional[pd.DataFrame]:
    root = Path("physics_model_v3_previous_outputs")
    bases = sorted([p for p in root.iterdir() if p.is_dir()], key=version_number, reverse=True) if root.exists() else []
    for base in bases:
        candidate = base / "predicted_mean_trajectory.csv"
        if candidate.exists():
            return load_phase_file(candidate, "physics_v3_previous", "takeoff")
    return None


def collect_v4() -> Optional[pd.DataFrame]:
    root = Path("physics_model_v4_simple_mc_outputs")
    base = latest_version_dir(root)
    if base is None:
        return None
    frames = []
    for phase in ("takeoff", "cruise", "landing"):
        candidate = base / phase / f"{phase}_predicted_mean_trajectory.csv"
        frame = load_phase_file(candidate, "physics_v4_simple_mc", phase)
        if frame is not None:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def collect_v6() -> Optional[pd.DataFrame]:
    root = Path("physics_model_v6_realistic_outputs")
    base = latest_version_dir(root)
    if base is None:
        return None
    frames = []
    for phase in ("takeoff", "cruise", "landing"):
        frame = load_phase_file(base / f"{phase}_predicted_mean_trajectory_v6.csv", "physics_v6_realistic", phase)
        if frame is not None:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def collect_physics_model_outputs(root: Path, version: Optional[int]) -> Optional[pd.DataFrame]:
    base = resolve_physics_output_dir(root, version)
    if base is None:
        return None
    selected_version = version_number(base)
    model_name = f"physics_model_outputs_v{selected_version}" if selected_version >= 0 else "physics_model_outputs"
    frames = []
    for phase in ("takeoff", "cruise", "landing"):
        candidates = [
            base / f"{phase}_predicted_mean_trajectory_v6.csv",
            base / f"{phase}_predicted_mean_trajectory.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not path.exists():
            matches = []
            for pattern in (
                f"**/{phase}_predicted_mean_trajectory_v6.csv",
                f"**/{phase}_predicted_mean_trajectory.csv",
            ):
                matches.extend(sorted(base.glob(pattern)))
            path = matches[-1] if matches else path
        frame = load_phase_file(path, model_name, phase)
        if frame is not None:
            frame["physics_source_dir"] = str(base)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def collect_physics_predictions(
    root: Path = Path(CFG.physics_output_root),
    version: Optional[int] = CFG.physics_output_version,
) -> Dict[str, pd.DataFrame]:
    outputs = {}
    frame = collect_physics_model_outputs(root, version)
    if frame is not None and len(frame) > 0:
        outputs[str(frame["physics_model"].iloc[0])] = frame
    return outputs


def align_lstm_and_physics(lstm: pd.DataFrame, physics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    frames = []
    lstm_time_col = "phase_time_rel" if "phase_time_rel" in lstm.columns else "t_next"
    for phase in sorted(set(lstm["phase"]).intersection(set(physics["phase"]))):
        lstm_phase = lstm[lstm["phase"] == phase].sort_values(lstm_time_col).copy()
        physics_phase = physics[physics["phase"] == phase].sort_values("t_physics").copy()
        if lstm_phase.empty or physics_phase.empty:
            continue
        frames.append(
            pd.merge_asof(
                lstm_phase,
                physics_phase,
                left_on=lstm_time_col,
                right_on="t_physics",
                direction="nearest",
                tolerance=cfg.max_time_gap_seconds,
                suffixes=("", "_physics"),
            )
        )
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["phy_x", "phy_y", "phy_z"])
    merged["time_gap_seconds"] = (merged[lstm_time_col] - merged["t_physics"]).abs()
    cols = ["phy_x", "phy_y", "phy_z", "lstm_x", "lstm_y", "lstm_z", "true_x", "true_y", "true_z"]
    return merged.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).sort_values(["phase", "t_next"]).reset_index(drop=True)


def build_phase_physics_horizon(physics_df: pd.DataFrame, phase: str, t_grid: np.ndarray) -> Optional[np.ndarray]:
    phase_df = physics_df[physics_df["phase"] == phase].sort_values("t_physics").copy()
    if phase_df.empty:
        return None
    phase_df = phase_df.drop_duplicates(subset=["t_physics"], keep="last")
    t_phy = phase_df["t_physics"].to_numpy(dtype=float)
    if t_phy.size == 0:
        return None
    out = np.zeros((len(t_grid), 3), dtype=np.float32)
    for idx, col in enumerate(("phy_x", "phy_y", "phy_z")):
        vals = phase_df[col].to_numpy(dtype=float)
        if len(t_phy) == 1:
            out[:, idx] = vals[0]
        else:
            out[:, idx] = np.interp(t_grid, t_phy, vals, left=vals[0], right=vals[-1])
    return out


def condition_physics_horizon_to_sample(physics_horizon: np.ndarray, lstm_horizon: np.ndarray, phase: str) -> np.ndarray:
    conditioned = physics_horizon.astype(np.float32).copy()

    # Convert the phase template into a sample-conditioned prior by matching the
    # initial predicted state. This removes the large absolute-coordinate bias
    # that is especially damaging for landing altitude.
    conditioned[:, 0] = conditioned[:, 0] - conditioned[0, 0] + lstm_horizon[0, 0]
    conditioned[:, 1] = conditioned[:, 1] - conditioned[0, 1] + lstm_horizon[0, 1]
    conditioned[:, 2] = conditioned[:, 2] - conditioned[0, 2] + lstm_horizon[0, 2]

    if phase == "landing":
        phy_drop = float(conditioned[0, 2] - conditioned[-1, 2])
        lstm_drop = float(lstm_horizon[0, 2] - lstm_horizon[-1, 2])
        if abs(phy_drop) > 1e-6:
            scale = np.clip(lstm_drop / phy_drop, 0.35, 1.65)
            conditioned[:, 2] = conditioned[0, 2] - (conditioned[0, 2] - conditioned[:, 2]) * scale

    return conditioned


def predict_horizon_with_saved_mlp(
    physics_df: pd.DataFrame,
    lstm_horizon_pred: np.ndarray,
    phase_lbl: np.ndarray,
    cfg: Config,
    physics_model_name: str,
) -> Optional[np.ndarray]:
    mlp_root = Path(cfg.mlp_output_root)
    version_dir = latest_version_dir(mlp_root)
    if version_dir is None:
        print(f"[WARN] MLP output root not found; skipping MLP fusion horizon curve: {mlp_root}")
        return None

    model_dir = version_dir / physics_model_name
    checkpoint_path = model_dir / "models" / "fusion_mlp.pt"
    scalers_path = model_dir / "models" / "scalers.pkl"
    if not checkpoint_path.exists() or not scalers_path.exists():
        matches = sorted(version_dir.glob("**/models/fusion_mlp.pt"))
        if matches:
            checkpoint_path = matches[-1]
            scalers_path = checkpoint_path.parent / "scalers.pkl"
        if not checkpoint_path.exists() or not scalers_path.exists():
            print(f"[WARN] No saved MLP fusion model found in {version_dir}; skipping MLP fusion horizon curve")
            return None

    try:
        import MLP as mlp_source

        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        with open(scalers_path, "rb") as f:
            scaler_payload = pickle.load(f)
        x_scaler = scaler_payload["x_scaler"]
        y_scaler = scaler_payload["y_scaler"]
        mlp_cfg = checkpoint.get("config", {})
        model = mlp_source.FusionMLP(
            input_dim=len(checkpoint.get("input_features", mlp_source.INPUT_FEATURES)),
            hidden_layers=tuple(mlp_cfg.get("hidden_layers", mlp_source.CFG.hidden_layers)),
            dropout=float(mlp_cfg.get("dropout", mlp_source.CFG.dropout)),
        ).to(cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        residual_alpha = float(checkpoint.get("residual_alpha", 1.0))
        runtime_cfg = mlp_source.Config(device=cfg.device)
        return mlp_source.predict_horizon_with_mlp(
            model,
            x_scaler,
            y_scaler,
            physics_df,
            lstm_horizon_pred,
            phase_lbl,
            cfg.dt_seconds,
            runtime_cfg,
            residual_alpha,
        )
    except Exception as exc:
        print(f"[WARN] Could not compute MLP fusion horizon curve from {checkpoint_path}: {exc}")
        return None


def build_multistep_horizon_dataset(
    physics_df: pd.DataFrame,
    y_true: np.ndarray,
    y_lstm: np.ndarray,
    phase_lbl: np.ndarray,
    file_lbl: np.ndarray,
    t_start: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    t_grid = np.arange(cfg.horizon_steps, dtype=float) * cfg.dt_seconds
    phase_horizons: Dict[str, np.ndarray] = {}
    for phase in sorted({normalize_phase(p) for p in phase_lbl.tolist()}):
        horizon = build_phase_physics_horizon(physics_df, phase, t_grid)
        if horizon is not None:
            phase_horizons[phase] = horizon

    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    phase_ids: List[int] = []
    meta_rows: List[Dict[str, object]] = []

    for i in range(len(y_true)):
        phase = normalize_phase(phase_lbl[i])
        phy = phase_horizons.get(phase)
        if phy is None:
            continue
        lstm_block = y_lstm[i]
        true_block = y_true[i]
        conditioned_phy = condition_physics_horizon_to_sample(phy, lstm_block, phase)
        feature_block = np.concatenate([conditioned_phy, lstm_block], axis=1).reshape(-1)
        x_blocks.append(feature_block.astype(np.float32))
        y_blocks.append(true_block.astype(np.float32))
        phase_ids.append(PHASE_TO_ID.get(phase, 1))
        meta_rows.append(
            {
                "phase": phase,
                "anchor_index": int(i),
                "file": str(file_lbl[i]),
                "t_start": float(t_start[i]),
                "t_end": float(t_start[i] + (cfg.horizon_steps - 1) * cfg.dt_seconds),
            }
        )

    if not x_blocks:
        return (
            np.empty((0, cfg.horizon_steps * 6), dtype=np.float32),
            np.empty((0, cfg.horizon_steps, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DataFrame(),
        )

    return (
        np.stack(x_blocks).astype(np.float32),
        np.stack(y_blocks).astype(np.float32),
        np.asarray(phase_ids, dtype=np.int64),
        pd.DataFrame(meta_rows),
    )


def build_multistep_anchors(aligned: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    phase_ids: List[int] = []
    meta_rows: List[Dict[str, object]] = []

    for phase, phase_df in aligned.groupby("phase", sort=False):
        phase_df = phase_df.sort_values("t_next").reset_index(drop=True)
        if len(phase_df) < cfg.horizon_steps:
            continue
        features = phase_df[["phy_x", "phy_y", "phy_z", "lstm_x", "lstm_y", "lstm_z"]].to_numpy(dtype=np.float32)
        targets = phase_df[["true_x", "true_y", "true_z"]].to_numpy(dtype=np.float32)
        times = phase_df["t_next"].to_numpy(dtype=float)
        for start in range(0, len(phase_df) - cfg.horizon_steps + 1):
            end = start + cfg.horizon_steps
            x_blocks.append(features[start:end].reshape(-1))
            y_blocks.append(targets[start:end])
            phase_ids.append(PHASE_TO_ID.get(phase, 1))
            meta_rows.append({"phase": phase, "anchor_index": int(start), "t_start": float(times[start]), "t_end": float(times[end - 1])})

    if not x_blocks:
        return (
            np.empty((0, cfg.horizon_steps * 6), dtype=np.float32),
            np.empty((0, cfg.horizon_steps, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DataFrame(),
        )
    return np.stack(x_blocks).astype(np.float32), np.stack(y_blocks).astype(np.float32), np.asarray(phase_ids, dtype=np.int64), pd.DataFrame(meta_rows)


def split_temporal(n: int, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    n_train = max(1, int(n * cfg.train_ratio))
    n_val = max(1, int(n * cfg.val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def phase_speed_bounds(phase_ids: torch.Tensor, cfg: Config) -> torch.Tensor:
    bounds = torch.full_like(phase_ids.float(), cfg.vmax_cruise)
    bounds = torch.where(phase_ids == PHASE_TO_ID["takeoff"], torch.tensor(cfg.vmax_takeoff, device=phase_ids.device), bounds)
    bounds = torch.where(phase_ids == PHASE_TO_ID["landing"], torch.tensor(cfg.vmax_landing, device=phase_ids.device), bounds)
    return bounds


class PhysicsInformedLoss(nn.Module):
    def __init__(
        self,
        cfg: Config,
        x_scaler: Optional[StandardScaler] = None,
        y_scaler: Optional[StandardScaler] = None,
    ):
        super().__init__()
        self.cfg = cfg
        weights = torch.tensor([cfg.horizon_discount**i for i in range(cfg.horizon_steps)], dtype=torch.float32)
        self.register_buffer("horizon_weights", weights.view(1, cfg.horizon_steps, 1))
        if x_scaler is not None and y_scaler is not None:
            self.register_buffer("x_mean", torch.as_tensor(x_scaler.mean_, dtype=torch.float32).view(1, -1))
            self.register_buffer("x_scale", torch.as_tensor(x_scaler.scale_, dtype=torch.float32).view(1, -1))
            self.register_buffer("y_mean", torch.as_tensor(y_scaler.mean_, dtype=torch.float32).view(1, -1))
            self.register_buffer("y_scale", torch.as_tensor(y_scaler.scale_, dtype=torch.float32).view(1, -1))
        else:
            self.register_buffer("x_mean", torch.empty(0, dtype=torch.float32))
            self.register_buffer("x_scale", torch.empty(0, dtype=torch.float32))
            self.register_buffer("y_mean", torch.empty(0, dtype=torch.float32))
            self.register_buffer("y_scale", torch.empty(0, dtype=torch.float32))

    def _prediction_to_meters(self, pred: torch.Tensor) -> torch.Tensor:
        if self.y_scale.numel() == 0:
            return pred
        flat = pred.reshape(pred.shape[0], -1)
        raw = flat * self.y_scale.to(pred.device) + self.y_mean.to(pred.device)
        return raw.view(pred.shape[0], self.cfg.horizon_steps, 3)

    def _physics_to_meters(self, features: torch.Tensor) -> torch.Tensor:
        if self.x_scale.numel() == 0:
            return features.view(-1, self.cfg.horizon_steps, 6)[:, :, 0:3]
        flat = features.reshape(features.shape[0], -1)
        raw = flat * self.x_scale.to(features.device) + self.x_mean.to(features.device)
        return raw.view(features.shape[0], self.cfg.horizon_steps, 6)[:, :, 0:3]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        phase_ids: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        data = torch.mean(self.horizon_weights.to(pred.device) * ((pred - target) ** 2))
        pred_m = self._prediction_to_meters(pred)
        physics_m = self._physics_to_meters(features)
        physics_guidance = torch.mean(self.horizon_weights.to(pred.device) * ((pred_m - physics_m) ** 2))

        increments = pred_m[:, 1:, :] - pred_m[:, :-1, :]
        speeds = torch.linalg.norm(increments, dim=-1) / self.cfg.dt_seconds
        vmax = phase_speed_bounds(phase_ids, self.cfg).view(-1, 1)
        speed_penalty = torch.mean(torch.relu(speeds - vmax) ** 2) if speeds.numel() else pred.new_tensor(0.0)

        d2 = pred_m[:, 2:, :] - 2.0 * pred_m[:, 1:-1, :] + pred_m[:, :-2, :]
        acc_penalty = torch.mean(d2**2) if d2.numel() else pred.new_tensor(0.0)
        if d2.numel() and self.y_scale.numel() > 0:
            accel = d2 / max(self.cfg.dt_seconds**2, 1e-12)
            xy_over = torch.relu(torch.abs(accel[:, :, :2]) - self.cfg.accel_limit_xy)
            z_over = torch.relu(torch.abs(accel[:, :, 2]) - self.cfg.accel_limit_z)
            norm_over = torch.relu(torch.linalg.norm(accel, dim=-1) - self.cfg.max_integration_accel)
            acc_bound_penalty = torch.mean(xy_over**2) + torch.mean(z_over**2) + torch.mean(norm_over**2)
        else:
            acc_bound_penalty = pred.new_tensor(0.0)

        d3 = pred_m[:, 3:, :] - 3.0 * pred_m[:, 2:-1, :] + 3.0 * pred_m[:, 1:-2, :] - pred_m[:, :-3, :]
        jerk_penalty = torch.mean(d3**2) if d3.numel() else pred.new_tensor(0.0)

        dz_rate = increments[:, :, 2] / self.cfg.dt_seconds
        takeoff_mask = phase_ids == PHASE_TO_ID["takeoff"]
        landing_mask = phase_ids == PHASE_TO_ID["landing"]
        vertical_terms = []
        if self.y_scale.numel() > 0 and dz_rate.numel() and bool(torch.any(takeoff_mask)):
            takeoff_rate = dz_rate[takeoff_mask]
            vertical_terms.append(torch.relu(self.cfg.takeoff_vz_min - takeoff_rate) ** 2)
            vertical_terms.append(torch.relu(takeoff_rate - self.cfg.takeoff_vz_max) ** 2)
        if self.y_scale.numel() > 0 and dz_rate.numel() and bool(torch.any(landing_mask)):
            landing_rate = dz_rate[landing_mask]
            vertical_terms.append(torch.relu(self.cfg.landing_vz_min - landing_rate) ** 2)
            vertical_terms.append(torch.relu(landing_rate - self.cfg.landing_vz_max) ** 2)
        if vertical_terms:
            vertical_penalty = torch.mean(torch.cat([term.reshape(-1) for term in vertical_terms]))
        elif bool(torch.any(landing_mask)) and dz_rate.numel():
            vertical_penalty = torch.mean(torch.relu(torch.abs(dz_rate[landing_mask]) - self.cfg.landing_descent_rate_max) ** 2)
        else:
            vertical_penalty = pred.new_tensor(0.0)

        total = (
            data
            + self.cfg.lambda_phy * physics_guidance
            + self.cfg.lambda_spd * speed_penalty
            + self.cfg.lambda_acc * acc_penalty
            + self.cfg.lambda_acc_bound * acc_bound_penalty
            + self.cfg.lambda_jrk * jerk_penalty
            + self.cfg.lambda_vz * vertical_penalty
        )
        parts = {
            "data": float(data.detach().cpu()),
            "physics_guidance": float(physics_guidance.detach().cpu()),
            "speed": float(speed_penalty.detach().cpu()),
            "acceleration": float(acc_penalty.detach().cpu()),
            "acceleration_bounds": float(acc_bound_penalty.detach().cpu()),
            "jerk": float(jerk_penalty.detach().cpu()),
            "vertical_rate": float(vertical_penalty.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total, parts


def average_parts(parts: List[Dict[str, float]]) -> Dict[str, float]:
    if not parts:
        return {
            "data": 0.0,
            "physics_guidance": 0.0,
            "speed": 0.0,
            "acceleration": 0.0,
            "acceleration_bounds": 0.0,
            "jerk": 0.0,
            "vertical_rate": 0.0,
            "total": 0.0,
        }
    return {key: float(np.mean([p[key] for p in parts])) for key in parts[0].keys()}


def scale_blocks(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit(x[train_idx]).transform(x).astype(np.float32)
    y_flat = y.reshape(len(y), -1)
    y_scaled = y_scaler.fit(y_flat[train_idx]).transform(y_flat).astype(np.float32).reshape(y.shape)
    return x_scaled, y_scaled, x_scaler, y_scaler


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    phase_ids: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    shuffle: bool,
    phase_balanced: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x[idx]).float(),
        torch.from_numpy(y[idx]).float(),
        torch.from_numpy(phase_ids[idx]).long(),
    )
    if phase_balanced and shuffle:
        counts = np.bincount(phase_ids[idx], minlength=len(PHASE_TO_ID)).astype(float)
        counts[counts == 0.0] = 1.0
        weights = 1.0 / counts[phase_ids[idx]]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(idx),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    phase_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
) -> Tuple[PIMLDataBackboneNet, List[Dict[str, float]], List[Dict[str, float]]]:
    model = PIMLDataBackboneNet(cfg.horizon_steps).to(cfg.device)
    criterion = PhysicsInformedLoss(cfg, x_scaler=x_scaler, y_scaler=y_scaler).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = make_loader(
        x_scaled,
        y_scaled,
        phase_ids,
        train_idx,
        cfg.batch_size,
        shuffle=True,
        phase_balanced=cfg.phase_balanced_sampling,
    )
    val_loader = make_loader(x_scaled, y_scaled, phase_ids, val_idx, cfg.batch_size, shuffle=False)

    best_state = None
    best_val = math.inf
    stale = 0
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_parts = []
        for xb, yb, pb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            pb = pb.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = criterion(model(xb), yb, pb, xb)
            loss.backward()
            optimizer.step()
            train_parts.append(parts)

        model.eval()
        val_parts = []
        with torch.no_grad():
            for xb, yb, pb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                pb = pb.to(cfg.device)
                _, parts = criterion(model(xb), yb, pb, xb)
                val_parts.append(parts)

        train_epoch = average_parts(train_parts)
        val_epoch = average_parts(val_parts)
        train_epoch["epoch"] = epoch
        val_epoch["epoch"] = epoch
        train_history.append(train_epoch)
        val_history.append(val_epoch)

        if val_epoch["total"] < best_val - 1e-8:
            best_val = val_epoch["total"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_history, val_history


def predict(model: PIMLDataBackboneNet, x_scaled: np.ndarray, idx: np.ndarray, y_scaler: StandardScaler, cfg: Config) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(torch.from_numpy(x_scaled[idx]).float(), batch_size=cfg.batch_size, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            preds.append(model(xb.to(cfg.device)).detach().cpu().numpy())
    pred_scaled = np.concatenate(preds, axis=0)
    pred_flat = pred_scaled.reshape(len(idx), -1)
    return y_scaler.inverse_transform(pred_flat).reshape(len(idx), cfg.horizon_steps, 3)


def residual_blend(
    piml_pred: np.ndarray,
    x_raw: np.ndarray,
    phase_ids: np.ndarray,
    phase_alphas: Dict[str, float],
    cfg: Config,
) -> np.ndarray:
    lstm = baseline_from_input(x_raw, "lstm_v2", cfg)
    out = lstm.copy()
    for phase_id, phase in ID_TO_PHASE.items():
        mask = phase_ids == phase_id
        if not np.any(mask):
            continue
        alpha = float(phase_alphas.get(phase, phase_alphas.get("default", 1.0)))
        out[mask] = lstm[mask] + alpha * (piml_pred[mask] - lstm[mask])
    return out


def calibrate_phase_residual_alphas(
    y_true: np.ndarray,
    piml_pred: np.ndarray,
    x_raw: np.ndarray,
    phase_ids: np.ndarray,
    cfg: Config,
) -> Dict[str, float]:
    if not cfg.calibrate_phase_residual:
        return {"default": 1.0}

    lstm = baseline_from_input(x_raw, "lstm_v2", cfg)
    alphas: Dict[str, float] = {}
    grid_steps = max(2, int(cfg.residual_alpha_grid_steps))
    grid = np.linspace(0.0, 1.0, grid_steps)
    for phase_id, phase in ID_TO_PHASE.items():
        mask = phase_ids == phase_id
        if not np.any(mask):
            continue
        best_alpha = 1.0
        best_error = math.inf
        residual = piml_pred[mask] - lstm[mask]
        for alpha in grid:
            blended = lstm[mask] + float(alpha) * residual
            error = float(np.mean(np.linalg.norm(blended - y_true[mask], axis=-1)))
            if error < best_error:
                best_error = error
                best_alpha = float(alpha)
        alphas[phase] = best_alpha

    if alphas:
        alphas["default"] = float(np.mean(list(alphas.values())))
    else:
        alphas["default"] = 1.0
    return alphas


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    dist = np.linalg.norm(err, axis=-1)
    final_dist = dist[:, -1]
    return {
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "ade": float(np.mean(dist)),
        "fde": float(np.mean(final_dist)),
        "median_error": float(np.median(dist)),
        "p90_error": float(np.percentile(dist, 90)),
        "max_error": float(np.max(dist)),
        "rmse_x": float(np.sqrt(np.mean(err[:, :, 0] ** 2))),
        "rmse_y": float(np.sqrt(np.mean(err[:, :, 1] ** 2))),
        "rmse_z": float(np.sqrt(np.mean(err[:, :, 2] ** 2))),
    }


def baseline_from_input(x_raw: np.ndarray, source: str, cfg: Config) -> np.ndarray:
    reshaped = x_raw.reshape(len(x_raw), cfg.horizon_steps, 6)
    if source == "physics":
        return reshaped[:, :, 0:3]
    if source == "lstm_v2":
        return reshaped[:, :, 3:6]
    if source == "static_average":
        return 0.5 * (reshaped[:, :, 0:3] + reshaped[:, :, 3:6])
    raise ValueError(source)


def evaluate_split(split: str, y_true: np.ndarray, y_pred: np.ndarray, x_raw: np.ndarray, cfg: Config) -> List[Dict[str, object]]:
    rows = []
    predictors = {
        "piml_fusion": y_pred,
        "lstm_v2": baseline_from_input(x_raw, "lstm_v2", cfg),
        "physics": baseline_from_input(x_raw, "physics", cfg),
        "static_average": baseline_from_input(x_raw, "static_average", cfg),
    }
    for name, values in predictors.items():
        row = {"split": split, "predictor": name, "n_anchors": int(len(y_true)), "horizon_steps": cfg.horizon_steps}
        row.update(metric_dict(y_true, values))
        rows.append(row)
    return rows


def save_loss_plot(train_history: List[Dict[str, float]], val_history: List[Dict[str, float]], outpath: Path) -> None:
    train_df = pd.DataFrame(train_history)
    val_df = pd.DataFrame(val_history)
    plt.figure(figsize=(9, 5))
    plt.plot(train_df["epoch"], train_df["total"], label="train total")
    plt.plot(val_df["epoch"], val_df["total"], label="validation total")
    plt.plot(val_df["epoch"], val_df["data"], label="validation data", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Scaled physics-informed loss")
    plt.title("PIML Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_loss_components_plot(val_history: List[Dict[str, float]], outpath: Path) -> None:
    val_df = pd.DataFrame(val_history)
    plt.figure(figsize=(10, 5))
    for col in ("data", "physics_guidance", "speed", "acceleration", "acceleration_bounds", "jerk", "vertical_rate"):
        if col in val_df.columns:
            plt.plot(val_df["epoch"], val_df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Component value")
    plt.title("Validation Loss Components")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_horizon_error_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_raw: np.ndarray,
    outpath: Path,
    cfg: Config,
    title: str,
    mlp_pred: Optional[np.ndarray] = None,
) -> None:
    time_sec = np.arange(1, cfg.horizon_steps + 1, dtype=float) * float(cfg.dt_seconds)
    predictors = {
        "PIML fusion": y_pred,
        "LSTM": baseline_from_input(x_raw, "lstm_v2", cfg),
        "Physics": baseline_from_input(x_raw, "physics", cfg),
    }
    if mlp_pred is not None:
        predictors["MLP fusion"] = mlp_pred
    plt.figure(figsize=(10, 5))
    for label, values in predictors.items():
        err = np.linalg.norm(values - y_true, axis=-1)
        plt.plot(time_sec, np.mean(err, axis=0), label=label)
    plt.xlabel("Forecast horizon time (s)")
    plt.ylabel("Mean Euclidean error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_anchor_validation_plot(y_true: np.ndarray, y_pred: np.ndarray, x_raw: np.ndarray, outpath: Path, cfg: Config, title: str) -> None:
    if len(y_true) == 0:
        return
    anchor = min(3, len(y_true) - 1)
    steps = np.arange(1, cfg.horizon_steps + 1)
    lstm = baseline_from_input(x_raw, "lstm_v2", cfg)
    phy = baseline_from_input(x_raw, "physics", cfg)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, label, idx in zip(axs, ("x", "y", "z"), range(3)):
        ax.plot(steps, y_true[anchor, :, idx], label="true", linewidth=2)
        ax.plot(steps, y_pred[anchor, :, idx], label="PIML fusion")
        ax.plot(steps, lstm[anchor, :, idx], label="LSTM v2", alpha=0.8)
        ax.plot(steps, phy[anchor, :, idx], label="physics", alpha=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axs[0].legend(ncol=4, fontsize=9)
    axs[-1].set_xlabel("Forecast horizon step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_error_histogram(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path, title: str) -> None:
    err = np.linalg.norm(y_pred - y_true, axis=-1).reshape(-1)
    plt.figure(figsize=(8, 5))
    plt.hist(err, bins=40, alpha=0.85)
    plt.xlabel("Euclidean position error")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_trajectory_overlay(y_true: np.ndarray, y_pred: np.ndarray, x_raw: np.ndarray, outpath: Path, cfg: Config, title: str) -> None:
    if len(y_true) == 0:
        return
    anchor = min(3, len(y_true) - 1)
    lstm = baseline_from_input(x_raw, "lstm_v2", cfg)
    phy = baseline_from_input(x_raw, "physics", cfg)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(y_true[anchor, :, 0], y_true[anchor, :, 1], y_true[anchor, :, 2], label="true", linewidth=2)
    ax.plot(y_pred[anchor, :, 0], y_pred[anchor, :, 1], y_pred[anchor, :, 2], label="PIML fusion")
    ax.plot(lstm[anchor, :, 0], lstm[anchor, :, 1], lstm[anchor, :, 2], label="LSTM v2", alpha=0.8)
    ax.plot(phy[anchor, :, 0], phy[anchor, :, 1], phy[anchor, :, 2], label="physics", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def flatten_predictions(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, x_raw: np.ndarray, cfg: Config) -> pd.DataFrame:
    lstm = baseline_from_input(x_raw, "lstm_v2", cfg)
    phy = baseline_from_input(x_raw, "physics", cfg)
    rows = []
    for anchor in range(len(y_true)):
        for step in range(cfg.horizon_steps):
            rows.append(
                {
                    "anchor": int(anchor),
                    "phase": meta.iloc[anchor]["phase"],
                    "t_start": float(meta.iloc[anchor]["t_start"]),
                    "horizon_step": int(step + 1),
                    "time_sec": float(step * cfg.dt_seconds),
                    "true_x": y_true[anchor, step, 0],
                    "true_y": y_true[anchor, step, 1],
                    "true_z": y_true[anchor, step, 2],
                    "piml_x": y_pred[anchor, step, 0],
                    "piml_y": y_pred[anchor, step, 1],
                    "piml_z": y_pred[anchor, step, 2],
                    "lstm_x": lstm[anchor, step, 0],
                    "lstm_y": lstm[anchor, step, 1],
                    "lstm_z": lstm[anchor, step, 2],
                    "phy_x": phy[anchor, step, 0],
                    "phy_y": phy[anchor, step, 1],
                    "phy_z": phy[anchor, step, 2],
                }
            )
    out = pd.DataFrame(rows)
    out["piml_error_euclidean"] = np.linalg.norm(
        out[["piml_x", "piml_y", "piml_z"]].to_numpy(dtype=float)
        - out[["true_x", "true_y", "true_z"]].to_numpy(dtype=float),
        axis=1,
    )
    return out


def save_phase_metrics(flat_pred: pd.DataFrame, outpath: Path) -> None:
    rows = []
    for phase, phase_df in flat_pred.groupby("phase", sort=False):
        y_true = phase_df[["true_x", "true_y", "true_z"]].to_numpy(dtype=float).reshape(-1, 1, 3)
        for name, cols in {
            "piml_fusion": ["piml_x", "piml_y", "piml_z"],
            "lstm_v2": ["lstm_x", "lstm_y", "lstm_z"],
            "physics": ["phy_x", "phy_y", "phy_z"],
        }.items():
            y_pred = phase_df[cols].to_numpy(dtype=float).reshape(-1, 1, 3)
            row = {"phase": phase, "predictor": name, "n_points": int(len(phase_df))}
            row.update(metric_dict(y_true, y_pred))
            rows.append(row)
    pd.DataFrame(rows).to_csv(outpath, index=False)


def save_phase_rmse_bar(metrics_by_phase: pd.DataFrame, outpath: Path) -> None:
    valid = metrics_by_phase[metrics_by_phase["predictor"] == "piml_fusion"].copy()
    if valid.empty:
        return
    order = ["takeoff", "cruise", "landing"]
    valid["phase"] = pd.Categorical(valid["phase"], categories=order, ordered=True)
    valid = valid.sort_values("phase")
    plt.figure(figsize=(8, 5))
    plt.bar(valid["phase"].astype(str), valid["rmse"], color=["#2f7f6f", "#386cb0", "#d95f02"][: len(valid)])
    plt.ylabel("RMSE")
    plt.title("Phase Comparison: PIML Fusion")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def representative_anchor_by_phase(flat_pred: pd.DataFrame, phase: str, axis_name: Optional[str] = None) -> Optional[int]:
    phase_df = flat_pred[flat_pred["phase"] == phase].copy()
    if phase_df.empty:
        return None
    if axis_name in {"x", "y", "z"}:
        err_col = f"{axis_name}_axis_abs_err"
        phase_df[err_col] = np.abs(phase_df[f"piml_{axis_name}"] - phase_df[f"true_{axis_name}"])
        anchor_err = (
            phase_df.groupby("anchor", sort=False)[err_col]
            .mean()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
    else:
        anchor_err = (
            phase_df.groupby("anchor", sort=False)["piml_error_euclidean"]
            .mean()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
    if anchor_err.empty:
        return None
    target = float(anchor_err.median())
    return int((anchor_err - target).abs().idxmin())


def save_phase_grid_real_vs_predicted(flat_pred: pd.DataFrame, outpath: Path) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    phases = ["takeoff", "cruise", "landing"]
    horizon_limit = float(flat_pred["time_sec"].max()) if not flat_pred.empty else 0.0
    for i, phase in enumerate(phases):
        phase_df = flat_pred[flat_pred["phase"] == phase].copy()
        for j, axis_name in enumerate(["x", "y", "z"]):
            ax = axs[i, j]
            if phase_df.empty:
                ax.set_title(phase.capitalize())
                ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0.0, horizon_limit)
                ax.set_ylabel(f"{axis_name.upper()} (m)")
                ax.grid(True, alpha=0.25)
                continue
            for _, anchor_df in phase_df.groupby("anchor", sort=False):
                anchor_df = anchor_df.sort_values("time_sec")
                ax.plot(anchor_df["time_sec"], anchor_df[f"true_{axis_name}"], "-", alpha=0.08, color="#1f77b4", linewidth=0.9)
                ax.plot(anchor_df["time_sec"], anchor_df[f"piml_{axis_name}"], ":", alpha=0.08, color="#d62728", linewidth=0.9)
            mean_df = phase_df.groupby("time_sec", sort=True).mean(numeric_only=True).reset_index()
            ax.plot(mean_df["time_sec"], mean_df[f"true_{axis_name}"], "-", color="#1f77b4", linewidth=2.0, label="Real")
            ax.plot(mean_df["time_sec"], mean_df[f"piml_{axis_name}"], ":", color="#d62728", linewidth=2.0, label="Predicted")
            ax.set_title(f"{phase.capitalize()} (n={phase_df['anchor'].nunique()})")
            ax.set_ylabel(f"{axis_name.upper()} (m)")
            ax.set_xlim(0.0, horizon_limit)
            ax.grid(True, alpha=0.25)
            if i == 0 and j == 0:
                ax.legend()
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def save_phase_grid_representative(flat_pred: pd.DataFrame, outpath: Path) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    phases = ["takeoff", "cruise", "landing"]
    horizon_limit = float(flat_pred["time_sec"].max()) if not flat_pred.empty else 0.0
    for i, phase in enumerate(phases):
        for j, axis_name in enumerate(["x", "y", "z"]):
            rep_anchor = representative_anchor_by_phase(flat_pred, phase, axis_name)
            ax = axs[i, j]
            if rep_anchor is None:
                ax.set_title(phase.capitalize())
                ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0.0, horizon_limit)
                ax.set_ylabel(f"{axis_name.upper()} (m)")
                ax.grid(True, alpha=0.25)
                continue
            anchor_df = flat_pred[flat_pred["anchor"] == rep_anchor].sort_values("time_sec")
            ax.plot(anchor_df["time_sec"], anchor_df[f"true_{axis_name}"], "-", color="#1f77b4", linewidth=2.0, label="Real")
            ax.plot(anchor_df["time_sec"], anchor_df[f"piml_{axis_name}"], ":", color="#d62728", linewidth=2.0, label="Predicted")
            ax.set_title(f"{phase.capitalize()} ({axis_name.upper()})")
            ax.set_ylabel(f"{axis_name.upper()} (m)")
            ax.set_xlim(0.0, horizon_limit)
            ax.grid(True, alpha=0.25)
            if i == 0 and j == 0:
                ax.legend()
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def save_phase_grid_position_error(flat_pred: pd.DataFrame, outpath: Path) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    phases = ["takeoff", "cruise", "landing"]
    horizon_limit = float(flat_pred["time_sec"].max()) if not flat_pred.empty else 0.0
    for i, phase in enumerate(phases):
        phase_df = flat_pred[flat_pred["phase"] == phase].copy()
        for j, axis_name in enumerate(["x", "y", "z"]):
            ax = axs[i, j]
            if phase_df.empty:
                ax.set_title(phase.capitalize())
                ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0.0, horizon_limit)
                ax.set_ylabel(f"{axis_name.upper()} Error (m)")
                ax.grid(True, alpha=0.25)
                continue
            phase_df[f"{axis_name}_err"] = phase_df[f"piml_{axis_name}"] - phase_df[f"true_{axis_name}"]
            for _, anchor_df in phase_df.groupby("anchor", sort=False):
                anchor_df = anchor_df.sort_values("time_sec")
                ax.plot(anchor_df["time_sec"], anchor_df[f"{axis_name}_err"], "-", alpha=0.08, color="#9467bd", linewidth=0.9)
            mean_df = phase_df.groupby("time_sec", sort=True).mean(numeric_only=True).reset_index()
            ax.plot(mean_df["time_sec"], mean_df[f"{axis_name}_err"], "-", color="#4c2c92", linewidth=2.0)
            ax.set_title(f"{phase.capitalize()} (n={phase_df['anchor'].nunique()})")
            ax.set_ylabel(f"{axis_name.upper()} Error (m)")
            ax.set_xlim(0.0, horizon_limit)
            ax.grid(True, alpha=0.25)
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Position Error Over Time")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def save_phase_grid_euclidean_error(flat_pred: pd.DataFrame, outpath: Path) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    phases = ["takeoff", "cruise", "landing"]
    horizon_limit = float(flat_pred["time_sec"].max()) if not flat_pred.empty else 0.0
    for i, phase in enumerate(phases):
        ax = axs[i]
        phase_df = flat_pred[flat_pred["phase"] == phase].copy()
        if phase_df.empty:
            ax.set_title(phase.capitalize())
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0.0, horizon_limit)
            ax.set_ylabel("3D Error (m)")
            ax.grid(True, alpha=0.25)
            continue
        for _, anchor_df in phase_df.groupby("anchor", sort=False):
            anchor_df = anchor_df.sort_values("time_sec")
            ax.plot(anchor_df["time_sec"], anchor_df["piml_error_euclidean"], "-", alpha=0.08, color="#2ca02c", linewidth=0.9)
        mean_df = phase_df.groupby("time_sec", sort=True)["piml_error_euclidean"].mean().reset_index()
        ax.plot(mean_df["time_sec"], mean_df["piml_error_euclidean"], "-", color="#1b7f1b", linewidth=2.0)
        ax.set_title(f"{phase.capitalize()} (n={phase_df['anchor'].nunique()})")
        ax.set_ylabel("3D Error (m)")
        ax.set_xlim(0.0, horizon_limit)
        ax.grid(True, alpha=0.25)
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Euclidean Position Error Over Time")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def load_lstm_predictions_for_data_dir(
    lstm_version_dir: Path,
    data_dir: Path,
    horizon_steps: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Validation data directory not found: {data_dir}")

    import lstm_unified_nohyp as lstm_source

    run_info_path = lstm_version_dir / "run_info.json"
    if not run_info_path.exists():
        raise FileNotFoundError(f"Missing LSTM run_info.json: {run_info_path}")
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    saved_cfg = run_info.get("config", {})
    for key, value in saved_cfg.items():
        if hasattr(lstm_source.CFG, key):
            setattr(lstm_source.CFG, key, value)

    all_dfs = lstm_source.load_all_csvs(str(data_dir))
    big_df, feature_name_map = lstm_source.build_unified_dataframe(all_dfs, lstm_source.CFG)
    x_raw, y_true, phase_lbl, file_lbl, t_start, _dt, feature_cols = lstm_source.create_sequences_all(
        big_df, feature_name_map, lstm_source.CFG
    )
    if len(x_raw) == 0:
        raise RuntimeError(f"No LSTM validation sequences generated from {data_dir}")

    artifacts_dir = lstm_version_dir / "artifacts"
    models_dir = lstm_version_dir / "models"
    with open(artifacts_dir / "scalers.pkl", "rb") as f:
        scaler_payload = pickle.load(f)
    x_scaler = scaler_payload["x_scaler"]
    y_scaler = scaler_payload["y_scaler"]
    trained_features = list(scaler_payload.get("feature_cols", feature_cols))
    if list(feature_cols) != trained_features:
        raise ValueError(
            "Validation feature columns do not match the saved LSTM model. "
            f"Expected {trained_features}, got {feature_cols}."
        )

    ckpt_path = models_dir / "best_all.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("cfg", saved_cfg)
    hidden_size = int(ckpt_cfg.get("hidden_size", lstm_source.CFG.hidden_size))
    num_layers = int(ckpt_cfg.get("num_layers", lstm_source.CFG.num_layers))
    dropout = float(ckpt_cfg.get("dropout", lstm_source.CFG.dropout))

    x_scaled = x_scaler.transform(x_raw.reshape(-1, x_raw.shape[-1])).reshape(x_raw.shape).astype(np.float32)
    full_horizon = int(y_true.shape[1])
    model = lstm_source.LSTMSeq2Seq(
        input_size=x_raw.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=full_horizon,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x_scaled).float()), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).detach().cpu().numpy())
    y_pred_scaled = np.concatenate(preds, axis=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 3)).reshape(y_pred_scaled.shape)
    h = min(int(horizon_steps), y_true.shape[1], y_pred.shape[1])
    return y_true[:, :h, :], y_pred[:, :h, :], phase_lbl.astype(str), file_lbl.astype(str), t_start.astype(float), data_dir


def save_external_validation_outputs(
    name: str,
    model: PIMLDataBackboneNet,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    phase_alphas: Dict[str, float],
    physics_df: pd.DataFrame,
    model_dir: Path,
    cfg: Config,
    y_true_ext: np.ndarray,
    y_lstm_ext: np.ndarray,
    phase_ext: np.ndarray,
    file_ext: np.ndarray,
    t_start_ext: np.ndarray,
) -> Optional[Dict[str, object]]:
    x_ext, y_ext, _phase_ids_ext, meta_ext = build_multistep_horizon_dataset(
        physics_df, y_true_ext, y_lstm_ext, phase_ext, file_ext, t_start_ext, cfg
    )
    if len(x_ext) == 0:
        print(f"[WARN] {name}: no external validation anchors after physics/LSTM alignment")
        return None

    root = model_dir / name
    plots_dir = root / "plots"
    artifacts_dir = root / "artifacts"
    metrics_dir = root / "metrics"
    for path in (plots_dir, artifacts_dir, metrics_dir):
        ensure_dir(path)

    x_ext_scaled = x_scaler.transform(x_ext).astype(np.float32)
    idx = np.arange(len(x_ext))
    pred_ext_raw = predict(model, x_ext_scaled, idx, y_scaler, cfg)
    external_phase_alphas = phase_alphas
    if cfg.calibrate_external_residual:
        external_phase_alphas = calibrate_phase_residual_alphas(
            y_ext,
            pred_ext_raw,
            x_ext,
            _phase_ids_ext,
            cfg,
        )
    pred_ext = residual_blend(pred_ext_raw, x_ext, _phase_ids_ext, external_phase_alphas, cfg)
    flat_ext = flatten_predictions(meta_ext, y_ext, pred_ext, x_ext, cfg)
    meta_ext.to_csv(artifacts_dir / "multistep_anchor_metadata.csv", index=False)
    flat_ext.to_csv(artifacts_dir / "piml_multistep_predictions.csv", index=False)

    metrics_rows = evaluate_split(name, y_ext, pred_ext, x_ext, cfg)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_dir / "metrics_summary.csv", index=False)
    with open(metrics_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)
    save_phase_metrics(flat_ext, metrics_dir / "metrics_by_phase.csv")
    phase_metrics_df = pd.read_csv(metrics_dir / "metrics_by_phase.csv")

    mlp_ext = predict_horizon_with_saved_mlp(
        physics_df,
        baseline_from_input(x_ext, "lstm_v2", cfg),
        meta_ext["phase"].to_numpy(dtype=str),
        cfg,
        name,
    )
    save_horizon_error_plot(y_ext, pred_ext, x_ext, plots_dir / "horizon_error.png", cfg, f"{name}: Horizon Error", mlp_ext)
    save_anchor_validation_plot(y_ext, pred_ext, x_ext, plots_dir / "anchor_validation_axes.png", cfg, f"{name}: Anchor Validation")
    save_error_histogram(y_ext, pred_ext, plots_dir / "error_histogram.png", f"{name}: PIML Error Distribution")
    save_trajectory_overlay(y_ext, pred_ext, x_ext, plots_dir / "3d_trajectory_overlay.png", cfg, f"{name}: 3D Multi-Step Forecast")
    save_phase_rmse_bar(phase_metrics_df, plots_dir / "phase_comparison_bar_chart.png")
    save_phase_grid_real_vs_predicted(flat_ext, plots_dir / "real_vs_predicted_position_over_time_all_phase.png")
    save_phase_grid_representative(flat_ext, plots_dir / "real_vs_predicted_position_over_time_representative_phase.png")
    save_phase_grid_position_error(flat_ext, plots_dir / "position_error_over_time_all_phase.png")
    save_phase_grid_euclidean_error(flat_ext, plots_dir / "euclidean_error_over_time_all_phase.png")

    piml_row = metrics_df[metrics_df["predictor"] == "piml_fusion"].iloc[0].to_dict()
    return {
        f"{name}_anchors": int(len(x_ext)),
        f"{name}_piml_ade": float(piml_row["ade"]),
        f"{name}_piml_fde": float(piml_row["fde"]),
        f"{name}_piml_rmse": float(piml_row["rmse"]),
        f"{name}_phase_residual_alphas": json.dumps(external_phase_alphas, sort_keys=True),
    }


def process_physics_model(
    model_name: str,
    physics_df: pd.DataFrame,
    lstm_df: pd.DataFrame,
    out_root: Path,
    cfg: Config,
    horizon_y_true: np.ndarray,
    horizon_y_lstm: np.ndarray,
    horizon_phase_lbl: np.ndarray,
    horizon_file_lbl: np.ndarray,
    horizon_t_start: np.ndarray,
    validation_payload: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]] = None,
) -> Optional[Dict[str, object]]:
    physics_source_dir = (
        str(physics_df["physics_source_dir"].iloc[0])
        if "physics_source_dir" in physics_df.columns and len(physics_df) > 0
        else str(Path(cfg.physics_output_root))
    )
    aligned = align_lstm_and_physics(lstm_df, physics_df, cfg)
    x_raw, y_raw, phase_ids, meta = build_multistep_horizon_dataset(
        physics_df,
        horizon_y_true,
        horizon_y_lstm,
        horizon_phase_lbl,
        horizon_file_lbl,
        horizon_t_start,
        cfg,
    )
    if len(x_raw) < cfg.min_anchors:
        print(f"[WARN] {model_name}: only {len(x_raw)} multi-step anchors; skipping")
        return None

    model_dir = out_root / model_name
    plots_dir = model_dir / "plots"
    artifacts_dir = model_dir / "artifacts"
    metrics_dir = model_dir / "metrics"
    models_dir = model_dir / "models"
    for path in (plots_dir, artifacts_dir, metrics_dir, models_dir):
        ensure_dir(path)

    train_idx, val_idx, test_idx = split_temporal(len(x_raw), cfg)
    x_scaled, y_scaled, x_scaler, y_scaler = scale_blocks(x_raw, y_raw, train_idx)
    model, train_history, val_history = train_model(
        x_scaled,
        y_scaled,
        x_scaler,
        y_scaler,
        phase_ids,
        train_idx,
        val_idx,
        cfg,
    )

    all_idx = np.arange(len(x_raw))
    all_pred_raw = predict(model, x_scaled, all_idx, y_scaler, cfg)
    train_pred_raw = all_pred_raw[train_idx]
    val_pred_raw = all_pred_raw[val_idx]
    test_pred_raw = all_pred_raw[test_idx]
    phase_alphas = calibrate_phase_residual_alphas(
        y_raw[val_idx],
        val_pred_raw,
        x_raw[val_idx],
        phase_ids[val_idx],
        cfg,
    )
    all_pred = residual_blend(all_pred_raw, x_raw, phase_ids, phase_alphas, cfg)
    train_pred = all_pred[train_idx]
    val_pred = all_pred[val_idx]
    test_pred = all_pred[test_idx]

    aligned.to_csv(artifacts_dir / "aligned_lstm_physics_rows.csv", index=False)
    meta.to_csv(artifacts_dir / "multistep_anchor_metadata.csv", index=False)
    flat_pred = flatten_predictions(meta, y_raw, all_pred, x_raw, cfg)
    flat_pred.to_csv(artifacts_dir / "piml_multistep_predictions.csv", index=False)

    metrics_rows: List[Dict[str, object]] = []
    metrics_rows.extend(evaluate_split("train", y_raw[train_idx], train_pred, x_raw[train_idx], cfg))
    metrics_rows.extend(evaluate_split("validation", y_raw[val_idx], val_pred, x_raw[val_idx], cfg))
    metrics_rows.extend(evaluate_split("test", y_raw[test_idx], test_pred, x_raw[test_idx], cfg))
    metrics_rows.extend(evaluate_split("all", y_raw, all_pred, x_raw, cfg))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_dir / "metrics_summary.csv", index=False)
    with open(metrics_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)
    save_phase_metrics(flat_pred, metrics_dir / "metrics_by_phase.csv")
    phase_metrics_df = pd.read_csv(metrics_dir / "metrics_by_phase.csv")
    pd.DataFrame(train_history).to_csv(metrics_dir / "train_loss_history.csv", index=False)
    pd.DataFrame(val_history).to_csv(metrics_dir / "validation_loss_history.csv", index=False)

    with open(models_dir / "scalers.pkl", "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, f)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": ["phy_x", "phy_y", "phy_z", "lstm_x", "lstm_y", "lstm_z"],
            "target_features": ["true_x", "true_y", "true_z"],
            "backbone": "data_driven_lstm_horizon",
            "physics_role": "guided_correction_and_regularization",
            "physics_source_dir": physics_source_dir,
            "physics_output_version": cfg.physics_output_version,
            "phase_residual_alphas": phase_alphas,
            "config": asdict(cfg),
        },
        models_dir / "piml_2_fusion.pt",
    )

    save_loss_plot(train_history, val_history, plots_dir / "loss_curve.png")
    save_loss_components_plot(val_history, plots_dir / "validation_loss_components.png")
    save_horizon_error_plot(y_raw, all_pred, x_raw, plots_dir / "horizon_error_all.png", cfg, f"{model_name}: Horizon Error")
    test_mlp_pred = predict_horizon_with_saved_mlp(
        physics_df,
        baseline_from_input(x_raw[test_idx], "lstm_v2", cfg),
        np.array([ID_TO_PHASE.get(int(p), str(p)) for p in phase_ids[test_idx]], dtype=str),
        cfg,
        model_name,
    )
    save_horizon_error_plot(
        y_raw[test_idx],
        test_pred,
        x_raw[test_idx],
        plots_dir / "horizon_error_test.png",
        cfg,
        f"{model_name}: Test Horizon Error",
        test_mlp_pred,
    )
    save_anchor_validation_plot(y_raw[test_idx], test_pred, x_raw[test_idx], plots_dir / "test_anchor_validation_axes.png", cfg, f"{model_name}: Test Anchor Validation")
    save_error_histogram(y_raw[test_idx], test_pred, plots_dir / "test_error_histogram.png", f"{model_name}: Test PIML Error Distribution")
    save_trajectory_overlay(y_raw[test_idx], test_pred, x_raw[test_idx], plots_dir / "test_3d_trajectory_overlay.png", cfg, f"{model_name}: Test 3D Multi-Step Forecast")
    save_phase_rmse_bar(phase_metrics_df, plots_dir / "phase_comparison_bar_chart.png")
    save_phase_grid_real_vs_predicted(flat_pred, plots_dir / "real_vs_predicted_position_over_time_all_phase.png")
    save_phase_grid_representative(flat_pred, plots_dir / "real_vs_predicted_position_over_time_representative_phase.png")
    save_phase_grid_position_error(flat_pred, plots_dir / "position_error_over_time_all_phase.png")
    save_phase_grid_euclidean_error(flat_pred, plots_dir / "euclidean_error_over_time_all_phase.png")

    test_piml = metrics_df[(metrics_df["split"] == "test") & (metrics_df["predictor"] == "piml_fusion")].iloc[0].to_dict()
    external_summary = {}
    if validation_payload is not None:
        y_true_ext, y_lstm_ext, phase_ext, file_ext, t_start_ext, validation_source = validation_payload
        external = save_external_validation_outputs(
            "simple_test_validation",
            model,
            x_scaler,
            y_scaler,
            phase_alphas,
            physics_df,
            model_dir,
            cfg,
            y_true_ext,
            y_lstm_ext,
            phase_ext,
            file_ext,
            t_start_ext,
        )
        if external is not None:
            external_summary.update(external)
            external_summary["simple_test_validation_source"] = str(validation_source)
    print(
        f"[OK] {model_name}: {len(x_raw)} anchors, "
        f"phase alphas={phase_alphas}, "
        f"test ADE={test_piml['ade']:.4f}, test FDE={test_piml['fde']:.4f}, test RMSE={test_piml['rmse']:.4f}"
    )
    return {
        "physics_model": model_name,
        "physics_source_dir": physics_source_dir,
        "n_aligned_rows": int(len(aligned)),
        "n_anchors": int(len(x_raw)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "horizon_steps": int(cfg.horizon_steps),
        "epochs_ran": int(len(train_history)),
        "best_validation_total_loss": float(min(h["total"] for h in val_history)),
        "phase_residual_alphas": json.dumps(phase_alphas, sort_keys=True),
        "test_piml_ade": float(test_piml["ade"]),
        "test_piml_fde": float(test_piml["fde"]),
        "test_piml_rmse": float(test_piml["rmse"]),
        **external_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train data-driven-backbone physics-informed multi-step hybrid models.")
    parser.add_argument("--lstm-output-root", default=CFG.lstm_output_root)
    parser.add_argument("--physics-output-root", default=CFG.physics_output_root)
    parser.add_argument(
        "--physics-version",
        type=parse_physics_version,
        default=CFG.physics_output_version,
        help="Physics output version to use, e.g. 20 or v20. Use 'latest' to follow the newest output directory.",
    )
    parser.add_argument("--mlp-output-root", default=CFG.mlp_output_root)
    parser.add_argument("--validation-data-dir", default=CFG.validation_data_dir)
    parser.add_argument("--output-root", default=CFG.output_root)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--dt-seconds", type=float, default=None)
    parser.add_argument("--min-anchors", type=int, default=CFG.min_anchors)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--no-external-residual-calibration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inferred_dt, inferred_horizon = infer_lstm_timing(Path(args.lstm_output_root))
    cfg = Config(
        lstm_output_root=args.lstm_output_root,
        physics_output_root=args.physics_output_root,
        physics_output_version=args.physics_version,
        mlp_output_root=args.mlp_output_root,
        validation_data_dir=args.validation_data_dir,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        horizon_steps=args.horizon_steps if args.horizon_steps is not None else inferred_horizon,
        min_anchors=args.min_anchors,
        dt_seconds=args.dt_seconds if args.dt_seconds is not None else inferred_dt,
        device=args.device,
        calibrate_external_residual=not args.no_external_residual_calibration,
    )
    set_seed(cfg.seed)

    start = time.time()
    out_dir = get_next_version_dir(Path(cfg.output_root))
    lstm_df, lstm_artifact = load_latest_lstm_predictions(Path(cfg.lstm_output_root))
    horizon_y_true, horizon_y_lstm, horizon_phase_lbl, horizon_file_lbl, horizon_t_start, horizon_version_dir = load_latest_lstm_horizon_artifacts(
        Path(cfg.lstm_output_root),
        cfg.horizon_steps,
    )
    validation_payload = None
    if cfg.validation_data_dir:
        validation_path = Path(cfg.validation_data_dir)
        if validation_path.exists():
            validation_payload = load_lstm_predictions_for_data_dir(
                horizon_version_dir,
                validation_path,
                cfg.horizon_steps,
                cfg.device,
            )
        else:
            print(f"[WARN] Validation data directory not found; skipping external validation: {validation_path}")

    physics_outputs = collect_physics_predictions(Path(cfg.physics_output_root), cfg.physics_output_version)
    if not physics_outputs:
        version_label = "latest" if cfg.physics_output_version is None else f"v{cfg.physics_output_version}"
        raise RuntimeError(f"No usable physics-model prediction outputs were found for {version_label}.")

    summaries = []
    for model_name, physics_df in physics_outputs.items():
        summary = process_physics_model(
            model_name,
            physics_df,
            lstm_df,
            out_dir,
            cfg,
            horizon_y_true,
            horizon_y_lstm,
            horizon_phase_lbl,
            horizon_file_lbl,
            horizon_t_start,
            validation_payload,
        )
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError("No physics model had enough aligned rows for multi-step PIML training.")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "piml_2_model_summary.csv", index=False)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "runtime_seconds": time.time() - start,
                "config": asdict(cfg),
                "lstm_artifact": str(lstm_artifact),
                "lstm_horizon_source": str(horizon_version_dir),
                "physics_output_root": str(cfg.physics_output_root),
                "physics_output_version": cfg.physics_output_version,
                "validation_data_dir": str(cfg.validation_data_dir),
                "output_dir": str(out_dir),
                "physics_models_processed": list(summary_df["physics_model"]),
            },
            f,
            indent=2,
        )

    print(f"[DONE] Wrote PIML_2 outputs to {out_dir}")


if __name__ == "__main__":
    main()
