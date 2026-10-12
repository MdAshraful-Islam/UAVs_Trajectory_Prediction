#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM v2: train ONE model on all flight data together (phase included as input),
then evaluate/predict:
1) per phase (takeoff, cruise, landing)
2) full flight (continuous next-step trajectory for one test flight)
"""

import os
import re
import glob
import json
import time
import math
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    data_dir: str = "data/full_dataset"  # folder next to this script
    output_root: str = "output_lstm"

    time_col_candidates: Tuple[str, ...] = ("time", "Time", "timestamp", "t")
    phase_col_candidates: Tuple[str, ...] = ("Phase", "phase", "phase ", "flight_phase", "FlightPhase")
    x_col_candidates: Tuple[str, ...] = ("x_position", "x", "pos_x", "position_x")
    y_col_candidates: Tuple[str, ...] = ("y_position", "y", "pos_y", "position_y")
    z_col_candidates: Tuple[str, ...] = ("z_position", "z", "pos_z", "position_z", "alt", "altitude")

    extra_feature_candidates: Tuple[str, ...] = (
        "wind_speed", "wind_angle",
        "x_twist_linear", "y_twist_linear", "z_twist_linear",
        "x_linear_acceleration", "y_linear_acceleration", "z_linear_acceleration",
        "battery_voltage", "battery_current",
        "x_orientation", "y_orientation", "z_orientation", "w",
    )

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    horizon_seconds: float = 30.0
    lookback_seconds: float = 6.0

    batch_size: int = 128
    epochs: int = 30
    lr: float = 0.001
    weight_decay: float = 1e-6
    grad_clip: float = 1.0

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.22

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    early_stopping_patience: int = 10


CFG = Config()
PHASES: Tuple[str, ...] = ("takeoff", "cruise", "landing")

OPTIONAL_FEATURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "wind_speed": ("wind_speed", "windSpeed", "windspeed"),
    "wind_angle": ("wind_angle", "windAngle", "wind_dir", "wind_direction"),
    "battery_voltage": ("battery_voltage", "batteryVoltage", "voltage"),
    "battery_current": ("battery_current", "batteryCurrent", "current"),
    "orientation_x": ("orientation_x", "x_orientation"),
    "orientation_y": ("orientation_y", "y_orientation"),
    "orientation_z": ("orientation_z", "z_orientation"),
    "orientation_w": ("orientation_w", "w"),
    "velocity_x": ("velocity_x", "x_twist_linear"),
    "velocity_y": ("velocity_y", "y_twist_linear"),
    "velocity_z": ("velocity_z", "z_twist_linear"),
    "angular_x": ("angular_x", "x_twist_angular"),
    "angular_y": ("angular_y", "y_twist_angular"),
    "angular_z": ("angular_z", "z_twist_angular"),
    "linear_acceleration_x": ("linear_acceleration_x", "x_linear_acceleration"),
    "linear_acceleration_y": ("linear_acceleration_y", "y_linear_acceleration"),
    "linear_acceleration_z": ("linear_acceleration_z", "z_linear_acceleration"),
}


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_next_version_dir(output_root: str) -> str:
    candidates = [output_root, output_root + "_runs", output_root + "_results"]
    last_err = None
    for root in candidates:
        try:
            ensure_dir(root)
            vals = []
            for d in os.listdir(root):
                m = re.match(r"v(\d+)$", d)
                if m and os.path.isdir(os.path.join(root, d)):
                    vals.append(int(m.group(1)))
            nxt = 1 if not vals else max(vals) + 1
            out = os.path.join(root, f"v{nxt}")
            ensure_dir(out)
            return out
        except PermissionError as e:
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError("Could not create a versioned output directory.")


def find_first_existing_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def normalize_phase(s: str) -> str:
    if s is None:
        return "unknown"
    t = str(s).strip().lower()
    if "take" in t or "ascend" in t:
        return "takeoff"
    if "land" in t or "descend" in t:
        return "landing"
    if "cruise" in t or "hover" in t or "turn" in t:
        return "cruise"
    return "unknown"


def compute_dt_seconds(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.size < 5:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        return 1.0
    return float(np.median(dt))


class Seq2SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        h_last = h_n[-1]
        dec_in = h_last.unsqueeze(1).repeat(1, self.horizon, 1)
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))
        return self.head(dec_out)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def ade(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.linalg.norm(y_true - y_pred, axis=-1)
    return float(np.mean(d))


def fde(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.linalg.norm(y_true[:, -1, :] - y_pred[:, -1, :], axis=-1)
    return float(np.mean(d))


def save_loss_plot(train_losses: List[float], val_losses: List[float], outpath: str):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.title("Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_error_over_time_plot(t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, outpath: str, title: str):
    err = np.linalg.norm(y_true - y_pred, axis=-1)
    m = np.mean(err, axis=0)
    p50 = np.percentile(err, 50, axis=0)
    p90 = np.percentile(err, 90, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(t, m, label="mean")
    plt.plot(t, p50, label="p50")
    plt.plot(t, p90, label="p90")
    plt.xlabel("future time (s)")
    plt.ylabel("euclidean error")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_hist(y_true: np.ndarray, y_pred: np.ndarray, outpath: str, title: str):
    err = np.linalg.norm(y_true - y_pred, axis=-1).reshape(-1)
    plt.figure(figsize=(8, 5))
    plt.hist(err, bins=60)
    plt.title(title)
    plt.xlabel("euclidean error")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_3d_overlay(y_true: np.ndarray, y_pred: np.ndarray, outpath: str, title: str):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(y_true[:, 0], y_true[:, 1], y_true[:, 2], label="true")
    ax.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], "--", label="pred")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_phase_rmse_bar(phase_df: pd.DataFrame, outpath: str):
    if "status" in phase_df.columns:
        valid = phase_df[phase_df["status"] != "no_test_samples"].copy()
    else:
        valid = phase_df.copy()
    if valid.empty or "rmse_xyz" not in valid.columns:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(valid["phase"], valid["rmse_xyz"], color=["#2f7f6f", "#386cb0", "#d95f02"][: len(valid)])
    plt.ylabel("RMSE XYZ")
    plt.title("Phase Comparison: LSTM v2")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_all_sample_rmse_horizon(y_true: np.ndarray, y_pred: np.ndarray, dt: float, outpath: str):
    t = np.arange(y_true.shape[1]) * dt
    err = y_true - y_pred
    rmse_x = np.sqrt(np.mean(err[:, :, 0] ** 2, axis=0))
    rmse_y = np.sqrt(np.mean(err[:, :, 1] ** 2, axis=0))
    rmse_z = np.sqrt(np.mean(err[:, :, 2] ** 2, axis=0))
    rmse_3d = np.sqrt(np.mean(np.linalg.norm(err, axis=-1) ** 2, axis=0))
    plt.figure(figsize=(10, 5))
    plt.plot(t, rmse_x, label="RMSE X")
    plt.plot(t, rmse_y, label="RMSE Y")
    plt.plot(t, rmse_z, label="RMSE Z")
    plt.plot(t, rmse_3d, label="RMSE 3D", linewidth=2)
    plt.xlabel("future time (s)")
    plt.ylabel("RMSE")
    plt.title("RMSE Over Prediction Horizon")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def comparison_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], color="black", linestyle="-", label="Real"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2.0, label="Predicted"),
    ]


def trajectory_velocity(traj: np.ndarray, dt: float) -> np.ndarray:
    if traj.shape[0] < 2:
        return np.zeros_like(traj)
    return np.gradient(traj, dt, axis=0)


def build_case_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phases: np.ndarray,
    files: np.ndarray,
    dt: float,
) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    case_outputs: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    t = np.arange(y_true.shape[1], dtype=float) * dt

    for i in range(y_true.shape[0]):
        real = np.asarray(y_true[i], dtype=float)
        pred = np.asarray(y_pred[i], dtype=float)
        real_vel = trajectory_velocity(real, dt)
        pred_vel = trajectory_velocity(pred, dt)
        err = pred - real
        vel_err = pred_vel - real_vel
        err_3d = np.linalg.norm(err, axis=1)

        err_df = pd.DataFrame(
            {
                "time_rel": t,
                "real_x": real[:, 0],
                "real_y": real[:, 1],
                "real_z": real[:, 2],
                "x": pred[:, 0],
                "y": pred[:, 1],
                "z": pred[:, 2],
                "real_vx": real_vel[:, 0],
                "real_vy": real_vel[:, 1],
                "real_vz": real_vel[:, 2],
                "vx": pred_vel[:, 0],
                "vy": pred_vel[:, 1],
                "vz": pred_vel[:, 2],
                "error_x": err[:, 0],
                "error_y": err[:, 1],
                "error_z": err[:, 2],
                "error_vx": vel_err[:, 0],
                "error_vy": vel_err[:, 1],
                "error_vz": vel_err[:, 2],
                "error_3d": err_3d,
            }
        )

        phase = str(phases[i])
        csv_name = str(files[i]).replace("\\", "/")
        case_outputs.append({"phase": phase, "csv": csv_name, "error": err_df})
        detail_rows.append(
            {
                "phase": phase,
                "csv": csv_name,
                "rmse_3d_m": float(np.sqrt(np.mean(err_3d ** 2))),
                "mae_3d_m": float(np.mean(err_3d)),
                "final_error_m": float(err_3d[-1]),
                "rmse_velocity_mps": float(np.sqrt(np.mean(np.sum(vel_err ** 2, axis=1)))),
            }
        )

    return case_outputs, pd.DataFrame(detail_rows)


def representative_case_for_phase(cases: List[Dict[str, object]]) -> Optional[Tuple[Dict[str, object], Dict[str, float]]]:
    scored_cases = []
    for case in cases:
        err = case["error"]
        error_3d = err["error_3d"].to_numpy(dtype=float)
        error_3d = error_3d[np.isfinite(error_3d)]
        if error_3d.size == 0:
            continue
        rmse_3d = float(np.sqrt(np.mean(error_3d**2)))
        scored_cases.append(
            (
                rmse_3d,
                case,
                {
                    "rmse_3d_m": rmse_3d,
                    "mae_3d_m": float(np.mean(error_3d)),
                    "final_error_m": float(error_3d[-1]),
                    "n_points": float(error_3d.size),
                },
            )
        )
    if not scored_cases:
        return None

    median_rmse = float(np.median([score for score, _, _ in scored_cases]))
    _, case, metrics = min(scored_cases, key=lambda item: abs(item[0] - median_rmse))
    return case, metrics


def save_representative_phase_position_plot(case_outputs: List[Dict[str, object]], out_dir: str) -> None:
    by_phase = {phase: [case for case in case_outputs if case["phase"] == phase] for phase in PHASES}
    selected_rows = []

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        selected = representative_case_for_phase(by_phase[phase])
        for j, col in enumerate(["x", "y", "z"]):
            ax = axs[i, j]
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(["X", "Y", "Z"][j] + " (m)")
            ax.set_title(f"{phase.capitalize()} {['X', 'Y', 'Z'][j]}")

            if selected is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            case, metrics = selected
            err = case["error"]
            t = err["time_rel"]
            ax.plot(t, err[f"real_{col}"], "-", color="#1f77b4", alpha=0.9)
            ax.plot(t, err[col], ":", color="#d62728", alpha=0.95, linewidth=2.0)
            if i == 0 and j == 0:
                ax.legend(handles=comparison_legend_handles(), loc="best")

        if selected is not None:
            case, metrics = selected
            selected_rows.append(
                {
                    "phase": phase,
                    "csv": case["csv"],
                    "rmse_3d_m": metrics["rmse_3d_m"],
                    "mae_3d_m": metrics["mae_3d_m"],
                    "final_error_m": metrics["final_error_m"],
                    "n_points": int(metrics["n_points"]),
                }
            )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "real_vs_predicted_position_over_time_representative_phase.png"), dpi=220)
    plt.close(fig)

    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(os.path.join(out_dir, "representative_phase_cases.csv"), index=False)


def save_extended_plot_suite(case_outputs: List[Dict[str, object]], detail: pd.DataFrame, out_dir: str) -> None:
    if not case_outputs or detail.empty:
        return

    by_phase = {phase: [case for case in case_outputs if case["phase"] == phase] for phase in PHASES}
    class_legend = comparison_legend_handles()
    unique_labels = sorted({str(case["csv"]) for case in case_outputs})
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    color_map = {label: cmap(i % cmap.N) for i, label in enumerate(unique_labels)}

    save_representative_phase_position_plot(case_outputs, out_dir)

    fig = plt.figure(figsize=(15, 10))
    for i, phase in enumerate(PHASES, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for case in by_phase[phase]:
            err = case["error"]
            color = color_map[str(case["csv"])]
            ax.plot(err["real_x"], err["real_y"], err["real_z"], "-", alpha=0.35, color=color)
            ax.plot(err["x"], err["y"], err["z"], ":", alpha=0.80, color=color, linewidth=1.6)
        ax.set_xlabel("X local (m)")
        ax.set_ylabel("Y local (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{phase.capitalize()}")
        if i == 1:
            ax.legend(handles=class_legend, loc="best")
    fig.suptitle("Real vs Predicted 3D Trajectory")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "real_vs_predicted_3d_trajectory.png"), dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            for j, col in enumerate(["x", "y", "z"]):
                color = color_map[str(case["csv"])]
                axs[i, j].plot(t, err[f"real_{col}"], "-", alpha=0.35, color=color)
                axs[i, j].plot(t, err[col], ":", alpha=0.85, color=color, linewidth=1.6)
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["X", "Y", "Z"][j] + " (m)")
                axs[i, j].grid(True, alpha=0.25)
                if i == 0 and j == 0:
                    axs[i, j].legend(handles=class_legend, loc="best")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Real vs Predicted Position Over Time")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "real_vs_predicted_position_over_time.png"), dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            for j, col in enumerate(["vx", "vy", "vz"]):
                color = color_map[str(case["csv"])]
                axs[i, j].plot(t, err[f"real_{col}"], "-", alpha=0.35, color=color)
                axs[i, j].plot(t, err[col], ":", alpha=0.85, color=color, linewidth=1.6)
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["Vx", "Vy", "Vz"][j] + " (m/s)")
                axs[i, j].grid(True, alpha=0.25)
                if i == 0 and j == 0:
                    axs[i, j].legend(handles=class_legend, loc="best")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Real vs Predicted Velocity Over Time")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "real_vs_predicted_velocity_over_time.png"), dpi=220)
    plt.close(fig)

    for phase, cases in by_phase.items():
        if not cases:
            continue
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for case in cases:
            err = case["error"]
            color = color_map[str(case["csv"])]
            ax.plot(err["real_x"], err["real_y"], err["real_z"], "-", alpha=0.35, color=color)
            ax.plot(err["x"], err["y"], err["z"], ":", alpha=0.75, color=color, linewidth=1.6)
        ax.set_xlabel("X local (m)")
        ax.set_ylabel("Y local (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"3D Trajectory: {phase.capitalize()} - all real flights")
        ax.legend(handles=class_legend, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"3d_trajectory_{phase}.png"), dpi=220)
        plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            for j, col in enumerate(["error_x", "error_y", "error_z"]):
                axs[i, j].plot(t, err[col], alpha=0.65, color=color_map[str(case["csv"])])
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["X", "Y", "Z"][j] + " Error (m)")
                axs[i, j].grid(True, alpha=0.25)
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "position_error_over_time.png"), dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            for j, col in enumerate(["error_vx", "error_vy", "error_vz"]):
                axs[i, j].plot(t, err[col], alpha=0.65, color=color_map[str(case["csv"])])
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["Vx", "Vy", "Vz"][j] + " Error (m/s)")
                axs[i, j].grid(True, alpha=0.25)
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "velocity_error_over_time.png"), dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            for j, col in enumerate(["x", "y", "z"]):
                color = color_map[str(case["csv"])]
                axs[i, j].plot(t, err[col], ":", alpha=0.5, color=color)
                axs[i, j].plot(t, err[f"real_{col}"], "-", alpha=0.4, color=color)
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["X", "Y", "Z"][j] + " (m)")
                axs[i, j].grid(True, alpha=0.25)
                if i == 0 and j == 0:
                    axs[i, j].legend(handles=class_legend, loc="best")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "absolute_position_over_time.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    detail.boxplot(column="rmse_3d_m", by="phase", ax=ax)
    ax.set_title("RMSE 3D Distribution by Phase")
    ax.set_ylabel("RMSE 3D (m)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rmse_distribution_by_phase.png"), dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for case in case_outputs:
        err = case["error"]
        t = err["time_rel"]
        for ax, col, label in zip(axes, ["error_x", "error_y", "error_z"], ["X Error (m)", "Y Error (m)", "Z Error (m)"]):
            ax.plot(t, err[col], alpha=0.18, linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
    axes[0].set_title("All Flights: Position Error Over Time")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_flights_position_error_over_time.png"), dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for case in case_outputs:
        err = case["error"]
        t = err["time_rel"]
        for ax, col, label in zip(axes, ["error_vx", "error_vy", "error_vz"], ["Vx Error (m/s)", "Vy Error (m/s)", "Vz Error (m/s)"]):
            ax.plot(t, err[col], alpha=0.18, linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
    axes[0].set_title("All Flights: Velocity Error Over Time")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_flights_velocity_error_over_time.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for case in case_outputs:
        err = case["error"]
        ax.plot(err["time_rel"], err["error_3d"], alpha=0.18, linewidth=0.8)
    ax.set_title("All Flights: 3D Position Error Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("3D Error (m)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_flights_3d_error_over_time.png"), dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for case in case_outputs:
        err = case["error"]
        t = err["time_rel"]
        color = color_map[str(case["csv"])]
        for ax, col, label in zip(axes, ["x", "y", "z"], ["X (m)", "Y (m)", "Z (m)"]):
            ax.plot(t, err[f"real_{col}"], "-", alpha=0.18, color=color, linewidth=0.9)
            ax.plot(t, err[col], ":", alpha=0.45, color=color, linewidth=1.1)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
    axes[0].set_title("All Flights: Real Solid vs Predicted Dotted Position")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_flights_real_vs_predicted_position_over_time.png"), dpi=220)
    plt.close(fig)


def load_all_csvs(data_dir: str) -> List[Tuple[str, pd.DataFrame]]:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    out = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=None, engine="python")
            label = os.path.relpath(f, data_dir).replace("\\", "/")
            out.append((label, df))
        except Exception as e:
            print(f"[WARN] Failed reading {f}: {e}")
    return out


def build_unified_dataframe(all_dfs: List[Tuple[str, pd.DataFrame]], cfg: Config) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames = []
    feature_name_map: Dict[str, str] = {}
    for fname, df in all_dfs:
        time_col = find_first_existing_col(df, cfg.time_col_candidates)
        phase_col = find_first_existing_col(df, cfg.phase_col_candidates)
        x_col = find_first_existing_col(df, cfg.x_col_candidates)
        y_col = find_first_existing_col(df, cfg.y_col_candidates)
        z_col = find_first_existing_col(df, cfg.z_col_candidates)

        if any(v is None for v in [time_col, phase_col, x_col, y_col, z_col]):
            print(f"[WARN] Skipping {fname}: missing required cols.")
            continue

        tmp = df.copy()
        tmp["_file"] = fname

        tmp["time_std"] = pd.to_numeric(tmp[time_col], errors="coerce")
        tmp["phase_std"] = tmp[phase_col].apply(normalize_phase)
        tmp["x_std"] = pd.to_numeric(tmp[x_col], errors="coerce")
        tmp["y_std"] = pd.to_numeric(tmp[y_col], errors="coerce")
        tmp["z_std"] = pd.to_numeric(tmp[z_col], errors="coerce")

        feature_name_map.setdefault("x_std", x_col)
        feature_name_map.setdefault("y_std", y_col)
        feature_name_map.setdefault("z_std", z_col)
        feature_name_map.setdefault("phase_takeoff", f"{phase_col}_takeoff")
        feature_name_map.setdefault("phase_cruise", f"{phase_col}_cruise")
        feature_name_map.setdefault("phase_landing", f"{phase_col}_landing")

        for internal_name, aliases in OPTIONAL_FEATURE_ALIASES.items():
            cc = find_first_existing_col(tmp, aliases)
            if cc is not None:
                tmp[internal_name] = pd.to_numeric(tmp[cc], errors="coerce")
                feature_name_map.setdefault(internal_name, cc)

        keep = ["time_std", "phase_std", "x_std", "y_std", "z_std", "_file"] + [
            c for c in OPTIONAL_FEATURE_ALIASES if c in tmp.columns
        ]
        tmp = tmp[keep]
        tmp = tmp.dropna(subset=["time_std", "x_std", "y_std", "z_std"])
        frames.append(tmp)

    if not frames:
        raise RuntimeError("No usable CSVs found.")

    big = pd.concat(frames, axis=0, ignore_index=True)
    big = big.sort_values(["_file", "time_std"]).reset_index(drop=True)

    # phase one-hot as input features
    big["phase_takeoff"] = (big["phase_std"] == "takeoff").astype(float)
    big["phase_cruise"] = (big["phase_std"] == "cruise").astype(float)
    big["phase_landing"] = (big["phase_std"] == "landing").astype(float)
    return big, feature_name_map


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "wind_speed" in out.columns and "wind_angle" in out.columns:
        spd = out["wind_speed"].to_numpy(dtype=float)
        ang = np.deg2rad(out["wind_angle"].to_numpy(dtype=float))
        out["wind_x"] = spd * np.cos(ang)
        out["wind_y"] = spd * np.sin(ang)
    return out


def create_sequences_all(
    df: pd.DataFrame, feature_name_map: Dict[str, str], cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, List[str]]:
    df = add_wind_components(df)

    base_feats = ["x_std", "y_std", "z_std", "phase_takeoff", "phase_cruise", "phase_landing"]
    optional = []
    for c in ["wind_speed", "wind_angle", "wind_x", "wind_y"] + list(OPTIONAL_FEATURE_ALIASES):
        if c in df.columns and c not in base_feats and c not in optional:
            optional.append(c)
    feature_internal_cols = base_feats + optional
    feature_cols = [feature_name_map.get(c, c) for c in feature_internal_cols]

    dts = []
    for _, g in df.groupby("_file"):
        t = g["time_std"].to_numpy(dtype=float)
        if t.size > 5:
            dts.append(compute_dt_seconds(t))
    dt = float(np.median(dts)) if dts else 1.0

    H = max(1, int(round(cfg.horizon_seconds / dt)))
    L = max(2, int(round(cfg.lookback_seconds / dt)))

    X_list, Y_list = [], []
    phase_list, file_list, tnext_list = [], [], []

    for fname, g in df.groupby("_file"):
        g = g.sort_values("time_std").reset_index(drop=True)
        t = g["time_std"].to_numpy(dtype=float)
        feats = g[feature_internal_cols].to_numpy(dtype=float)
        phases = g["phase_std"].astype(str).to_numpy()
        pos = g[["x_std", "y_std", "z_std"]].to_numpy(dtype=float)

        mask = np.all(np.isfinite(feats), axis=1) & np.isfinite(t) & np.all(np.isfinite(pos), axis=1)
        t = t[mask]
        feats = feats[mask]
        phases = phases[mask]
        pos = pos[mask]

        if len(t) < (L + H + 1):
            continue

        for i in range(0, len(t) - (L + H) + 1):
            X_list.append(feats[i:i + L, :])
            Y_list.append(pos[i + L:i + L + H, :])
            phase_list.append(phases[i + L - 1])  # phase at prediction anchor
            file_list.append(fname)
            tnext_list.append(t[i + L])           # time at first predicted step

    if not X_list:
        return (
            np.zeros((0, 1, len(feature_cols))),
            np.zeros((0, 1, 3)),
            np.array([]),
            np.array([]),
            np.array([]),
            dt,
            feature_cols,
        )

    return (
        np.stack(X_list, axis=0),
        np.stack(Y_list, axis=0),
        np.asarray(phase_list),
        np.asarray(file_list),
        np.asarray(tnext_list, dtype=float),
        dt,
        feature_cols,
    )


def split_by_file(
    X: np.ndarray, Y: np.ndarray, phase_lbl: np.ndarray, file_lbl: np.ndarray, tnext: np.ndarray, cfg: Config
):
    files = np.unique(file_lbl)
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(files)

    n = len(files)
    n_train = max(1, int(math.floor(cfg.train_ratio * n)))
    n_val = max(1, int(math.floor(cfg.val_ratio * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_files = set(files[:n_train])
    val_files = set(files[n_train:n_train + n_val])
    test_files = set(files[n_train + n_val:])

    if not test_files:
        test_files = {files[-1]}

    m_train = np.array([f in train_files for f in file_lbl])
    m_val = np.array([f in val_files for f in file_lbl])
    m_test = np.array([f in test_files for f in file_lbl])

    return {
        "train": (X[m_train], Y[m_train], phase_lbl[m_train], file_lbl[m_train], tnext[m_train]),
        "val": (X[m_val], Y[m_val], phase_lbl[m_val], file_lbl[m_val], tnext[m_val]),
        "test": (X[m_test], Y[m_test], phase_lbl[m_test], file_lbl[m_test], tnext[m_test]),
    }


def fit_scalers(Xtr: np.ndarray, Ytr: np.ndarray):
    xs = Xtr.reshape(-1, Xtr.shape[-1])
    ys = Ytr.reshape(-1, Ytr.shape[-1])
    x_scaler = StandardScaler().fit(xs)
    y_scaler = StandardScaler().fit(ys)
    return x_scaler, y_scaler


def apply_scalers(X: np.ndarray, Y: np.ndarray, x_scaler: StandardScaler, y_scaler: StandardScaler):
    Xs = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    Ys = y_scaler.transform(Y.reshape(-1, Y.shape[-1])).reshape(Y.shape)
    return Xs, Ys


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse_xyz": mse(y_true, y_pred),
        "rmse_xyz": rmse(y_true, y_pred),
        "mae_xyz": mae(y_true, y_pred),
        "ade": ade(y_true, y_pred),
        "fde": fde(y_true, y_pred),
        "mse_x": mse(y_true[:, :, 0], y_pred[:, :, 0]),
        "mse_y": mse(y_true[:, :, 1], y_pred[:, :, 1]),
        "mse_z": mse(y_true[:, :, 2], y_pred[:, :, 2]),
        "rmse_x": rmse(y_true[:, :, 0], y_pred[:, :, 0]),
        "rmse_y": rmse(y_true[:, :, 1], y_pred[:, :, 1]),
        "rmse_z": rmse(y_true[:, :, 2], y_pred[:, :, 2]),
        "mae_x": mae(y_true[:, :, 0], y_pred[:, :, 0]),
        "mae_y": mae(y_true[:, :, 1], y_pred[:, :, 1]),
        "mae_z": mae(y_true[:, :, 2], y_pred[:, :, 2]),
    }


def train_and_predict_all(
    X: np.ndarray,
    Y: np.ndarray,
    phase_lbl: np.ndarray,
    file_lbl: np.ndarray,
    tnext: np.ndarray,
    dt: float,
    feature_cols: List[str],
    out_root: str,
    cfg: Config,
):
    ensure_dir(out_root)
    ensure_dir(os.path.join(out_root, "plots"))
    ensure_dir(os.path.join(out_root, "models"))
    ensure_dir(os.path.join(out_root, "metrics"))
    ensure_dir(os.path.join(out_root, "artifacts"))

    splits = split_by_file(X, Y, phase_lbl, file_lbl, tnext, cfg)
    Xtr, Ytr, _, _, _ = splits["train"]
    Xva, Yva, _, _, _ = splits["val"]
    Xte, Yte, Pte, Fte, Tte = splits["test"]

    if Xtr.shape[0] < 20 or Xva.shape[0] < 5 or Xte.shape[0] < 5:
        raise RuntimeError("Not enough samples after split. Add more files or reduce horizon/lookback.")

    x_scaler, y_scaler = fit_scalers(Xtr, Ytr)
    Xtr_s, Ytr_s = apply_scalers(Xtr, Ytr, x_scaler, y_scaler)
    Xva_s, Yva_s = apply_scalers(Xva, Yva, x_scaler, y_scaler)
    Xte_s, Yte_s = apply_scalers(Xte, Yte, x_scaler, y_scaler)

    train_loader = DataLoader(Seq2SeqDataset(Xtr_s, Ytr_s), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(Seq2SeqDataset(Xva_s, Yva_s), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(Seq2SeqDataset(Xte_s, Yte_s), batch_size=cfg.batch_size, shuffle=False)

    H = Y.shape[1]
    F = X.shape[2]
    model = LSTMSeq2Seq(F, cfg.hidden_size, cfg.num_layers, cfg.dropout, H).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_ckpt = os.path.join(out_root, "models", "best_all.pt")
    tr_losses, va_losses = [], []
    stale_epochs = 0

    for ep in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        ntr = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            yp = model(xb)
            loss = criterion(yp, yb)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
            ntr += xb.size(0)
        tr_loss /= max(1, ntr)

        model.eval()
        va_loss = 0.0
        nva = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                yp = model(xb)
                loss = criterion(yp, yb)
                va_loss += float(loss.item()) * xb.size(0)
                nva += xb.size(0)
        va_loss /= max(1, nva)

        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        if va_loss < best_val:
            best_val = va_loss
            stale_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg.__dict__, "feature_cols": feature_cols, "dt": dt},
                best_ckpt,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.early_stopping_patience:
                print(f"[all] early stopping at epoch {ep + 1:03d} | best val {best_val:.6f}")
                break
        print(f"[all] epoch {ep + 1:03d}/{cfg.epochs} | train {tr_loss:.6f} | val {va_loss:.6f}")

    ck = torch.load(best_ckpt, map_location=cfg.device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    preds_s, trues_s = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(cfg.device)
            yp = model(xb).cpu().numpy()
            preds_s.append(yp)
            trues_s.append(yb.numpy())
    Ypred_s = np.concatenate(preds_s, axis=0)
    Ytrue_s = np.concatenate(trues_s, axis=0)

    Ypred = y_scaler.inverse_transform(Ypred_s.reshape(-1, 3)).reshape(Ypred_s.shape)
    Ytrue = y_scaler.inverse_transform(Ytrue_s.reshape(-1, 3)).reshape(Ytrue_s.shape)

    save_loss_plot(tr_losses, va_losses, os.path.join(out_root, "plots", "loss_all.png"))
    t_future = np.arange(H) * dt
    save_error_over_time_plot(
        t_future, Ytrue, Ypred, os.path.join(out_root, "plots", "error_over_time_all.png"), "All-data Error over Horizon"
    )
    save_all_sample_rmse_horizon(
        Ytrue, Ypred, dt, os.path.join(out_root, "plots", "rmse_over_prediction_horizon_all.png")
    )
    save_hist(Ytrue, Ypred, os.path.join(out_root, "plots", "error_hist_all.png"), "All-data Error Histogram")
    save_3d_overlay(
        Ytrue[0], Ypred[0], os.path.join(out_root, "plots", "traj_3d_overlay_all.png"), "All-data 3D Overlay (sample)"
    )

    overall = eval_metrics(Ytrue, Ypred)
    overall.update(
        {
            "status": "ok",
            "n_total": int(X.shape[0]),
            "n_train": int(Xtr.shape[0]),
            "n_val": int(Xva.shape[0]),
            "n_test": int(Xte.shape[0]),
            "dt_seconds": float(dt),
            "lookback_steps": int(X.shape[1]),
            "horizon_steps": int(H),
            "horizon_seconds_effective": float(H * dt),
            "feature_count": int(F),
            "features": feature_cols,
        }
    )

    with open(os.path.join(out_root, "metrics", "metrics_all.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # Per-phase metrics on same test predictions
    rows = []
    for ph in ["takeoff", "cruise", "landing"]:
        m = (Pte == ph)
        if np.sum(m) == 0:
            rows.append({"phase": ph, "status": "no_test_samples"})
            continue
        mm = eval_metrics(Ytrue[m], Ypred[m])
        mm["phase"] = ph
        mm["n_test"] = int(np.sum(m))
        rows.append(mm)
        save_error_over_time_plot(
            t_future,
            Ytrue[m],
            Ypred[m],
            os.path.join(out_root, "plots", f"error_over_time_{ph}.png"),
            f"{ph.capitalize()} Error over Horizon",
        )

    phase_df = pd.DataFrame(rows)
    phase_df.to_csv(os.path.join(out_root, "metrics", "metrics_by_phase.csv"), index=False)
    save_phase_rmse_bar(phase_df, os.path.join(out_root, "plots", "phase_comparison_bar_chart.png"))

    case_outputs, detail_df = build_case_outputs(Ytrue, Ypred, Pte, Fte, dt)
    detail_df.to_csv(os.path.join(out_root, "metrics", "per_sample_metrics.csv"), index=False)
    save_extended_plot_suite(case_outputs, detail_df, os.path.join(out_root, "plots"))

    # Full-flight next-step continuous prediction for one test flight
    uniq_test_files = pd.Series(Fte).value_counts().index.tolist()
    if uniq_test_files:
        pick = uniq_test_files[0]
        m = (Fte == pick)
        d = pd.DataFrame(
            {
                "file": Fte[m],
                "phase": Pte[m],
                "t_next": Tte[m],
                "true_x": Ytrue[m, 0, 0],
                "true_y": Ytrue[m, 0, 1],
                "true_z": Ytrue[m, 0, 2],
                "pred_x": Ypred[m, 0, 0],
                "pred_y": Ypred[m, 0, 1],
                "pred_z": Ypred[m, 0, 2],
            }
        ).sort_values("t_next")
        d["err_euclidean"] = np.sqrt(
            (d["true_x"] - d["pred_x"]) ** 2 + (d["true_y"] - d["pred_y"]) ** 2 + (d["true_z"] - d["pred_z"]) ** 2
        )
        d.to_csv(os.path.join(out_root, "artifacts", "full_flight_nextstep_predictions.csv"), index=False)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for ax, c in zip(axes, ["x", "y", "z"]):
            ax.plot(d["t_next"], d[f"true_{c}"], label=f"true_{c}")
            ax.plot(d["t_next"], d[f"pred_{c}"], "--", label=f"pred_{c}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel("time")
        fig.suptitle(f"Full-flight next-step prediction ({pick})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, "plots", "real_vs_predicted_position_over_time.png"), dpi=200)
        plt.close(fig)

        plt.figure(figsize=(7, 6))
        plt.plot(d["true_x"], d["true_y"], label="true_xy")
        plt.plot(d["pred_x"], d["pred_y"], "--", label="pred_xy")
        plt.title(f"Full-flight XY trajectory ({pick})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "plots", "real_vs_predicted_xy_trajectory.png"), dpi=200)
        plt.close()

    with open(os.path.join(out_root, "artifacts", "scalers.pkl"), "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "feature_cols": feature_cols, "dt": dt}, f)
    np.save(os.path.join(out_root, "artifacts", "Ytrue_test.npy"), Ytrue)
    np.save(os.path.join(out_root, "artifacts", "Ypred_test.npy"), Ypred)
    np.save(os.path.join(out_root, "artifacts", "Pte_test.npy"), Pte)
    np.save(os.path.join(out_root, "artifacts", "Fte_test.npy"), Fte)
    np.save(os.path.join(out_root, "artifacts", "Tte_test.npy"), Tte)


def main():
    set_seed(CFG.seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, CFG.data_dir)
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Data directory not found: {data_dir}")

    all_dfs = load_all_csvs(data_dir)
    if len(all_dfs) == 0:
        raise RuntimeError(f"No CSV files found in: {data_dir}")

    version_dir = get_next_version_dir(os.path.join(script_dir, CFG.output_root))
    ensure_dir(version_dir)

    big_df, feature_name_map = build_unified_dataframe(all_dfs, CFG)
    X, Y, phase_lbl, file_lbl, tnext, dt, feature_cols = create_sequences_all(big_df, feature_name_map, CFG)
    if X.shape[0] == 0:
        raise RuntimeError("No sequences generated. Reduce lookback/horizon or check input data.")

    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": CFG.device,
        "data_dir": data_dir,
        "csv_files": [f for f, _ in all_dfs],
        "config": CFG.__dict__,
        "n_sequences": int(X.shape[0]),
        "lookback_steps": int(X.shape[1]),
        "horizon_steps": int(Y.shape[1]),
        "dt": float(dt),
        "features": feature_cols,
    }
    with open(os.path.join(version_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(
        f"All-data training | samples={X.shape[0]} | L={X.shape[1]} | "
        f"H={Y.shape[1]} | F={X.shape[2]} | dt={dt:.3f}s"
    )
    train_and_predict_all(X, Y, phase_lbl, file_lbl, tnext, dt, feature_cols, version_dir, CFG)

    print("\nDONE.")
    print(f"Outputs saved to: {version_dir}")


if __name__ == "__main__":
    main()
