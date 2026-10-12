"""mlp_sr.py — SR fusion MLP: best physics (physics_model_sr) + best LSTM (v12).

Single consolidated script:
  - All trajectories in a local METER frame (LSTM v12 predicts lon/lat degrees +
    altitude; the WGS84 conversion of physics_model_sr.add_local_meter_frame is
    applied per flight so metrics are true 3D meters).
  - Physics input: per-sample physics-SR damped-trend-prior horizon (quadratic
    fit on the 10 s of real data before each anchor -> phase-scaled damped
    extrapolation; identical equations to physics_model_sr.simulate_trend_phase).
  - LSTM input: v12 test horizon artifacts (Lstm_sr_outputs/v12).
  - Fusion model: MatchedHorizonMLP architecture (H*12 -> 334 -> 167 -> H*3,
    dropout 0.10, discounted MSE 0.98^i, Adam 1e-3/wd 1e-5), trained as a
    RESIDUAL head on top of the LSTM horizon (--no-residual to disable).
  - Outputs versioned like the physics model (mlp_sr_outputs/v1, v2, ...) with
    the same evaluation files, plot types, and colors as physics_model_sr.

Usage:
  python3 mlp_sr.py --epochs 12          # train (resumes mlp_sr_outputs/checkpoint.npz)
  python3 mlp_sr.py --eval               # full evaluation + plots -> next vN folder
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import physics_model_sr as phy

# ----------------------------- configuration -----------------------------
LSTM_DIR = Path("Lstm_sr_outputs/v12")
DATA_DIR = Path("data/full_dataset")
OUTPUT_ROOT = Path("mlp_sr_outputs")
DT = 0.12
HORIZON = 250
WARMUP_S = 10.0
SEED = 42
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
DISCOUNT = 0.98
DROPOUT = 0.10
PATIENCE = 15
H1, H2 = 334, 167
PHASES = ("takeoff", "cruise", "landing")
MODEL_NAME = "mlp_sr_fusion"
RESIDUAL_BASE = "lstm"  # "lstm" (MLP-SR) or "physics" (PIML-2-SR: physics main, learned correction)
FEATURES = "joint"      # "joint" ([phy, lstm, phy-lstm, phy+lstm]) or "physics_only" (phy horizon only)
N_PLOT_CASES = 12   # sampled cases per phase for spaghetti/3D plots
N_ALL_CASES = 40    # sampled cases per phase for "all flights" plots
_CACHE: dict = {}


def comparison_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="black", linestyle="-", label="Real"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2.0, label="Predicted"),
    ]


def get_next_version_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for p in root.iterdir() if p.is_dir() and (m := re.fullmatch(r"v(\d+)", p.name))]
    out = root / f"v{max(nums) + 1 if nums else 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------- data pipeline -----------------------------
def meters_factors(lat0: float) -> tuple[float, float]:
    lr = math.radians(lat0)
    m_lat = 111_132.92 - 559.82 * math.cos(2 * lr) + 1.175 * math.cos(4 * lr)
    m_lon = 111_412.84 * math.cos(lr) - 93.5 * math.cos(3 * lr)
    return m_lon, m_lat


def trend_horizon(p0, v0, acc, phase, t_grid, prm) -> np.ndarray:
    v0 = v0.copy()
    acc = np.clip(acc, -prm.trend_accel_clip, prm.trend_accel_clip)
    if phase == "takeoff":
        v0[2] = float(np.clip(max(v0[2], prm.takeoff_vz_min), prm.takeoff_vz_min, prm.takeoff_vz_max))
        acc = acc * prm.takeoff_trend_accel_scale
    elif phase == "landing":
        v0[2] = float(np.clip(min(v0[2], prm.landing_vz_max), prm.landing_vz_min, prm.landing_vz_max))
        acc = acc * prm.landing_trend_accel_scale
    else:
        v0[2] = 0.0
        acc = acc * prm.cruise_trend_accel_scale
        acc[2] = 0.0
    tau = max(float(prm.trend_accel_tau_s), 1e-6)
    pos_integral = tau * (t_grid - tau * (1.0 - np.exp(-t_grid / tau)))
    return p0[None, :] + t_grid[:, None] * v0[None, :] + pos_integral[:, None] * acc[None, :]


def load_dataset(lstm_dir: Path = LSTM_DIR):
    """Meter-frame truth/LSTM horizons + per-sample physics-SR trend horizons (cached)."""
    key = str(lstm_dir)
    if key in _CACHE:
        return _CACHE[key]
    cache_file = OUTPUT_ROOT / "dataset_cache.npz"
    if cache_file.exists():
        c = np.load(cache_file, allow_pickle=True)
        out = (c["y_true"], c["y_lstm"], c["y_phy"], c["phase_lbl"].astype(str), c["file_lbl"].astype(str), c["t_start"])
        _CACHE[key] = out
        return out

    art = lstm_dir / "artifacts"
    y_true_deg = np.load(art / "Ytrue_test.npy").astype(np.float64)
    y_lstm_deg = np.load(art / "Ypred_test.npy").astype(np.float64)
    phase_lbl = np.array([str(p).strip().lower() for p in np.load(art / "Pte_test.npy", allow_pickle=True)])
    file_lbl = np.load(art / "Fte_test.npy", allow_pickle=True).astype(str)
    t_start = np.load(art / "Tte_test.npy").astype(float)

    t_grid = np.arange(HORIZON, dtype=float) * DT
    prm = phy.RealisticUAVParams()
    y_true = np.empty_like(y_true_deg, dtype=np.float32)
    y_lstm = np.empty_like(y_lstm_deg, dtype=np.float32)
    y_phy = np.empty_like(y_true)

    for fname in np.unique(file_lbl):
        df = pd.read_csv(DATA_DIR / fname)
        lon = df["position_x"].to_numpy(float)
        lat = df["position_y"].to_numpy(float)
        alt = df["position_z"].to_numpy(float)
        t = df["time"].to_numpy(float)
        m_lon, m_lat = meters_factors(float(lat[0]))
        lon0, lat0 = float(lon[0]), float(lat[0])
        fx, fy = (lon - lon0) * m_lon, (lat - lat0) * m_lat
        idx = np.where(file_lbl == fname)[0]
        for arr_deg, arr_m in ((y_true_deg, y_true), (y_lstm_deg, y_lstm)):
            arr_m[idx, :, 0] = (arr_deg[idx, :, 0] - lon0) * m_lon
            arr_m[idx, :, 1] = (arr_deg[idx, :, 1] - lat0) * m_lat
            arr_m[idx, :, 2] = arr_deg[idx, :, 2]
        for i in idx:
            t0 = t_start[i]
            m = (t >= t0 - WARMUP_S - 1e-9) & (t <= t0 + 1e-9)
            if m.sum() < 5:
                m = np.zeros(len(t), bool)
                m[np.argsort(np.abs(t - t0))[:8]] = True
            tw = t[m] - t[m][0]
            v0, acc = np.zeros(3), np.zeros(3)
            if np.ptp(tw) > 1e-6:
                for axis, series in enumerate((fx[m], fy[m], alt[m])):
                    co = np.polyfit(tw, series, 2)
                    v0[axis] = co[1] + 2.0 * co[0] * tw[-1]
                    acc[axis] = 2.0 * co[0]
            y_phy[i] = trend_horizon(y_true[i, 0].astype(float), v0, acc, phase_lbl[i], t_grid, prm).astype(np.float32)

    keep = np.isin(phase_lbl, PHASES)
    out = (y_true[keep], y_lstm[keep], y_phy[keep], phase_lbl[keep], file_lbl[keep], t_start[keep])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, y_true=out[0], y_lstm=out[1], y_phy=out[2],
                        phase_lbl=out[3], file_lbl=out[4], t_start=out[5])
    _CACHE[key] = out
    return out


def joint_features(y_phy, y_lstm) -> np.ndarray:
    if FEATURES == "physics_only":
        return y_phy.reshape(len(y_phy), -1).astype(np.float32)
    joint = np.concatenate([y_phy, y_lstm, y_phy - y_lstm, y_phy + y_lstm], axis=2)
    return joint.reshape(len(joint), -1).astype(np.float32)


def split_temporal(n: int):
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    return np.arange(0, n_tr), np.arange(n_tr, n_tr + n_va), np.arange(n_tr + n_va, n)


# ----------------------------- numpy MLP -----------------------------
class NumpyMLP:
    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator):
        def init(a, b):
            lim = np.sqrt(6.0 / (a + b))
            return rng.uniform(-lim, lim, size=(a, b)).astype(np.float32)

        self.params = {"W1": init(d_in, H1), "b1": np.zeros(H1, np.float32),
                       "W2": init(H1, H2), "b2": np.zeros(H2, np.float32),
                       "W3": init(H2, d_out), "b3": np.zeros(d_out, np.float32)}
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def forward(self, x, train=False, rng=None):
        p = self.params
        h1 = np.maximum(x @ p["W1"] + p["b1"], 0.0)
        if train:
            mask1 = (rng.random(h1.shape) >= DROPOUT).astype(np.float32) / (1 - DROPOUT)
            h1 *= mask1
        h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0.0)
        if train:
            mask2 = (rng.random(h2.shape) >= DROPOUT).astype(np.float32) / (1 - DROPOUT)
            h2 *= mask2
        out = h2 @ p["W3"] + p["b3"]
        return (out, (x, h1, h2, mask1, mask2)) if train else out

    def backward(self, grad_out, cache):
        x, h1, h2, mask1, mask2 = cache
        p = self.params
        g = {"W3": h2.T @ grad_out, "b3": grad_out.sum(0)}
        dh2 = (grad_out @ p["W3"].T) * (h2 > 0) * mask2
        g["W2"], g["b2"] = h1.T @ dh2, dh2.sum(0)
        dh1 = (dh2 @ p["W2"].T) * (h1 > 0) * mask1
        g["W1"], g["b1"] = x.T @ dh1, dh1.sum(0)
        return g

    def adam_step(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, p in self.params.items():
            gk = grads[k] + WEIGHT_DECAY * p
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            p -= LR * (self.m[k] / (1 - b1**self.t)) / (np.sqrt(self.v[k] / (1 - b2**self.t)) + eps)


def discounted_loss(pred_s, targ_s, w):
    d = (pred_s - targ_s).reshape(-1, HORIZON, 3)
    return float(np.mean(w * d * d)), (2.0 * w * d / d.size).reshape(len(d), -1)


# ----------------------------- training -----------------------------
def run_training(args) -> None:
    rng = np.random.default_rng(SEED)
    y_true, y_lstm, y_phy, phase_lbl, file_lbl, t_start = load_dataset(args.lstm_dir)
    x = joint_features(y_phy, y_lstm)
    residual = not args.no_residual
    base = y_lstm if RESIDUAL_BASE == "lstm" else y_phy
    y = ((y_true - base) if residual else y_true).reshape(len(y_true), -1)
    tr, va, te = split_temporal(len(x))
    w = (DISCOUNT ** np.arange(HORIZON, dtype=np.float32)).reshape(1, HORIZON, 1)

    ckpt_path = OUTPUT_ROOT / "checkpoint.npz"
    model = NumpyMLP(x.shape[1], y.shape[1], rng)
    history, best_val, best_epoch, epoch0 = [], np.inf, -1, 0
    if ckpt_path.exists():
        ck = np.load(ckpt_path, allow_pickle=True)
        for k in model.params:
            model.params[k] = ck[f"p_{k}"]
            model.m[k] = ck[f"m_{k}"]
            model.v[k] = ck[f"v_{k}"]
        model.t = int(ck["t"])
        xm, xs, ym, ys = ck["xm"], ck["xs"], ck["ym"], ck["ys"]
        history = list(ck["history"])
        best_val, best_epoch, epoch0 = float(ck["best_val"]), int(ck["best_epoch"]), int(ck["epoch"])
        best_params = {k: ck[f"best_{k}"] for k in model.params}
        print(f"resumed at epoch {epoch0}, best val {best_val:.6f} @ {best_epoch}")
    else:
        xm, xs = x[tr].mean(0), x[tr].std(0) + 1e-8
        ym, ys = y[tr].mean(0), y[tr].std(0) + 1e-8
        best_params = {k: v.copy() for k, v in model.params.items()}

    xs_all = (x - xm) / xs
    ys_all = (y - ym) / ys

    stop = False
    for epoch in range(epoch0, epoch0 + args.epochs):
        t0 = time.time()
        order = rng.permutation(tr)
        tr_loss, nb = 0.0, 0
        for i in range(0, len(order), BATCH):
            bi = order[i : i + BATCH]
            pred, cache = model.forward(xs_all[bi], train=True, rng=rng)
            loss, grad = discounted_loss(pred, ys_all[bi], w)
            model.adam_step(model.backward(grad.astype(np.float32), cache))
            tr_loss += loss
            nb += 1
        val_loss, _ = discounted_loss(model.forward(xs_all[va]), ys_all[va], w)
        history.append({"epoch": epoch, "train_loss": tr_loss / nb, "val_loss": val_loss})
        improved = val_loss < best_val - 1e-7
        if improved:
            best_val, best_epoch = val_loss, epoch
            best_params = {k: v.copy() for k, v in model.params.items()}
        print(f"epoch {epoch}: train {tr_loss/nb:.6f} val {val_loss:.6f}{' *' if improved else ''} ({time.time()-t0:.1f}s)")
        if epoch - best_epoch >= PATIENCE:
            print(f"early stop: no val improvement for {PATIENCE} epochs")
            stop = True
            break

    save = {"t": model.t, "epoch": history[-1]["epoch"] + 1 if history else 0, "best_val": best_val,
            "best_epoch": best_epoch, "xm": xm, "xs": xs, "ym": ym, "ys": ys,
            "history": np.array(history, dtype=object), "converged": stop, "residual": residual}
    for k in model.params:
        save[f"p_{k}"] = model.params[k]
        save[f"m_{k}"] = model.m[k]
        save[f"v_{k}"] = model.v[k]
        save[f"best_{k}"] = best_params[k]
    np.savez_compressed(ckpt_path, **save)
    print(f"checkpoint saved (epoch {save['epoch']}, best val {best_val:.6f} @ {best_epoch}, converged={stop})")


# ----------------------------- evaluation -----------------------------
def predict_test(x, te, y_base, args):
    ck = np.load(OUTPUT_ROOT / "checkpoint.npz", allow_pickle=True)
    residual = bool(ck["residual"]) if "residual" in ck.files else (not args.no_residual)
    model = NumpyMLP(x.shape[1], HORIZON * 3, np.random.default_rng(SEED))
    for k in model.params:
        model.params[k] = ck[f"best_{k}"]
    xm, xs, ym, ys = ck["xm"], ck["xs"], ck["ym"], ck["ys"]
    preds = []
    for i in range(0, len(te), 512):
        bi = te[i : i + 512]
        preds.append(model.forward((x[bi] - xm) / xs) * ys + ym)
    y_mlp = np.concatenate(preds).reshape(-1, HORIZON, 3).astype(np.float32)
    if residual:
        y_mlp = y_mlp + y_base[te]
    return y_mlp, ck, residual


def sample_indices(rmse_arr, ph, per_phase):
    """Evenly-spread (by RMSE) case indices per phase."""
    out = {}
    for phase in PHASES:
        m = np.where(ph == phase)[0]
        order = m[np.argsort(rmse_arr[m])]
        take = np.unique(np.linspace(0, len(order) - 1, min(per_phase, len(order))).astype(int))
        out[phase] = order[take]
    return out


def run_evaluation(args) -> None:
    out_dir = get_next_version_dir(OUTPUT_ROOT)
    y_true, y_lstm, y_phy, phase_lbl, file_lbl, t_start = load_dataset(args.lstm_dir)
    x = joint_features(y_phy, y_lstm)
    tr, va, te = split_temporal(len(x))
    y_base = y_lstm if RESIDUAL_BASE == "lstm" else y_phy
    y_mlp, ck, residual = predict_test(x, te, y_base, args)

    yt, yl, yp = y_true[te], y_lstm[te], y_phy[te]
    ph, fl = phase_lbl[te], file_lbl[te]
    t_grid = np.arange(HORIZON) * DT

    # velocities via finite differences
    v_real = np.gradient(yt, DT, axis=1)
    v_pred = np.gradient(y_mlp, DT, axis=1)

    err = y_mlp - yt
    dist = np.linalg.norm(err, axis=2)
    vel_err = np.linalg.norm(v_pred - v_real, axis=2)
    rmse_arr = np.sqrt((dist**2).mean(1))

    # ---------------- detail / summary CSVs (physics naming) ----------------
    detail = pd.DataFrame({
        "csv": fl, "phase": ph, "t_start": t_start[te], "n_points": HORIZON,
        "rmse_3d_m": rmse_arr, "mae_3d_m": dist.mean(1),
        "rmse_x_m": np.sqrt((err[:, :, 0] ** 2).mean(1)),
        "rmse_y_m": np.sqrt((err[:, :, 1] ** 2).mean(1)),
        "rmse_z_m": np.sqrt((err[:, :, 2] ** 2).mean(1)),
        "final_error_m": dist[:, -1],
        "rmse_velocity_mps": np.sqrt((vel_err**2).mean(1)),
    })
    detail.to_csv(out_dir / "mlp_model_detail_metrics.csv", index=False)

    summary = (
        detail.groupby("phase", as_index=False)
        .agg(cases=("rmse_3d_m", "count"), mean_rmse_3d_m=("rmse_3d_m", "mean"),
             median_rmse_3d_m=("rmse_3d_m", "median"), mean_mae_3d_m=("mae_3d_m", "mean"),
             mean_final_error_m=("final_error_m", "mean"))
        .sort_values("mean_rmse_3d_m")
    )
    summary.to_csv(out_dir / "mlp_model_phase_summary.csv", index=False)
    model_summary = pd.DataFrame([{
        "model": MODEL_NAME, "cases": len(detail),
        "mean_rmse_3d_m": detail["rmse_3d_m"].mean(), "median_rmse_3d_m": detail["rmse_3d_m"].median(),
        "mean_mae_3d_m": detail["mae_3d_m"].mean(), "mean_final_error_m": detail["final_error_m"].mean(),
    }])
    model_summary.to_csv(out_dir / "mlp_model_summary.csv", index=False)

    # baseline comparison summary (extra, kept from before)
    comp_rows = []
    for name, pred in {MODEL_NAME: y_mlp, "lstm_v12": yl, "physics_sr_prior": yp}.items():
        d = np.linalg.norm(pred - yt, axis=2)
        r = np.sqrt((d**2).mean(1))
        comp_rows.append({"model": name, "cases": len(r), "mean_rmse_3d_m": r.mean(),
                          "median_rmse_3d_m": np.median(r), "mean_mae_3d_m": d.mean(),
                          "mean_final_error_m": d[:, -1].mean()})
    pd.DataFrame(comp_rows).to_csv(out_dir / "mlp_model_comparison_summary.csv", index=False)

    # ---------------- phase RMSE bar (physics plot_summary style) ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    s2 = summary.set_index("phase").reindex(PHASES).reset_index()
    ax.bar(s2["phase"], s2["mean_rmse_3d_m"], color=["#238b45", "#386cb0", "#d95f02"])
    ax.set_ylabel("Mean RMSE 3D (m)")
    ax.set_title("MLP Model by Phase")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "mlp_model_phase_rmse_bar.png", dpi=220)
    plt.close(fig)

    # ---------------- representative phase position grid ----------------
    class_legend = comparison_legend_handles()
    rep_rows = []
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        m = np.where(ph == phase)[0]
        med = float(np.median(rmse_arr[m]))
        rep = m[np.argmin(np.abs(rmse_arr[m] - med))]
        rep_rows.append({"phase": phase, "csv": fl[rep], "rmse_3d_m": float(rmse_arr[rep]),
                         "mae_3d_m": float(dist[rep].mean()), "final_error_m": float(dist[rep, -1]),
                         "n_points": HORIZON})
        for j, k in enumerate(range(3)):
            ax = axs[i, j]
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(["X", "Y", "Z"][j] + " (m)")
            ax.set_title(f"{phase.capitalize()} {['X', 'Y', 'Z'][j]}")
            ax.plot(t_grid, yt[rep, :, k], "-", color="#1f77b4", alpha=0.9)
            ax.plot(t_grid, y_mlp[rep, :, k], ":", color="#d62728", alpha=0.95, linewidth=2.0)
            if i == 0 and j == 0:
                ax.legend(handles=class_legend, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "real_vs_predicted_position_over_time_representative_phase.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(rep_rows).to_csv(out_dir / "representative_phase_cases.csv", index=False)

    # ---------------- sampled case sets + per-flight colors ----------------
    plot_sets = sample_indices(rmse_arr, ph, N_PLOT_CASES)
    all_sets = sample_indices(rmse_arr, ph, N_ALL_CASES)
    all_idx = np.concatenate(list(all_sets.values()))
    unique_labels = sorted(set(fl))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    color_map = {label: cmap(i % cmap.N) for i, label in enumerate(unique_labels)}

    # ---------------- real vs predicted 3D trajectory (1x3) ----------------
    fig = plt.figure(figsize=(15, 10))
    for i, phase in enumerate(PHASES, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for c in plot_sets[phase]:
            color = color_map[fl[c]]
            ax.plot(yt[c, :, 0], yt[c, :, 1], yt[c, :, 2], "-", alpha=0.35, color=color)
            ax.plot(y_mlp[c, :, 0], y_mlp[c, :, 1], y_mlp[c, :, 2], ":", alpha=0.80, color=color, linewidth=1.6)
        ax.set_xlabel("X local (m)")
        ax.set_ylabel("Y local (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{phase.capitalize()}")
        if i == 1:
            ax.legend(handles=class_legend, loc="best")
    fig.suptitle("Real vs Predicted 3D Trajectory")
    fig.tight_layout()
    fig.savefig(out_dir / "real_vs_predicted_3d_trajectory.png", dpi=220)
    plt.close(fig)

    # ---------------- real vs predicted position over time (3x3) ----------------
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for c in plot_sets[phase]:
            for j in range(3):
                axs[i, j].plot(t_grid, yt[c, :, j], "-", alpha=0.35)
                axs[i, j].plot(t_grid, y_mlp[c, :, j], ":", alpha=0.85, linewidth=1.6)
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["X", "Y", "Z"][j] + " (m)")
                axs[i, j].grid(True, alpha=0.25)
                if i == 0 and j == 0:
                    axs[i, j].legend(handles=class_legend, loc="best")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Real vs Predicted Position Over Time")
    fig.tight_layout()
    fig.savefig(out_dir / "real_vs_predicted_position_over_time.png", dpi=220)
    plt.close(fig)

    # ---------------- real vs predicted velocity over time (3x3) ----------------
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for c in plot_sets[phase]:
            for j in range(3):
                axs[i, j].plot(t_grid, v_real[c, :, j], "-", alpha=0.35)
                axs[i, j].plot(t_grid, v_pred[c, :, j], ":", alpha=0.85, linewidth=1.6)
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["Vx", "Vy", "Vz"][j] + " (m/s)")
                axs[i, j].grid(True, alpha=0.25)
                if i == 0 and j == 0:
                    axs[i, j].legend(handles=class_legend, loc="best")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Real vs Predicted Velocity Over Time")
    fig.tight_layout()
    fig.savefig(out_dir / "real_vs_predicted_velocity_over_time.png", dpi=220)
    plt.close(fig)

    # ---------------- per-phase 3D trajectories ----------------
    for phase in PHASES:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for c in plot_sets[phase]:
            ax.plot(yt[c, :, 0], yt[c, :, 1], yt[c, :, 2], "-", alpha=0.35)
            ax.plot(y_mlp[c, :, 0], y_mlp[c, :, 1], y_mlp[c, :, 2], ":", alpha=0.75, linewidth=1.6)
        ax.set_xlabel("X local (m)")
        ax.set_ylabel("Y local (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"3D Trajectory: {phase.capitalize()} - all real flights")
        ax.legend(handles=class_legend, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"3d_trajectory_{phase}.png", dpi=220)
        plt.close(fig)

    # ---------------- position / velocity error over time (3x3) ----------------
    for cols, ylab, fname in [((0, 1, 2), ["X", "Y", "Z"], "position_error_over_time.png"),
                              ((0, 1, 2), ["Vx", "Vy", "Vz"], "velocity_error_over_time.png")]:
        src = err if fname.startswith("position") else (v_pred - v_real)
        unit = "(m)" if fname.startswith("position") else "(m/s)"
        fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
        for i, phase in enumerate(PHASES):
            for c in plot_sets[phase]:
                for j in range(3):
                    axs[i, j].plot(t_grid, src[c, :, j], alpha=0.65)
                    axs[i, j].set_title(f"{phase.capitalize()}")
                    axs[i, j].set_ylabel(f"{ylab[j]} Error {unit}")
                    axs[i, j].grid(True, alpha=0.25)
        for ax in axs.flat:
            ax.set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=220)
        plt.close(fig)

    # ---------------- absolute position over time (3x3) ----------------
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for c in plot_sets[phase]:
            for j in range(3):
                axs[i, j].plot(t_grid, y_mlp[c, :, j], ":", alpha=0.5)
                axs[i, j].plot(t_grid, yt[c, :, j], "-", alpha=0.4)
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["X", "Y", "Z"][j] + " (m)")
                axs[i, j].grid(True, alpha=0.25)
                if i == 0 and j == 0:
                    axs[i, j].legend(handles=class_legend, loc="best")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "absolute_position_over_time.png", dpi=220)
    plt.close(fig)

    # ---------------- phase comparison bar chart ----------------
    phase_metrics = (
        detail.groupby("phase", as_index=False)
        .agg(RMSE_3D=("rmse_3d_m", "mean"), MAE_3D=("mae_3d_m", "mean"),
             FinalErr=("final_error_m", "mean"), RMSE_Vel=("rmse_velocity_mps", "mean"))
    )
    metrics = ["RMSE_3D", "MAE_3D", "FinalErr", "RMSE_Vel"]
    xpos = np.arange(len(PHASES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, metric in enumerate(metrics):
        vals = [float(phase_metrics.loc[phase_metrics["phase"] == p, metric].iloc[0]) for p in PHASES]
        ax.bar(xpos + idx * width, vals, width, label=metric)
    ax.set_xticks(xpos + width * 1.5)
    ax.set_xticklabels([p.capitalize() for p in PHASES])
    ax.set_ylabel("Error")
    ax.set_title("Phase Comparison: MLP Model Key Metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_comparison_bar_chart.png", dpi=220)
    plt.close(fig)

    # ---------------- RMSE distributions ----------------
    fig, ax = plt.subplots(figsize=(12, 6))
    detail.boxplot(column="rmse_3d_m", by="phase", ax=ax)
    ax.set_title("RMSE 3D Distribution by Phase")
    ax.set_ylabel("RMSE 3D (m)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_distribution_by_phase.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    detail.boxplot(column="rmse_3d_m", by="phase", ax=ax)
    ax.set_title("MLP Model RMSE Distribution by Phase")
    ax.set_ylabel("RMSE 3D (m)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_dir / "mlp_model_rmse_distribution_boxplot.png", dpi=220)
    plt.close(fig)

    # ---------------- per-flight bars ----------------
    grp = detail.groupby(["phase", "csv"], as_index=False).agg(
        rmse_3d_m=("rmse_3d_m", "mean"), final_error_m=("final_error_m", "mean"))
    grp = grp.sort_values(["phase", "csv"])
    labels = [f"{r.phase[:1].upper()}-{r.csv.replace('flights_', '').replace('.csv', '')}" for r in grp.itertuples()]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(labels, grp["rmse_3d_m"], color="#386cb0")
    ax.set_ylabel("RMSE 3D (m)")
    ax.set_title("Per-flight Validation RMSE")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "per_flight_rmse_bar.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(labels, grp["final_error_m"], color="#b7372f")
    ax.set_ylabel("Final Error (m)")
    ax.set_title("Per-flight Final Prediction Error")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "per_flight_final_error_bar.png", dpi=220)
    plt.close(fig)

    # ---------------- all flights spaghetti plots ----------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for c in all_idx:
        for ax, j, label in zip(axes, range(3), ["X Error (m)", "Y Error (m)", "Z Error (m)"]):
            ax.plot(t_grid, err[c, :, j], alpha=0.18, linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
    axes[0].set_title("All Flights: Position Error Over Time")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "all_flights_position_error_over_time.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    verr = v_pred - v_real
    for c in all_idx:
        for ax, j, label in zip(axes, range(3), ["Vx Error (m/s)", "Vy Error (m/s)", "Vz Error (m/s)"]):
            ax.plot(t_grid, verr[c, :, j], alpha=0.18, linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
    axes[0].set_title("All Flights: Velocity Error Over Time")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "all_flights_velocity_error_over_time.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for c in all_idx:
        ax.plot(t_grid, dist[c], alpha=0.18, linewidth=0.8)
    ax.set_title("All Flights: 3D Position Error Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("3D Error (m)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "all_flights_3d_error_over_time.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for c in all_idx:
        color = color_map[fl[c]]
        for ax, j, label in zip(axes, range(3), ["X (m)", "Y (m)", "Z (m)"]):
            ax.plot(t_grid, yt[c, :, j], "-", alpha=0.18, color=color, linewidth=0.9)
            ax.plot(t_grid, y_mlp[c, :, j], ":", alpha=0.45, color=color, linewidth=1.1)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
    axes[0].set_title("All Flights: Real Solid vs Predicted Dotted Position")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "all_flights_real_vs_predicted_position_over_time.png", dpi=220)
    plt.close(fig)

    # ---------------- euclidean error over time (physics v2 style) ----------------
    mean_all, std_all = dist.mean(0), dist.std(0)
    pd.DataFrame({"time_s": t_grid, "mean_euclidean_error_m": mean_all, "std_euclidean_error_m": std_all,
                  "n_cases": len(dist)}).to_csv(out_dir / "mlp_mean_euclidean_error_over_time.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_grid, mean_all, color="tab:blue", linewidth=2.0, label=f"Mean (n={len(dist)} cases)")
    ax.fill_between(t_grid, np.maximum(mean_all - std_all, 0.0), mean_all + std_all, color="tab:blue", alpha=0.2, label="±1 std")
    ax.set_xlabel("Prediction time (s)")
    ax.set_ylabel("Euclidean position error (m)")
    ax.set_title("Mean Euclidean Error over 30 s Prediction Horizon")
    ax.set_xlim(0, HORIZON * DT)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mlp_mean_euclidean_error_over_time.png", dpi=220)
    plt.close(fig)

    raw_rows = []
    for c in all_idx:
        for ti, ei in zip(t_grid, dist[c]):
            raw_rows.append({"time_s": ti, "csv": fl[c], "phase": ph[c], "euclidean_error_m": ei})
    pd.DataFrame(raw_rows).to_csv(out_dir / "mlp_euclidean_error_over_time_raw.csv", index=False)

    phase_cols = {"takeoff": "tab:green", "cruise": "tab:blue", "landing": "tab:red"}
    fig, ax = plt.subplots(figsize=(10, 6))
    pw = {"time_s": t_grid}
    for phase in PHASES:
        m = ph == phase
        mp, sp = dist[m].mean(0), dist[m].std(0)
        pw[f"{phase}_mean_error_m"], pw[f"{phase}_std_error_m"] = mp, sp
        ax.plot(t_grid, mp, color=phase_cols[phase], linewidth=2.0, label=f"{phase} (n={m.sum()})")
        ax.fill_between(t_grid, np.maximum(mp - sp, 0.0), mp + sp, color=phase_cols[phase], alpha=0.12)
    ax.set_xlabel("Prediction time (s)")
    ax.set_ylabel("Euclidean position error (m)")
    ax.set_title("Phase-wise Mean Euclidean Error over 30 s Prediction Horizon")
    ax.set_xlim(0, HORIZON * DT)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase_wise_euclidean_error_over_time.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(pw).to_csv(out_dir / "phase_wise_euclidean_error_over_time.csv", index=False)

    # model comparison error over time (extra, kept)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, pred in {MODEL_NAME: y_mlp, "lstm_v12": yl, "physics_sr_prior": yp}.items():
        ax.plot(t_grid, np.linalg.norm(pred - yt, axis=2).mean(0), linewidth=2.0, label=name)
    ax.set_xlabel("Prediction time (s)")
    ax.set_ylabel("Mean Euclidean position error (m)")
    ax.set_title(f"Mean Euclidean Error over 30 s — MLP-SR vs LSTM v12 vs physics prior (n={len(te)})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "euclidean_error_over_time_model_comparison.png", dpi=220)
    plt.close(fig)

    # ---------------- phase mean predicted trajectories ----------------
    for phase in PHASES:
        m = ph == phase
        pd.DataFrame({"time": t_grid, "x": y_mlp[m, :, 0].mean(0), "y": y_mlp[m, :, 1].mean(0),
                      "z": y_mlp[m, :, 2].mean(0)}).to_csv(out_dir / f"{phase}_predicted_mean_trajectory_v6.csv", index=False)

    # ---------------- training curve ----------------
    hist = pd.DataFrame(list(ck["history"]))
    if len(hist):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(hist["epoch"], hist["train_loss"], label="train")
        ax.plot(hist["epoch"], hist["val_loss"], label="validation")
        ax.axvline(int(ck["best_epoch"]), color="gray", linestyle="--", linewidth=1, label=f"best epoch {int(ck['best_epoch'])}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Discounted MSE (scaled)")
        ax.set_yscale("log")
        ax.set_title("MLP-SR training curve")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_curve.png", dpi=220)
        plt.close(fig)
        hist.to_csv(out_dir / "training_history.csv", index=False)

    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "model": MODEL_NAME,
                   "residual_head": residual, "residual_base": RESIDUAL_BASE,
                   "units": "meters (local WGS84 frame per flight)",
                   "physics_input": "per-sample physics-SR damped-trend prior (10 s warmup)",
                   "lstm_source": str(args.lstm_dir), "data_dir": str(DATA_DIR),
                   "horizon_steps": HORIZON, "dt_seconds": DT, "test_samples": int(len(te)),
                   "plot_cases_per_phase": N_PLOT_CASES, "all_flights_cases_per_phase": N_ALL_CASES,
                   "best_epoch": int(ck["best_epoch"]), "best_val_loss": float(ck["best_val"]),
                   "architecture": f"{x.shape[1]}-{H1}-{H2}-{HORIZON*3}", "features": FEATURES,
                   "split": "temporal 70/15/15 (PIML_2 protocol)"}, f, indent=2)

    print("=== MLP Model Summary ===")
    print(model_summary.round(4).to_string(index=False))
    print("\n=== MLP Model Phase Summary ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SR fusion MLP (numpy): train / evaluate with versioned outputs.")
    ap.add_argument("--lstm-dir", type=Path, default=LSTM_DIR)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--no-residual", action="store_true", help="Predict full trajectory instead of LSTM residual.")
    args = ap.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.eval:
        run_evaluation(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
