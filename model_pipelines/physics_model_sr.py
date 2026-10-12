"""physics_model_sr.py — SR configuration of the realistic UAV physics model.

Derived from Physics_model.py (v20 best configuration) with physically
realistic parameters for a 3.6 kg multirotor:
  - mass_kg        = 3.6
  - max_thrust_n   = 88.29   (thrust-to-weight ~ 2.5)
  - area_xy_m2     = 0.12    (scaled frontal area for a 3.6 kg airframe)
  - area_z_m2      = 0.18
  - cd_xy / cd_z   = 0.90 / 1.10  (bluff-body multirotor drag coefficients)
  - damped-trend gray-box prior ENABLED by default (v20 improvement)
  - prediction horizon = 30 s, dt = 0.1 s
Falls back to a pure-numpy rotation implementation if scipy is unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
try:
    from scipy.spatial.transform import Rotation as R
except ModuleNotFoundError:
    class _NumpyRotation:
        """Minimal numpy stand-in for scipy.spatial.transform.Rotation (xyzw quaternions)."""

        def __init__(self, q):
            import numpy as _np
            q = _np.asarray(q, dtype=float)
            self._q = q / _np.linalg.norm(q)

        @classmethod
        def from_quat(cls, q):
            return cls(q)

        @classmethod
        def from_rotvec(cls, rv):
            import numpy as _np, math as _m
            rv = _np.asarray(rv, dtype=float)
            angle = float(_np.linalg.norm(rv))
            if angle < 1e-12:
                return cls(_np.array([0.0, 0.0, 0.0, 1.0]))
            axis = rv / angle
            s = _m.sin(angle / 2.0)
            return cls(_np.array([axis[0] * s, axis[1] * s, axis[2] * s, _m.cos(angle / 2.0)]))

        @classmethod
        def from_matrix(cls, m):
            import numpy as _np, math as _m
            m = _np.asarray(m, dtype=float)
            t = float(_np.trace(m))
            if t > 0:
                s = _m.sqrt(t + 1.0) * 2
                qw = 0.25 * s
                qx = (m[2, 1] - m[1, 2]) / s
                qy = (m[0, 2] - m[2, 0]) / s
                qz = (m[1, 0] - m[0, 1]) / s
            elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = _m.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
                qw = (m[2, 1] - m[1, 2]) / s
                qx = 0.25 * s
                qy = (m[0, 1] + m[1, 0]) / s
                qz = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = _m.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
                qw = (m[0, 2] - m[2, 0]) / s
                qx = (m[0, 1] + m[1, 0]) / s
                qy = 0.25 * s
                qz = (m[1, 2] + m[2, 1]) / s
            else:
                s = _m.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
                qw = (m[1, 0] - m[0, 1]) / s
                qx = (m[0, 2] + m[2, 0]) / s
                qy = (m[1, 2] + m[2, 1]) / s
                qz = 0.25 * s
            import numpy as _np2
            return cls(_np2.array([qx, qy, qz, qw]))

        def as_quat(self):
            return self._q.copy()

        def as_matrix(self):
            import numpy as _np
            x, y, z, w = self._q
            return _np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )

        def as_rotvec(self):
            import numpy as _np, math as _m
            x, y, z, w = self._q
            if w < 0:
                x, y, z, w = -x, -y, -z, -w
            n = _m.sqrt(x * x + y * y + z * z)
            if n < 1e-12:
                return _np.zeros(3)
            angle = 2.0 * _m.atan2(n, w)
            return _np.array([x, y, z]) / n * angle

        def as_euler(self, seq, degrees=False):
            import numpy as _np, math as _m
            if seq != "zyx":
                raise NotImplementedError(seq)
            x, y, z, w = self._q
            yaw = _m.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            sinp = 2 * (w * y - z * x)
            pitch = _m.copysign(_m.pi / 2, sinp) if abs(sinp) >= 1 else _m.asin(sinp)
            roll = _m.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            e = _np.array([yaw, pitch, roll])
            return _np.degrees(e) if degrees else e

        def inv(self):
            import numpy as _np
            x, y, z, w = self._q
            return _NumpyRotation(_np.array([-x, -y, -z, w]))

        def magnitude(self):
            import numpy as _np
            return float(_np.linalg.norm(self.as_rotvec()))

        def __mul__(self, other):
            import numpy as _np
            x1, y1, z1, w1 = self._q
            x2, y2, z2, w2 = other._q
            return _NumpyRotation(
                _np.array(
                    [
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    ]
                )
            )

    R = _NumpyRotation


PHASES = ("takeoff", "cruise", "landing")
MODEL_NAME = "physics_model_sr_3p6kg_realistic"
DEFAULT_DATA_DIR = Path("data/physics_test/test_top10") 
DEFAULT_OUTPUT_DIR = Path("physics_model_sr_outputs")


@dataclass
class RealisticUAVParams:
    mass_kg: float = 3.6
    gravity: float = 9.81
    rho_air: float = 1.225
    area_xy_m2: float = 0.12  # realistic frontal area for 3.6 kg airframe
    area_z_m2: float = 0.18
    cd_xy: float = 0.90
    cd_z: float = 1.10
    max_thrust_n: float = 88.29  # thrust-to-weight ~ 2.5
    max_tilt_deg: float = 35.0
    attitude_tau_s: float = 0.30
    velocity_tau_s: float = 4.0
    altitude_kp: float = 0.22
    altitude_kd: float = 0.85
    horizontal_drag_gain: float = 0.015
    wind_gain: float = 0.25
    takeoff_vz_min: float = 0.5
    takeoff_vz_max: float = 2.4
    landing_vz_min: float = -2.3
    landing_vz_max: float = -0.35
    accel_limit_xy: float = 1.8
    accel_limit_z: float = 1.4
    max_integration_accel: float = 2.5
    takeoff_velocity_tau_s: float = 4.0
    cruise_velocity_tau_s: float = 4.0
    landing_velocity_tau_s: float = 3.0
    takeoff_altitude_gain: float = 0.08
    landing_altitude_gain: float = 0.18
    use_trend_prior: bool = True  # v20 best-config improvement, on by default
    trend_warmup_s: float = 10.0
    trend_accel_tau_s: float = 40.0
    cruise_trend_accel_scale: float = 0.10
    takeoff_trend_accel_scale: float = 0.30
    landing_trend_accel_scale: float = 0.20
    trend_accel_clip: float = 0.40
    use_6dof: bool = False
    inertia_xx_kg_m2: float = 0.082
    inertia_yy_kg_m2: float = 0.082
    inertia_zz_kg_m2: float = 0.145
    angular_damping_nms: float = 0.18
    angular_rate_tau_s: float = 1.5


def create_versioned_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    versions = []
    for path in base_dir.iterdir():
        if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit():
            versions.append(int(path.name[1:]))
    next_version = 1 if not versions else max(versions) + 1
    out_dir = base_dir / f"v{next_version}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def safe_label_from_path(csv_path: Path, data_dir: Path) -> str:
    try:
        label = str(csv_path.relative_to(data_dir))
    except ValueError:
        label = csv_path.name
    return label.replace("\\", "/")


def safe_stem_from_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {
        "position_x": "x_position",
        "position_y": "y_position",
        "position_z": "z_position",
        "orientation_x": "x_orientation",
        "orientation_y": "y_orientation",
        "orientation_z": "z_orientation",
        "orientation_w": "w",
        "velocity_x": "x_twist_linear",
        "velocity_y": "y_twist_linear",
        "velocity_z": "z_twist_linear",
        "angular_x": "x_twist_angular",
        "angular_y": "y_twist_angular",
        "angular_z": "z_twist_angular",
        "linear_acceleration_x": "x_linear_acceleration",
        "linear_acceleration_y": "y_linear_acceleration",
        "linear_acceleration_z": "z_linear_acceleration",
    }
    for src, dst in aliases.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)
    df["phase"] = df["phase"].astype(str).str.strip().str.lower()
    return df.sort_values("time").reset_index(drop=True)


def add_local_meter_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    lon = df["x_position"].to_numpy(dtype=float)
    lat = df["y_position"].to_numpy(dtype=float)
    df["source_x_position"] = df["x_position"]
    df["source_y_position"] = df["y_position"]

    if np.nanmax(np.abs(lon)) > 180.0 or np.nanmax(np.abs(lat)) > 90.0:
        return df

    lon0 = float(lon[0])
    lat0 = float(lat[0])
    lat0_rad = math.radians(lat0)
    meters_per_deg_lat = 111_132.92 - 559.82 * math.cos(2 * lat0_rad) + 1.175 * math.cos(4 * lat0_rad)
    meters_per_deg_lon = 111_412.84 * math.cos(lat0_rad) - 93.5 * math.cos(3 * lat0_rad)
    df["x_position"] = (lon - lon0) * meters_per_deg_lon
    df["y_position"] = (lat - lat0) * meters_per_deg_lat
    return df


def get_phase_segment(df: pd.DataFrame, phase: str, horizon_s: float) -> pd.DataFrame | None:
    phase_df = df[df["phase"] == phase].copy().reset_index(drop=True)
    if phase_df.empty:
        return None
    t0 = float(phase_df["time"].iloc[0])
    seg = phase_df[phase_df["time"] <= t0 + horizon_s + 1e-9].copy().reset_index(drop=True)
    if len(seg) < 3:
        return None
    seg["time_rel"] = seg["time"] - t0
    return seg


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 4:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if not np.all(np.isfinite(q)):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def comparison_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="black", linestyle="-", label="Real"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2.0, label="Predicted"),
    ]


def slerp_quat(q_current: np.ndarray, q_target: np.ndarray, alpha: float) -> np.ndarray:
    r_current = R.from_quat(normalize_quat_xyzw(q_current))
    r_target = R.from_quat(normalize_quat_xyzw(q_target))
    r_delta = r_current.inv() * r_target
    r_step = R.from_rotvec(np.clip(alpha, 0.0, 1.0) * r_delta.as_rotvec())
    return normalize_quat_xyzw((r_current * r_step).as_quat())


def integrate_rigid_body_attitude(
    q_xyzw: np.ndarray,
    omega_body: np.ndarray,
    target_omega_body: np.ndarray,
    params: RealisticUAVParams,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance quaternion and body rates with Euler's rigid-body equations."""
    inertia = np.array(
        [params.inertia_xx_kg_m2, params.inertia_yy_kg_m2, params.inertia_zz_kg_m2],
        dtype=float,
    )
    inertia = np.maximum(inertia, 1e-6)
    omega = np.nan_to_num(np.asarray(omega_body, dtype=float), nan=0.0)
    target = np.nan_to_num(np.asarray(target_omega_body, dtype=float), nan=0.0)
    angular_momentum = inertia * omega
    control_torque = inertia * (target - omega) / max(params.angular_rate_tau_s, dt)
    damping_torque = -params.angular_damping_nms * omega
    omega_dot = (control_torque + damping_torque - np.cross(omega, angular_momentum)) / inertia
    omega_next = np.clip(omega + omega_dot * dt, -math.radians(240.0), math.radians(240.0))
    dq = R.from_rotvec(0.5 * (omega + omega_next) * dt)
    q_next = normalize_quat_xyzw((R.from_quat(normalize_quat_xyzw(q_xyzw)) * dq).as_quat())
    return q_next, omega_next


def initial_body_rate(seg: pd.DataFrame, warmup_s: float) -> np.ndarray:
    cols = ["x_twist_angular", "y_twist_angular", "z_twist_angular"]
    if not all(col in seg.columns for col in cols):
        return np.zeros(3, dtype=float)
    warm = seg[seg["time_rel"] <= warmup_s]
    if warm.empty:
        warm = seg.iloc[:1]
    rate = np.nanmedian(warm[cols].to_numpy(dtype=float), axis=0)
    return np.clip(np.nan_to_num(rate, nan=0.0), -math.radians(240.0), math.radians(240.0))


def quat_from_body_z_and_yaw(body_z_world: np.ndarray, yaw_rad: float) -> np.ndarray:
    z_b = body_z_world / max(np.linalg.norm(body_z_world), 1e-9)
    x_heading = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    y_b = np.cross(z_b, x_heading)
    if np.linalg.norm(y_b) < 1e-6:
        y_b = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0])
    y_b = y_b / np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    rot = np.column_stack([x_b, y_b, z_b])
    return normalize_quat_xyzw(R.from_matrix(rot).as_quat())


def initial_yaw_from_quat(q_xyzw: np.ndarray) -> float:
    return float(R.from_quat(normalize_quat_xyzw(q_xyzw)).as_euler("zyx")[0])


def wind_vector(row: pd.Series, params: RealisticUAVParams) -> np.ndarray:
    speed = float(row.get("wind_speed", 0.0)) * params.wind_gain
    angle = math.radians(float(row.get("wind_angle", 0.0)))
    return np.array([speed * math.cos(angle), speed * math.sin(angle), 0.0])


def warm_start(seg: pd.DataFrame, warmup_s: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    warm = seg[seg["time_rel"] <= warmup_s].copy()
    if len(warm) < 3:
        warm = seg.iloc[: min(6, len(seg))].copy()
    t = warm["time_rel"].to_numpy(dtype=float)
    xyz = warm[["x_position", "y_position", "z_position"]].to_numpy(dtype=float)
    vel = warm[["x_twist_linear", "y_twist_linear", "z_twist_linear"]].to_numpy(dtype=float)
    v_est = np.nanmean(vel, axis=0)
    a_est = np.zeros(3)
    if len(warm) >= 3 and np.ptp(t) > 1e-6:
        v_fit = []
        for axis in range(3):
            v_fit.append(float(np.polyfit(t, xyz[:, axis], 1)[0]))
            a_est[axis] = float(np.polyfit(t, vel[:, axis], 1)[0])
        v_fit = np.array(v_fit)
        # Some logs contain body-frame or corrupted twist values that disagree with
        # the measured position changes by orders of magnitude. For trajectory
        # prediction, the local position derivative is the safer warm-start signal.
        if np.linalg.norm(v_est - v_fit) > 4.0 * max(np.linalg.norm(v_fit), 1.0) + 3.0:
            v_est = v_fit
            a_est = np.zeros(3)
        else:
            v_est = 0.75 * v_fit + 0.25 * v_est
    return v_est.astype(float), np.clip(a_est.astype(float), -0.4, 0.4)


def trend_start(seg: pd.DataFrame, params: RealisticUAVParams) -> tuple[np.ndarray, np.ndarray]:
    warm = seg[seg["time_rel"] <= params.trend_warmup_s].copy()
    if len(warm) < 5:
        warm = seg.iloc[: min(8, len(seg))].copy()
    t = warm["time_rel"].to_numpy(dtype=float)
    xyz = warm[["x_position", "y_position", "z_position"]].to_numpy(dtype=float)
    v_est = np.zeros(3)
    a_est = np.zeros(3)
    if len(warm) >= 4 and np.ptp(t) > 1e-6:
        for axis in range(3):
            coeff = np.polyfit(t, xyz[:, axis], 2)
            v_est[axis] = float(coeff[1])
            a_est[axis] = float(2.0 * coeff[0])
    else:
        v_est, a_est = warm_start(seg)
    return v_est.astype(float), np.clip(a_est.astype(float), -params.trend_accel_clip, params.trend_accel_clip)


def phase_targets(seg: pd.DataFrame, phase: str, params: RealisticUAVParams) -> dict[str, np.ndarray | float]:
    row0 = seg.iloc[0]
    v_est, a_est = warm_start(seg)
    pos0 = row0[["x_position", "y_position", "z_position"]].to_numpy(dtype=float)

    if phase == "takeoff":
        vz_target = float(np.clip(max(v_est[2], params.takeoff_vz_min), params.takeoff_vz_min, params.takeoff_vz_max))
        v_target = np.array([v_est[0], v_est[1], vz_target])
        z_target = pos0[2] + vz_target * float(seg["time_rel"].iloc[-1])
    elif phase == "landing":
        vz_target = float(np.clip(min(v_est[2], params.landing_vz_max), params.landing_vz_min, params.landing_vz_max))
        v_target = np.array([v_est[0], v_est[1], vz_target])
        z_target = max(0.0, pos0[2] + vz_target * float(seg["time_rel"].iloc[-1]))
    else:
        v_target = np.array([v_est[0], v_est[1], 0.0])
        z_target = pos0[2]

    return {"velocity": v_target, "z": float(z_target), "accel_bias": a_est}


def aerodynamic_drag(vel: np.ndarray, wind: np.ndarray, params: RealisticUAVParams) -> np.ndarray:
    v_rel = vel - wind
    drag_xy = -0.5 * params.rho_air * params.cd_xy * params.area_xy_m2 * np.linalg.norm(v_rel[:2]) * np.array([v_rel[0], v_rel[1], 0.0])
    drag_z = np.array([0.0, 0.0, -0.5 * params.rho_air * params.cd_z * params.area_z_m2 * abs(v_rel[2]) * v_rel[2]])
    return drag_xy + drag_z


def controller_accel(
    pos: np.ndarray,
    vel: np.ndarray,
    phase: str,
    targets: dict[str, np.ndarray | float],
    params: RealisticUAVParams,
) -> np.ndarray:
    v_target = np.asarray(targets["velocity"], dtype=float)
    accel_bias = np.asarray(targets["accel_bias"], dtype=float)
    z_target = float(targets["z"])

    tau = params.velocity_tau_s
    if phase == "takeoff":
        tau = params.takeoff_velocity_tau_s
    elif phase == "cruise":
        tau = params.cruise_velocity_tau_s
    elif phase == "landing":
        tau = params.landing_velocity_tau_s

    acc_cmd = (v_target - vel) / max(tau, 1e-6)
    acc_cmd[:2] += accel_bias[:2]
    acc_cmd[:2] -= params.horizontal_drag_gain * vel[:2] * np.abs(vel[:2])

    if phase == "cruise":
        acc_cmd[2] = params.altitude_kp * (z_target - pos[2]) - params.altitude_kd * vel[2]
    else:
        altitude_gain = params.takeoff_altitude_gain if phase == "takeoff" else params.landing_altitude_gain
        acc_cmd[2] += altitude_gain * (z_target - pos[2]) / max(abs(z_target - pos[2]) + 1.0, 1.0)

    return np.clip(
        acc_cmd,
        [-params.accel_limit_xy, -params.accel_limit_xy, -params.accel_limit_z],
        [params.accel_limit_xy, params.accel_limit_xy, params.accel_limit_z],
    )


def simulate_phase(seg: pd.DataFrame, phase: str, params: RealisticUAVParams, dt: float) -> pd.DataFrame:
    row0 = seg.iloc[0]
    t_end = float(seg["time_rel"].iloc[-1])
    steps = max(1, int(round(t_end / dt)))
    t_rel = np.arange(steps + 1) * dt
    t_rel[-1] = min(t_rel[-1], t_end)

    pos = row0[["x_position", "y_position", "z_position"]].to_numpy(dtype=float)
    # Several logs contain body-frame or noisy twist values that make the
    # rollout drift immediately if used directly as the initial inertial state.
    # Cruise and landing benefit from the robust local-position warm start.
    # Takeoff often has useful measured twist at lift-off, so keep it when it
    # is plausible and fall back to the robust estimate otherwise.
    warm_vel, _accel_bias = warm_start(seg)
    raw_vel = row0[["x_twist_linear", "y_twist_linear", "z_twist_linear"]].to_numpy(dtype=float)
    if phase == "takeoff" and np.all(np.isfinite(raw_vel)) and np.linalg.norm(raw_vel) < 15.0:
        vel = raw_vel
    else:
        vel = warm_vel
    q = normalize_quat_xyzw(row0[["x_orientation", "y_orientation", "z_orientation", "w"]].to_numpy(dtype=float))
    yaw0 = initial_yaw_from_quat(q)
    targets = phase_targets(seg, phase, params)
    wind = wind_vector(row0, params)

    if params.use_trend_prior:
        return simulate_trend_phase(seg, phase, params, dt, q, yaw0)

    records = []
    for i, tau in enumerate(t_rel):
        rot = R.from_quat(q).as_matrix()
        body_z = rot[:, 2]
        drag = aerodynamic_drag(vel, wind, params)
        acc_cmd = controller_accel(pos, vel, phase, targets, params)

        desired_force = params.mass_kg * (acc_cmd + np.array([0.0, 0.0, params.gravity])) - drag
        thrust = float(np.clip(np.linalg.norm(desired_force), 0.0, params.max_thrust_n))
        desired_body_z = desired_force / max(np.linalg.norm(desired_force), 1e-9)

        max_tilt = math.radians(params.max_tilt_deg)
        tilt = math.acos(float(np.clip(desired_body_z[2], -1.0, 1.0)))
        if tilt > max_tilt:
            horizontal = desired_body_z[:2]
            h_norm = max(np.linalg.norm(horizontal), 1e-9)
            desired_body_z = np.array(
                [
                    math.sin(max_tilt) * horizontal[0] / h_norm,
                    math.sin(max_tilt) * horizontal[1] / h_norm,
                    math.cos(max_tilt),
                ]
            )

        q_target = quat_from_body_z_and_yaw(desired_body_z, yaw0)
        records.append(
            {
                "time": float(row0["time"]) + tau,
                "time_rel": tau,
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
                "vx": vel[0],
                "vy": vel[1],
                "vz": vel[2],
                "qx": q[0],
                "qy": q[1],
                "qz": q[2],
                "qw": q[3],
                "thrust_n": thrust,
                "wind_x": wind[0],
                "wind_y": wind[1],
                "wind_z": wind[2],
            }
        )

        if i == len(t_rel) - 1:
            break

        step_dt = float(t_rel[i + 1] - tau)
        q = slerp_quat(q, q_target, step_dt / max(params.attitude_tau_s, step_dt))
        body_z = R.from_quat(q).as_matrix()[:, 2]
        drag = aerodynamic_drag(vel, wind, params)
        acc = (thrust * body_z + drag) / params.mass_kg - np.array([0.0, 0.0, params.gravity])
        acc_lim = float(params.max_integration_accel)
        acc = np.clip(acc, [-acc_lim, -acc_lim, -acc_lim], [acc_lim, acc_lim, acc_lim])
        vel = vel + acc * step_dt
        pos = pos + vel * step_dt

    return pd.DataFrame(records)


def simulate_trend_phase(
    seg: pd.DataFrame,
    phase: str,
    params: RealisticUAVParams,
    dt: float,
    q0: np.ndarray,
    yaw0: float,
) -> pd.DataFrame:
    row0 = seg.iloc[0]
    t_end = float(seg["time_rel"].iloc[-1])
    steps = max(1, int(round(t_end / dt)))
    t_rel = np.arange(steps + 1) * dt
    t_rel[-1] = min(t_rel[-1], t_end)

    pos0 = row0[["x_position", "y_position", "z_position"]].to_numpy(dtype=float)
    v0, accel = trend_start(seg, params)
    if phase == "takeoff":
        v0[2] = float(np.clip(max(v0[2], params.takeoff_vz_min), params.takeoff_vz_min, params.takeoff_vz_max))
        accel *= params.takeoff_trend_accel_scale
    elif phase == "landing":
        v0[2] = float(np.clip(min(v0[2], params.landing_vz_max), params.landing_vz_min, params.landing_vz_max))
        accel *= params.landing_trend_accel_scale
    else:
        v0[2] = 0.0
        accel[2] = 0.0
        accel *= params.cruise_trend_accel_scale

    tau = max(float(params.trend_accel_tau_s), dt)
    accel_integral = tau * (1.0 - np.exp(-t_rel / tau))
    pos_integral = tau * (t_rel - tau * (1.0 - np.exp(-t_rel / tau)))
    xyz = pos0[None, :] + t_rel[:, None] * v0[None, :] + pos_integral[:, None] * accel[None, :]
    vel = v0[None, :] + accel_integral[:, None] * accel[None, :]

    q = normalize_quat_xyzw(q0)
    omega = initial_body_rate(seg, params.trend_warmup_s)
    target_omega = omega.copy()
    records = []
    for index, (tau_rel, pos, vel_i) in enumerate(zip(t_rel, xyz, vel)):
        speed = float(np.linalg.norm(vel_i))
        if speed > 1e-6:
            desired_force = params.mass_kg * (np.array([0.0, 0.0, params.gravity]) + np.array([0.0, 0.0, 0.0]))
            q = slerp_quat(q, quat_from_body_z_and_yaw(np.array([0.0, 0.0, 1.0]), yaw0), dt / max(params.attitude_tau_s, dt))
            thrust = float(np.clip(np.linalg.norm(desired_force), 0.0, params.max_thrust_n))
        else:
            thrust = float(params.mass_kg * params.gravity)
        records.append(
            {
                "time": float(row0["time"]) + float(tau_rel),
                "time_rel": float(tau_rel),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "vx": float(vel_i[0]),
                "vy": float(vel_i[1]),
                "vz": float(vel_i[2]),
                "qx": float(q[0]),
                "qy": float(q[1]),
                "qz": float(q[2]),
                "qw": float(q[3]),
                "thrust_n": thrust,
                "wind_x": 0.0,
                "wind_y": 0.0,
                "wind_z": 0.0,
                "wx": float(omega[0]),
                "wy": float(omega[1]),
                "wz": float(omega[2]),
            }
        )
        if params.use_6dof and index < len(t_rel) - 1:
            step_dt = float(t_rel[index + 1] - tau_rel)
            # Logged rate is the short-horizon control target; damping naturally
            # removes uncommanded rotation as the rollout horizon increases.
            q, omega = integrate_rigid_body_attitude(q, omega, target_omega, params, step_dt)
    return pd.DataFrame(records)


def align_real(pred: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    t_pred = pred["time_rel"].to_numpy(dtype=float)
    t_real = real["time_rel"].to_numpy(dtype=float)
    out = pd.DataFrame({"time_rel": t_pred})
    for raw, clean in [
        ("x_position", "x"),
        ("y_position", "y"),
        ("z_position", "z"),
        ("x_twist_linear", "vx"),
        ("y_twist_linear", "vy"),
        ("z_twist_linear", "vz"),
    ]:
        out[clean] = np.interp(t_pred, t_real, real[raw].to_numpy(dtype=float))
    return out


def align_real_quaternion(pred: pd.DataFrame, real: pd.DataFrame) -> np.ndarray:
    t_pred = pred["time_rel"].to_numpy(dtype=float)
    t_real = real["time_rel"].to_numpy(dtype=float)
    cols = ["x_orientation", "y_orientation", "z_orientation", "w"]
    q = np.column_stack(
        [np.interp(t_pred, t_real, real[col].to_numpy(dtype=float)) for col in cols]
    )
    return np.array([normalize_quat_xyzw(row) for row in q])


def evaluate_prediction(pred: pd.DataFrame, real: pd.DataFrame, csv_name: str, phase: str, runtime_s: float) -> dict[str, float | str]:
    real_i = align_real(pred, real)
    p = pred[["x", "y", "z"]].to_numpy(dtype=float)
    r = real_i[["x", "y", "z"]].to_numpy(dtype=float)
    err = p - r
    dist = np.linalg.norm(err, axis=1)
    pv = pred[["vx", "vy", "vz"]].to_numpy(dtype=float)
    rv = real_i[["vx", "vy", "vz"]].to_numpy(dtype=float)
    vdist = np.linalg.norm(pv - rv, axis=1)
    return {
        "csv": csv_name,
        "phase": phase,
        "n_points": len(pred),
        "runtime_s": runtime_s,
        "rmse_3d_m": float(np.sqrt(np.mean(dist**2))),
        "mae_3d_m": float(np.mean(dist)),
        "rmse_x_m": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_y_m": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "rmse_z_m": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "final_error_m": float(dist[-1]),
        "rmse_velocity_mps": float(np.sqrt(np.mean(vdist**2))),
    }


def build_error_frame(pred: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    real_i = align_real(pred, real)
    out = pred[["time", "time_rel", "x", "y", "z", "vx", "vy", "vz", "qx", "qy", "qz", "qw"]].copy()
    for col in ["x", "y", "z", "vx", "vy", "vz"]:
        out[f"real_{col}"] = real_i[col]
        out[f"error_{col}"] = out[col] - out[f"real_{col}"]

    q_real = align_real_quaternion(pred, real)
    q_pred = out[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
    ori_error_deg = []
    heading_error_deg = []
    for q_p, q_r in zip(q_pred, q_real):
        r_p = R.from_quat(normalize_quat_xyzw(q_p))
        r_r = R.from_quat(normalize_quat_xyzw(q_r))
        ori_error_deg.append((r_p.inv() * r_r).magnitude() * 180.0 / np.pi)
        yaw_p = r_p.as_euler("zyx", degrees=True)[0]
        yaw_r = r_r.as_euler("zyx", degrees=True)[0]
        heading_error_deg.append(abs((yaw_p - yaw_r + 180.0) % 360.0 - 180.0))

    out["error_3d"] = np.linalg.norm(out[["error_x", "error_y", "error_z"]].to_numpy(dtype=float), axis=1)
    out["error_velocity"] = np.linalg.norm(out[["error_vx", "error_vy", "error_vz"]].to_numpy(dtype=float), axis=1)
    out["orientation_error_deg"] = ori_error_deg
    out["heading_error_deg"] = heading_error_deg
    return out


def plot_phase_prediction(pred: pd.DataFrame, real: pd.DataFrame, out_dir: Path, csv_stem: str, phase: str) -> None:
    phase_dir = out_dir / "plots" / csv_stem / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    real_i = align_real(pred, real)
    t = pred["time_rel"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(real_i["x"], real_i["y"], real_i["z"], "-", label="Real")
    ax.plot(pred["x"], pred["y"], pred["z"], ":", linewidth=2.0, label="Predicted")
    ax.set_xlabel("X local (m)")
    ax.set_ylabel("Y local (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{csv_stem} {phase} 3D trajectory")
    ax.legend(handles=comparison_legend_handles(), loc="best")
    fig.tight_layout()
    fig.savefig(phase_dir / "trajectory_3d.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, col, label in zip(axes, ["x", "y", "z"], ["X", "Y", "Z"]):
        ax.plot(t, real_i[col], "-", label="Real")
        ax.plot(t, pred[col], ":", linewidth=2.0, label="Predicted")
        ax.set_ylabel(f"{label} (m)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(handles=comparison_legend_handles(), loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(phase_dir / "position_components.png", dpi=220)
    plt.close(fig)

    err = pred[["x", "y", "z"]].to_numpy(dtype=float) - real_i[["x", "y", "z"]].to_numpy(dtype=float)
    err_dist = np.linalg.norm(err, axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, err[:, 0], label="X error")
    ax.plot(t, err[:, 1], label="Y error")
    ax.plot(t, err[:, 2], label="Z error")
    ax.plot(t, err_dist, label="3D error", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_title(f"{csv_stem} {phase} error over time")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(phase_dir / "error_over_time.png", dpi=220)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, phase_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["phase"], summary["mean_rmse_3d_m"], color=["#238b45", "#386cb0", "#d95f02"])
    ax.set_ylabel("Mean RMSE 3D (m)")
    ax.set_title("Physics Model by Phase")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "physics_model_phase_rmse_bar.png", dpi=220)
    plt.close(fig)


def representative_case_for_phase(cases: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, float]] | None:
    scored_cases = []
    for case in cases:
        err = case["error"]
        error_3d = err["error_3d"].to_numpy(dtype=float)
        error_3d = error_3d[np.isfinite(error_3d)]
        if error_3d.size == 0:
            continue
        rmse = float(np.sqrt(np.mean(error_3d**2)))
        scored_cases.append(
            (
                rmse,
                case,
                {
                    "rmse_3d_m": rmse,
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


def plot_representative_phase_position(case_outputs: list[dict[str, object]], out_dir: Path) -> None:
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
    fig.savefig(out_dir / "real_vs_predicted_position_over_time_representative_phase.png", dpi=220)
    plt.close(fig)

    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(out_dir / "representative_phase_cases.csv", index=False)


def plot_v5_compatible_outputs(case_outputs: list[dict[str, object]], detail: pd.DataFrame, out_dir: Path) -> None:
    by_phase = {phase: [case for case in case_outputs if case["phase"] == phase] for phase in PHASES}
    class_legend = comparison_legend_handles()
    unique_labels = sorted({str(case["csv"]) for case in case_outputs})
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    color_map = {label: cmap(i % cmap.N) for i, label in enumerate(unique_labels)}

    plot_representative_phase_position(case_outputs, out_dir)

    fig = plt.figure(figsize=(15, 10))
    for i, phase in enumerate(PHASES, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for case in by_phase[phase]:
            err = case["error"]
            color = color_map[str(case["csv"])]
            ax.plot(
                err["real_x"],
                err["real_y"],
                err["real_z"],
                "-",
                alpha=0.35,
                color=color,
            )
            ax.plot(
                err["x"],
                err["y"],
                err["z"],
                ":",
                alpha=0.80,
                color=color,
                linewidth=1.6,
            )
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

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            for j, col in enumerate(["x", "y", "z"]):
                axs[i, j].plot(t, err[f"real_{col}"], "-", alpha=0.35)
                axs[i, j].plot(t, err[col], ":", alpha=0.85, linewidth=1.6)
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

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            for j, col in enumerate(["vx", "vy", "vz"]):
                axs[i, j].plot(t, err[f"real_{col}"], "-", alpha=0.35)
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
    fig.savefig(out_dir / "real_vs_predicted_velocity_over_time.png", dpi=220)
    plt.close(fig)

    for phase, cases in by_phase.items():
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for case in cases:
            err = case["error"]
            label_suffix = str(case["csv"]).replace(".csv", "")
            ax.plot(err["real_x"], err["real_y"], err["real_z"], "-", alpha=0.35)
            ax.plot(err["x"], err["y"], err["z"], ":", alpha=0.75, linewidth=1.6)
        ax.set_xlabel("X local (m)")
        ax.set_ylabel("Y local (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"3D Trajectory: {phase.capitalize()} - all real flights")
        ax.legend(handles=class_legend, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"3d_trajectory_{phase}.png", dpi=220)
        plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            label = str(case["csv"]).replace(".csv", "")
            for j, col in enumerate(["error_x", "error_y", "error_z"]):
                axs[i, j].plot(t, err[col], alpha=0.65, label=label)
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["X", "Y", "Z"][j] + " Error (m)")
                axs[i, j].grid(True, alpha=0.25)
                #axs[i, j].legend(fontsize=7, loc="best")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "position_error_over_time.png", dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            label = str(case["csv"]).replace(".csv", "")
            for j, col in enumerate(["error_vx", "error_vy", "error_vz"]):
                axs[i, j].plot(t, err[col], alpha=0.65, label=label)
                axs[i, j].set_title(f"{phase.capitalize()}")
                axs[i, j].set_ylabel(["Vx", "Vy", "Vz"][j] + " Error (m/s)")
                axs[i, j].grid(True, alpha=0.25)
                #axs[i, j].legend(fontsize=7, loc="best")
    for ax in axs.flat:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "velocity_error_over_time.png", dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            axs[i].plot(err["time_rel"], err["orientation_error_deg"], alpha=0.45)
        axs[i].set_title(f"{phase.capitalize()}")
        axs[i].set_ylabel("Ori Error (deg)")
        axs[i].grid(True, alpha=0.25)
    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "orientation_error_over_time.png", dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
    for i, phase in enumerate(PHASES):
        for case in by_phase[phase]:
            err = case["error"]
            t = err["time_rel"]
            for j, col in enumerate(["x", "y", "z"]):
                axs[i, j].plot(t, err[col], ":", alpha=0.5)
                axs[i, j].plot(t, err[f"real_{col}"], "-", alpha=0.4)
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

    phase_metrics = (
        detail.groupby("phase", as_index=False)
        .agg(
            RMSE_3D=("rmse_3d_m", "mean"),
            MAE_3D=("mae_3d_m", "mean"),
            FinalErr=("final_error_m", "mean"),
            RMSE_Vel=("rmse_velocity_mps", "mean"),
        )
    )
    metrics = ["RMSE_3D", "MAE_3D", "FinalErr", "RMSE_Vel"]
    x = np.arange(len(PHASES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, metric in enumerate(metrics):
        vals = [
            float(phase_metrics.loc[phase_metrics["phase"] == phase, metric].iloc[0])
            for phase in PHASES
        ]
        ax.bar(x + idx * width, vals, width, label=metric)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([phase.capitalize() for phase in PHASES])
    ax.set_ylabel("Error")
    ax.set_title("Phase Comparison: Physics Model Key Metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_comparison_bar_chart.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    detail.boxplot(column="rmse_3d_m", by="phase", ax=ax)
    ax.set_title("RMSE 3D Distribution by Phase")
    ax.set_ylabel("RMSE 3D (m)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_distribution_by_phase.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    sorted_detail = detail.sort_values(["phase", "csv"])
    labels = [f"{row.phase[:1].upper()}-{row.csv.replace('flights_', '').replace('.csv', '')}" for row in sorted_detail.itertuples()]
    ax.bar(labels, sorted_detail["rmse_3d_m"], color="#386cb0")
    ax.set_ylabel("RMSE 3D (m)")
    ax.set_title("Per-flight Validation RMSE")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "per_flight_rmse_bar.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(labels, sorted_detail["final_error_m"], color="#b7372f")
    ax.set_ylabel("Final Error (m)")
    ax.set_title("Per-flight Final Prediction Error")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "per_flight_final_error_bar.png", dpi=220)
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
    fig.savefig(out_dir / "all_flights_position_error_over_time.png", dpi=220)
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
    fig.savefig(out_dir / "all_flights_velocity_error_over_time.png", dpi=220)
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
    fig.savefig(out_dir / "all_flights_3d_error_over_time.png", dpi=220)
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
    fig.savefig(out_dir / "all_flights_real_vs_predicted_position_over_time.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    detail.boxplot(column="rmse_3d_m", by="phase", ax=ax)
    ax.set_title("Physics Model RMSE Distribution by Phase")
    ax.set_ylabel("RMSE 3D (m)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_dir / "physics_model_rmse_distribution_boxplot.png", dpi=220)
    plt.close(fig)


def save_phase_mean_trajectories(case_outputs: list[dict[str, object]], out_dir: Path, dt: float) -> None:
    for phase in PHASES:
        cases = [case for case in case_outputs if case["phase"] == phase]
        if not cases:
            continue
        t_max = max(float(case["pred"]["time_rel"].iloc[-1]) for case in cases)
        t_grid = np.arange(0.0, t_max + dt * 0.5, dt)
        rows = {"time": t_grid}
        for axis, col in enumerate(["x", "y", "z"]):
            stacked = []
            for case in cases:
                pred = case["pred"]
                stacked.append(
                    np.interp(
                        t_grid,
                        pred["time_rel"].to_numpy(dtype=float),
                        pred[col].to_numpy(dtype=float),
                    )
                )
            rows[col] = np.mean(np.vstack(stacked), axis=0)
        pd.DataFrame(rows).to_csv(out_dir / f"{phase}_predicted_mean_trajectory_v6.csv", index=False)


def plot_euclidean_error_over_time(case_outputs: list[dict[str, object]], out_dir: Path, dt: float, horizon_s: float) -> None:
    """Euclidean (3D) position error vs time: overall mean curve and phase-wise curves."""
    t_grid = np.arange(0.0, horizon_s + dt * 0.5, dt)

    raw_rows = []
    curves: dict[str, np.ndarray] = {}
    for case in case_outputs:
        err = case["error"]
        t = err["time_rel"].to_numpy(dtype=float)
        e = err["error_3d"].to_numpy(dtype=float)
        interp = np.interp(t_grid, t, e)
        interp[t_grid > t[-1] + dt] = np.nan  # do not extrapolate beyond case duration
        curves[(case["csv"], case["phase"])] = interp
        for ti, ei in zip(t_grid, interp):
            if not np.isnan(ei):
                raw_rows.append({"time_s": ti, "csv": case["csv"], "phase": case["phase"], "euclidean_error_m": ei})

    pd.DataFrame(raw_rows).to_csv(out_dir / "physics_euclidean_error_over_time_raw.csv", index=False)

    # --- overall mean +/- std ---
    all_stack = np.vstack(list(curves.values()))
    mean_all = np.nanmean(all_stack, axis=0)
    std_all = np.nanstd(all_stack, axis=0)
    n_all = np.sum(~np.isnan(all_stack), axis=0)
    pd.DataFrame(
        {"time_s": t_grid, "mean_euclidean_error_m": mean_all, "std_euclidean_error_m": std_all, "n_cases": n_all}
    ).to_csv(out_dir / "physics_mean_euclidean_error_over_time.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_grid, mean_all, color="tab:blue", linewidth=2.0, label=f"Mean (n={int(n_all.max())} cases)")
    ax.fill_between(t_grid, np.maximum(mean_all - std_all, 0.0), mean_all + std_all, color="tab:blue", alpha=0.2, label="±1 std")
    ax.set_xlabel("Prediction time (s)")
    ax.set_ylabel("Euclidean position error (m)")
    ax.set_title(f"Mean Euclidean Error over {horizon_s:.0f} s Prediction Horizon")
    ax.set_xlim(0, horizon_s)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "physics_mean_euclidean_error_over_time.png", dpi=220)
    plt.close(fig)

    # --- phase-wise mean curves ---
    phase_cols = {"takeoff": "tab:green", "cruise": "tab:blue", "landing": "tab:red"}
    phase_frame = {"time_s": t_grid}
    fig, ax = plt.subplots(figsize=(10, 6))
    for phase in PHASES:
        stack = [c for (csv_name, ph), c in curves.items() if ph == phase]
        if not stack:
            continue
        stack = np.vstack(stack)
        mean_p = np.nanmean(stack, axis=0)
        std_p = np.nanstd(stack, axis=0)
        phase_frame[f"{phase}_mean_error_m"] = mean_p
        phase_frame[f"{phase}_std_error_m"] = std_p
        ax.plot(t_grid, mean_p, color=phase_cols[phase], linewidth=2.0, label=f"{phase} (n={stack.shape[0]})")
        ax.fill_between(t_grid, np.maximum(mean_p - std_p, 0.0), mean_p + std_p, color=phase_cols[phase], alpha=0.12)
    ax.set_xlabel("Prediction time (s)")
    ax.set_ylabel("Euclidean position error (m)")
    ax.set_title(f"Phase-wise Mean Euclidean Error over {horizon_s:.0f} s Prediction Horizon")
    ax.set_xlim(0, horizon_s)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase_wise_euclidean_error_over_time.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(phase_frame).to_csv(out_dir / "phase_wise_euclidean_error_over_time.csv", index=False)


def run_evaluation(
    data_dir: Path,
    output_dir: Path,
    horizon_s: float,
    dt: float,
    make_phase_plots: bool,
    file_names: list[str] | None = None,
    params: RealisticUAVParams | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = params or RealisticUAVParams()
    rows = []
    case_outputs = []

    csv_paths = sorted(data_dir.rglob("flights_*.csv"))
    if file_names:
        selected = set(file_names)
        csv_paths = [
            path
            for path in csv_paths
            if path.name in selected or safe_label_from_path(path, data_dir) in selected
        ]
        matched = {path.name for path in csv_paths} | {safe_label_from_path(path, data_dir) for path in csv_paths}
        missing = selected - matched
        if missing:
            raise FileNotFoundError(f"Requested CSV files were not found in {data_dir}: {sorted(missing)}")

    for csv_path in csv_paths:
        csv_label = safe_label_from_path(csv_path, data_dir)
        df = add_local_meter_frame(pd.read_csv(csv_path))
        for phase in PHASES:
            seg = get_phase_segment(df, phase, horizon_s)
            if seg is None:
                continue
            start = time.perf_counter()
            pred = simulate_phase(seg, phase, params, dt)
            runtime_s = time.perf_counter() - start
            rows.append(evaluate_prediction(pred, seg, csv_label, phase, runtime_s))
            case_outputs.append(
                {
                    "csv": csv_label,
                    "phase": phase,
                    "pred": pred,
                    "real": seg,
                    "error": build_error_frame(pred, seg),
                }
            )
            if make_phase_plots:
                plot_phase_prediction(pred, seg, output_dir, safe_stem_from_label(csv_label), phase)

    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "physics_model_detail_metrics.csv", index=False)
    summary = (
        detail.groupby("phase", as_index=False)
        .agg(
            cases=("rmse_3d_m", "count"),
            mean_rmse_3d_m=("rmse_3d_m", "mean"),
            median_rmse_3d_m=("rmse_3d_m", "median"),
            mean_mae_3d_m=("mae_3d_m", "mean"),
            mean_final_error_m=("final_error_m", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
        )
        .sort_values("mean_rmse_3d_m")
    )
    summary.to_csv(output_dir / "physics_model_phase_summary.csv", index=False)

    model_summary = pd.DataFrame(
        [
            {
                "model": f"{MODEL_NAME}_6dof" if params.use_6dof else MODEL_NAME,
                "cases": len(detail),
                "mean_rmse_3d_m": detail["rmse_3d_m"].mean(),
                "median_rmse_3d_m": detail["rmse_3d_m"].median(),
                "mean_mae_3d_m": detail["mae_3d_m"].mean(),
                "mean_final_error_m": detail["final_error_m"].mean(),
                "mean_runtime_s": detail["runtime_s"].mean(),
            }
        ]
    )
    model_summary.to_csv(output_dir / "physics_model_summary.csv", index=False)
    plot_summary(summary, detail, output_dir)
    plot_v5_compatible_outputs(case_outputs, detail, output_dir)
    save_phase_mean_trajectories(case_outputs, output_dir, dt)
    plot_euclidean_error_over_time(case_outputs, output_dir, dt, horizon_s)

    print("=== Physics Model Summary ===")
    print(model_summary.round(4).to_string(index=False))
    print("\n=== Physics Model Phase Summary ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complex realistic UAV physics model with real-data evaluation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--mass-kg", type=float, default=None)
    parser.add_argument("--max-thrust-n", type=float, default=None)
    parser.add_argument("--drag-scale", type=float, default=1.0)
    parser.add_argument("--wind-gain", type=float, default=None)
    parser.add_argument("--trend-prior", action="store_true", help="Force-enable the damped-trend gray-box prior (already on by default).")
    parser.add_argument("--no-trend-prior", action="store_true", help="Disable the damped-trend gray-box prior.")
    parser.add_argument("--trend-warmup-s", type=float, default=None)
    parser.add_argument("--six-dof", action="store_true", help="Enable rigid-body quaternion/body-rate dynamics.")
    parser.add_argument("--skip-phase-plots", action="store_true")
    parser.add_argument("--files", nargs="*", default=None, help="Optional CSV file names to evaluate.")
    parser.add_argument(
        "--no-versioned-output",
        action="store_true",
        help="Write directly to --output-dir instead of creating the next vN subfolder.",
    )
    args = parser.parse_args()

    if args.no_versioned_output or (args.output_dir.name.startswith("v") and args.output_dir.name[1:].isdigit()):
        output_dir = args.output_dir
    else:
        output_dir = create_versioned_output_dir(args.output_dir)

    params = RealisticUAVParams()
    if args.mass_kg is not None:
        params.mass_kg = float(args.mass_kg)
    if args.max_thrust_n is not None:
        params.max_thrust_n = float(args.max_thrust_n)
    if args.wind_gain is not None:
        params.wind_gain = float(args.wind_gain)
    params.cd_xy *= float(args.drag_scale)
    params.cd_z *= float(args.drag_scale)
    if args.no_trend_prior:
        params.use_trend_prior = False
    elif args.trend_prior:
        params.use_trend_prior = True
    if args.trend_warmup_s is not None:
        params.trend_warmup_s = float(args.trend_warmup_s)
    if args.six_dof:
        params.use_6dof = True
        # Existing SR sensitivity results put the RMSE/MAE optimum near +40%
        # from the original 10 s warm-up.
        if args.trend_warmup_s is None:
            params.trend_warmup_s = 14.0

    run_evaluation(args.data_dir, output_dir, args.horizon, args.dt, not args.skip_phase_plots, args.files, params=params)
    (output_dir / "physics_model_parameters.json").write_text(
        json.dumps(
            {
                "horizon_s": float(args.horizon),
                "integration_dt_s": float(args.dt),
                "parameters": asdict(params),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
