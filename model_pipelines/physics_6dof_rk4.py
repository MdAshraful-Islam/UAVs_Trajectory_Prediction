"""Phase-aware closed-loop 6-DOF Physics-SR model with RK4 integration.

The 13-state rigid-body plant is integrated directly.  A damped phase trend is
used only to construct the reference followed by the thrust/attitude controller.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import physics_model_sr as phy


MODEL_NAME = "physics_model_sr_6dof_rk4_control"


@dataclass
class ControllerParams:
    position_kp_xy: float = 0.45
    position_kp_z: float = 0.65
    velocity_kd_xy: float = 1.15
    velocity_kd_z: float = 1.35
    attitude_kp: float = 7.0
    angular_rate_kd: float = 2.8
    max_reference_accel: float = 2.0


def quat_product_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def reference_at(reference: pd.DataFrame, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = reference["time_rel"].to_numpy(float)
    pos = np.array([np.interp(t, grid, reference[c]) for c in ("x", "y", "z")])
    vel = np.array([np.interp(t, grid, reference[c]) for c in ("vx", "vy", "vz")])
    ref_vel = reference[["vx", "vy", "vz"]].to_numpy(float)
    edge = 2 if len(grid) >= 3 else 1
    acc_values = np.gradient(ref_vel, grid, axis=0, edge_order=edge)
    acc = np.array([np.interp(t, grid, acc_values[:, i]) for i in range(3)])
    return pos, vel, acc


def control_and_derivative(
    t: float,
    state: np.ndarray,
    reference: pd.DataFrame,
    yaw_ref: float,
    wind: np.ndarray,
    plant: phy.RealisticUAVParams,
    control: ControllerParams,
) -> tuple[np.ndarray, float, np.ndarray]:
    p, v = state[0:3], state[3:6]
    q = phy.normalize_quat_xyzw(state[6:10])
    omega = state[10:13]
    p_ref, v_ref, a_ref = reference_at(reference, t)

    kp = np.array([control.position_kp_xy, control.position_kp_xy, control.position_kp_z])
    kd = np.array([control.velocity_kd_xy, control.velocity_kd_xy, control.velocity_kd_z])
    a_cmd = a_ref + kp * (p_ref - p) + kd * (v_ref - v)
    a_cmd = np.clip(a_cmd, -control.max_reference_accel, control.max_reference_accel)

    drag = phy.aerodynamic_drag(v, wind, plant)
    desired_force = plant.mass_kg * (a_cmd + np.array([0.0, 0.0, plant.gravity])) - drag
    thrust = float(np.clip(np.linalg.norm(desired_force), 0.0, plant.max_thrust_n))
    desired_z = desired_force / max(np.linalg.norm(desired_force), 1e-9)
    max_tilt = math.radians(plant.max_tilt_deg)
    tilt = math.acos(float(np.clip(desired_z[2], -1.0, 1.0)))
    if tilt > max_tilt:
        horizontal = desired_z[:2] / max(np.linalg.norm(desired_z[:2]), 1e-9)
        desired_z = np.array([*(math.sin(max_tilt) * horizontal), math.cos(max_tilt)])
    q_target = phy.quat_from_body_z_and_yaw(desired_z, yaw_ref)

    attitude_error = (phy.R.from_quat(q).inv() * phy.R.from_quat(q_target)).as_rotvec()
    inertia = np.array([plant.inertia_xx_kg_m2, plant.inertia_yy_kg_m2, plant.inertia_zz_kg_m2])
    torque = inertia * (control.attitude_kp * attitude_error - control.angular_rate_kd * omega)
    torque -= plant.angular_damping_nms * omega

    rotation = phy.R.from_quat(q).as_matrix()
    acceleration = (thrust * rotation[:, 2] + drag) / plant.mass_kg
    acceleration -= np.array([0.0, 0.0, plant.gravity])
    omega_dot = (torque - np.cross(omega, inertia * omega)) / inertia
    q_dot = 0.5 * quat_product_xyzw(q, np.array([omega[0], omega[1], omega[2], 0.0]))
    derivative = np.concatenate([v, acceleration, q_dot, omega_dot])
    return derivative, thrust, torque


def rk4_step(t, state, dt, derivative):
    k1 = derivative(t, state)[0]
    k2 = derivative(t + 0.5 * dt, state + 0.5 * dt * k1)[0]
    k3 = derivative(t + 0.5 * dt, state + 0.5 * dt * k2)[0]
    k4 = derivative(t + dt, state + dt * k3)[0]
    out = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    out[6:10] = phy.normalize_quat_xyzw(out[6:10])
    return out


def simulate_phase_rk4(
    seg: pd.DataFrame,
    phase: str,
    plant: phy.RealisticUAVParams,
    dt: float,
    controller: ControllerParams | None = None,
) -> pd.DataFrame:
    controller = controller or ControllerParams()
    row0 = seg.iloc[0]
    t_end = float(seg["time_rel"].iloc[-1])
    times = np.arange(0.0, t_end + 0.5 * dt, dt)
    times[-1] = min(times[-1], t_end)
    q0 = phy.normalize_quat_xyzw(
        row0[["x_orientation", "y_orientation", "z_orientation", "w"]].to_numpy(float)
    )
    yaw_ref = phy.initial_yaw_from_quat(q0)
    omega0 = phy.initial_body_rate(seg, plant.trend_warmup_s)
    v0, _ = phy.trend_start(seg, plant)
    state = np.concatenate([
        row0[["x_position", "y_position", "z_position"]].to_numpy(float), v0, q0, omega0
    ])
    wind = phy.wind_vector(row0, plant)

    reference_params = phy.RealisticUAVParams(**asdict(plant))
    reference_params.use_6dof = False
    reference = phy.simulate_trend_phase(seg, phase, reference_params, dt, q0, yaw_ref)
    derivative = lambda t, x: control_and_derivative(
        t, x, reference, yaw_ref, wind, plant, controller
    )

    records = []
    for i, t in enumerate(times):
        _, thrust, torque = derivative(float(t), state)
        records.append({
            "time": float(row0["time"]) + float(t), "time_rel": float(t),
            "x": state[0], "y": state[1], "z": state[2],
            "vx": state[3], "vy": state[4], "vz": state[5],
            "qx": state[6], "qy": state[7], "qz": state[8], "qw": state[9],
            "wx": state[10], "wy": state[11], "wz": state[12],
            "thrust_n": thrust, "torque_x": torque[0], "torque_y": torque[1], "torque_z": torque[2],
            "wind_x": wind[0], "wind_y": wind[1], "wind_z": wind[2],
        })
        if i < len(times) - 1:
            state = rk4_step(float(t), state, float(times[i + 1] - t), derivative)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=phy.DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("physics_model_sr_6dof_rk4_outputs"))
    parser.add_argument("--horizon", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--skip-phase-plots", action="store_true")
    args = parser.parse_args()

    output = phy.create_versioned_output_dir(args.output_dir)
    plant = phy.RealisticUAVParams(use_6dof=True, trend_warmup_s=14.0)
    controller = ControllerParams()
    phy.simulate_phase = lambda seg, phase, params, dt: simulate_phase_rk4(
        seg, phase, params, dt, controller
    )
    phy.MODEL_NAME = MODEL_NAME
    phy.run_evaluation(
        args.data_dir, output, args.horizon, args.dt,
        not args.skip_phase_plots, params=plant,
    )
    (output / "model_configuration.json").write_text(
        json.dumps({"plant": asdict(plant), "controller": asdict(controller), "integrator": "RK4"}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
