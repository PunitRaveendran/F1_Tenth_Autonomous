#!/usr/bin/env python3
"""
F1TENTH Multi-Track Benchmark: Nuerburgring Circuit
Author: Punit Raveendran (punitraveendran@gmail.com)

Evaluates tracking controllers on the Nuerburgring track:
  1. Pure Pursuit (Baseline: Fixed L, Fixed Speed)
  2. Pure Pursuit (Speed Profiled from Raceline)
  3. Advanced Pure Pursuit (Adaptive L(v) + Segment Interpolation)
  4. Stanley Controller (Baseline Untuned)
  5. Stanley Controller (Optuna Tuned)
  6. LQR Controller (Discrete-Time State Space Feedback)

Outputs:
  - benchmark_nuerburgring_results.csv
  - benchmark_nuerburgring_summary.png
"""

import os
import csv
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.linalg import solve_discrete_are
import gymnasium as gym
from f1tenth_gym.envs.env_config import EnvConfig
import stanley_tuner as st

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def load_track_geometry(raceline_path):
    """Loads TUM format raceline: s, x, y, psi, kappa, vx, ax"""
    rows = []
    with open(raceline_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for r in reader:
            if not r or r[0].startswith('#') or r[0].strip() == 's_m' or len(r) < 6:
                continue
            try:
                # s, x, y, psi, kappa, vx
                rows.append([float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[0])])
            except ValueError:
                continue
    geom = np.array(rows) # [x, y, psi, kappa, s]
    return geom

class NuerburgringBenchmark:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.map_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'Nuerburgring')
        self.raceline_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'Nuerburgring_map_raceline.csv')

        geom = load_track_geometry(self.raceline_path)
        self.xy = geom[:, :2]
        self.psi = geom[:, 2]
        self.kappa_raw = geom[:, 3]
        self.kappa_abs = np.maximum(np.abs(self.kappa_raw), 1e-6)
        
        s = geom[:, 4]
        ds = np.diff(s)
        median_ds = np.median(ds[ds > 0]) if np.any(ds > 0) else 0.2
        ds[ds <= 0] = median_ds
        self.ds = ds

        self.kdtree = KDTree(self.xy)
        self.wheelbase = 0.33
        self.steer_max = 0.41

        # Compute physics speed profiles on Nuerburgring
        self.speeds_std = st.compute_speed_profile(self.kappa_abs, self.ds, 8.0, 7.0, 5.0, 9.0)
        self.speeds_tuned = st.compute_speed_profile(self.kappa_abs, self.ds, 8.0, 9.5, 7.0, 11.0)

        # Columns: [0: x, 1: y, 2: psi, 3: speed, 4: kappa]
        self.wp_std = np.column_stack([self.xy, self.psi, self.speeds_std, self.kappa_raw])
        self.wp_tuned = np.column_stack([self.xy, self.psi, self.speeds_tuned, self.kappa_raw])
        self.n_pts = len(self.wp_std)
        print(f"Loaded Nuerburgring: {self.n_pts} waypoints, speed range {self.speeds_std.min():.2f} - {self.speeds_std.max():.2f} m/s")

    def get_env(self):
        config = EnvConfig(map_name=self.map_path, num_agents=1, render_enabled=False)
        return gym.make('f1tenth_gym:f1tenth-v0', config=config)

    # ── Controllers ──

    def ctrl_pure_pursuit_baseline(self, state, wp_table):
        """Fixed lookahead L=1.6m, Constant Speed 5.5 m/s"""
        cx, cy, spd, yaw = state[0], state[1], abs(state[3]), state[4]
        L = 1.6
        _, idx = self.kdtree.query([cx, cy])
        target = wp_table[(idx + 8) % self.n_pts]
        dx = target[0] - cx
        dy = target[1] - cy
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        alpha = np.arctan2(local_y, local_x)
        steer = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), L)
        steer = np.clip(steer, -self.steer_max, self.steer_max)
        return float(steer), 5.5, idx

    def ctrl_pure_pursuit_profiled(self, state, wp_table):
        """Fixed lookahead L=1.5m, Speed Profile from Raceline"""
        cx, cy, spd, yaw = state[0], state[1], abs(state[3]), state[4]
        L = 1.5
        _, idx = self.kdtree.query([cx, cy])
        target = wp_table[(idx + 8) % self.n_pts]
        dx = target[0] - cx
        dy = target[1] - cy
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        alpha = np.arctan2(local_y, local_x)
        steer = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), L)
        steer = np.clip(steer, -self.steer_max, self.steer_max)
        target_v = float(np.clip(wp_table[(idx + 3) % self.n_pts, 3] * 0.85, 0.5, 8.0))
        return float(steer), target_v, idx

    def ctrl_advanced_pure_pursuit(self, state, wp_table):
        """Adaptive lookahead L(v) + Circle-Segment Interpolation"""
        cx, cy, spd, yaw = state[0], state[1], abs(state[3]), state[4]
        L = float(np.clip(1.1 + (spd / 8.0) * (2.1 - 1.1), 1.1, 2.1))
        _, idx = self.kdtree.query([cx, cy])

        target = None
        for i in range(idx, idx + 120):
            w = wp_table[i % self.n_pts]
            d = np.hypot(w[0] - cx, w[1] - cy)
            dx = w[0] - cx
            dy = w[1] - cy
            local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
            if d >= L and local_x > 0.0:
                prev_w = wp_table[(i - 1) % self.n_pts]
                v_seg = w[:2] - prev_w[:2]
                w_seg = prev_w[:2] - np.array([cx, cy])
                c = np.dot(w_seg, w_seg) - L ** 2
                b = 2.0 * np.dot(v_seg, w_seg)
                a = np.dot(v_seg, v_seg)
                disc = b * b - 4 * a * c
                if disc >= 0.0 and a > 1e-9:
                    t_cand = (-b + np.sqrt(disc)) / (2.0 * a)
                    if 0.0 <= t_cand <= 1.0:
                        target = prev_w[:2] + t_cand * v_seg
                        break
                target = w[:2]
                break

        if target is None:
            target = wp_table[(idx + 8) % self.n_pts, :2]

        dx = target[0] - cx
        dy = target[1] - cy
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        alpha = np.arctan2(local_y, local_x)
        steer = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), L) * 1.1
        steer = np.clip(steer, -self.steer_max, self.steer_max)
        target_v = float(np.clip(wp_table[(idx + 3) % self.n_pts, 3] * 0.90, 0.5, 8.0))
        return float(steer), target_v, idx

    def ctrl_stanley_baseline(self, state, wp_table):
        """Stanley Untuned: k_gain=1.5, k_soft=1.0, speed_scale=0.85"""
        cx, cy, spd, yaw = state[0], state[1], abs(state[3]), state[4]
        fx = cx + self.wheelbase * np.cos(yaw)
        fy = cy + self.wheelbase * np.sin(yaw)
        _, idx = self.kdtree.query([fx, fy])
        wp = wp_table[idx]
        theta_e = normalize_angle(wp[2] - yaw)
        e_fa = np.sin(wp[2]) * (fx - wp[0]) - np.cos(wp[2]) * (fy - wp[1])
        steer = theta_e + np.arctan2(1.5 * e_fa, spd + 1.0) + np.arctan(self.wheelbase * wp[4])
        steer = np.clip(steer, -self.steer_max, self.steer_max)
        target_v = float(np.clip(wp_table[(idx + 4) % self.n_pts, 3] * 0.85, 0.5, 8.0))
        return float(steer), target_v, idx

    def ctrl_stanley_tuned(self, state, wp_table):
        """Stanley Tuned: k_gain=3.0, k_soft=0.30, speed_scale=1.05"""
        cx, cy, spd, yaw = state[0], state[1], abs(state[3]), state[4]
        fx = cx + self.wheelbase * np.cos(yaw)
        fy = cy + self.wheelbase * np.sin(yaw)
        _, idx = self.kdtree.query([fx, fy])
        wp = wp_table[idx]
        theta_e = normalize_angle(wp[2] - yaw)
        e_fa = np.sin(wp[2]) * (fx - wp[0]) - np.cos(wp[2]) * (fy - wp[1])
        steer = theta_e + np.arctan2(3.0 * e_fa, spd + 0.3) + np.arctan(self.wheelbase * wp[4])
        steer = np.clip(steer, -self.steer_max, self.steer_max)
        target_v = float(np.clip(wp_table[(idx + 4) % self.n_pts, 3] * 1.05, 0.5, 8.0 * 1.05))
        return float(steer), target_v, idx

    def ctrl_lqr(self, state, wp_table):
        """LQR Controller: Discrete DARE State-Feedback"""
        cx, cy, spd, yaw = state[0], state[1], abs(state[3]), state[4]
        v = max(spd, 0.8)
        _, idx = self.kdtree.query([cx, cy])
        wp = wp_table[idx]
        dx = cx - wp[0]
        dy = cy - wp[1]
        wp_psi = wp[2]
        wp_k = wp[4]

        e_lat = -np.sin(wp_psi) * dx + np.cos(wp_psi) * dy
        theta_e = normalize_angle(yaw - wp_psi)

        dt = 0.01
        A = np.array([[1.0, v * dt],
                      [0.0, 1.0]])
        B = np.array([[0.0],
                      [(v / self.wheelbase) * dt]])
        Q = np.diag([2.2, 0.9])
        R = np.array([[0.20]])

        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        delta_fb = -float(np.squeeze(K @ np.array([[e_lat], [theta_e]])))
        delta_ff = np.arctan(self.wheelbase * wp_k)
        steer = np.clip(delta_fb + delta_ff, -self.steer_max, self.steer_max)
        target_v = float(np.clip(wp_table[(idx + 4) % self.n_pts, 3] * 0.95, 0.5, 8.0))
        return float(steer), target_v, idx

    # ── Evaluation Loop ──

    def evaluate_controller(self, name, ctrl_fn, wp_table, max_steps=12000):
        print(f"--> [Nuerburgring] Running: {name} ...")
        env = self.get_env()
        obs, info = env.reset()
        done = False

        traj_x = []
        traj_y = []
        speeds = []
        cte_list = []
        theta_e_list = []
        steer_list = []

        step = 0
        completed = False
        lap_time = 999.0

        while not done and step < max_steps:
            step += 1
            st_arr = obs['agent_0']['state']
            cx, cy, spd, yaw = st_arr[0], st_arr[1], abs(st_arr[3]), st_arr[4]

            steer, target_v, closest_idx = ctrl_fn(st_arr, wp_table)

            wp = wp_table[closest_idx]
            cte = float(np.hypot(cx - wp[0], cy - wp[1]))
            theta_e = float(abs(normalize_angle(wp[2] - yaw)))

            traj_x.append(cx)
            traj_y.append(cy)
            speeds.append(spd)
            cte_list.append(cte)
            theta_e_list.append(theta_e)
            steer_list.append(steer)

            obs, _, terminated, truncated, _ = env.step(np.array([[steer, target_v]]))
            done = terminated or truncated

            if obs['agent_0']['collision']:
                print(f"    [!] Collision at step {step} ({obs['agent_0']['sim_time']:.2f}s)!")
                break

            if obs['agent_0']['lap_count'] >= 1 and obs['agent_0']['lap_time'] > 10.0:
                lap_time = float(np.squeeze(obs['agent_0']['lap_time']))
                completed = True
                print(f"    [+] Lap completed in {lap_time:.2f}s!")
                break

        env.close()
        del env
        import gc
        gc.collect()
        time.sleep(0.2)

        if completed:
            status = "COMPLETED"
            rmse_cte = float(np.sqrt(np.mean(np.square(cte_list))))
            max_cte = float(np.max(cte_list))
            rmse_theta_deg = float(np.degrees(np.sqrt(np.mean(np.square(theta_e_list)))))
            avg_speed = float(np.mean(speeds))
            max_speed = float(np.max(speeds))
            steer_diff = np.diff(steer_list)
            steer_chatter = float(np.std(steer_diff)) if len(steer_diff) > 0 else 0.0
        else:
            status = "COLLISION" if obs['agent_0']['collision'] else "TIMEOUT"
            lap_time = 999.0
            rmse_cte = float(np.sqrt(np.mean(np.square(cte_list)))) if len(cte_list) > 0 else 99.0
            max_cte = float(np.max(cte_list)) if len(cte_list) > 0 else 99.0
            rmse_theta_deg = float(np.degrees(np.sqrt(np.mean(np.square(theta_e_list))))) if len(theta_e_list) > 0 else 99.0
            avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
            max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0
            steer_chatter = float(np.std(np.diff(steer_list))) if len(steer_list) > 1 else 0.0

        return {
            "name": name,
            "status": status,
            "lap_time": round(float(lap_time), 2),
            "avg_speed": round(float(avg_speed), 2),
            "max_speed": round(float(max_speed), 2),
            "rmse_cte": round(float(rmse_cte), 3),
            "max_cte": round(float(max_cte), 3),
            "rmse_theta_deg": round(float(rmse_theta_deg), 2),
            "steer_chatter": round(float(steer_chatter), 4),
            "traj_x": traj_x,
            "traj_y": traj_y,
            "cte_list": cte_list,
            "speeds": speeds
        }

    def run_all(self):
        controllers = [
            ("Pure Pursuit (Baseline: Fixed L, Fixed Speed)", self.ctrl_pure_pursuit_baseline, self.wp_std),
            ("Pure Pursuit (Speed Profiled)", self.ctrl_pure_pursuit_profiled, self.wp_std),
            ("Advanced Pure Pursuit (Adaptive L(v) + Interp)", self.ctrl_advanced_pure_pursuit, self.wp_std),
            ("Stanley (Baseline Untuned)", self.ctrl_stanley_baseline, self.wp_std),
            ("Stanley (Optuna Tuned)", self.ctrl_stanley_tuned, self.wp_tuned),
            ("LQR (Discrete State Feedback)", self.ctrl_lqr, self.wp_std),
        ]
        results = []
        for name, fn, wp_table in controllers:
            res = self.evaluate_controller(name, fn, wp_table)
            results.append(res)
        return results

    def save_and_plot(self, results):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "benchmark_nuerburgring_results.csv")

        headers = ["Controller", "Status", "Lap Time (s)", "Avg Speed (m/s)", "Max Speed (m/s)",
                   "RMSE Cross-Track Error (m)", "Max Cross-Track Error (m)",
                   "RMSE Heading Error (deg)", "Steering Chatter (rad/s)"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in results:
                writer.writerow([
                    r["name"], r["status"], r["lap_time"], r["avg_speed"], r["max_speed"],
                    r["rmse_cte"], r["max_cte"], r["rmse_theta_deg"], r["steer_chatter"]
                ])
        print(f"\n[+] Saved Nuerburgring results to: {csv_path}")

        # Visual Plots
        fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=200)
        completed_res = [r for r in results if r["status"] == "COMPLETED"]
        names = [r["name"].split('(')[0].strip() for r in completed_res]
        times = [r["lap_time"] for r in completed_res]
        colors = ['#4A90E2', '#50E3C2', '#B8E986', '#F5A623', '#D0021B', '#9013FE']

        # 1. Lap Times
        axes[0].barh(names, times, color=colors[:len(names)], edgecolor='black', alpha=0.85)
        axes[0].set_xlabel("Lap Time (s)", fontsize=12, fontweight='bold')
        axes[0].set_title("Nuerburgring Lap Times (Lower is Better)", fontsize=13, fontweight='bold')
        axes[0].grid(axis='x', linestyle='--', alpha=0.6)
        for i, v in enumerate(times):
            axes[0].text(v + 0.3, i, f"{v:.2f}s", va='center', fontweight='bold', fontsize=11)

        # 2. Tracking Error
        ctes = [r["rmse_cte"] for r in completed_res]
        axes[1].barh(names, ctes, color='#FF6B6B', edgecolor='black', alpha=0.85)
        axes[1].set_xlabel("Cross-Track Error RMSE (m)", fontsize=12, fontweight='bold')
        axes[1].set_title("Path Tracking Fidelity (Lower is Better)", fontsize=13, fontweight='bold')
        axes[1].grid(axis='x', linestyle='--', alpha=0.6)
        for i, v in enumerate(ctes):
            axes[1].text(v + 0.005, i, f"{v:.3f}m", va='center', fontweight='bold', fontsize=11)

        # 3. Trajectory Overlay
        axes[2].plot(self.xy[:, 0], self.xy[:, 1], 'k--', linewidth=1.5, label='Optimal Raceline', alpha=0.7)
        for r in completed_res:
            short_name = r["name"].split('(')[0].strip()
            axes[2].plot(r["traj_x"], r["traj_y"], linewidth=1.2, label=short_name)

        axes[2].set_title("Nuerburgring Trajectories vs Raceline", fontsize=13, fontweight='bold')
        axes[2].set_xlabel("X (m)", fontsize=11)
        axes[2].set_ylabel("Y (m)", fontsize=11)
        axes[2].legend(loc='lower right', fontsize=8)
        axes[2].axis('equal')
        axes[2].grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(script_dir, "benchmark_nuerburgring_summary.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[+] Saved visualization to: {plot_path}")

if __name__ == "__main__":
    runner = NuerburgringBenchmark()
    results = runner.run_all()
    runner.save_and_plot(results)
