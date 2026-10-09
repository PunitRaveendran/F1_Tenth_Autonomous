"""
Optuna Hyperparameter Tuner for Pure Pursuit Controller on F1TENTH Gym.
Tunes lookahead distances, velocity scaling, and steering gains to minimize lap time.
"""

import optuna
import gymnasium as gym
import numpy as np
import csv
import os
import time
import threading
from scipy.spatial import KDTree
from f1tenth_gym.envs.env_config import EnvConfig

_tls = threading.local()
_ready = False
_lock = threading.Lock()
_waypoints = None
_kdtree = None
_map_path = None

def load_raceline(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for r in reader:
            if not r or r[0].startswith('#') or r[0].strip() == 's_m' or len(r) < 6:
                continue
            try:
                # s, x, y, psi, kappa, vx, ax
                rows.append([float(r[1]), float(r[2]), 0.0, float(r[5]), float(r[4]), float(r[0])])
            except ValueError:
                continue
    geom = np.array(rows)
    n = len(geom)
    for i in range(n):
        dx = geom[(i + 1) % n, 0] - geom[i, 0]
        dy = geom[(i + 1) % n, 1] - geom[i, 1]
        geom[i, 2] = np.arctan2(dy, dx)
    return geom  # x, y, psi, v, kappa, s

def init_globals():
    global _ready, _waypoints, _kdtree, _map_path
    if _ready:
        return
    with _lock:
        if _ready:
            return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _map_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'Spielberg')
        raceline_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'raceline.csv')
        _waypoints = load_raceline(raceline_path)
        _kdtree = KDTree(_waypoints[:, :2])
        _ready = True
        print(f"Loaded {len(_waypoints)} waypoints for Pure Pursuit Tuner. Ready.")

def get_env():
    if not hasattr(_tls, 'env'):
        config = EnvConfig(map_name=_map_path, num_agents=1, render_enabled=False)
        _tls.env = gym.make('f1tenth_gym:f1tenth-v0', config=config)
    return _tls.env

def run_pure_pursuit_lap(env, waypoints, kdtree, L_min, L_max, speed_scale, steer_gain, max_lat_accel=6.5):
    obs, info = env.reset()
    done = False
    wb = 0.33
    steer_max = 0.41
    n = len(waypoints)

    max_steps = 3500
    step = 0

    while not done and step < max_steps:
        step += 1
        st = obs['agent_0']['state']
        cx, cy, spd, yaw = st[0], st[1], abs(st[3]), st[4]

        # 1. Adaptive lookahead
        lookahead = L_min + (spd / 8.0) * (L_max - L_min)
        lookahead = float(np.clip(lookahead, L_min, L_max))

        # 2. Nearest point
        _, closest_idx = kdtree.query([cx, cy])

        # 3. Find lookahead target with segment interpolation
        target_found = False
        target_pt = None
        for i in range(closest_idx, closest_idx + 120):
            idx = i % n
            wp = waypoints[idx]
            d = np.hypot(wp[0] - cx, wp[1] - cy)

            dx = wp[0] - cx
            dy = wp[1] - cy
            local_x = np.cos(yaw) * dx + np.sin(yaw) * dy

            if d >= lookahead and local_x > 0.0:
                prev_idx = (i - 1) % n
                prev_wp = waypoints[prev_idx]
                v_seg = wp[:2] - prev_wp[:2]
                w_seg = prev_wp[:2] - np.array([cx, cy])
                c = np.dot(w_seg, w_seg) - lookahead ** 2
                b = 2.0 * np.dot(v_seg, w_seg)
                a = np.dot(v_seg, v_seg)
                disc = b * b - 4 * a * c
                if disc >= 0.0 and a > 1e-9:
                    t_cand = (-b + np.sqrt(disc)) / (2.0 * a)
                    if 0.0 <= t_cand <= 1.0:
                        target_pt = prev_wp[:2] + t_cand * v_seg
                        target_found = True
                        break
                target_pt = wp[:2]
                target_found = True
                break

        if not target_found:
            target_pt = waypoints[(closest_idx + 8) % n, :2]

        # Transform target to vehicle frame
        dx = target_pt[0] - cx
        dy = target_pt[1] - cy
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy

        L_actual = np.hypot(local_x, local_y)
        if L_actual > 1e-3:
            curvature = 2.0 * local_y / (L_actual ** 2)
            steering = np.arctan(wb * curvature) * steer_gain
        else:
            steering = 0.0
        steering = float(np.clip(steering, -steer_max, steer_max))

        # Velocity control (latency compensation: read ahead + curvature clamp)
        speed_idx = (closest_idx + 3) % n
        raw_speed = waypoints[speed_idx, 3] * speed_scale
        kappa = abs(waypoints[speed_idx, 4])
        if kappa > 1e-3:
            grip_speed = np.sqrt(max_lat_accel / kappa)
            target_v = min(raw_speed, grip_speed)
        else:
            target_v = raw_speed

        target_v = float(np.clip(target_v, 0.5, 8.0 * speed_scale))

        obs, _, terminated, truncated, _ = env.step(np.array([[steering, target_v]]))
        done = terminated or truncated

        if obs['agent_0']['collision']:
            return 999.0
        if obs['agent_0']['lap_count'] >= 1 and obs['agent_0']['lap_time'] > 10.0:
            return obs['agent_0']['lap_time']

    return 999.0

def objective(trial):
    init_globals()
    L_min       = trial.suggest_float("L_min", 0.6, 1.4)
    L_max       = trial.suggest_float("L_max", 1.4, 2.8)
    speed_scale = trial.suggest_float("speed_scale", 0.75, 1.15)
    steer_gain  = trial.suggest_float("steer_gain", 0.85, 1.35)
    max_lat     = trial.suggest_float("max_lat_accel", 5.0, 9.0)

    env = get_env()
    return run_pure_pursuit_lap(env, _waypoints, _kdtree, L_min, L_max, speed_scale, steer_gain, max_lat)

if __name__ == "__main__":
    print("=" * 60)
    print("  Optuna Pure Pursuit Tuner (Spielberg)")
    print("=" * 60)
    init_globals()
    study = optuna.create_study(direction="minimize", study_name="pure_pursuit_optuna")
    study.optimize(objective, n_trials=75, n_jobs=8)

    valid = [t for t in study.trials if t.value is not None and t.value < 900.0]
    valid.sort(key=lambda t: t.value)
    print(f"\nTop Pure Pursuit Trials (out of {len(study.trials)}):")
    for i, t in enumerate(valid[:5]):
        print(f"  {i+1}. {t.value:.2f}s | L_min={t.params['L_min']:.2f}, L_max={t.params['L_max']:.2f}, "
              f"speed_scale={t.params['speed_scale']:.3f}, steer_gain={t.params['steer_gain']:.2f}, max_lat={t.params['max_lat_accel']:.2f}")

    best = study.best_params
    print("\nBest Parameters:", best)
    print(f"Best Lap Time: {study.best_value:.2f}s")
