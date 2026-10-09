import optuna
import gymnasium as gym
import numpy as np
import csv
import os
import time
import threading
import shutil
from scipy.spatial import KDTree

from f1tenth_gym.envs.env_config import EnvConfig

# ── Geometry loader (immutable, shared across threads) ──────────────

def load_raceline_geometry(path):
    """Load raceline CSV → array of [x, y, psi, kappa, s]."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if not row or row[0].startswith('#') or row[0].strip() == 's_m' or len(row) < 6:
                continue
            try:
                rows.append([float(row[1]), float(row[2]), 0.0,
                             float(row[4]), float(row[0])])
            except ValueError:
                continue
    geom = np.array(rows)
    n = len(geom)
    for i in range(n):
        dx = geom[(i + 1) % n, 0] - geom[i, 0]
        dy = geom[(i + 1) % n, 1] - geom[i, 1]
        geom[i, 2] = np.arctan2(dy, dx)
    return geom  # cols: x, y, psi, kappa, s


# ── Numba-accelerated speed profile ───────────────────────────────

def _speed_profile_core(kappa_abs, ds, v_max, a_lat, a_lon, a_brake):
    """Pure-numba forward/backward pass — no Python overhead."""
    n = len(kappa_abs)
    v = np.empty(n)

    # Curvature speed limit
    for i in range(n):
        v[i] = min(np.sqrt(a_lat / kappa_abs[i]), v_max)

    # Forward pass (acceleration limit)
    for i in range(1, n):
        d = ds[i - 1]
        v_new = np.sqrt(max(0.0, v[i - 1] ** 2 + 2.0 * a_lon * d))
        if v_new < v[i]:
            v[i] = v_new

    # Backward pass (braking limit)
    for i in range(n - 2, -1, -1):
        d = ds[i]
        v_new = np.sqrt(max(0.0, v[i + 1] ** 2 + 2.0 * a_brake * d))
        if v_new < v[i]:
            v[i] = v_new

    # Clip
    for i in range(n):
        if v[i] < 0.5:
            v[i] = 0.5
        elif v[i] > v_max:
            v[i] = v_max

    return v


def compute_speed_profile(kappa_abs, ds, v_max, a_lat_max, a_lon_max, a_lon_brake):
    """Wrapper that calls the numba kernel."""
    return _speed_profile_core(kappa_abs, ds, v_max, a_lat_max, a_lon_max, a_lon_brake)


# ── Stanley controller ─────────────────────────────────────────────

def _normalize(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def run_stanley_lap(env, waypoints, kdtree, k_gain, k_soft, speed_scale, lookahead):
    obs, info = env.reset()
    done = False
    wb = 0.33
    steer_max = 0.41
    n = len(waypoints)

    while not done:
        st = obs['agent_0']['state']
        cx, cy, spd, yaw = st[0], st[1], abs(st[3]), st[4]

        fx = cx + wb * np.cos(yaw)
        fy = cy + wb * np.sin(yaw)

        # KDTree lookup: O(log n) instead of O(n)
        _, idx = kdtree.query([fx, fy])
        wp_x, wp_y, wp_psi, wp_v, wp_k = waypoints[idx]

        theta_e = _normalize(wp_psi - yaw)
        e_fa = np.sin(wp_psi) * (fx - wp_x) - np.cos(wp_psi) * (fy - wp_y)

        delta = theta_e + np.arctan2(k_gain * e_fa, spd + k_soft) + np.arctan(wb * wp_k)
        steering = np.clip(delta, -steer_max, steer_max)

        target_v = waypoints[(idx + lookahead) % n, 3] * speed_scale
        if abs(theta_e) > np.pi / 3:
            target_v *= 0.5
        target_v = np.clip(target_v, 0.5, 8.0 * speed_scale)

        obs, _, terminated, truncated, _ = env.step(np.array([[steering, target_v]]))
        done = terminated or truncated

        if obs['agent_0']['collision']:
            return 999.0
        if obs['agent_0']['lap_count'] >= 1 and obs['agent_0']['lap_time'] > 10.0:
            return obs['agent_0']['lap_time']

    return 999.0


# ── Thread-safe globals ────────────────────────────────────────────

_tls = threading.local()
_ready = False         # gate flag — set LAST after all arrays are initialized
_geometry = None       # [x, y, psi, kappa, s]
_kappa_abs = None      # pre-computed abs(kappa), clamped
_ds = None             # pre-computed arc-length diffs
_xy = None             # pre-computed (x,y) for building waypoints
_psi = None            # pre-computed psi
_kappa_raw = None      # pre-computed kappa (signed)
_kdtree = None         # shared KDTree on (x,y) — geometry is constant
_map_path = None
_lock = threading.Lock()


def _init_globals():
    global _ready, _geometry, _kappa_abs, _ds, _xy, _psi, _kappa_raw, _kdtree, _map_path
    if _ready:
        return
    with _lock:
        if _ready:
            return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _map_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'Spielberg')
        raceline_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'Spielberg_map_raceline.csv')
        geom = load_raceline_geometry(raceline_path)

        # Pre-compute arrays (immutable, safe to share)
        _xy = geom[:, :2].copy()
        _psi = geom[:, 2].copy()
        _kappa_raw = geom[:, 3].copy()
        _kappa_abs = np.maximum(np.abs(_kappa_raw), 1e-6)

        s = geom[:, 4]
        ds = np.diff(s)
        median_ds = np.median(ds[ds > 0]) if np.any(ds > 0) else 0.2
        ds[ds <= 0] = median_ds
        _ds = ds

        # KDTree on (x,y) — built once, reused by all threads & trials
        _kdtree = KDTree(_xy)

        _geometry = geom  # set second-to-last
        _ready = True      # gate opens — all arrays guaranteed initialized
        print(f"Loaded {len(_geometry)} points. Ready.")


def _get_env():
    """One env per thread — avoids race conditions."""
    if not hasattr(_tls, 'env'):
        config = EnvConfig(map_name=_map_path, num_agents=1, render_enabled=False)
        _tls.env = gym.make('f1tenth_gym:f1tenth-v0', config=config)
    return _tls.env


# ── Optuna objective ───────────────────────────────────────────────

def objective(trial):
    _init_globals()

    # Controller params
    k_gain      = trial.suggest_float("k_gain", 1.0, 5.0)
    k_soft      = trial.suggest_float("k_soft", 0.01, 2.0)
    speed_scale = trial.suggest_float("speed_scale", 0.85, 1.25)
    lookahead   = trial.suggest_int("lookahead", 1, 10)

    # Speed profile physics params
    a_lat_max   = trial.suggest_float("a_lat_max", 5.0, 11.0)
    a_lon_max   = trial.suggest_float("a_lon_max", 3.0, 9.0)
    a_lon_brake = trial.suggest_float("a_lon_brake", 5.0, 14.0)

    # Compute speed profile (numba-fast)
    speeds = compute_speed_profile(_kappa_abs, _ds, 8.0, a_lat_max, a_lon_max, a_lon_brake)

    # Build waypoints: [x, y, psi, vx, kappa]
    waypoints = np.column_stack([_xy, _psi, speeds, _kappa_raw])

    env = _get_env()
    return run_stanley_lap(env, waypoints, _kdtree, k_gain, k_soft, speed_scale, lookahead)


# ── Regenerate raceline CSV with optimal physics ───────────────────

def regenerate_raceline(best_params):
    """Rewrite the raceline CSV with the optimized speed profile."""
    _init_globals()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps')
    raceline_path = os.path.join(map_dir, 'Spielberg_map_raceline.csv')

    speeds = compute_speed_profile(
        _kappa_abs, _ds,
        8.0, best_params['a_lat_max'], best_params['a_lon_max'], best_params['a_lon_brake']
    )

    # Recompute ax
    ax = np.diff(speeds ** 2) / (2 * _ds + 1e-9)
    ax = np.append(ax, ax[-1])

    s_vals = _geometry[:, 4]

    # Read original headers
    with open(raceline_path, 'r') as f:
        lines = f.readlines()
    headers = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('s_m'):
            headers.append(line)
        else:
            break

    with open(raceline_path, 'w', newline='') as f:
        for h in headers:
            f.write(h)
        for i in range(len(_geometry)):
            f.write(f"{s_vals[i]:.6f};{_xy[i, 0]:.6f};{_xy[i, 1]:.6f};"
                    f"{_psi[i]:.6f};{_kappa_raw[i]:.6f};"
                    f"{speeds[i]:.4f};{ax[i]:.4f}\n")

    shutil.copyfile(raceline_path, os.path.join(map_dir, 'raceline.csv'))
    print(f"✅ Raceline updated: {raceline_path}")
    print(f"   Speed range: {speeds.min():.2f} – {speeds.max():.2f} m/s")


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Enhanced Optuna Stanley Tuner v2")
    print("  Tuning 7 params: controller (4) + physics (3)")
    print("  Optimizations: KDTree + Numba JIT + thread-local envs")
    print("=" * 60)

    study = optuna.create_study(direction="minimize", study_name="stanley_v2")

    start = time.time()
    try:
        study.optimize(objective, n_trials=200, n_jobs=16)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    elapsed = time.time() - start
    best = study.best_params

    valid_trials = [t for t in study.trials if t.value is not None and t.value < 900.0]
    valid_trials.sort(key=lambda t: t.value)
    top_5 = valid_trials[:5]

    print("\n" + "=" * 60)
    print(f"  Finished in {elapsed:.1f}s  |  Top {len(top_5)} Laps:")
    for i, t in enumerate(top_5):
        print(f"\n  {i+1}. {t.value:.2f}s  (Trial {t.number})")
        print(f"     k_gain: {t.params['k_gain']:.4f}  |  k_soft: {t.params['k_soft']:.4f}  |  speed_scale: {t.params['speed_scale']:.4f}  |  lookahead: {t.params['lookahead']}")
        print(f"     a_lat_max: {t.params['a_lat_max']:.2f}  |  a_lon_max: {t.params['a_lon_max']:.2f}  |  a_lon_brake: {t.params['a_lon_brake']:.2f}")
    print("=" * 60)

    best = study.best_params
    print("\n── Regenerating raceline with absolute optimal physics... ──")
    regenerate_raceline(best)

    print("=" * 60)
