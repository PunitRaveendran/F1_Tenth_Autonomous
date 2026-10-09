import numpy as np
import csv

# ── F1TENTH Car Parameters (tune these) ──────────────────
V_MAX       = 8.0   # max speed m/s
A_LAT_MAX   = 7.0   # max lateral acceleration m/s²  (F1Tenth tires on grip surface)
A_LON_MAX   = 5.0   # max longitudinal accel m/s²    (VESC motor limit)
A_LON_BRAKE = 9.0   # max braking deceleration m/s²  (slightly conservative for tracking stability)
# ─────────────────────────────────────────────────────────

import argparse
import os
import shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MAP_DIR = os.path.join(_SCRIPT_DIR, 'src', 'f1tenth_gym_ros', 'maps')

parser = argparse.ArgumentParser(description="Generate speed profile for raceline")
parser.add_argument('--map_name', type=str, default='Spielberg_map', help='Name of the map (without .yaml/.png)')
parser.add_argument('--map_dir', type=str, default=_DEFAULT_MAP_DIR, help='Directory containing the maps')
args = parser.parse_args()

INPUT_CSV  = os.path.join(_SCRIPT_DIR, 'global_racetrajectory_optimization', 'outputs', 'traj_race_cl.csv')
OUTPUT_CSV = os.path.join(args.map_dir, f"{args.map_name}_raceline.csv")
DEFAULT_OUTPUT_CSV = os.path.join(args.map_dir, "raceline.csv")

# Load TUM output (semicolon delimited, 3 header lines)
data = np.genfromtxt(INPUT_CSV, delimiter=';', skip_header=3)
s       = data[:, 0]  # arc length
x       = data[:, 1]
y       = data[:, 2]
psi     = data[:, 3]
kappa   = data[:, 4]  # curvature
n       = len(x)

# Step 1 — curvature-based max speed
kappa_safe = np.abs(kappa)
kappa_safe[kappa_safe < 1e-6] = 1e-6  # avoid division by zero
v_curv = np.sqrt(A_LAT_MAX / kappa_safe)
v_max_arr = np.minimum(v_curv, V_MAX)

# Step 2 — forward pass (acceleration limit)
v_fwd = v_max_arr.copy()
ds = np.diff(s)
for i in range(1, n):
    d = ds[i-1] if i < len(ds)+1 else ds[-1]
    v_fwd[i] = min(v_max_arr[i], np.sqrt(v_fwd[i-1]**2 + 2 * A_LON_MAX * d))

# Step 3 — backward pass (braking limit)
v_final = v_fwd.copy()
for i in range(n-2, -1, -1):
    d = ds[i] if i < len(ds) else ds[-1]
    v_final[i] = min(v_final[i], np.sqrt(v_final[i+1]**2 + 2 * A_LON_BRAKE * d))

v_final = np.clip(v_final, 0.5, V_MAX)

# Compute acceleration
ax = np.diff(v_final**2) / (2 * np.diff(s) + 1e-9)
ax = np.append(ax, ax[-1])

# Save raceline CSV
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['# Format: TUM raceline'])
    writer.writerow(['#'])
    writer.writerow(['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2'])
    for i in range(n):
        writer.writerow([
            round(s[i], 6), round(x[i], 6), round(y[i], 6),
            round(psi[i], 6), round(kappa[i], 6),
            round(v_final[i], 4), round(ax[i], 4)
        ])

print(f"Saved {n} waypoints to {OUTPUT_CSV}")
print(f"Speed range: {v_final.min():.2f} to {v_final.max():.2f} m/s")
print(f"Avg speed: {v_final.mean():.2f} m/s")

# Copy to default raceline.csv for backward compatibility
shutil.copyfile(OUTPUT_CSV, DEFAULT_OUTPUT_CSV)
print(f"Also copied to {DEFAULT_OUTPUT_CSV} for external nodes.")

# Quick stats per section
print(f"\nSlow corners (<3 m/s): {(v_final < 3).sum()} waypoints")
print(f"Medium (3-6 m/s):      {((v_final>=3)&(v_final<6)).sum()} waypoints")
print(f"Fast (>6 m/s):         {(v_final >= 6).sum()} waypoints")
