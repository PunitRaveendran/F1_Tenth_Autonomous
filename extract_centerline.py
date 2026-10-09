import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
import yaml
import csv
import matplotlib.pyplot as plt

import argparse
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MAP_DIR = os.path.join(_SCRIPT_DIR, 'src', 'f1tenth_gym_ros', 'maps')

parser = argparse.ArgumentParser(description="Extract centerline from a map image")
parser.add_argument('--map_name', type=str, default='Spielberg_map', help='Name of the map (without .yaml/.png)')
parser.add_argument('--map_dir', type=str, default=_DEFAULT_MAP_DIR, help='Directory containing the maps')
args = parser.parse_args()

# ── Config ──────────────────────────────────────────────
MAP_IMG  = os.path.join(args.map_dir, f"{args.map_name}.png")
MAP_YAML = os.path.join(args.map_dir, f"{args.map_name}.yaml")
OUT_CSV  = os.path.join(args.map_dir, f"{args.map_name}_centerline.csv")
# ────────────────────────────────────────────────────────

# Load map metadata
with open(MAP_YAML) as f:
    meta = yaml.safe_load(f)

resolution = meta['resolution']
origin     = meta['origin']  # [x, y, theta]
free_thresh = meta.get('free_thresh', 0.196)
negate      = meta.get('negate', 0)

# Load image
img = cv2.imread(MAP_IMG, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Cannot load {MAP_IMG}")

if negate:
    img = 255 - img

# Extract full free space (>200 means free, typical for ROS maps 254/255 are free)
free_mask_full = (img > 200).astype(np.uint8) * 255

kernel = np.ones((3,3), np.uint8)
free_mask_full = cv2.morphologyEx(free_mask_full, cv2.MORPH_OPEN, kernel, iterations=1)

# Flood fill from world origin (0,0) to extract ONLY the track (ignoring outside boundaries)
px_origin = int((0 - origin[0]) / resolution)
py_origin = img.shape[0] - int((0 - origin[1]) / resolution)

# Use floodFill to color the track as 128
mask = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)
try:
    cv2.floodFill(free_mask_full, mask, (px_origin, py_origin), 128)
except Exception as e:
    print(f"Warning: floodfill failed from {px_origin}, {py_origin}. Ensure it's inside the track.")

free_mask = (free_mask_full == 128).astype(np.uint8)

# Clean up noise
free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Skeletonize to get centerline
skeleton = skeletonize(free_mask).astype(np.uint8)

# Get skeleton pixel coordinates
ys, xs = np.where(skeleton > 0)
points_px = np.column_stack([xs, ys])

if len(points_px) == 0:
    raise RuntimeError("No skeleton found — check map thresholds")

print(f"Found {len(points_px)} skeleton points")

# Convert pixels → world coordinates
def px_to_world(px, py, resolution, origin):
    wx = px * resolution + origin[0]
    wy = (img.shape[0] - py) * resolution + origin[1]
    return wx, wy

def world_to_px(wx, wy, resolution, origin):
    px = (wx - origin[0]) / resolution
    py = img.shape[0] - (wy - origin[1]) / resolution
    return int(px), int(py)

# Calculate width at each pixel BEFORE ordering and conversions
dist_px = ndimage.distance_transform_edt(free_mask)
widths_unordered = dist_px[ys, xs] * resolution * 2

world_points = []
for px, py in points_px:
    wx, wy = px_to_world(px, py, resolution, origin)
    world_points.append([wx, wy])
world_points = np.array(world_points)

# Order points along the track using nearest neighbor
def order_points_indices(pts):
    ordered_idx = [0]
    remaining = list(range(1, len(pts)))
    while remaining:
        last = pts[ordered_idx[-1]]
        dists = np.linalg.norm(pts[remaining] - last, axis=1)
        idx = np.argmin(dists)
        ordered_idx.append(remaining.pop(idx))
    return ordered_idx

print("Ordering points along track...")
ordered_idx = order_points_indices(world_points)
ordered = world_points[ordered_idx]
ordered_widths = widths_unordered[ordered_idx]

# Fix 1 & 2: Spline smoothing + equal arc-length resampling at 0.4m intervals
# Close the loop
closed_ordered = np.vstack((ordered, ordered[0]))
closed_ordered_widths = np.append(ordered_widths, ordered_widths[0])

# Fit spline, s is smoothing factor
tck, u = splprep([closed_ordered[:, 0], closed_ordered[:, 1]], s=2.0, per=1)

# Fit 1D spline for widths using the same 'u' parameterization
from scipy.interpolate import splrep
tck_w = splrep(u, closed_ordered_widths, s=0, per=True)

# Evaluate spline at fine resolution
u_fine = np.linspace(0, 1, 10000)
x_fine, y_fine = splev(u_fine, tck)
w_fine = splev(u_fine, tck_w)

# Compute arc length
dx = np.diff(x_fine)
dy = np.diff(y_fine)
ds = np.sqrt(dx**2 + dy**2)
s = np.insert(np.cumsum(ds), 0, 0)

# Resample at 0.4m intervals
total_length = s[-1]
s_target = np.arange(0, total_length, 0.4)
x_resampled = np.interp(s_target, s, x_fine)
y_resampled = np.interp(s_target, s, y_fine)
w_resampled = np.interp(s_target, s, w_fine)

unnormalized_waypoints = np.column_stack((x_resampled, y_resampled))

# Fix 3: Keep in global map frame (removed normalization)
waypoints = unnormalized_waypoints

# Fix 4: Smooth + clamp widths (carried through via interpolation)
raw_widths = w_resampled

# 15-point moving average (with wrap-around since it's a closed track)
window = 15
padded_widths = np.pad(raw_widths, (window//2, window//2), mode='wrap')
smooth_widths = np.convolve(padded_widths, np.ones(window)/window, mode='valid')

# Clamp between 1.0 and 1.2
final_widths = np.clip(smooth_widths, 1.0, 1.2)

# Save CSV
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
    for i, (x, y) in enumerate(waypoints):
        w = final_widths[i] / 2
        writer.writerow([round(x, 6), round(y, 6), round(w, 3), round(w, 3)])

print(f"Saved {len(waypoints)} waypoints to {OUT_CSV}")

# Visualize
plt.figure(figsize=(10, 10))
plt.imshow(free_mask, cmap='gray', origin='upper')

# Plot original skeleton points in light red
skel_ys, skel_xs = np.where(skeleton > 0)
plt.scatter(skel_xs, skel_ys, s=1, c='lightcoral', alpha=0.5, label='Skeleton')

# Plot resampled waypoints in bold red
# Map resampled waypoints back to pixels for plotting
wp_px = []
for wx, wy in unnormalized_waypoints:
    px, py = world_to_px(wx, wy, resolution, origin)
    wp_px.append([px, py])
wp_px = np.array(wp_px)

plt.scatter(wp_px[:, 0], wp_px[:, 1], s=10, c='red', label='Resampled 0.4m')
plt.title('Spline-Smoothed Centerline (0.4m interval)')
plt.legend()
_preview_path = os.path.join(_SCRIPT_DIR, 'centerline_preview.png')
plt.savefig(_preview_path)
plt.close()
print(f"Preview saved to {_preview_path}")
