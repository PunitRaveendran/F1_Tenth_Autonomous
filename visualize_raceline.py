#!/usr/bin/env python3
"""
Visualize the raceline overlaid on the map image, color-coded by speed.
Also shows the centerline for comparison.

Usage:
    python3 visualize_raceline.py
    python3 visualize_raceline.py --map_name Spielberg_map
    python3 visualize_raceline.py --map_name Spielberg_map --no-centerline
"""

import argparse
import os
import sys
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import yaml


def load_map(map_dir, map_name):
    """Load the map image and metadata."""
    yaml_path = os.path.join(map_dir, f"{map_name}.yaml")
    img_path = os.path.join(map_dir, f"{map_name}.png")

    if not os.path.exists(yaml_path):
        print(f"Error: Map YAML not found: {yaml_path}")
        sys.exit(1)
    if not os.path.exists(img_path):
        print(f"Error: Map image not found: {img_path}")
        sys.exit(1)

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot read image: {img_path}")
        sys.exit(1)

    return img, meta


def world_to_px(wx, wy, resolution, origin, img_height):
    """Convert world coordinates to pixel coordinates."""
    px = (wx - origin[0]) / resolution
    py = img_height - (wy - origin[1]) / resolution
    return px, py


def load_raceline(csv_path):
    """Load a raceline CSV (s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2)."""
    x, y, v = [], [], []
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if not row or row[0].startswith('#') or row[0] == 's_m' or len(row) < 6:
                continue
            try:
                x.append(float(row[1]))
                y.append(float(row[2]))
                v.append(float(row[5]))
            except ValueError:
                continue
    return np.array(x), np.array(y), np.array(v)


def load_centerline(csv_path):
    """Load a centerline CSV (x_m, y_m, w_tr_right_m, w_tr_left_m)."""
    x, y = [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if not row or len(row) < 2:
                continue
            x.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(x), np.array(y)


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_map_dir = os.path.join(_script_dir, 'src', 'f1tenth_gym_ros', 'maps')

    parser = argparse.ArgumentParser(description="Visualize raceline on map")
    parser.add_argument('--map_name', type=str, default='Spielberg_map',
                        help='Name of the map (without .yaml/.png)')
    parser.add_argument('--map_dir', type=str,
                        default=_default_map_dir,
                        help='Directory containing the maps')
    parser.add_argument('--no-centerline', action='store_true',
                        help='Do not show the centerline')
    args = parser.parse_args()

    # ── Load map ──────────────────────────────────────────
    img, meta = load_map(args.map_dir, args.map_name)
    resolution = meta['resolution']
    origin = meta['origin']

    # ── Load raceline ─────────────────────────────────────
    # Try map-specific raceline first, then fall back to default
    raceline_path = os.path.join(args.map_dir, f"{args.map_name}_raceline.csv")
    if not os.path.exists(raceline_path):
        raceline_path = os.path.join(args.map_dir, "raceline.csv")
    if not os.path.exists(raceline_path):
        print(f"Error: No raceline found. Run ./generate_raceline.sh {args.map_name} first.")
        sys.exit(1)

    print(f"Loading raceline from: {raceline_path}")
    rx, ry, rv = load_raceline(raceline_path)

    # ── Load centerline (optional) ────────────────────────
    show_centerline = not args.no_centerline
    cx, cy = None, None
    if show_centerline:
        cl_path = os.path.join(args.map_dir, f"{args.map_name}_centerline.csv")
        if not os.path.exists(cl_path):
            cl_path = os.path.join(args.map_dir, "centerline.csv")
        if os.path.exists(cl_path):
            print(f"Loading centerline from: {cl_path}")
            cx, cy = load_centerline(cl_path)
        else:
            print("No centerline file found, skipping.")
            show_centerline = False

    # ── Convert world → pixel ─────────────────────────────
    rpx, rpy = [], []
    for wx, wy in zip(rx, ry):
        px, py = world_to_px(wx, wy, resolution, origin, img.shape[0])
        rpx.append(px)
        rpy.append(py)
    rpx, rpy = np.array(rpx), np.array(rpy)

    cpx, cpy = None, None
    if show_centerline and cx is not None:
        cpx, cpy = [], []
        for wx, wy in zip(cx, cy):
            px, py = world_to_px(wx, wy, resolution, origin, img.shape[0])
            cpx.append(px)
            cpy.append(py)
        cpx, cpy = np.array(cpx), np.array(cpy)

    # ── Plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 14))

    # Show map as background
    ax.imshow(img, cmap='gray', origin='upper', alpha=0.6)

    # Plot centerline as a thin dashed white line
    if show_centerline and cpx is not None:
        # Close the loop for centerline
        cpx_closed = np.append(cpx, cpx[0])
        cpy_closed = np.append(cpy, cpy[0])
        ax.plot(cpx_closed, cpy_closed, '--', color='white', linewidth=1.0,
                alpha=0.7, label='Centerline', zorder=2)

    # Plot raceline as a color-coded line (speed)
    # Close the loop
    rpx_closed = np.append(rpx, rpx[0])
    rpy_closed = np.append(rpy, rpy[0])
    rv_closed = np.append(rv, rv[0])

    # Create line segments for LineCollection
    points = np.array([rpx_closed, rpy_closed]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color map: slow = red, fast = green
    norm = mcolors.Normalize(vmin=rv.min(), vmax=rv.max())
    cmap = plt.cm.RdYlGn  # Red → Yellow → Green

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.5, zorder=3)
    lc.set_array(rv_closed[:-1])
    ax.add_collection(lc)

    # Add colorbar
    cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Speed (m/s)', fontsize=12)

    # Mark start point
    ax.plot(rpx[0], rpy[0], 'o', color='cyan', markersize=10, zorder=5,
            label='Start / Finish')

    # Title and legend
    ax.set_title(f'Raceline Visualization — {args.map_name}\n'
                 f'Speed range: {rv.min():.1f} – {rv.max():.1f} m/s  |  '
                 f'Avg: {rv.mean():.1f} m/s  |  {len(rv)} waypoints',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_axis_off()

    plt.tight_layout()

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'raceline_visualization.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print(f"\nSaved visualization to: {out_path}")

    # Also try to show interactively
    try:
        plt.show()
    except Exception:
        print("(Non-interactive backend, saved to file only)")


if __name__ == '__main__':
    main()
