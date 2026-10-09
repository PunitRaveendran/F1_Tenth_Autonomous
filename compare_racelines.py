import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_raceline(csv_path):
    """Load a raceline CSV (s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2)."""
    s, x, y, v = [], [], [], []
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if not row or row[0].startswith('#') or row[0].strip() == 's_m' or len(row) < 6:
                continue
            try:
                s.append(float(row[0]))
                x.append(float(row[1]))
                y.append(float(row[2]))
                v.append(float(row[5]))
            except ValueError:
                continue
    return np.array(s), np.array(x), np.array(y), np.array(v)

def analyze_raceline(name, x, y, v):
    # compute distances between points
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    # distance between last and first point (assuming closed loop)
    d_close = np.sqrt((x[0]-x[-1])**2 + (y[0]-y[-1])**2)
    
    total_length = np.sum(distances) + d_close
    
    # Calculate expected laptime (sum of dt = ds / v)
    # Using v[:-1] for distances, and v[-1] for d_close
    dt = distances / v[:-1]
    dt_close = d_close / v[-1]
    laptime = np.sum(dt) + dt_close
    
    avg_speed = np.mean(v)
    max_speed = np.max(v)
    min_speed = np.min(v)
    
    print(f"=== {name} ===")
    print(f"Points:      {len(x)}")
    print(f"Path length: {total_length:.2f} m")
    print(f"Est Laptime: {laptime:.2f} s")
    print(f"Avg Speed:   {avg_speed:.2f} m/s")
    print(f"Max Speed:   {max_speed:.2f} m/s")
    print(f"Min Speed:   {min_speed:.2f} m/s")
    print("")
    return total_length, laptime

def main():
    generated_csv = "src/f1tenth_gym_ros/maps/Spielberg_map_raceline.csv"
    official_csv = "Spielberg_raceline.csv"
    
    gen_s, gen_x, gen_y, gen_v = load_raceline(generated_csv)
    off_s, off_x, off_y, off_v = load_raceline(official_csv)
    
    analyze_raceline("Generated Raceline", gen_x, gen_y, gen_v)
    analyze_raceline("Official Raceline", off_x, off_y, off_v)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Plot official
    plt.plot(off_x, off_y, label='Official Raceline', color='blue', linewidth=2, linestyle='--')
    
    # Plot generated
    plt.plot(gen_x, gen_y, label='Generated Raceline', color='red', linewidth=2, alpha=0.8)
    
    plt.title('Raceline Comparison: Generated vs Official')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    out_img = "raceline_comparison.png"
    plt.savefig(out_img, dpi=150)
    print(f"Saved plot to {out_img}")

if __name__ == '__main__':
    main()
