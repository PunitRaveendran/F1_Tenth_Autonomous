# 🏁 Racing on a New Map — Step-by-Step Guide

Complete checklist for taking a new map → generating a raceline → tuning with Optuna → racing.

---

## Step 0: Get Your Map Files

You need **two files** for any map:
- `my_track.png` — the occupancy grid image (white = free, black = walls, gray = unknown)
- `my_track.yaml` — metadata file pointing to the PNG

The YAML should look like this (use Spielberg as a reference):
```yaml
image: my_track.png
resolution: 0.05796        # meters per pixel
origin: [-84.85, -36.30, 0.0]  # [x, y, theta] of the bottom-left pixel in world coords
negate: 0
occupied_thresh: 0.45
free_thresh: 0.196
```

### Option A: Download a Map from the Internet

If you already have a map (e.g. from the F1TENTH community, a competition, or an online source), just make sure you have both the `.png` and `.yaml` files ready. Skip to **Step 1**.

### Option B: Map the Track Yourself Using SLAM

If you're starting from scratch and need to map a physical track (or an unknown simulated environment), use the car's LiDAR to build the map.

1. **First, tell the simulator which environment to load.** Open [sim.yaml](file:///d:/github/f1tenth/src/f1tenth_gym_ros/config/sim.yaml) and update the map path:
   ```yaml
   map_path: 'maps/my_track'
   ```
   Also set `sx`, `sy`, `stheta` to a valid starting pose on that track.

2. **Launch the SLAM mapping stack:**
   ```bash
   ros2 launch f1tenth_gym_ros slam_mapping_launch.py
   ```

3. **In a new terminal, launch teleop to manually drive the car:**
   ```bash
   ros2 run f1tenth_gym_ros teleop_ackermann_node --ros-args -p max_forward_speed:=2.0
   ```
   Use WASD or arrow keys to slowly drive a **complete, closed loop** around the track.

4. **Once the map looks complete and closed in RViz, save it.** Open a new terminal:
   ```bash
   cd ~/f1tenth_ws/src/f1tenth_gym_ros/maps
   ros2 run nav2_map_server map_saver_cli -f my_track
   ```
   This generates `my_track.png` and `my_track.yaml` directly in your maps folder. You can now close the SLAM and teleop terminals.

> [!TIP]
> Drive slowly and make sure you close the loop cleanly — gaps in the track boundary will break the centerline extraction later.

---

## Step 1: Place Map Files in the Maps Folder

Copy both files into:
```
d:\github\f1tenth\src\f1tenth_gym_ros\maps\
```

For example, if your map is called `monza`:
```
src/f1tenth_gym_ros/maps/monza.png
src/f1tenth_gym_ros/maps/monza.yaml
```

> [!TIP]
> Make sure the `image:` field inside the `.yaml` matches the `.png` filename exactly.

---

## Step 2: Generate the Raceline

This is a single command that runs 4 sub-steps automatically:
1. Extract centerline from the map image (skeletonization)
2. Run TUM Global Trajectory Optimization (minimum-curvature racing line)
3. Compute physics-based speed profile
4. Generate a visualization image

**Run from the repo root (`d:\github\f1tenth`):**

```bash
./generate_raceline.sh monza
```

> [!NOTE]
> On Windows/WSL, run this inside your Linux environment where Python + dependencies are installed.

### What it does under the hood:
| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1/4 | [extract_centerline.py](file:///d:/github/f1tenth/extract_centerline.py) | `monza.png` + `monza.yaml` | `monza_centerline.csv` |
| 2/4 | [main_globaltraj.py](file:///d:/github/f1tenth/global_racetrajectory_optimization/main_globaltraj.py) | centerline → `f1tenth_track.csv` | `outputs/traj_race_cl.csv` |
| 3/4 | [speed_profile.py](file:///d:/github/f1tenth/speed_profile.py) | `traj_race_cl.csv` | `monza_raceline.csv` + `raceline.csv` |
| 4/4 | [visualize_raceline.py](file:///d:/github/f1tenth/visualize_raceline.py) | raceline + map | `raceline_visualization.png` |

### ✅ Verify:
- Open `raceline_visualization.png` — you should see a color-coded line (green=fast, red=slow) overlaid on your track
- Check `src/f1tenth_gym_ros/maps/monza_raceline.csv` exists and has data
- Check `src/f1tenth_gym_ros/maps/raceline.csv` was updated (the default file the car reads)

---

## Step 3: Configure Optuna for the New Map

You need to edit **two paths** in [stanley_tuner.py](file:///d:/github/f1tenth/stanley_tuner.py).

### Edit `_init_globals()` (around line 148–151):

**Before (Spielberg hardcoded):**
```python
_map_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'Spielberg')
raceline_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'Spielberg_map_raceline.csv')
```

**After (your new map):**
```python
_map_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'monza')
raceline_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'monza_raceline.csv')
```

> [!IMPORTANT]
> `_map_path` should be the map base name **without** `.png`/`.yaml` extension. The gym loads both files automatically from this base path.

### Also update `regenerate_raceline()` (around line 215):

**Before:**
```python
raceline_path = os.path.join(map_dir, 'Spielberg_map_raceline.csv')
```

**After:**
```python
raceline_path = os.path.join(map_dir, 'monza_raceline.csv')
```

---

## Step 4: Run Optuna

```bash
cd d:\github\f1tenth    # or ~/f1tenth_ws on Linux
python3 stanley_tuner.py
```

This runs **200 trials across 16 threads** (~2 minutes). Each trial:
1. Picks random controller params (k_gain, k_soft, speed_scale, lookahead)
2. Picks random physics params (a_lat_max, a_lon_max, a_lon_brake)
3. Computes a speed profile with those physics
4. Simulates a full lap with the Stanley controller
5. Returns the lap time (or 999 if crashed)

### Output you'll see:
```
══════════════════════════════════════════════════════════
  Finished in 120.3s  |  Top 5 Laps:

  1. 14.52s  (Trial 147)
     k_gain: 3.7769  |  k_soft: 0.3439  |  speed_scale: 1.2401  |  lookahead: 3
     a_lat_max: 8.42  |  a_lon_max: 6.31  |  a_lon_brake: 11.28
══════════════════════════════════════════════════════════
```

It also **automatically rewrites** the raceline CSV with the optimal speed profile.

---

## Step 5: Update the Stanley Node with Optuna's Best Params

Open [stanley_node.py](file:///d:/github/f1tenth/src/gap_follow/gap_follow/stanley_node.py) and update **lines 23–28** with the best values from Optuna:

```python
self.declare_parameter('k_gain',        3.7769)   # ← from Optuna
self.declare_parameter('k_soft',        0.3439)   # ← from Optuna
self.declare_parameter('speed_scale',   1.2401)   # ← from Optuna
self.declare_parameter('lookahead',     3)        # ← from Optuna
```

Then rebuild:
```bash
cd ~/f1tenth_ws
colcon build --symlink-install
```

---

## Step 6: Point the Simulator at the New Map

Edit [sim.yaml](file:///d:/github/f1tenth/src/f1tenth_gym_ros/config/sim.yaml), **line 51**:

```yaml
map_path: 'maps/monza'
```

Also update the **starting pose** (`sx`, `sy`, `stheta` on lines 63–65) so the car spawns on the track, not in a wall. A safe default for most maps:
```yaml
sx: 0.0
sy: 0.0
stheta: 0.0
```

> [!TIP]
> If `(0,0)` is inside a wall on your map, look at your map's `origin` in the YAML and pick a point that's on the track. You can eyeball this from the raceline CSV — grab the first waypoint's `x_m` and `y_m` values.

---

## Step 7: Race! 🏎️💨

### Terminal 1 — Simulator:
```bash
cd ~/f1tenth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py launch_rviz:=true
```

### Terminal 2 — Stanley Controller:
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run gap_follow stanley_node
```

### Terminal 3 — Lap Timer (optional):
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run gap_follow lap_timer_node
```

---

## Quick Reference — File Checklist

For a map called `monza`, you should have these files when everything is done:

| File | Created By |
|------|-----------|
| `maps/monza.png` | You (Step 1) |
| `maps/monza.yaml` | You (Step 1) |
| `maps/monza_centerline.csv` | `generate_raceline.sh` (Step 2) |
| `maps/monza_raceline.csv` | `generate_raceline.sh` → updated by Optuna (Steps 2+4) |
| `maps/raceline.csv` | Auto-copied from above (default file the car reads) |
| `raceline_visualization.png` | `generate_raceline.sh` (Step 2) |
| `centerline_preview.png` | `generate_raceline.sh` (Step 2) |

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `Error: Map files not found!` | Make sure both `.png` and `.yaml` are in `src/f1tenth_gym_ros/maps/` and the name matches exactly |
| `floodfill failed` in centerline extraction | The world origin `(0,0)` isn't inside the track. Edit the `origin` in your YAML so that `(0,0)` falls on drivable space |
| `No skeleton found` | Map thresholds are wrong, or the track isn't a closed loop. Check that white pixels form a connected ring |
| All Optuna trials return `999.0` | The car is crashing every lap. Check the map_path in `stanley_tuner.py` is correct, and verify the raceline is valid |
| Car spawns in a wall | Update `sx`, `sy`, `stheta` in `sim.yaml` to coordinates on the track |
