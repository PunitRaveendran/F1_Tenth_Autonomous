# F1TENTH: How to Race on a New Map 🏎️💨

This is your ultimate reference guide for taking a brand-new map (from the internet or from your own SLAM session) and getting the car to race on it at the absolute physical limits.

Follow these steps exactly in order.

---

## Step 1: Map the Track Using SLAM (If you don't already have a map)

If you are starting from scratch and need to map a physical track (or an unknown simulated environment), you will use the car's LiDAR to build the map.

1. Launch the SLAM mapping stack:
   ```bash
   ros2 launch f1tenth_gym_ros slam_mapping_launch.py
   ```
2. In a new terminal, launch the teleop node so you can manually drive the car:
   ```bash
   ros2 run f1tenth_gym_ros teleop_ackermann_node --ros-args -p max_forward_speed:=2.0
   ```
   *(Use WASD or arrow keys to slowly drive a complete, closed loop around the track).*
3. Once the map looks complete and closed in RViz, save it. Open a new terminal and run:
   ```bash
   cd ~/f1tenth_ws/src/f1tenth_gym_ros/maps
   ros2 run nav2_map_server map_saver_cli -f my_new_track
   ```
This will generate `my_new_track.png` and `my_new_track.yaml` directly in your maps folder. You can now close the SLAM and teleop terminals.

---

## Step 2: Add Downloaded Map Files (Optional)

If you downloaded a map from the internet instead of mapping it yourself, you need to place the `.png` and `.yaml` files in the maps directory.

1. Move both of these files into the simulator's maps directory:
   ```bash
   cp my_new_track.png ~/f1tenth_ws/src/f1tenth_gym_ros/maps/
   cp my_new_track.yaml ~/f1tenth_ws/src/f1tenth_gym_ros/maps/
   ```

*(Note: Ensure the name inside the `.yaml` file points to the correct `.png` image if you renamed the files).*

---

## Step 3: Generate the Optimal Raceline

Before the car can race, it needs to know the fastest geometric path through the track and exactly how fast it can go without spinning out. 

We have a single script that extracts the track boundaries, computes the minimum-curvature racing line, and generates the speed profile.

1. Navigate to your workspace root:
   ```bash
   cd ~/f1tenth_ws
   ```
2. Run the generator script with your map's base name (without the `.png` or `.yaml` extension):
   ```bash
   ./generate_raceline.sh my_new_track
   ```
3. **Verify it worked**: Open `raceline_visualization.png` in your file explorer. You should see your track with a colorful line overlaid on it (Green = Fast, Red = Slow). 

*(The script automatically saves the output to `maps/my_new_track_raceline.csv` and copies it to `maps/raceline.csv` so the car nodes can find it).*

---

## Step 4: Find the Perfect Controller Parameters (Optuna)

Every track is different. Some have long straights where the car can push harder, and some have tight hairpins requiring tighter tracking. 

We will use the background AI tuner to simulate hundreds of headless laps and find the perfect parameters for your new track.

1. Open `~/f1tenth_ws/stanley_tuner.py` in your code editor.
2. Update the map name and raceline path inside the `objective` function and `get_env_and_waypoints` to point to your new map:
   ```python
   config = EnvConfig(map_name='my_new_track', num_agents=1, render_enabled=False)
   # ...
   script_dir = os.path.dirname(os.path.abspath(__file__))
   raceline_path = os.path.join(script_dir, 'src', 'f1tenth_gym_ros', 'maps', 'my_new_track_raceline.csv')
   ```
3. Run the tuner script:
   ```bash
   cd ~/f1tenth_ws
   python3 stanley_tuner.py
   ```
4. The tuner will take ~2 minutes. When it finishes, it will print out the **Best Parameters** (e.g., `k_gain: 2.15`, `k_soft: 0.40`, `speed_scale: 1.08`) and the estimated fastest lap time.

---

## Step 5: Update Your Code with the New Parameters

Now that you have the mathematically perfect parameters for the new track, put them into the car's brain.

1. Open `~/f1tenth_ws/src/gap_follow/gap_follow/stanley_node.py`.
2. Update the parameters around line 23 with the values from Optuna:
   ```python
   self.declare_parameter('k_gain',        2.15)   # Use value from Optuna
   self.declare_parameter('k_soft',        0.40)   # Use value from Optuna
   self.declare_parameter('speed_scale',   1.08)   # Use value from Optuna
   ```
3. Save the file. Since you edited a ROS 2 node, you need to quickly rebuild the workspace:
   ```bash
   cd ~/f1tenth_ws
   colcon build --symlink-install
   ```

---

## Step 6: Tell the Simulator to Load the New Map

You need to configure the simulation to spawn the car on your new track.

1. Open the simulator config file: `~/f1tenth_ws/src/f1tenth_gym_ros/config/sim.yaml`
2. Change the `map_path` to your new map's base name:
   ```yaml
   map_path: 'maps/my_new_track'
   ```

*(You don't need to rebuild the workspace for editing `.yaml` config files, but it's good practice to close and reopen the simulator).*

---

## Step 7: 🏁 Race!

You are fully configured! It's time to run it.

1. **Terminal 1: Launch the Simulator**
   ```bash
   cd ~/f1tenth_ws
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   ros2 launch f1tenth_gym_ros gym_bridge_launch.py launch_rviz:=true
   ```

2. **Terminal 2: Launch the Stanley Controller**
   ```bash
   cd ~/f1tenth_ws
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   ros2 run gap_follow stanley_node
   ```

3. **Terminal 3: Launch the Lap Timer (Optional)**
   To see your real lap times print to the console:
   ```bash
   cd ~/f1tenth_ws
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   ros2 run gap_follow lap_timer_node
   ```

Sit back and watch the car crush the track! 🏎️💨
