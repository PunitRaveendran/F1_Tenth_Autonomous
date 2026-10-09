# F1TENTH — How It Works & What Needs Improving

A high-level overview of the racing stack for teammates. No code details — just concepts, what works, what doesn't, and where to focus.

---

## The Big Picture

We have a simulated 1/10th scale race car that drives around a track autonomously. The simulator gives us LiDAR scans and odometry (position + speed), and we send back steering + speed commands. Everything runs on ROS 2 Jazzy (Ubuntu 24.04), no Docker.

```
  LiDAR scan + Odom  ──→  [ Our Algorithms ]  ──→  Steering + Speed  ──→  Simulator
```

---

## What We Have (Working ✅)

### 1. Solo Time-Trial Racing ✅

The car drives around the track as fast as possible by following an **optimal raceline**.

**How it works:**

1. **Map** → We have a map image of the track (like a top-down floor plan)
2. **Centerline extraction** → A script finds the middle of the track from the map using computer vision (skeletonization)
3. **Trajectory optimization** → A TUM optimizer finds the fastest line through the track (cutting corners, using full width)
4. **Speed profile** → For every point on the line, we calculate the max safe speed based on how curvy that section is (slow in corners, fast on straights)
5. **Pure Pursuit** → The controller steers the car to follow this pre-computed racing line at the computed speeds

This whole pipeline (steps 1–4) is automated with one command: `./generate_raceline.sh <map_name>`

**What works well:**
- The raceline generation pipeline works reliably for any map
- Pure Pursuit tracking is solid — the car completes laps consistently
- Adaptive lookahead adjusts based on speed (looks further ahead on straights)
- Lap timer tracks progress accurately

### 2. SLAM Mapping ✅

We can drive the car manually (keyboard) around an unknown track and build a map from LiDAR data. This map then feeds into the raceline pipeline above.

### 3. Follow-the-Gap (Reactive Driving) ✅

A reactive planner that drives using only live LiDAR — no map, no raceline. It finds the biggest open gap in front and steers toward it. Works as a safety fallback.

---

## What We're Building (Not Done ❌)

### 4. Head-to-Head Overtaking ❌ — THE MAIN THING TO IMPROVE

This is the big challenge: racing against an opponent car and overtaking it.

**The idea (3 new components):**

#### a) Opponent Detection (partially working)
- **What it does:** Looks at the LiDAR scan and tries to find the other car
- **How:** Groups LiDAR points into clusters by finding "breakpoints" (sudden jumps in range). Then filters clusters by size and shape — a car-sized cluster near the track = opponent
- **What works:** Detects the opponent reliably on straight sections when there's clear separation from walls
- **What doesn't work:**
  - ❌ **False positives near walls** — when the opponent is close to a wall, its LiDAR signature blends with the wall and the detector either misses it or detects part of the wall as the opponent
  - ❌ **Misses in corners** — the opponent's profile changes shape in turns and doesn't always pass the geometry filters
  - ❌ **No tracking/prediction** — we detect frame-by-frame with basic hysteresis (3 hits to confirm, 5 misses to clear). No Kalman filter or velocity estimation. If detection flickers, the whole downstream pipeline stutters
  - 💡 **Ideas:** Add a Kalman filter or particle filter to track the opponent's position and velocity. Use the opponent's raceline position to predict where it will be. Consider using the known opponent URDF size instead of generic width filters

#### b) Overtake Path Planning — "Spliner" (code exists, not validated)
- **What it does:** When an opponent is detected ahead, generates a smooth curved path that goes around it and returns to the raceline
- **How:** 
  1. Figures out where the opponent is on the raceline
  2. Checks which side (left or right) has more room using a distance-to-wall map
  3. Creates 5 waypoints: current position → approach → pass-around-apex → exit → back-on-raceline
  4. Fits a smooth cubic spline through those points
  5. Assigns speeds (slower at the apex of the overtake, full speed on entry/exit)
- **What doesn't work:**
  - ❌ **Not tested in practice** — the paths may collide with walls on tight sections
  - ❌ **Static planning** — plans once based on where opponent is RIGHT NOW, doesn't account for opponent moving during the maneuver
  - ❌ **Fixed lateral offset (0.6m)** — may not be enough on some sections, too much on others
  - ❌ **No abort mechanism** — once committed, there's no way to say "this path is bad, go back"
  - 💡 **Ideas:** Replan continuously as opponent moves. Vary the lateral offset based on available space. Add a safety check against predicted opponent position (not just walls). Implement an abort condition that returns to TRAILING if the path becomes unsafe

#### c) State Machine (code exists, transitions not tuned)
- **What it does:** Decides WHAT the car should be doing at any moment
- **States:**
  - **RACELINE** — Normal racing, follow the optimal line (default)
  - **TRAILING** — Opponent detected ahead; follow raceline but slow down to match their speed
  - **OVERTAKE** — Committed to passing; follow the spliner's overtake path
  - **RETURN** — Passed the opponent; blend back to raceline
  - **EMERGENCY** — Something very close ahead; hard brake + steer toward biggest gap
- **What doesn't work:**
  - ❌ **Transition thresholds are guesses** — e.g., "enter TRAILING at 4m" and "commit to OVERTAKE at 3m" are made-up numbers, not tuned
  - ❌ **No opponent speed estimation** — in TRAILING, we estimate opponent speed as "half our speed" which is wrong
  - ❌ **RETURN state is too simple** — just follows raceline, doesn't smoothly blend from wherever the overtake ended
  - ❌ **Full cycle never tested** — the RACELINE → TRAILING → OVERTAKE → RETURN → RACELINE loop hasn't been run successfully end-to-end
  - 💡 **Ideas:** Tune thresholds in simulation with opponent at various speeds. Use opponent detector's position history to estimate actual opponent speed. Implement a proper return trajectory (another spline) instead of jumping straight to raceline

---

## Component Status Summary

| Component | Status | Confidence |
|-----------|--------|------------|
| Raceline generation pipeline | ✅ Working | High |
| Pure Pursuit (solo racing) | ✅ Working | High |
| Lap timer | ✅ Working | High |
| SLAM mapping | ✅ Working | High |
| Follow-the-Gap (reactive) | ✅ Working | Medium |
| Simulator bridge + Foxglove | ✅ Working | High |
| Opponent detection | 🟡 Partial | Low — works on straights, fails near walls/corners |
| Spliner (overtake path) | 🔴 Unvalidated | Low — code runs, paths not tested for safety |
| State machine (FSM) | 🔴 Unvalidated | Low — runs but full cycle never completed |
| End-to-end overtake | 🔴 Not working | Not attempted — all 3 pieces above need to work first |

---

## Priority Improvements (suggested order)

### P0 — Make opponent detection reliable
Without solid detection, nothing downstream works. Focus on:
1. Reduce false positives near walls (tighter geometry filtering, or use wall-subtraction from static map)
2. Add tracking (Kalman filter) so detections are smooth even if individual frames miss
3. Estimate opponent velocity from tracked positions

### P1 — Validate overtake paths in simulation
Once detection is stable:
1. Manually place opponent on track, run spliner, visualize the generated path
2. Check if paths stay within track boundaries on ALL sections (not just straights)
3. Tune lateral offset, approach/exit distances per-section if needed
4. Test with car actually following the spline path

### P2 — Tune the state machine
Once paths are safe:
1. Run full stack, observe when transitions happen, adjust thresholds
2. Get opponent speed estimation working (use position tracking from P0)
3. Add an abort mechanism: if overtake path becomes dangerous mid-maneuver, return to TRAILING
4. Handle edge cases: what if opponent speeds up during overtake? what if there's a wall on both sides?

### P3 — Performance optimization
Once overtaking works:
1. Optimize raceline speeds (currently conservative)
2. Tune speed during overtake (currently 70% at apex — could be faster)
3. Consider multiple overtake strategies (inside vs outside depending on corner)

---

## How to Run Things

### Quick solo test
```bash
# Terminal 1: Simulator
ros2 launch f1tenth_gym_ros gym_bridge_launch.py launch_rviz:=true

# Terminal 2: Driver
ros2 run gap_follow pure_pursuit_node --ros-args -p speed_scale:=0.5

# Terminal 3: Lap timer (optional)
ros2 run gap_follow lap_timer_node
```

### Head-to-head test (current state — will likely not overtake successfully)
```bash
# Terminal 1: Simulator (make sure num_agent: 2 in sim.yaml)
ros2 launch f1tenth_gym_ros gym_bridge_launch.py launch_rviz:=true

# Terminal 2: Opponent (drives slow)
ros2 run gap_follow opponent_driver_node --ros-args -p speed_scale:=0.5

# Terminal 3: Detector
ros2 run gap_follow opponent_detector_node

# Terminal 4: Path planner
ros2 run gap_follow spliner_node

# Terminal 5: Our car (state machine replaces pure_pursuit)
ros2 run gap_follow state_machine_node
```

### Generate raceline for a new map
```bash
./generate_raceline.sh <map_name>
```

---

## Key Files to Know

| What | Where |
|------|-------|
| Simulator config (map, agents, LiDAR) | `src/f1tenth_gym_ros/config/sim.yaml` |
| Pure Pursuit controller | `src/gap_follow/gap_follow/pure_pursuit_node.py` |
| Opponent detector | `src/gap_follow/gap_follow/opponent_detector_node.py` |
| Overtake path planner | `src/gap_follow/gap_follow/spliner_node.py` |
| State machine (FSM) | `src/gap_follow/gap_follow/state_machine_node.py` |
| Opponent driver (for testing) | `src/gap_follow/gap_follow/opponent_driver_node.py` |
| Raceline + maps | `src/f1tenth_gym_ros/maps/` |
| Raceline generation pipeline | `generate_raceline.sh` |
