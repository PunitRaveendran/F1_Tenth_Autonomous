# F1TENTH Controller Benchmark & Ablation Study

**Author:** Punit Raveendran (punitraveendran@gmail.com)  
**Platform:** F1TENTH Autonomous Racing  
**Track:** Red Bull Ring (Spielberg)  
**Simulation Environment:** Gymnasium `f1tenth_gym:f1tenth-v0`  
**Evaluation Metrics:** Lap Time ($T_{lap}$), Average & Peak Velocity ($\bar{v}, v_{max}$), Cross-Track Error ($\text{RMSE}_{cte}$), Heading Error ($\text{RMSE}_{\theta}$), Steering Jerk / Chatter ($\sigma_{\dot{\delta}}$)

---

## 1. Executive Summary & Benchmark Leaderboard

The table below presents the quantitative performance comparison across all 8 evaluated controller variants tested under identical vehicle dynamics and friction conditions on the Spielberg circuit.

| Rank | Controller Variant | Category | Lap Time (s) | Δ vs Baseline | Avg Speed (m/s) | Top Speed (m/s) | $\text{RMSE}_{cte}$ (m) | Max CTE (m) | Heading Error $\text{RMSE}_\theta$ (deg) | Steering Chatter $\sigma_{\dot{\delta}}$ (rad/s) | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Stanley (Optuna Tuned)** | Non-Linear Feedback + Feedforward | **42.04s** | **-20.29s** | **8.10** | **8.40** | 0.361 | 0.553 | 2.55° | 0.0091 | COMPLETED |
| 🥈 | **Optuna-Tuned Pure Pursuit** | Bayesian Optimized Geometric | **46.05s** | **-16.28s** | 7.38 | 7.84 | 0.149 | 0.448 | 3.63° | 0.0024 | COMPLETED |
| 🥉 | **LQR (Discrete State Feedback)** | Optimal Control (DARE) | **47.45s** | **-14.88s** | 7.16 | 7.60 | **0.066** | **0.123** | **2.08°** | 0.0179 | COMPLETED |
| 4 | **Advanced Pure Pursuit** | Geometric + Interpolation | **50.09s** | **-12.24s** | 6.79 | 7.20 | 0.127 | 0.375 | 2.76° | **0.0008** | COMPLETED |
| 5 | **Stanley (Untuned Baseline)** | Geometric Feedback ($k=1.5$) | **52.68s** | **-9.65s** | 6.43 | 6.80 | 0.344 | 0.534 | 1.48° | 0.0084 | COMPLETED |
| 6 | **Pure Pursuit (Speed Profiled)** | Geometric ($L=1.5m$) | **52.94s** | **-9.39s** | 6.42 | 6.80 | 0.099 | 0.276 | 2.31° | 0.0019 | COMPLETED |
| 7 | **Pure Pursuit (Baseline)** | Fixed $L=1.6m$, Constant $5.5m/s$ | **62.33s** | **0.00s** | 5.47 | 5.50 | 0.103 | 0.461 | 1.65° | 0.0013 | COMPLETED |
| 8 | **Follow The Gap (Reactive LiDAR)**| Reactive Obstacle Avoidance | N/A | — | 3.12 | 4.80 | N/A | N/A | N/A | 0.0361 | COLLISION (Turn 3) |

> **Key Visual Artifact:**  
> The complete benchmark comparison plot (Lap Times, Tracking Error distributions, and Trajectory overlays) is saved at:  
> `benchmark_summary.png`

---

## 2. Pure Pursuit Ablation Study

An ablation study systematically isolates and strips/adds specific algorithmic components to quantify their standalone impact. Starting from the naive baseline, four progressive upgrades were added:

```
[Level 0] Baseline Pure Pursuit (Constant 5.5 m/s, Fixed L=1.6m)
   │
   ├──▶ [Level 1] + Friction/Curvature Speed Profiling
   │       Gain: -9.39s | Accelerates on straights, brakes into hairpins
   │
   ├──▶ [Level 2] + Adaptive Velocity Lookahead L(v) & Segment Interpolation
   │       Gain: -2.85s | Cuts corner clipping; smooths steering chatter by 58%
   │
   └──▶ [Level 3] + Bayesian Hyperparameter Optimization (Optuna TPE)
           Gain: -4.04s | Optimal friction extraction, tightest high-speed tracking
```

### Detailed Ablation Matrix

| Exp ID | Configuration | Added Feature / Component | Lap Time (s) | Step Gain (s) | Cumulative Gain (s) | $\text{RMSE}_{cte}$ (m) | Steering Chatter ($\sigma_{\dot{\delta}}$) | Qualitative Failure / Limitation |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PP-0** | Baseline Pure Pursuit | Fixed Lookahead $L=1.6m$, Fixed Velocity $5.5 m/s$ | **62.33s** | — | — | 0.103m | 0.0013 rad/s | Severe under-utilization of straightaways; lacks braking before hairpins. |
| **PP-1** | Profiled Pure Pursuit | + Physics-based Curvature Speed Profiling ($a_{lat}=7.0, a_{lon}=5.0$) | **52.94s** | **-9.39s** | **-9.39s** | 0.099m | 0.0019 rad/s | High speed reached on straights, but constant lookahead cuts sharp corners at high speed. |
| **PP-2** | Advanced Pure Pursuit | + Adaptive Lookahead $L(v) \in [1.1, 2.1]m$ & Segment Circle-Line Interpolation | **50.09s** | **-2.85s** | **-12.24s** | 0.127m | **0.0008 rad/s** | Eliminates waypoint discretization jumping; lowest actuator chatter of any controller. |
| **PP-3** | Optuna Pure Pursuit | + Bayesian TPE Hyperparameter Search ($L_{min}, L_{max}, \text{steer\_gain}, \text{speed\_scale}$) | **46.05s** | **-4.04s** | **-16.28s** | 0.149m | 0.0024 rad/s | Pushes car to the friction limit ($7.84 m/s$ peak); slight lateral drift due to lack of CTE feedback. |

### Ablation Takeaways:
1. **Speed profiling yields the largest raw delta:** Introducing a 3-pass physics profile (curvature limit, acceleration pass, deceleration pass) shaved **9.39 seconds** alone.
2. **Segment interpolation dramatically improves steering smoothness:** Advanced Pure Pursuit achieved the lowest steering chatter in the entire benchmark ($\sigma_{\dot{\delta}} = 0.0008\text{ rad/s}$), as circle-line intersection prevents sudden discrete angle steps when jumping between waypoints.
3. **Pure Pursuit hits an architectural ceiling:** Even with automated Optuna tuning, Pure Pursuit plateaus at **46.05s** because the algorithm steers toward a future point with zero direct feedback on current cross-track deviation ($e_{cte}$).

---

## 3. Mathematical Formulations of Evaluated Controllers

### 3.1 Pure Pursuit (Geometric Tracking)
Pure Pursuit fits a circular arc from the vehicle's rear axle to a lookahead target point $(x_t, y_t)$ situated at distance $L$ ahead on the path.

$$\alpha = \arctan2(y_t - y, x_t - x) - \psi$$
$$\kappa_{arc} = \frac{2 \sin(\alpha)}{L}$$
$$\delta = \arctan(L_{wb} \cdot \kappa_{arc})$$

**Advanced Adaptive Extension:**
$$L(v) = \text{clip}\left(L_{min} + \frac{v}{v_{max}}(L_{max} - L_{min}), L_{min}, L_{max}\right)$$

---

### 3.2 Stanley Controller (Nonlinear Geometric Error Feedback + Feedforward)
The Stanley controller calculates error at the **front axle** $(x_{fa}, y_{fa}) = (x + L_{wb}\cos\psi, y + L_{wb}\sin\psi)$ and combines three complementary terms:

$$\delta(t) = \theta_e(t) + \arctan\left(\frac{k \cdot e_{fa}(t)}{v(t) + k_{soft}}\right) + \arctan(L_{wb} \cdot \kappa_{ref})$$

Where:
- $\theta_e = \psi_{ref} - \psi$: Heading error.
- $e_{fa} = \sin(\psi_{ref})(x_{fa} - x_{ref}) - \cos(\psi_{ref})(y_{fa} - y_{ref})$: Cross-track error at the front axle.
- $\arctan(L_{wb} \cdot \kappa_{ref})$: Feedforward curvature term to pre-steer into corners.

**Why Optuna Tuning unlocks 42.04s:**
By optimizing $k_{gain}=3.0, k_{soft}=0.30$ and scaling physics profile limits ($a_{lat}=9.5 m/s^2, a_{lon}=7.0 m/s^2$), Stanley eliminates corner-entry understeer while maintaining high-speed straightaway stability.

---

### 3.3 Linear Quadratic Regulator (LQR - Optimal Control)
LQR minimizes an infinite-horizon quadratic cost function balancing tracking error and control effort:

$$J = \sum_{k=0}^{\infty} \left( x_k^T Q x_k + u_k^T R u_k \right)$$

#### State-Space Kinematic Error Model:
$$x_k = \begin{bmatrix} e_{lat}(k) \\ \theta_e(k) \end{bmatrix}, \quad u_k = \delta_{fb}(k)$$

$$x_{k+1} = A x_k + B u_k$$

Where:
$$A = \begin{bmatrix} 1 & v \cdot dt \\ 0 & 1 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ \frac{v}{L_{wb}} dt \end{bmatrix}$$

Weights:
$$Q = \begin{bmatrix} 2.2 & 0 \\ 0 & 0.9 \end{bmatrix}, \quad R = [0.20]$$

The Discrete Algebraic Riccati Equation (DARE) is solved:
$$P = A^T P A - (A^T P B)(R + B^T P B)^{-1}(B^T P A) + Q$$
$$K = (R + B^T P B)^{-1} B^T P A$$

The total control law combines optimal feedback and kinematic feedforward:
$$\delta(t) = -K \begin{bmatrix} e_{lat} \\ \theta_e \end{bmatrix} + \arctan(L_{wb} \cdot \kappa_{ref})$$

**Why LQR has the Highest Path Tracking Fidelity ($\text{RMSE}_{cte} = 0.066m$):**
LQR continuously optimizes the trade-off between lateral error and heading rate, producing the tightest path tracking of all tested algorithms (maximum cross-track error was only $0.123m$ over the entire 430m track).

---

### 3.4 Follow The Gap (Reactive LiDAR)
Follow The Gap is an obstacle-avoidance algorithm that requires **zero precomputed map or raceline**:

1. **Window Filtering:** LiDAR scan is clipped to front $180^\circ$ ($[-90^\circ, +90^\circ]$) and capped at depth $D_{cap} = 3.5m$.
2. **Safety Bubble:** Identifies the closest point $(d_{min}, \theta_{min})$ and zeroes out all readings within bubble radius $r_{bubble} = 0.5m$:
   $$r_{idx} = \frac{r_{bubble}}{\Delta \theta_{lidar}}$$
3. **Max Continuous Gap Search:** Finds the largest continuous non-zero ray sequence $[i_{start}, i_{end}]$.
4. **Deepest Target Heading:** Chooses the center of the deepest plateau:
   $$\theta_{target} = \theta_{min} + i_{best} \cdot \Delta \theta_{lidar}$$
5. **Steering Command:** $\delta = \text{clip}(k_{steer} \cdot \theta_{target}, -\delta_{max}, \delta_{max})$.

**Failure Mode on Spielberg Circuit:**  
FTG reliably cleared straightaways and chicanes, reaching Step 1164 ($3.12 m/s$ average speed). However, at Turn 3 (Remus hairpin), the walls converge rapidly. Without global map awareness or predictive lookahead braking, reactive LiDAR suffers from outside-wall perspective distortion, leading to barrier collision.

---

## 4. Deep Comparative Analysis & Engineering Trade-offs

### 4.1 Speed vs. Tracking Precision vs. Smoothness

```
                   High Speed (Low Lap Time)
                          ▲
                          │     ★ Stanley (Optuna) [42.04s]
                          │
                          │     ★ Optuna PP [46.05s]
                          │
                          │     ★ LQR [47.45s, RMSE=0.066m]
                          │
                          │     ★ Advanced PP [50.09s, Chatter=0.0008]
                          │
  High Tracking Precision ┼──────────────────────────────► Low Steering Chatter
  (LQR: 0.066m CTE)       │                              (Advanced PP: 0.0008 rad/s)
                          │
                          │     ★ Pure Pursuit Baseline [62.33s]
                          │
```

- **Pure Pursuit:** Easiest to implement, extremely smooth, but suffers from corner cutting and cannot compensate for steady-state cross-track error.
- **Stanley:** Exceptional for high-speed aggressive racing when tuned with curvature feedforward; converges quickly to the raceline.
- **LQR:** The definitive winner for **tracking fidelity** ($\text{RMSE}_{cte} = 0.066m$). If the goal is staying on the center of the raceline without violating track limits, LQR is mathematically superior.
- **Follow The Gap:** Indispensable when dynamic obstacles (opponents, debris) appear, but unsuitable for solo high-speed time trials on complex road circuits.

---

## 5. How to Reproduce All Experiments

All code is fully automated and self-contained in the repository.

### 1. Run the Unified Benchmark Suite
Runs all 8 controller variants headless at ~20x real-time speed, logs all telemetry, saves `benchmark_results.csv`, and generates `benchmark_summary.png`:

```bash
python3 benchmark_suite.py
```

### 2. Run Pure Pursuit Optuna Tuner
Executes 75 Bayesian optimization trials on lookahead bounds, velocity scale, and steering gains:

```bash
python3 pure_pursuit_tuner.py
```

### 3. Run Stanley Optuna Tuner
Executes 200 Bayesian optimization trials tuning both Stanley control law and tire friction speed limits:

```bash
python3 stanley_tuner.py
```

### 4. Run Live ROS 2 Nodes in Simulator
Launch the Gym simulator and run the controllers as ROS 2 nodes:

```bash
# Terminal 1: Launch ROS 2 Simulation Bridge
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py

# Terminal 2: Run Advanced Pure Pursuit Node
python3 advanced_pure_pursuit.py

# Or Run Follow The Gap Node
python3 basic_driver.py
```

---

## 6. Multi-Track Benchmark: Nürburgring Circuit

To evaluate controller generalization across different track topologies, the exact same controllers were evaluated on the **Nürburgring circuit** (570m length, 2,190 waypoints, flowing high-speed sweeping corners).

### Nürburgring Leaderboard

| Rank | Controller Variant | Category | Lap Time (s) | Δ vs Baseline | Avg Speed (m/s) | Top Speed (m/s) | $\text{RMSE}_{cte}$ (m) | Max CTE (m) | Steering Chatter $\sigma_{\dot{\delta}}$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Stanley (Optuna Tuned)** | Non-Linear Feedback + Feedforward | **55.22s** | **-25.36s** | **7.98** | **8.40** | 0.362 | 0.538 | 0.0110 rad/s |
| 🥈 | **LQR (Discrete State Feedback)** | Optimal Control (DARE) | **62.99s** | **-17.59s** | 6.96 | 7.60 | **0.068** | **0.134** | 0.0218 rad/s |
| 🥉 | **Advanced Pure Pursuit** | Adaptive $L(v)$ + Interp | **66.56s** | **-14.02s** | 6.61 | 7.20 | 0.139 | 0.392 | **0.0010 rad/s** |
| 4 | **Stanley (Baseline Untuned)** | Geometric Feedback ($k=1.5$) | **69.63s** | **-10.95s** | 6.26 | 6.80 | 0.344 | 0.532 | 0.0103 rad/s |
| 5 | **Pure Pursuit (Speed Profiled)** | Geometric ($L=1.5m$) | **70.35s** | **-10.23s** | 6.25 | 6.80 | 0.106 | 0.284 | 0.0023 rad/s |
| 6 | **Pure Pursuit (Baseline)** | Fixed $L=1.6m, v=5.5m/s$ | **80.58s** | **0.00s** | 5.48 | 5.50 | 0.117 | 0.449 | 0.0015 rad/s |

### Multi-Track Generalization Insights:
1. **Stanley maintains speed supremacy across both circuits:** Tuned Stanley is 25.36s faster than baseline on Nürburgring and 20.29s faster on Spielberg.
2. **LQR holds unmatched cross-track consistency:** On both Spielberg ($\text{RMSE}_{cte} = 0.066m$) and Nürburgring ($\text{RMSE}_{cte} = 0.068m$), LQR tracks the centerline within under $7\text{cm}$ average error, confirming that state-space optimal control is universally the most accurate trajectory tracker.
3. **Advanced Pure Pursuit retains minimum steering chatter:** The circle-line interpolation yields the smoothest steering rate on both tracks ($\sigma_{\dot{\delta}} \le 0.001\text{ rad/s}$).

To reproduce the Nürburgring benchmark:
```bash
python3 benchmark_nuerburgring.py
```
*(Artifacts saved to `benchmark_nuerburgring_results.csv` and `benchmark_nuerburgring_summary.png`)*

