import gymnasium as gym
import numpy as np
import f1tenth_gym

# The maps in f1tenth_gym are usually passed by their path or name
try:
    env = gym.make('f1tenth_gym:f1tenth-v0',
                   config={'map': 'Levine', 'num_agents': 1})
    obs, info = env.reset(seed=123)
    print("obs keys:", obs.keys())
    action = np.array([[0.0, 1.0]]) # steer, speed
    obs, reward, done, truncated, info = env.step(action)
    print("step successful!")
except Exception as e:
    print(f"Error: {e}")
