"""
Quick sanity check for swarm_env.py -- same pattern as your earlier test.py.
Run this after creating swarm_env.py to confirm everything is wired correctly
before building the training loop on top of it.
"""
from pettingzoo.test import parallel_api_test
from swarm import SwarmCoverageEnv

env = SwarmCoverageEnv(grid_size=10, n_drones=3, n_obstacles=10, obs_window=3, max_steps=50, seed=0)

print("Running PettingZoo's official API compliance test...")
parallel_api_test(env, num_cycles=200)
print("Passed.\n")

obs, infos = env.reset(seed=1)
print("Agents:", env.agents)
print("Observation shape per drone:", obs["drone_0"].shape)
print("Action space:", env.action_space("drone_0"))
print()
print("A random layout:")
env.render()

print("\nEverything is working!")