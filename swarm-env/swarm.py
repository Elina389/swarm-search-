"""
Multi-agent grid-coverage environment for the swarm search project.
Implements PettingZoo's ParallelEnv interface so it plugs directly into
TorchRL's PettingZoo wrapper and other multi-agent RL tooling.
"""
import functools
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


class SwarmCoverageEnv(ParallelEnv):
    metadata = {"name": "swarm_coverage_v0"}

    def __init__(self, grid_size=10, n_drones=3, n_obstacles=10, obs_window=3,
                 max_steps=50, step_penalty=0.02, seed=None, bbox=None, osm_tags=None):
        """
        bbox : tuple(west, south, east, north), optional
            Lon/lat bounding box (EPSG:4326) of a real-world area. When set,
            obstacles are pulled from OpenStreetMap (buildings/water/forest
            by default) and rasterized onto the grid instead of being placed
            randomly, and stay fixed across resets since they represent a
            real place. Requires osmnx/geopandas/shapely -- see
            osm_obstacles.py.
        osm_tags : dict, optional
            Overrides the default OSM tag filter (see osm_obstacles.DEFAULT_TAGS).
        """
        self.grid_size = grid_size
        self.n_drones = n_drones
        self.n_obstacles = n_obstacles
        self.obs_window = obs_window
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.rng = np.random.default_rng(seed)
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up down left right stay

        self.bbox = bbox
        self.osm_tags = osm_tags
        self._fixed_obstacles = None
        self._cell_bounds = None
        if bbox is not None:
            from osm_obstacles import obstacles_from_bbox
            self._fixed_obstacles, self._cell_bounds = obstacles_from_bbox(
                bbox, grid_size, tags=osm_tags
            )

        self.possible_agents = [f"drone_{i}" for i in range(n_drones)]
        self.agents = self.possible_agents[:]
        self._obs_dim = (2 * obs_window + 1) ** 2 + 2

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]

        if self._fixed_obstacles is not None:
            # real-world layout from OSM -- fixed across episodes since it
            # represents an actual place, not something to memorize/overfit
            self.obstacles = set(self._fixed_obstacles)
        else:
            # a new random obstacle layout every episode -- this is what makes
            # generalization possible instead of memorizing one fixed map
            self.obstacles = set()
            while len(self.obstacles) < self.n_obstacles:
                r = self.rng.integers(0, self.grid_size)
                c = self.rng.integers(0, self.grid_size)
                self.obstacles.add((r, c))

        free_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                      if (r, c) not in self.obstacles]
        idx = self.rng.choice(len(free_cells), size=self.n_drones, replace=False)
        self.positions = {agent: free_cells[i] for agent, i in zip(self.agents, idx)}
        self.covered = set(self.positions.values())
        self.steps = 0

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _local_patch(self, pos):
        r0, c0 = pos
        w = self.obs_window
        patch = np.zeros((2 * w + 1, 2 * w + 1), dtype=np.float32)
        for dr in range(-w, w + 1):
            for dc in range(-w, w + 1):
                rr, cc = r0 + dr, c0 + dc
                if rr < 0 or rr >= self.grid_size or cc < 0 or cc >= self.grid_size or (rr, cc) in self.obstacles:
                    patch[dr + w, dc + w] = -1
                elif (rr, cc) in self.covered:
                    patch[dr + w, dc + w] = 1
                else:
                    patch[dr + w, dc + w] = 0
        return patch.flatten()

    def _get_obs(self, agent):
        pos = self.positions[agent]
        patch = self._local_patch(pos)
        norm_pos = np.array([pos[0] / self.grid_size, pos[1] / self.grid_size], dtype=np.float32)
        return np.concatenate([patch, norm_pos]).astype(np.float32)

    def step(self, actions):
        new_positions = {}
        for agent, action in actions.items():
            pos = self.positions[agent]
            dr, dc = self.action_deltas[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size or (nr, nc) in self.obstacles:
                new_positions[agent] = pos  # invalid move, stay put
            else:
                new_positions[agent] = (nr, nc)
        self.positions = new_positions

        # shared coverage reward: R_t = |C_new,t| - lambda
        new_cells = set(self.positions.values()) - self.covered
        shared_reward = len(new_cells) - self.step_penalty
        self.covered |= set(self.positions.values())
        self.steps += 1

        truncated = self.steps >= self.max_steps
        coverage_frac = len(self.covered) / (self.grid_size ** 2 - len(self.obstacles))

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        rewards = {agent: shared_reward for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {"coverage": coverage_frac} for agent in self.agents}

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        occupied = set(self.positions.values())
        rows = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if (r, c) in self.obstacles:
                    row.append("#")
                elif (r, c) in occupied:
                    row.append("D")
                else:
                    row.append(".")
            rows.append("".join(row))
        print("\n".join(rows))

    def close(self):
        pass