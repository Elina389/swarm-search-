"""
Multi-agent grid-coverage environment for the swarm search project.
Implements PettingZoo's ParallelEnv interface so it plugs directly into
TorchRL's PettingZoo wrapper and other multi-agent RL tooling.

IMPORTANT -- what "swarm coordination" actually means in this file today:

There is no movement policy yet. Whoever calls step() has to supply an
action per drone; right now that caller (see visualize_swarm.py) just picks
random actions. Nothing here decides how a drone should move -- that's the
piece a trained policy (or a hand-coded rule like potential fields) would
plug in later.

There is also no real drone-to-drone communication being modeled. Real
drones would have to transmit position/status over radio with limited
range, bandwidth, and reliability (mesh network, point-to-point RF, or
satellite fallback), and each drone would only know what it's actually
received. This simulator skips all of that: `self.positions`,
`self.covered`, and `self.battery` are single, global, always-up-to-date
Python attributes on one object, readable by everyone with zero latency
and zero data loss.
"Sharing" here just means every drone's reward is computed from that same
global state (see the shared_reward calculation in step()) -- it's a
simulator convenience, not a model of radio communication. If you want to
simulate the real constraints (limited range, dropped messages, per-drone
partial knowledge), that logic would need to be added on top of this env,
most likely by making `_get_obs` build each drone's observation from a
per-agent "last known teammate state" cache instead of the true global
`self.positions`/`self.covered`.
"""
import functools
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


class SwarmCoverageEnv(ParallelEnv):
    metadata = {"name": "swarm_coverage_v0"}

    def __init__(self, grid_size=10, n_drones=3, n_obstacles=10, obs_window=3,
                 max_steps=50, step_penalty=0.02, seed=None, bbox=None, osm_tags=None,
                 battery_drain_move=1.0, battery_drain_idle=0.2):
        """
        grid_size : int
            Number of rows/cols in the square grid the drones move on.
        n_drones : int
            How many agents ("drone_0", "drone_1", ...) exist in the swarm.
        n_obstacles : int
            Number of randomly-placed blocked cells, used only when `bbox`
            is not given -- ignored once real OSM obstacles are in play.
        obs_window : int
            Radius (in cells) of each drone's local observation patch. A
            value of 3 gives a 7x7 patch -- see _local_patch().
        max_steps : int
            Episode length -- the episode truncates once `steps` reaches this.
        step_penalty : float
            Small constant subtracted from the reward every step, so standing
            still (or being blocked) is never free -- see step().
        seed : int, optional
            Seeds the env's own RNG (obstacle/start-position randomization).
        battery_drain_move, battery_drain_idle : float
            Percent of battery consumed per step -- moving costs more than
            staying still, matching real drones (motors draw far more power
            than hovering/sitting). See self.battery in reset()/step().
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
        self.battery_drain_move = battery_drain_move
        self.battery_drain_idle = battery_drain_idle
        self.rng = np.random.default_rng(seed)

        # Discrete action -> (row delta, col delta). Index into this list is
        # exactly what action_space(agent) expects back from a policy.
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

    def cell_to_latlon(self, row, col):
        """Convert a grid cell to real (lat, lon) coordinates, for anything
        that needs to place a drone/obstacle on an actual map (e.g. a
        Leaflet frontend). Only works when the env was built with `bbox` --
        raises otherwise, since there's no real-world mapping without one.
        Returns the CENTER point of the cell, not its corner."""
        if self._cell_bounds is None:
            raise ValueError(
                "cell_to_latlon() requires the env to have been built with "
                "a bbox= argument -- there's no real-world location for a "
                "plain random grid."
            )
        west, south, east, north = self._cell_bounds(row, col)
        return ((south + north) / 2, (west + east) / 2)

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

        # Per-drone shared status, the kind of small state a real drone
        # would broadcast over radio (see module docstring -- "what
        # information actually gets shared"). Starts full at 100%.
        self.battery = {agent: 100.0 for agent in self.agents}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {"battery": self.battery[agent]} for agent in self.agents}
        return observations, infos

    def _local_patch(self, pos):
        """Build the (2w+1) x (2w+1) patch of cells centered on `pos`.

        Each cell in the patch is one of:
            -1 -> obstacle, or off the edge of the grid
             1 -> already covered (visited by any drone this episode)
             0 -> free and not yet covered

        This is the ONLY spatial information a drone gets -- it can't see
        the whole grid, and it can't see where teammates currently are.
        A real drone would only know this much of the world from its own
        onboard sensors (camera/lidar range), which is why the observation
        is local rather than global even though the simulator itself
        tracks everything globally under the hood (see the module
        docstring above for why the reward still uses global state).
        """
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
        """Full observation for one drone: its local patch (flattened) plus
        its own normalized (row, col) position -- nothing about teammates."""
        pos = self.positions[agent]
        patch = self._local_patch(pos)
        norm_pos = np.array([pos[0] / self.grid_size, pos[1] / self.grid_size], dtype=np.float32)
        return np.concatenate([patch, norm_pos]).astype(np.float32)

    def step(self, actions):
        """Advance the episode by one tick.

        `actions` is {agent_name: action_index}, one entry per drone --
        this env has no built-in movement strategy of its own. Whatever
        picked those actions (random sampling in visualize_swarm.py today,
        a trained network eventually) is the actual "movement strategy."
        This method just applies whatever it's told and computes the result.
        """
        new_positions = {}
        for agent, action in actions.items():
            pos = self.positions[agent]
            dr, dc = self.action_deltas[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size or (nr, nc) in self.obstacles:
                new_positions[agent] = pos  # invalid move (wall/obstacle/edge) -- stay put
                actually_moved = False
            else:
                new_positions[agent] = (nr, nc)
                actually_moved = (nr, nc) != pos

            # Drain battery every step -- moving costs more than idling,
            # same as a real drone's motors drawing more power than hovering
            # (a blocked move still counts as idle since the drone didn't
            # actually go anywhere). Floored at 0 -- no negative battery.
            drain = self.battery_drain_move if actually_moved else self.battery_drain_idle
            self.battery[agent] = max(0.0, self.battery[agent] - drain)
        self.positions = new_positions

        # Shared coverage reward: R_t = |newly covered cells this step| - step_penalty.
        # "Shared" means every drone gets this exact same number, regardless
        # of which drone actually reached the new cell -- so a drone that did
        # nothing this step still benefits if a teammate covered new ground.
        # This is what pushes a trained policy toward spreading out: two
        # drones sitting on the same covered cell contribute 0 new cells, so
        # redundancy is implicitly penalized without any explicit rule for it.
        # NOTE: this reward is computed from self.covered, which is global,
        # noise-free simulator state -- not something transmitted between
        # drones over any communication channel (see module docstring).
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
        infos = {
            agent: {"coverage": coverage_frac, "battery": self.battery[agent]}
            for agent in self.agents
        }

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