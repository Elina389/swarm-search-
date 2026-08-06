"""
Unknown-terrain sensing + exploration for SwarmCoverageEnv.

This is the "no map given" mode, in contrast to the known-map BFS policy in
movement_policy.py (which plans directly over env.obstacles because it is
handed an accurate map up front). Here, each drone must DISCOVER obstacles
live from limited-range sensing and remember them in an occupancy grid --
the harder, search-and-rescue-realistic version of the problem.

This file adds capability; it does not change SwarmCoverageEnv or the
existing known-map policies. The environment still holds the ground-truth
obstacle layout internally (it has to, in order to answer "what does the
sensor detect here" and to reject illegal moves), but this policy never
reads env.obstacles directly for planning -- it only ever consults its own
occupancy grid, which starts blank and fills in through sensing.

Three cell states in the occupancy grid:
    UNKNOWN  -- never sensed; the drone has no idea what is here
    FREE     -- sensed and confirmed traversable
    OCCUPIED -- sensed and confirmed blocked

------------------------------------------------------------------------
MATH USED (summary; see the accompanying explanation for detail)
------------------------------------------------------------------------
1. Distance sensing / field of view: a cell (r1,c1) is within sensing
   range of a drone at (r0,c0) iff the Euclidean distance
       sqrt((r1-r0)^2 + (c1-c0)^2) <= sensor_radius
   i.e. the sensed region is a disk of that radius (not a square).

2. Ray casting for occlusion: for each candidate cell in range we walk the
   straight line from the drone to that cell using Bresenham's line
   rasterization (integer-only line drawing). We mark cells FREE along the
   ray until the ray hits the first OCCUPIED cell, which we mark OCCUPIED
   and then stop -- everything behind an obstacle stays UNKNOWN, because a
   real range sensor cannot see through a wall. This "line integral until
   first hit" is the discrete-grid analogue of a lidar/sonar beam.

3. Occupancy grid update: this implementation is the DETERMINISTIC special
   case of a Bayesian occupancy grid. Real occupancy grids store a
   log-odds value per cell and update it with each noisy measurement:
       l_t(cell) = l_{t-1}(cell) + log( p(occ|z_t) / (1 - p(occ|z_t)) )
   Because sensing in this simulator is noise-free, that update collapses
   to l -> +infinity (certainly occupied) or -infinity (certainly free) on
   the first observation, i.e. a hard three-state grid. The log-odds form
   is the generalization you would switch to for real, noisy sensors.
"""
import numpy as np
from collections import deque

UNKNOWN = 0
FREE = 1
OCCUPIED = 2

# Must match SwarmCoverageEnv.action_deltas order: up, down, left, right, stay
_ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
STAY_ACTION = 4


def bresenham_line(r0, c0, r1, c1):
    """Integer cells along the straight line from (r0,c0) to (r1,c1),
    inclusive of both endpoints, via Bresenham's line algorithm -- the
    standard integer-only line rasterization. Used to trace a sensor ray
    across grid cells so we can stop it at the first obstacle (occlusion)."""
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


class OccupancyGrid:
    """The swarm's shared, live memory of the world -- a grid_size x
    grid_size array of UNKNOWN/FREE/OCCUPIED. Starts entirely UNKNOWN and
    fills in as drones sense. Shared across the swarm here (teammates pool
    what they've each seen), which is the optimistic end of the
    communication spectrum; a per-drone grid with limited sharing would be
    the more realistic, harder variant."""

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.state = np.full((grid_size, grid_size), UNKNOWN, dtype=np.int8)

    def in_bounds(self, r, c):
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def sense_from(self, pos, sensor_radius, true_obstacles):
        """Update the grid from one drone's sensor sweep at `pos`.

        `true_obstacles` is the environment's ground-truth obstacle set --
        used ONLY here, to answer "what would a real sensor detect," never
        for planning. Returns the list of cells whose state changed this
        sweep (handy for incremental rendering).

        Field of view = Euclidean disk of radius sensor_radius. Occlusion =
        Bresenham ray from the drone to each cell, stopping at the first
        obstacle hit. See the module docstring for the math.
        """
        r0, c0 = pos
        changed = []
        R = sensor_radius
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                if dr * dr + dc * dc > R * R:
                    continue  # outside the sensing disk (Euclidean range test)
                r1, c1 = r0 + dr, c0 + dc
                if not self.in_bounds(r1, c1):
                    continue
                # trace the beam; first obstacle blocks everything behind it
                for (rr, cc) in bresenham_line(r0, c0, r1, c1):
                    if (rr, cc) in true_obstacles:
                        if self.state[rr, cc] != OCCUPIED:
                            self.state[rr, cc] = OCCUPIED
                            changed.append((rr, cc, OCCUPIED))
                        break
                    else:
                        if self.state[rr, cc] != FREE:
                            self.state[rr, cc] = FREE
                            changed.append((rr, cc, FREE))
        return changed

    def frontier_cells(self):
        """FREE cells that border at least one UNKNOWN cell -- the boundary
        of explored space. Exploration = repeatedly heading to a frontier,
        sensing there to push the boundary outward, and repeating."""
        frontiers = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.state[r, c] != FREE:
                    continue
                for dr, dc in _ACTION_DELTAS[:4]:
                    nr, nc = r + dr, c + dc
                    if self.in_bounds(nr, nc) and self.state[nr, nc] == UNKNOWN:
                        frontiers.add((r, c))
                        break
        return frontiers

    def explored_fraction(self):
        """Fraction of the whole grid that is no longer UNKNOWN."""
        return float(np.count_nonzero(self.state != UNKNOWN)) / self.state.size


class UnknownTerrainPolicy:
    """Frontier-based exploration that plans ONLY over cells confirmed FREE.

    Safety guarantee (the "never crash" property): a drone's BFS traverses
    and steps into FREE cells only. It never routes through UNKNOWN or
    OCCUPIED cells, so the next move is always into ground already sensed as
    open. The map still expands, because each drone's sensor radius reaches
    PAST one step of movement -- standing on a frontier FREE cell senses the
    ring of UNKNOWN cells just beyond it, converting them to FREE/OCCUPIED
    before the drone ever tries to move there. So obstacles are always seen
    before they are reached.
    """

    def __init__(self, sensor_radius=3):
        self.sensor_radius = sensor_radius
        self.occ = None
        self._target = {}
        # agent -> short human-readable reason for its current action, for
        # the UI's hover tooltips. Purely explanatory; doesn't affect logic.
        self.last_reason = {}

    def reset(self, env):
        self.occ = OccupancyGrid(env.grid_size)
        self._target = {}
        # initial sweep from every drone's start cell
        for agent in env.agents:
            self.occ.sense_from(env.positions[agent], self.sensor_radius, env.obstacles)

    def _bfs_over_free(self, source):
        """BFS from `source` across FREE cells only. Returns (dist, parent)."""
        dist = {source: 0}
        parent = {}
        dq = deque([source])
        while dq:
            cur = dq.popleft()
            for dr, dc in _ACTION_DELTAS[:4]:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in dist:
                    continue
                r, c = nxt
                if not self.occ.in_bounds(r, c):
                    continue
                if self.occ.state[r, c] != FREE:
                    continue  # never traverse UNKNOWN or OCCUPIED -- safety
                dist[nxt] = dist[cur] + 1
                parent[nxt] = cur
                dq.append(nxt)
        return dist, parent

    def _first_step(self, source, target, parent):
        cur, step = target, target
        while cur != source:
            step = cur
            cur = parent[cur]
        return step

    def _frontier_cost(self, frontier, dist, priority, priority_weight):
        """Lower is better. Pure exploration = travel distance only. With a
        priority grid, subtract a reward proportional to how high-priority
        the frontier cell is, so drones prefer frontiers near discovered
        high-severity damage without ever giving up the FREE-only safety
        guarantee (the candidate set is still only reachable FREE cells)."""
        cost = dist[frontier]
        if priority is not None and priority_weight > 0:
            cost -= priority_weight * float(priority[frontier[0], frontier[1]])
        return cost

    def actions(self, env, priority=None, priority_weight=6.0):
        """Pick one action per drone.

        priority : optional 2-D grid (same shape as the map) of priority
            scores -- e.g. the Gaussian damage-priority density from
            damage.priority_density(). When given, frontier selection is
            biased toward high-priority areas: instead of always heading to
            the NEAREST frontier, a drone heads to the frontier minimizing
            (travel distance - priority_weight * priority-there). This is how
            the swarm routes toward severe damage clusters once early sensing
            reveals them. When None (the default), behaviour is unchanged:
            pure nearest-frontier exploration.

            NOTE: this only changes which FREE frontier cell a drone AIMS
            for. Pathing is still BFS over FREE cells only and every step is
            still into an already-sensed-free cell, so the no-crash guarantee
            is untouched -- priority affects goals, never whether a move is
            safe. (In a future RL version this same priority signal would
            instead enter the reward, so a trained policy would learn the
            routing rather than having it hand-coded here.)
        """
        if self.occ is None:
            self.reset(env)

        # 1. sense from every drone's current position (shared grid)
        for agent in env.agents:
            self.occ.sense_from(env.positions[agent], self.sensor_radius, env.obstacles)

        frontiers = self.occ.frontier_cells()
        claimed = set()
        actions = {}

        for agent in env.agents:
            pos = env.positions[agent]
            dist, parent = self._bfs_over_free(pos)

            # reachable frontier cells this drone could head for, minus ones
            # a teammate already claimed this tick (spreads the swarm out)
            reachable_frontiers = [
                f for f in frontiers
                if f in dist and f != pos and f not in claimed
            ]

            target = None
            prev = self._target.get(agent)
            if (prev is not None and prev in dist and prev in frontiers
                    and prev not in claimed):
                target = prev  # keep heading to the same frontier if still valid
            elif reachable_frontiers:
                target = min(
                    reachable_frontiers,
                    key=lambda f: self._frontier_cost(f, dist, priority, priority_weight),
                )

            self._target[agent] = target

            if target is None or target == pos:
                self.last_reason[agent] = (
                    "Idle -- no reachable unexplored area left in sensed-free space"
                )
                actions[agent] = STAY_ACTION
                continue

            # Explain the choice: did the damage-priority field divert this
            # drone away from the plain nearest frontier? If the priority-
            # aware pick differs from the pure-distance pick, it diverted.
            if priority is not None and reachable_frontiers:
                nearest = min(reachable_frontiers, key=lambda f: dist[f])
                if target != nearest and priority[target[0], target[1]] > 0:
                    self.last_reason[agent] = (
                        "Diverting to a high-priority damage cluster "
                        f"(~{dist[target]} steps) instead of the nearest frontier"
                    )
                else:
                    self.last_reason[agent] = (
                        f"Exploring -- heading to nearest frontier, {dist[target]} "
                        f"steps away (edge of sensed area)"
                    )
            else:
                self.last_reason[agent] = (
                    f"Exploring -- heading to nearest frontier, {dist[target]} "
                    f"steps away (edge of sensed area)"
                )

            claimed.add(target)
            nxt = self._first_step(pos, target, parent)
            delta = (nxt[0] - pos[0], nxt[1] - pos[1])
            action = STAY_ACTION
            for a, d in enumerate(_ACTION_DELTAS[:4]):
                if d == delta:
                    action = a
                    break
            actions[agent] = action

        return actions
