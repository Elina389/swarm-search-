"""
Rule-based "potential field" movement policy for SwarmCoverageEnv.

This is the first REAL decision-making logic in this project -- everything
before this picked actions randomly. No training/RL involved here; it's
hand-coded physics-like rules, the same family of approach real robotics
systems have used for decades before learned policies became common (see
"Layer 3: coordination logic" in the project notes).

Each drone picks a direction every step by combining three simple pushes,
each a 2D (row, col) vector:

  1. OBSTACLE AVOIDANCE -- steer away from every blocked cell (obstacle, or
     the edge of the grid) visible in the drone's own local observation
     patch, weighted by inverse-square distance so close obstacles matter
     far more than distant ones. This is the "recognize obstacles and react
     to them" behavior -- it reads exactly the -1 cells that were already
     being computed in swarm.py's _local_patch(), just previously unused.

  2. TEAMMATE SEPARATION -- steer away from nearby teammates, so the swarm
     spreads out across the map instead of piling onto the same territory.

  3. FRONTIER ATTRACTION -- steer toward uncovered, unblocked cells the
     drone can currently see (a pull toward every such cell in view,
     weighted by inverse-square distance, not just the single nearest one).

These three vectors are summed into one "resultant" push, which then gets
snapped to whichever of the env's 5 discrete actions (up/down/left/right/
stay) best matches its direction.

WHY TEAMMATE SEPARATION IS ALLOWED TO "CHEAT": it reads env.positions
directly, which is global simulator state -- a trained neural-network
policy would NOT be allowed to do this, since it can only see what
_get_obs() actually returns (see swarm.py's module docstring on why the
observation is deliberately local-only). This hand-coded rule gets a pass
because it's meant to model something a real drone plausibly WOULD know if
teammates broadcast their GPS position over radio (see "what information
actually gets shared" -- a position broadcast is small and cheap). It's a
reasonable non-learned baseline, not a substitute for solving the real
communication-constrained problem.

WHY THIS NEEDS A LITTLE MEMORY: pure potential fields (recomputing the push
from scratch every step, with no memory of where you just were) are known
to get stuck oscillating between two cells whenever the local forces reach
a stable-looking balance point -- e.g. an obstacle repulsion exactly
cancels a frontier pull, so the field flips the drone back and forth
forever instead of continuing to explore. This is a textbook local-minimum
problem with potential fields, not a bug specific to this grid. The fix
here is a short per-drone memory (`PotentialFieldPolicy`) that mildly
penalizes immediately reversing into the cell a drone just came from,
which is usually enough to break the cycle and let the frontier pull win.
"""
import numpy as np

# Must match SwarmCoverageEnv.action_deltas order exactly:
# 0=up, 1=down, 2=left, 3=right, 4=stay
_ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
STAY_ACTION = 4


def _is_blocked(env, pos):
    r, c = pos
    return r < 0 or r >= env.grid_size or c < 0 or c >= env.grid_size or (r, c) in env.obstacles


def _snap_to_action(push_vector, env=None, pos=None, stay_threshold=0.15):
    """Convert a continuous (drow, dcol) push vector into the discrete
    action whose movement direction best agrees with it (highest dot
    product). If the push is nearly zero -- no strong pull either way --
    stay put rather than picking an arbitrary direction.

    If `env` and `pos` are given, actions that would walk straight into an
    obstacle or off the grid are excluded from consideration entirely
    (falling back to the best still-legal action, or STAY if truly none are
    legal). This matters more than it looks: the weighted-sum push vector
    is only a soft preference -- a strong enough frontier pull on the far
    side of a wall can still numerically outweigh the wall's repulsion, so
    without this hard filter the policy can pick "walk into the obstacle"
    every single step, get silently rejected by env.step() (which just
    keeps the drone in place), and repeat forever since nothing about the
    situation ever changes. This is the exact bug that caused earlier
    versions of this policy to permanently freeze against a wall.
    """
    candidates = list(enumerate(_ACTION_DELTAS[:4]))  # exclude "stay" from ranking
    if env is not None and pos is not None:
        legal = [(a, d) for a, d in candidates
                  if not _is_blocked(env, (pos[0] + d[0], pos[1] + d[1]))]
        if legal:
            candidates = legal
        # if NOTHING is legal (fully boxed in), fall through and stay put

    if not candidates or np.linalg.norm(push_vector) < stay_threshold:
        return STAY_ACTION

    best_action, best_score = STAY_ACTION, -np.inf
    for action, (dr, dc) in candidates:
        score = push_vector[0] * dr + push_vector[1] * dc
        if score > best_score:
            best_action, best_score = action, score
    return best_action


def _compute_push(env, agent, obstacle_weight, teammate_weight, coverage_weight,
                   teammate_radius):
    """Sum the three potential-field forces for one drone. Pure function of
    current env state -- no memory of past positions (that lives in
    PotentialFieldPolicy, which is what actually calls this)."""
    pos = env.positions[agent]
    r0, c0 = pos
    w = env.obs_window
    push = np.zeros(2, dtype=np.float64)

    # --- 1. obstacle avoidance: repel from every blocked cell in view ----
    for dr in range(-w, w + 1):
        for dc in range(-w, w + 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r0 + dr, c0 + dc
            blocked = (
                rr < 0 or rr >= env.grid_size or cc < 0 or cc >= env.grid_size
                or (rr, cc) in env.obstacles
            )
            if blocked:
                dist = max(abs(dr), abs(dc))
                # vector points AWAY from the obstacle, stronger when closer
                push -= obstacle_weight * np.array([dr, dc]) / (dist ** 2)

    # --- 2. teammate separation: repel from nearby teammates -------------
    for other_agent, other_pos in env.positions.items():
        if other_agent == agent:
            continue
        dr, dc = other_pos[0] - r0, other_pos[1] - c0
        dist = max(abs(dr), abs(dc))
        if dist == 0:
            # Two drones landed on the exact same cell. A distance-based
            # repulsion vector is undefined here (dr, dc are both 0), so
            # without a special case these drones see IDENTICAL forces
            # every step (same obstacles, same frontier, same history) and
            # become perfect clones moving in lockstep forever -- a much
            # worse failure than ordinary oscillation. Give every agent a
            # distinct, deterministic escape direction (based on its index
            # in the swarm), weighted large enough to actually dominate the
            # (still-identical) obstacle/frontier pulls both agents share --
            # a small nudge here isn't enough, since two co-located drones'
            # frontier pull is identical too and would otherwise still snap
            # to the same discrete direction despite the "different" push.
            agent_idx = env.possible_agents.index(agent)
            angle = 2 * np.pi * agent_idx / max(len(env.possible_agents), 1)
            escape_strength = 10.0 * (obstacle_weight + coverage_weight)
            push += escape_strength * np.array([np.cos(angle), np.sin(angle)])
        elif dist <= teammate_radius:
            push -= teammate_weight * np.array([dr, dc]) / (dist ** 2)

    # --- 3. frontier attraction: pull toward visible uncovered cells -----
    frontier_push = np.zeros(2, dtype=np.float64)
    for dr in range(-w, w + 1):
        for dc in range(-w, w + 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r0 + dr, c0 + dc
            if rr < 0 or rr >= env.grid_size or cc < 0 or cc >= env.grid_size:
                continue
            if (rr, cc) in env.obstacles or (rr, cc) in env.covered:
                continue
            dist = max(abs(dr), abs(dc))
            frontier_push += np.array([dr, dc]) / (dist ** 2)
    push += coverage_weight * frontier_push

    return push


class PotentialFieldPolicy:
    """Stateful potential-field movement strategy with short-term memory.

    Use this instead of the plain per-call function when running a full
    rollout -- it remembers each drone's previous cell so it can penalize
    immediately reversing back into it, which is what actually prevents
    the 2-cell oscillation described in the module docstring. A stateless
    call (no memory of the past) can't fix that on its own.
    """

    def __init__(self, obstacle_weight=2.0, teammate_weight=1.5, coverage_weight=1.0,
                 teammate_radius=4, jitter=0.3, backtrack_penalty=1.5, rng=None):
        """
        obstacle_weight, teammate_weight, coverage_weight : float
            Relative strength of each push -- raise obstacle_weight if
            drones cut corners too close to buildings, raise
            teammate_weight if they cluster, raise coverage_weight to make
            them more aggressive about seeking new ground.
        teammate_radius : int
            Only teammates within this many cells contribute a repulsion
            push -- distant teammates don't affect a local decision.
        jitter : float
            Small random noise added to the push vector every step, so two
            forces that land in an exact tie don't deterministically repeat
            the same choice forever. Set to 0 for fully deterministic
            (but more oscillation-prone) behavior.
        backtrack_penalty : float
            How strongly to discourage immediately reversing into the cell
            a drone occupied last step -- see module docstring for why this
            is necessary. 0 disables it.
        rng : numpy.random.Generator, optional
            Source of the jitter noise. Defaults to a fresh unseeded
            generator; pass your own for reproducible demos.
        """
        self.obstacle_weight = obstacle_weight
        self.teammate_weight = teammate_weight
        self.coverage_weight = coverage_weight
        self.teammate_radius = teammate_radius
        self.jitter = jitter
        self.backtrack_penalty = backtrack_penalty
        self.rng = rng or np.random.default_rng()
        self._last_pos = {}  # agent -> previous (row, col), for anti-backtrack

    def action_for(self, env, agent):
        pos = env.positions[agent]
        push = _compute_push(
            env, agent,
            self.obstacle_weight, self.teammate_weight, self.coverage_weight,
            self.teammate_radius,
        )

        if self.backtrack_penalty > 0 and agent in self._last_pos:
            prev = self._last_pos[agent]
            delta = (prev[0] - pos[0], prev[1] - pos[1])  # direction back toward prev cell
            if delta != (0, 0):
                push -= self.backtrack_penalty * np.array(delta)

        if self.jitter > 0:
            push += self.rng.uniform(-self.jitter, self.jitter, size=2)

        self._last_pos[agent] = pos
        return _snap_to_action(push, env=env, pos=pos)

    def actions(self, env):
        """Compute actions for every currently active agent, in the
        {agent: action} dict format env.step() expects."""
        return {agent: self.action_for(env, agent) for agent in env.agents}


def potential_field_actions(env, rng=None, **kwargs):
    """Convenience one-shot function for a single step, with NO memory
    across calls -- prone to the 2-cell oscillation described in the module
    docstring if called repeatedly in a loop. Prefer PotentialFieldPolicy
    for anything longer than a single step (e.g. an actual rollout)."""
    if rng is None:
        rng = np.random.default_rng()
    actions = {}
    for agent in env.agents:
        push = _compute_push(
            env, agent,
            kwargs.get("obstacle_weight", 2.0),
            kwargs.get("teammate_weight", 1.5),
            kwargs.get("coverage_weight", 1.0),
            kwargs.get("teammate_radius", 4),
        )
        jitter = kwargs.get("jitter", 0.3)
        if jitter > 0:
            push += rng.uniform(-jitter, jitter, size=2)
        actions[agent] = _snap_to_action(push, env=env, pos=env.positions[agent])
    return actions


# =============================================================================
# BFS-based coverage policy -- structurally can't oscillate, unlike the
# potential-field policy above.
#
# WHY THIS EXISTS: potential fields are a sum of local force vectors, which
# means they can always land on a spot where forces balance and cancel out
# -- that's a fundamental property of the approach, not something you can
# patch away with more special cases (see all the anti-oscillation patches
# above: backtrack penalty, jitter, co-location escape, legal-move
# filtering -- each one fixed a real bug, but new local minima kept showing
# up because the underlying mechanism still allows them).
#
# This policy replaces "sum of forces, snap to nearest matching action" with
# actual graph search: for each drone, run a breadth-first search over the
# grid's free cells to find real shortest-path distances to every reachable
# cell, then walk the first step of the shortest path toward the nearest
# uncovered one. Every step taken is guaranteed to reduce that drone's
# distance to its current target by exactly 1 -- there is no way to end up
# looping between two cells forever, because "closer" is a strict integer
# count of graph hops, not a continuous force that can cancel out.
#
# COORDINATION: instead of teammate repulsion (a soft push), drones are
# assigned disjoint "territories" each step -- each uncovered cell is
# claimed by whichever drone can reach it in the fewest hops, so drones
# naturally spread out to cover different regions without needing an
# explicit repulsion force. Ties/co-located drones are resolved by
# processing agents in a fixed order and skipping cells already claimed.
# =============================================================================
from collections import deque


def _bfs_from(env, source):
    """Standard grid BFS from `source` over free (non-obstacle, in-bounds)
    cells. Returns (dist, parent) dicts covering every cell reachable from
    source. `parent[cell]` is the cell you'd come from on the shortest path
    from source to `cell` -- used to reconstruct the first step to take."""
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
            if r < 0 or r >= env.grid_size or c < 0 or c >= env.grid_size:
                continue
            if nxt in env.obstacles:
                continue
            dist[nxt] = dist[cur] + 1
            parent[nxt] = cur
            dq.append(nxt)
    return dist, parent


def _first_step_toward(source, target, parent):
    """Walk backward from `target` through the BFS parent chain until we
    reach `source`'s immediate successor on that path -- i.e. the actual
    next cell to move into this step."""
    cur = target
    step = target
    while cur != source:
        step = cur
        cur = parent[cur]
    return step


class BFSCoveragePolicy:
    """Coverage strategy using real shortest-path search instead of force
    summation. See module docstring ("BFS-based coverage policy") for why
    this structurally can't get stuck oscillating the way potential fields
    can. Each drone: finds every uncovered cell it can reach and how many
    steps away each one is, claims whichever reachable uncovered cells no
    other (earlier-processed) drone already claimed this step, and moves
    one step along the shortest path to its nearest claimed cell.
    """

    def __init__(self, sticky=True):
        """
        sticky : bool
            If True, a drone keeps heading toward the same target cell
            across steps as long as that cell is still uncovered and still
            reachable, instead of re-picking the globally nearest frontier
            cell every single step (which can otherwise cause a target to
            flip-flop between two equally-close cells). Recommended on.
        """
        self.sticky = sticky
        self._target = {}  # agent -> (row, col) it's currently heading toward
        # agent -> short human-readable reason for its current action, for
        # the UI's hover tooltips. Purely explanatory; doesn't affect logic.
        self.last_reason = {}

    def actions(self, env):
        claimed = set()
        actions = {}

        for agent in env.agents:
            pos = env.positions[agent]
            dist, parent = _bfs_from(env, pos)

            uncovered_reachable = [
                cell for cell in dist
                if cell not in env.covered and cell != pos and cell not in claimed
            ]

            target = None
            # Stick with the previous target if it's still a valid, reachable,
            # uncovered, unclaimed cell -- avoids re-deciding every step and
            # flip-flopping between two equally-close frontier cells.
            if self.sticky:
                prev_target = self._target.get(agent)
                if (prev_target is not None and prev_target in dist
                        and prev_target not in env.covered and prev_target not in claimed):
                    target = prev_target

            if target is None:
                if uncovered_reachable:
                    target = min(uncovered_reachable, key=lambda c: dist[c])
                else:
                    target = None  # nothing left to explore that this drone can reach

            self._target[agent] = target

            if target is None or target == pos:
                self.last_reason[agent] = (
                    "Idle -- all reachable cells already covered"
                )
                actions[agent] = STAY_ACTION
                continue

            self.last_reason[agent] = (
                f"Heading to nearest uncovered cell, {dist[target]} steps away "
                f"(BFS shortest path)"
            )

            claimed.add(target)
            next_cell = _first_step_toward(pos, target, parent)
            delta = (next_cell[0] - pos[0], next_cell[1] - pos[1])
            action = STAY_ACTION
            for a, d in enumerate(_ACTION_DELTAS[:4]):
                if d == delta:
                    action = a
                    break
            actions[agent] = action

        return actions
