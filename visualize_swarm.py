"""
Standalone live visualization for SwarmCoverageEnv.

Draws the grid with matplotlib -- obstacles as dark cells, covered cells
shaded green, drones as red dots -- and updates it live as the episode runs.
This file doesn't touch swarm_env.py at all; it just reads the env's public
attributes (obstacles, covered, positions, steps, grid_size) from the outside.

When the env was built with a real bbox, the grid overlay can optionally be
drawn on top of a real basemap image (via contextily, no API key needed) --
see basemap.py and the `use_basemap` flag on demo_rollout().

MOVEMENT STRATEGY: currently random. demo_rollout() below picks each
drone's action with `env.action_space(agent).sample()` -- i.e. every step,
every drone just rolls a die between up/down/left/right/stay. There is no
coordination logic, no potential-field repulsion, no learned policy. This
is intentional for now: it's a sanity check that the environment, rendering,
and reward all work correctly before a real decision-maker is plugged in.
The random rollout will "cover" a grid eventually, just inefficiently and
with drones frequently colliding onto the same cells.

To swap in real behavior, replace the `actions = {...}.sample()` line
inside demo_rollout() with a call to whatever's actually deciding moves:
  - Rule-based (potential fields / repulsion from teammates & obstacles,
    attraction to nearest uncovered cell) -- doesn't need any RL.
  - A trained policy network -- one forward pass per drone per step, using
    the observation from env._get_obs(agent) as input.

Run directly for a quick random-action demo:
    python visualize_swarm.py

Or import render_frame() and call it from inside your own eval/rollout loop
once you have a trained policy instead of random actions.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

# adjust this to match whatever your env file is actually named now
from swarm import SwarmCoverageEnv
from movement_policy import PotentialFieldPolicy, BFSCoveragePolicy


# --- small vector "plane" icon, nose pointing up by default -----------------
# drawn as a path instead of a photo so it can rotate cheaply to face
# whichever direction a drone is currently heading
_PLANE_VERTS = [
    (0.0, 1.0),     # nose
    (0.8, -0.55),   # right wingtip
    (0.0, -0.15),   # tail notch
    (-0.8, -0.55),  # left wingtip
    (0.0, 1.0),     # close back to nose
]
_PLANE_CODES = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
PLANE_PATH = Path(_PLANE_VERTS, _PLANE_CODES)

# grid movement delta (dr, dc) -> on-screen rotation angle in degrees,
# counterclockwise from "up" (matplotlib marker transforms are in screen
# space, so this doesn't need to account for the inverted row axis)
HEADING_ANGLES = {
    (-1, 0): 0,     # moved up    -> icon points up
    (0, -1): 90,    # moved left  -> icon points left
    (1, 0): 180,    # moved down  -> icon points down
    (0, 1): -90,    # moved right -> icon points right
}


def _plane_marker(angle_deg):
    return MarkerStyle(PLANE_PATH, transform=Affine2D().rotate_deg(angle_deg))


def render_frame(env, basemap_img=None, headings=None):
    """Draw one frame of the current environment state.

    basemap_img : optional RGBA array (from basemap.get_basemap) drawn behind
    the grid overlay -- obstacles/uncovered cells become semi-transparent so
    the real map shows through.
    headings : optional dict {agent: angle_degrees} controlling which way
    each drone's plane icon currently points (see HEADING_ANGLES / the
    tracking loop in demo_rollout). Defaults to pointing up if omitted.
    """
    extent = (0, env.grid_size, env.grid_size, 0)  # left, right, bottom, top

    grid = np.zeros((env.grid_size, env.grid_size), dtype=np.float32)
    for (r, c) in env.obstacles:
        grid[r, c] = -1
    for (r, c) in env.covered:
        if grid[r, c] != -1:
            grid[r, c] = 1

    plt.clf()
    ax = plt.gca()

    if basemap_img is not None:
        # real map underneath, stretched to fill the same coordinate space
        # as the grid so drone/cell positions line up with it
        ax.imshow(basemap_img, extent=extent)

        # build an RGBA overlay: uncovered cells are fully transparent so the
        # basemap shows through; obstacles/covered cells get a translucent
        # tint on top of it
        overlay = np.zeros((env.grid_size, env.grid_size, 4), dtype=np.float32)
        overlay[grid == -1] = (0.1, 0.1, 0.1, 0.55)   # obstacles: dark
        overlay[grid == 1] = (0.1, 0.8, 0.2, 0.4)     # covered: green
        ax.imshow(overlay, extent=extent)
    else:
        ax.imshow(grid, cmap="Greens", vmin=-1, vmax=1, extent=extent)

    for agent, pos in env.positions.items():
        x, y = pos[1] + 0.5, pos[0] + 0.5
        angle = headings.get(agent, 0) if headings else 0
        ax.plot(x, y, marker=_plane_marker(angle), markersize=16,
                 markerfacecolor="red", markeredgecolor="black", markeredgewidth=1)

        # Small per-drone status label -- this is the kind of compact state
        # a real drone would broadcast over radio (see swarm.py's module
        # docstring: "what information actually gets shared"). Right now
        # just battery %, but any per-agent stat added to env.battery (or a
        # similar dict) can be shown here the same way.
        battery = getattr(env, "battery", {}).get(agent)
        if battery is not None:
            color = "black" if battery > 30 else "red"
            ax.annotate(
                f"{battery:.0f}%",
                xy=(x, y), xytext=(x + 0.4, y - 0.4),
                fontsize=7, color=color, fontweight="bold",
                ha="left",
            )

    total_free = env.grid_size ** 2 - len(env.obstacles)
    coverage_pct = len(env.covered) / total_free
    ax.set_title(f"Step {env.steps} | Coverage: {coverage_pct:.0%}")
    plt.pause(0.05)


def demo_rollout(n_drones=3, grid_size=10, n_obstacles=10, max_steps=50, seed=0, bbox=None,
                  use_basemap=False, satellite=False, strategy="bfs"):
    """Runs one episode and visualizes it live.

    Pass bbox=(west, south, east, north) to search a real place instead of a
    random grid -- obstacles get pulled from OpenStreetMap. e.g. a few blocks
    near UC Berkeley: bbox=(-122.259, 37.870, -122.253, 37.875)

    Pass use_basemap=True (requires bbox and the `contextily` package, no API
    key needed) to draw the grid on top of a real street/satellite image
    instead of a blank canvas. satellite=True switches the basemap tiles
    from street map to satellite imagery.

    strategy : "random", "potential_field", or "bfs"
        "random" -- each drone samples a uniformly random action every step,
        ignoring its observation entirely. No obstacle awareness, no
        coordination. Useful as a baseline to compare against.
        "potential_field" -- hand-coded rule-based movement (see
        movement_policy.py): repels from obstacles it can see, repels from
        nearby teammates so the swarm spreads out, and steers toward visible
        uncovered ground. Prone to occasional local-minimum oscillation
        (a known limitation of potential fields in general).
        "bfs" (default) -- real shortest-path search: each drone finds the
        nearest reachable uncovered cell via breadth-first search and walks
        directly toward it, with drones claiming disjoint target cells so
        they naturally split up the territory. Roughly 2x the coverage of
        potential_field in testing, and structurally can't oscillate the
        way a force-based policy can.
    """
    env = SwarmCoverageEnv(
        grid_size=grid_size,
        n_drones=n_drones,
        n_obstacles=n_obstacles,
        max_steps=max_steps,
        seed=seed,
        bbox=bbox,
    )
    observations, infos = env.reset(seed=seed)

    basemap_img = None
    if use_basemap:
        if bbox is None:
            raise ValueError("use_basemap=True requires a bbox")
        from basemap import get_basemap
        import contextily as cx
        source = cx.providers.Esri.WorldImagery if satellite else None
        basemap_img = get_basemap(bbox, source=source)

    plt.ion()
    plt.figure(figsize=(5, 5))
    headings = {agent: 0 for agent in env.agents}
    render_frame(env, basemap_img=basemap_img, headings=headings)

    # Both stateful policies keep per-drone memory across steps (previous
    # cell for potential fields, current target cell for BFS) -- they need
    # to be created once outside the loop, not recreated every step.
    pf_policy = PotentialFieldPolicy() if strategy == "potential_field" else None
    bfs_policy = BFSCoveragePolicy() if strategy == "bfs" else None

    while env.agents:
        prev_positions = env.positions

        # --- movement strategy lives HERE -----------------------------
        if strategy == "potential_field":
            actions = pf_policy.actions(env)
        elif strategy == "bfs":
            actions = bfs_policy.actions(env)
        elif strategy == "random":
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        else:
            raise ValueError(f"unknown strategy: {strategy!r}")
        # ----------------------------------------------------------------

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # point each plane icon toward wherever it actually just moved --
        # keeps the previous heading if the move was blocked (stayed put)
        for agent, pos in env.positions.items():
            prev = prev_positions[agent]
            delta = (pos[0] - prev[0], pos[1] - prev[1])
            if delta in HEADING_ANGLES:
                headings[agent] = HEADING_ANGLES[delta]

        render_frame(env, basemap_img=basemap_img, headings=headings)
        time.sleep(0.05)

    plt.ioff()
    plt.show()
    final_coverage = list(infos.values())[0]["coverage"]
    print(f"Final coverage: {final_coverage:.1%}")


# Downtown San Jose, CA -- a real, recognizable slice of the metro area
# (Milpitas/the Bay to the north, Alum Rock foothills to the east). Small
# enough for OpenStreetMap's Overpass API to answer quickly with full
# building-level obstacles.
SAN_JOSE_BBOX = (-121.94, 37.32, -121.87, 37.37)  # west, south, east, north


def demo_san_jose(grid_size=25, n_drones=4, max_steps=120, seed=0, satellite=False,
                   strategy="bfs"):
    """Convenience wrapper: swarm search over real downtown San Jose, with
    real building obstacles and a basemap underneath the plane icons.
    Defaults to the BFS strategy for the best coverage (see demo_rollout's
    docstring for a comparison of strategies)."""
    demo_rollout(
        grid_size=grid_size,
        n_drones=n_drones,
        max_steps=max_steps,
        seed=seed,
        bbox=SAN_JOSE_BBOX,
        use_basemap=True,
        satellite=satellite,
        strategy=strategy,
    )


if __name__ == "__main__":
    demo_rollout()

    # To search a real place instead of a random grid, pass a bbox pulled
    # from OpenStreetMap obstacles (buildings/water/forest):
    # demo_rollout(grid_size=20, n_drones=3, max_steps=80,
    #              bbox=(-122.259, 37.870, -122.253, 37.875))

    # Same, but drawn on top of a real basemap image (no API key needed):
    # demo_rollout(grid_size=20, n_drones=3, max_steps=80,
    #              bbox=(-122.259, 37.870, -122.253, 37.875),
    #              use_basemap=True, satellite=True)

    # Matches the downtown San Jose map referenced in chat -- real buildings
    # as obstacles, real streets/imagery underneath, plane icons moving live:
    # demo_san_jose()
